import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np

# Mask 工具
from animatediff.utils.mask_util import AutoMaskGenerator

# 輔助函式：將生成的 Video Tensor (0~1) 轉回 VAE Latents
def encode_video_to_latents(video_tensor, vae):
    # video_tensor: [b, c, f, h, w], value range [0, 1]
    video_length = video_tensor.shape[2]
    
    # Rearrange to [b*f, c, h, w] for VAE
    x = rearrange(video_tensor, "b c f h w -> (b f) c h w")
    
    # Normalize to [-1, 1]
    x = 2.0 * x - 1.0
    
    # Encode
    latents = vae.encode(x).latent_dist.sample()
    latents = latents * 0.18215
    
    # Rearrange back to [b, c, f, h, w]
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    return latents

# 輔助函式：執行部分去噪 (SDEdit / Img2Img loop)
@torch.no_grad()
def run_denoising_loop(pipeline, prompt, n_prompt, latents, guidance_scale, video_length):
    """
    執行 Pipeline 的去噪過程 (支援從中間 Step 開始)。
    """
    vae = pipeline.vae
    device = pipeline.device
    scheduler = pipeline.scheduler
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    
    # 1. Encode Prompts
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    
    uncond_input = tokenizer(n_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    # 2. Denoising Loop
    # 注意：這裡直接使用 scheduler.timesteps，這些 steps 必須在外部設定好
    timesteps = scheduler.timesteps
    
    current_latents = latents.clone()
    
    for t in tqdm(timesteps, desc=f"Gen ({prompt[:10]}...):"):
        # Expand latents for CFG
        latent_model_input = torch.cat([current_latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=context,
            down_block_additional_residuals=None,
            mid_block_additional_residual=None,
        ).sample
        
        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Step
        current_latents = scheduler.step(noise_pred, t, current_latents).prev_sample

    # 3. Decode Latents
    latents = 1 / 0.18215 * current_latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = vae.decode(latents).sample
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    
    video = (video / 2 + 0.5).clamp(0, 1)
    return video

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []

    # 初始化控制器
    auto_masker = AutoMaskGenerator()

    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    # --- 參數設定 ---
    # Final Denoising Strength (Step 5)
    denoising_strength = getattr(args, "strength", 0.65)
    
    # [NEW] FG Init Strength (Step 1b): FG 生成時要破壞多少 BG 的結構
    # 建議 0.85 - 0.95。
    # 如果太低(如 0.6)，FG 會長得像 BG；如果 1.0，則完全隨機(等於原本的方法)
    strength_fg_init = getattr(args, "strength_fg_init", 0.85)

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()

        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=None,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        prompts      = model_config.prompt
        n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        config[model_idx].random_seed = []
        
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            
            if "|" in prompt:
                bg_prompt, fg_prompt = [p.strip() for p in prompt.split("|", 1)]
                combined_prompt_str = f"{bg_prompt}, {fg_prompt}"
                print(f"\n[Info] BG: {bg_prompt} | FG: {fg_prompt}")
            else:
                bg_prompt = "background" 
                fg_prompt = prompt
                combined_prompt_str = prompt

            if random_seed != -1: torch.manual_seed(random_seed); current_seed = random_seed
            else: torch.seed(); current_seed = torch.initial_seed()
            config[model_idx].random_seed.append(current_seed)
            
            # ==========================================
            # Step 1a: Generate BG Draft
            # ==========================================
            print(f"--- Step 1a: Generating BG Draft ---")
            torch.manual_seed(current_seed)
            bg_draft = pipeline(
                prompt = bg_prompt,
                negative_prompt = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale = model_config.guidance_scale,
                width = model_config.W,
                height = model_config.H,
                video_length = model_config.L,
            ).videos
            save_videos_grid(bg_draft, f"{savedir}/sample/{sample_idx}-1a_bg_draft.gif")
            
            # ==========================================
            # [MODIFIED] Step 1b: Generate FG Draft (From Noisy BG)
            # ==========================================
            print(f"--- Step 1b: Generating FG Draft (Initialized from BG) ---")
            
            # 1. Encode BG to Latents
            bg_latents = encode_video_to_latents(bg_draft.cuda(), vae)
            
            # 2. Add Noise to BG (create initial state for FG)
            pipeline.scheduler.set_timesteps(model_config.steps)
            t_start_fg_idx = int(len(pipeline.scheduler.timesteps) * (1 - strength_fg_init))
            t_start_fg = pipeline.scheduler.timesteps[t_start_fg_idx]
            
            torch.manual_seed(current_seed) # 保持 Seed 一致
            noise = torch.randn_like(bg_latents)
            noisy_bg_latents = pipeline.scheduler.add_noise(bg_latents, noise, t_start_fg)
            
            # 3. Generate FG using Denoising Loop
            # 設定 Scheduler 只跑剩下的部分
            full_timesteps = pipeline.scheduler.timesteps
            pipeline.scheduler.timesteps = full_timesteps[t_start_fg_idx:]
            
            torch.manual_seed(current_seed)
            fg_draft = run_denoising_loop(
                pipeline,
                prompt=fg_prompt, # 使用前景 Prompt
                n_prompt=n_prompt,
                latents=noisy_bg_latents, # 從加噪後的 BG 開始
                guidance_scale=model_config.guidance_scale,
                video_length=model_config.L
            )
            save_videos_grid(fg_draft.cpu(), f"{savedir}/sample/{sample_idx}-1b_fg_draft.gif")

            # ==========================================
            # Step 2: Generate Mask from FG
            # ==========================================
            print(f"--- Step 2: Generating Mask from FG ---")
            generated_mask = auto_masker.generate_mask_from_video_tensor(fg_draft)
            save_videos_grid(generated_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1).cpu(), f"{savedir}/sample/{sample_idx}-2_mask.gif")

            # ==========================================
            # Step 3: Composite (Collage)
            # ==========================================
            print(f"--- Step 3: Compositing (Collage) ---")
            mask_expanded = generated_mask.unsqueeze(1) # [b, 1, f, h, w]
            
            fg_draft = fg_draft.to(pipeline.device)
            bg_draft = bg_draft.to(pipeline.device)
            mask_expanded = mask_expanded.to(pipeline.device)
            
            composite_video = fg_draft * mask_expanded + bg_draft * (1.0 - mask_expanded)
            save_videos_grid(composite_video.cpu(), f"{savedir}/sample/{sample_idx}-3_composite.gif")

            # ==========================================
            # Step 4: Add Noise to Composite
            # ==========================================
            print(f"--- Step 4: Adding Noise to Composite ---")
            init_latents = encode_video_to_latents(composite_video, vae)
            
            pipeline.scheduler.set_timesteps(model_config.steps)
            t_start_index = int(len(pipeline.scheduler.timesteps) * (1 - denoising_strength))
            t_start = pipeline.scheduler.timesteps[t_start_index]
            
            torch.manual_seed(current_seed)
            noise = torch.randn_like(init_latents)
            noisy_latents = pipeline.scheduler.add_noise(init_latents, noise, t_start)
            
            # Visualize Noisy Input
            with torch.no_grad():
                temp_latents = 1 / 0.18215 * noisy_latents
                temp_latents = rearrange(temp_latents, "b c f h w -> (b f) c h w")
                noisy_vis = vae.decode(temp_latents).sample
                noisy_vis = rearrange(noisy_vis, "(b f) c h w -> b c f h w", f=model_config.L)
                noisy_vis = (noisy_vis / 2 + 0.5).clamp(0, 1)
                save_videos_grid(noisy_vis.cpu(), f"{savedir}/sample/{sample_idx}-4_noisy_input.gif")

            # ==========================================
            # Step 5: Final Generation
            # ==========================================
            print(f"--- Step 5: Final Generation (Harmonizing) ---")
            pipeline.scheduler.timesteps = full_timesteps[t_start_index:]
            
            torch.manual_seed(current_seed)
            final_output = run_denoising_loop(
                pipeline,
                prompt=combined_prompt_str,
                n_prompt=n_prompt,
                latents=noisy_latents,
                guidance_scale=model_config.guidance_scale,
                video_length=model_config.L
            )
            
            save_videos_grid(final_output.cpu(), f"{savedir}/sample/{sample_idx}-5_final_output.gif")
            print(f"[Success] Sample {sample_idx} completed.")
            
            samples.append(final_output.cpu())
            sample_idx += 1

    if len(samples) > 0:
        samples = torch.concat(samples)
        save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--H", type=int, default=256)
    
    # Final Fusion Strength
    parser.add_argument("--strength", type=float, default=0.1, help="Final fusion strength.")
    
    # FG Generation Strength (from BG)
    parser.add_argument("--strength_fg_init", type=float, default=0.8, help="Strength for FG generation from BG. Higher = less like BG.")
    
    parser.add_argument("--without-xformers", action="store_true")
    args = parser.parse_args()
    main(args)