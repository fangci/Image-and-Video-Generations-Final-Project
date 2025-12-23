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

def encode_video_to_latents(video_tensor, vae):
    video_length = video_tensor.shape[2]
    x = rearrange(video_tensor, "b c f h w -> (b f) c h w")
    x = 2.0 * x - 1.0
    latents = vae.encode(x).latent_dist.sample()
    latents = latents * 0.18215
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    return latents

# [MODIFIED] 修改回傳值：同時回傳 Video (給人看) 和 Latents (給機器接力)
@torch.no_grad()
def run_denoising_loop(pipeline, prompt, n_prompt, latents, guidance_scale, video_length):
    vae = pipeline.vae
    device = pipeline.device
    scheduler = pipeline.scheduler
    unet = pipeline.unet
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    uncond_input = tokenizer(n_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    timesteps = scheduler.timesteps
    current_latents = latents.clone()
    
    for t in tqdm(timesteps, desc=f"Gen: {prompt[:10]}..."):
        latent_model_input = torch.cat([current_latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=context,
        ).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        current_latents = scheduler.step(noise_pred, t, current_latents).prev_sample

    # [IMPORTANT] 保留乾淨的 Latents 供下一步驟使用
    # 注意：scheduler.step 出來的 latents 已經是 "Clean Latents" (Pred Original Sample)
    # 我們需要保留這個變數，不要做任何 scaling (因為 add_noise 需要 raw latents)
    clean_latents_for_next_step = current_latents.clone()

    # Decode for visualization
    # VAE Decode 需要 scale 放大
    latents_decoded = 1 / 0.18215 * current_latents
    latents_decoded = rearrange(latents_decoded, "b c f h w -> (b f) c h w")
    video = vae.decode(latents_decoded).sample
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    
    return video, clean_latents_for_next_step

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []

    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    strength_fg = getattr(args, "strength_fg", 0.75)
    strength_refine = getattr(args, "strength_refine", 0.4)

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
            model_config.get("motion_module", ""),
            model_config.get("motion_module_lora_configs", []),
            model_config.get("adapter_lora_path", ""),
            model_config.get("adapter_lora_scale", 1.0),
            model_config.get("dreambooth_path", ""),
            model_config.get("lora_model_path", ""),
            model_config.get("lora_alpha", 0.8),
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
            else:
                bg_prompt = "background" 
                fg_prompt = prompt
                combined_prompt_str = prompt

            if random_seed != -1: torch.manual_seed(random_seed); current_seed = random_seed
            else: torch.seed(); current_seed = torch.initial_seed()
            config[model_idx].random_seed.append(current_seed)
            
            # Step 1: Clean Background
            print(f"--- Step 1: Generating Clean Background ---")
            torch.manual_seed(current_seed)
            bg_output = pipeline(
                prompt = bg_prompt,
                negative_prompt = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale = model_config.guidance_scale,
                width = model_config.W,
                height = model_config.H,
                video_length = model_config.L,
            ).videos
            save_videos_grid(bg_output, f"{savedir}/sample/{sample_idx}-1_background.gif")
            
            # Step 2 & 3: Add FG
            print(f"--- Step 2 & 3: Adding FG to BG ---")
            bg_latents = encode_video_to_latents(bg_output.cuda(), vae)
            
            pipeline.scheduler.set_timesteps(model_config.steps)
            t_start_fg_idx = int(len(pipeline.scheduler.timesteps) * (1 - strength_fg))
            t_start_fg = pipeline.scheduler.timesteps[t_start_fg_idx]
            
            torch.manual_seed(current_seed)
            noise = torch.randn_like(bg_latents)
            noisy_bg_latents = pipeline.scheduler.add_noise(bg_latents, noise, t_start_fg)
            
            full_timesteps = pipeline.scheduler.timesteps
            pipeline.scheduler.timesteps = full_timesteps[t_start_fg_idx:]
            
            torch.manual_seed(current_seed)
            # [MODIFIED] 接收 latents
            fg_on_bg_video, fg_on_bg_latents = run_denoising_loop(
                pipeline,
                prompt=fg_prompt,
                n_prompt=n_prompt,
                latents=noisy_bg_latents,
                guidance_scale=model_config.guidance_scale,
                video_length=model_config.L
            )
            save_videos_grid(fg_on_bg_video.cpu(), f"{savedir}/sample/{sample_idx}-3_fg_added.gif")

            # Step 4 & 5: Refine
            print(f"--- Step 4 & 5: Refining (Latent Space) ---")

            # [FIXED] 這裡直接使用上一步傳回來的 Latents，不再 encode_video_to_latents
            current_latents = fg_on_bg_latents
            
            pipeline.scheduler.set_timesteps(model_config.steps)
            t_start_refine_idx = int(len(pipeline.scheduler.timesteps) * (1 - strength_refine))
            t_start_refine = pipeline.scheduler.timesteps[t_start_refine_idx]
            
            torch.manual_seed(current_seed)
            noise_refine = torch.randn_like(current_latents)
            # 使用 Clean Latents 直接加躁
            noisy_refine_latents = pipeline.scheduler.add_noise(current_latents, noise_refine, t_start_refine)
            
            pipeline.scheduler.timesteps = full_timesteps[t_start_refine_idx:]
            
            torch.manual_seed(current_seed)
            # 這裡我們不需要最後的 latents 了，所以只取 video
            final_output, _ = run_denoising_loop(
                pipeline,
                prompt=combined_prompt_str,
                n_prompt=n_prompt,
                latents=noisy_refine_latents,
                guidance_scale=model_config.guidance_scale,
                video_length=model_config.L
            )
            
            save_videos_grid(final_output.cpu(), f"{savedir}/sample/{sample_idx}-5_final.gif")
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
    parser.add_argument("--strength_fg", type=float, default=0.75)
    parser.add_argument("--strength_refine", type=float, default=0.4)
    parser.add_argument("--without-xformers", action="store_true")
    args = parser.parse_args()
    main(args)