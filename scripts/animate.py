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
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, load_weights, auto_download
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np

# 引入 Mask & KV 工具 (請確保 attention.py 已更新)
from animatediff.models.attention import KVCacheController
from animatediff.utils.mask_util import AttentionMaskController, AutoMaskGenerator

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
    mask_controller = AttentionMaskController.get_instance()
    kv_controller = KVCacheController.get_instance() # [NEW]
    auto_masker = AutoMaskGenerator()

    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

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
            # Phase 1: Draft Generation (Store KV)
            # ==========================================
            print(f"--- Step 1: Generating Draft Video (Storing KV) ---")
            mask_controller.active = False
            
            # [KV] 開啟 Store 模式
            kv_controller.clear()
            kv_controller.store_mode = True
            
            torch.manual_seed(current_seed)
            draft_output = pipeline(
                prompt = combined_prompt_str,
                negative_prompt = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale = model_config.guidance_scale,
                width = model_config.W,
                height = model_config.H,
                video_length = model_config.L,
            ).videos
            
            # [KV] 關閉 Store 模式
            kv_controller.store_mode = False
            
            save_videos_grid(draft_output, f"{savedir}/sample/{sample_idx}-1_draft.gif")
            
            # ==========================================
            # Phase 2: Mask Generation
            # ==========================================
            print(f"--- Step 2: Generating Mask ---")
            generated_mask = None
            if "|" in prompt:
                generated_mask = auto_masker.generate_mask_from_video_tensor(draft_output)
                save_videos_grid(generated_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1), f"{savedir}/sample/{sample_idx}-2_mask.gif")

            # ==========================================
            # Phase 3: Background Layer (KV Injection)
            # ==========================================
            bg_output = None
            if generated_mask is not None:
                print(f"--- Step 3: Generating Background (KV Injected) ---")
                mask_controller.active = False
                
                # [KV] 開啟 Inject 模式，並重置計數器
                kv_controller.inject_mode = True
                kv_controller.reset_counters()
                
                # 使用純雜訊 (latents=None 會自動生成) + 相同 Seed
                torch.manual_seed(current_seed)
                
                bg_output = pipeline(
                    prompt              = bg_prompt, # 只用背景 Prompt
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = model_config.W,
                    height              = model_config.H,
                    video_length        = model_config.L,
                ).videos
                
                save_videos_grid(bg_output, f"{savedir}/sample/{sample_idx}-3_layer_bg.gif")

            # ==========================================
            # Phase 4: Foreground Layer (KV Injection + Region Aware)
            # ==========================================
            if generated_mask is not None:
                print(f"--- Step 4: Generating Foreground (KV Injected) ---")
                
                # [Mask] 開啟 Region Aware，確保背景雜訊不干擾前景
                mask_controller.set_masks(generated_mask)
                
                # [KV] 重置計數器 (繼續 Inject)
                kv_controller.reset_counters()
                
                # 準備 Embeddings (同前)
                with torch.no_grad():
                    def encode_text(text):
                        tokens = tokenizer(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
                        return text_encoder(tokens)[0]
                    bg_embeds = encode_text(bg_prompt)
                    fg_embeds = encode_text(fg_prompt)
                    neg_embeds = encode_text(n_prompt)
                    combined_embeds = torch.cat([bg_embeds, fg_embeds], dim=1) 
                    combined_neg_embeds = torch.cat([neg_embeds, neg_embeds], dim=1)

                original_encode_prompt = pipeline._encode_prompt
                def custom_encode_override(prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
                    batch_neg = combined_neg_embeds.repeat(num_videos_per_prompt, 1, 1)
                    batch_pos = combined_embeds.repeat(num_videos_per_prompt, 1, 1)
                    return torch.cat([batch_neg, batch_pos])
                pipeline._encode_prompt = custom_encode_override

                # 生成
                torch.manual_seed(current_seed)
                fg_refined_context = pipeline(
                    prompt              = "DUMMY", 
                    negative_prompt     = None,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = model_config.W,
                    height              = model_config.H,
                    video_length        = model_config.L,
                ).videos
                
                pipeline._encode_prompt = original_encode_prompt
                
                # [KV] 結束，關閉 Inject
                kv_controller.inject_mode = False

                # Extraction
                mask_expanded = generated_mask.unsqueeze(1)
                fg_layer = fg_refined_context * mask_expanded
                save_videos_grid(fg_layer, f"{savedir}/sample/{sample_idx}-4_layer_fg.gif")
                
                # Save
                layer_dir = f"{savedir}/layers/{sample_idx}"
                os.makedirs(f"{layer_dir}/fg", exist_ok=True)
                os.makedirs(f"{layer_dir}/bg", exist_ok=True)
                
                fg_np = rearrange(fg_refined_context[0], "c f h w -> f h w c").cpu().numpy()
                bg_np = rearrange(bg_output[0], "c f h w -> f h w c").cpu().numpy()
                mask_np = rearrange(generated_mask[0], "f h w -> f h w").cpu().numpy()
                
                fg_np = ((fg_np + 1) * 127.5).astype(np.uint8)
                bg_np = ((bg_np + 1) * 127.5).astype(np.uint8)
                mask_np = (mask_np * 255).astype(np.uint8)
                
                for f in range(model_config.L):
                    Image.fromarray(bg_np[f]).save(f"{layer_dir}/bg/{f:04d}.png")
                    r, g, b = fg_np[f, :, :, 0], fg_np[f, :, :, 1], fg_np[f, :, :, 2]
                    alpha = mask_np[f]
                    Image.fromarray(np.dstack((r, g, b, alpha))).save(f"{layer_dir}/fg/{f:04d}.png")
                    
                print(f"[Success] Layers saved to {layer_dir}")

            samples.append(draft_output)
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
    parser.add_argument("--without-xformers", action="store_true")
    args = parser.parse_args()
    main(args)