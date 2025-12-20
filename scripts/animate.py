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

# 引入 Mask 工具 (請確保 animatediff/utils/mask_util.py 已經建立)
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

    # 初始化 Mask 工具
    mask_controller = AttentionMaskController.get_instance()
    # 只有在需要 layer separation 時才載入 rembg
    auto_masker = AutoMaskGenerator()

    # create validation pipeline
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

        controlnet = None 

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
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
            
            # 處理 Prompt 分隔
            if "|" in prompt:
                bg_prompt, fg_prompt = [p.strip() for p in prompt.split("|", 1)]
                combined_prompt_str = f"{bg_prompt}, {fg_prompt}"
                print(f"\n[Info] Detected Split Prompt:")
                print(f"  BG: {bg_prompt}")
                print(f"  FG: {fg_prompt}")
            else:
                print(f"\n[Warning] No '|' separator found. Using full prompt.")
                bg_prompt = "background" 
                fg_prompt = prompt
                combined_prompt_str = prompt

            # 設定 Seed
            if random_seed != -1: 
                torch.manual_seed(random_seed)
                current_seed = random_seed
            else: 
                torch.seed()
                current_seed = torch.initial_seed()
            
            config[model_idx].random_seed.append(current_seed)
            print(f"Sampling seed: {current_seed}")

            # ==========================================
            # Phase 1: Draft Generation
            # ==========================================
            print(f"--- Step 1/3: Generating Draft Video ---")
            mask_controller.active = False
            
            draft_output = pipeline(
                prompt = combined_prompt_str, # 這裡傳入正常的 string
                negative_prompt     = n_prompt,
                num_inference_steps = model_config.steps,
                guidance_scale      = model_config.guidance_scale,
                width               = model_config.W,
                height              = model_config.H,
                video_length        = model_config.L,
            ).videos

            save_videos_grid(draft_output, f"{savedir}/sample/{sample_idx}-1_draft.gif")
            
            # ==========================================
            # Phase 2: Auto Mask Generation
            # ==========================================
            print(f"--- Step 2/3: Generating Mask ---")
            if "|" in prompt:
                generated_mask = auto_masker.generate_mask_from_video_tensor(draft_output)
                mask_vis = generated_mask.unsqueeze(1).repeat(1, 3, 1, 1, 1)
                save_videos_grid(mask_vis, f"{savedir}/sample/{sample_idx}-2_mask.gif")
            else:
                print("Skipping mask generation.")
                generated_mask = None

            # ==========================================
            # Phase 3: Refined Generation (Fix applied)
            # ==========================================
            final_sample = draft_output 

            if generated_mask is not None and "|" in prompt:
                print(f"--- Step 3/3: Region-Aware Refinement ---")
                
                # 1. 啟動 Controller
                mask_controller.set_masks(generated_mask)
                
                # 2. 準備 Embeddings (BG + FG)
                with torch.no_grad():
                    def encode_text(text):
                        tokens = tokenizer(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
                        return text_encoder(tokens)[0]

                    bg_embeds = encode_text(bg_prompt)
                    fg_embeds = encode_text(fg_prompt)
                    neg_embeds = encode_text(n_prompt)

                    combined_embeds = torch.cat([bg_embeds, fg_embeds], dim=1) # [1, 154, Dim]
                    combined_neg_embeds = torch.cat([neg_embeds, neg_embeds], dim=1) # [1, 154, Dim]

                # 3. [FIX] 暫時替換 pipeline._encode_prompt
                # 因為 AnimateDiff pipeline 寫死了只接受 string prompt，我們用這招繞過它
                original_encode_prompt = pipeline._encode_prompt

                def custom_encode_override(prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
                    # 直接回傳我們準備好的 Embeddings
                    # 格式必須是 [Unconditional, Conditional]
                    batch_neg = combined_neg_embeds.repeat(num_videos_per_prompt, 1, 1)
                    batch_pos = combined_embeds.repeat(num_videos_per_prompt, 1, 1)
                    return torch.cat([batch_neg, batch_pos])

                # 掛載 Override
                pipeline._encode_prompt = custom_encode_override

                # 重置 Seed
                torch.manual_seed(current_seed)
                
                # 4. 執行 Pipeline (傳入 Dummy Prompt)
                final_sample = pipeline(
                    prompt              = "DUMMY_PROMPT_IGNORED", # 這裡隨便傳，因為會被 Override 忽略
                    negative_prompt     = None,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = model_config.W,
                    height              = model_config.H,
                    video_length        = model_config.L,
                ).videos

                # 5. 還原原始函數 (重要！)
                pipeline._encode_prompt = original_encode_prompt
                
                save_videos_grid(final_sample, f"{savedir}/sample/{sample_idx}-3_refined.gif")
            else:
                print("Skipping refinement.")

            samples.append(final_sample)
            
            short_prompt = "-".join((prompt.replace("/", "").replace("|", "").split(" ")[:10]))
            print(f"Saved sequence to {savedir}/sample/{sample_idx}-*.gif")
            
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
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)