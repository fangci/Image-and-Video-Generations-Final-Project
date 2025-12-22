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
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights, auto_download
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np

# [新增] 引入 rembg
from rembg import remove

# 引入 SharedAttentionState 確保環境初始化 (Optional)
from animatediff.models.attention import SharedAttentionState

# [新增] 輔助函數：建立棋盤格背景 (Checkerboard pattern)
def create_checkerboard_pattern(h, w, square_size=8):
    # 建立一個 HxW 的座標網格
    rows = torch.arange(h).unsqueeze(1).repeat(1, w)
    cols = torch.arange(w).unsqueeze(0).repeat(h, 1)
    
    # 計算棋盤格紋理 (0 或 1 交錯)
    pattern = ((rows // square_size) + (cols // square_size)) % 2
    
    # [修改] 調整顏色計算公式以變更淺灰色
    # 原本: pattern.float() * 0.5 + 0.5  => 結果是 0.5 (中灰) 和 1.0 (純白)
    # 現在: pattern.float() * 0.15 + 0.85 => 結果是 0.85 (淺灰) 和 1.0 (純白)
    # 您可以微調 0.85 這個基底值，越接近 1.0 灰色就越淺
    checkerboard = pattern.float() * 0.15 + 0.85
    
    # 擴展為 RGB 三通道 (3, H, W)
    return checkerboard.unsqueeze(0).repeat(3, 1, 1)

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []

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

        # load controlnet model (保持原邏輯)
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            auto_download(model_config.controlnet_path, is_dreambooth_lora=False)
            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name}
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

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
            
            # Seed Setting
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            current_seed = torch.initial_seed()
            config[model_idx].random_seed.append(current_seed)
            print(f"current seed: {current_seed}")

            # =========================================================
            # [修改] 邏輯判定：是否需要進行圖層式生成 (Multi-Stage)
            # 格式約定: "Foreground Prompt :: Background Prompt"
            # =========================================================
            if " :: " in prompt:
                print(f"Sampling with Layered Generation Mode...")
                fg_prompt, bg_prompt = prompt.split(" :: ")
                
                # 1. 準備共用的 Latents (保證動態一致)
                # Manually create latents to reuse
                init_latents = torch.randn(
                    (1, 4, model_config.L, model_config.H // 8, model_config.W // 8),
                    generator=torch.Generator("cuda").manual_seed(current_seed),
                    device="cuda", dtype=unet.dtype # 確保 dtype 一致
                )

                # 2. Stage 1: Generate Background & Record
                print(f"  [Stage 1] Generating Background: {bg_prompt}")
                bg_sample = pipeline(
                    bg_prompt,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = model_config.W,
                    height              = model_config.H,
                    video_length        = model_config.L,
                    latents             = init_latents.clone(), # Clone
                    attention_op_mode   = "record",             # <--- RECORD MODE
                    
                    controlnet_images = controlnet_images,
                    controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
                ).videos

                # 3. Stage 2: Generate Foreground & Inject
                # 強制加入黑色背景 Prompt 以便去背
                fg_prompt_final = f"{fg_prompt}, {bg_prompt} background"
                print(f"  [Stage 2] Generating Foreground: {fg_prompt_final}")
                
                fg_sample = pipeline(
                    fg_prompt_final,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = model_config.W,
                    height              = model_config.H,
                    video_length        = model_config.L,
                    latents             = init_latents.clone(), # Reuse Latents
                    attention_op_mode   = "inject",             # <--- INJECT MODE
                    
                    controlnet_images = controlnet_images,
                    controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
                ).videos

                # 4. Post-Process: Alpha Generation (使用 rembg)
                print("  [Stage 3] Generating Alpha Mask with Rembg...")
                fg_video = fg_sample[0] # (3, L, H, W), value [0, 1]
                
                alpha_masks_list = []
                
                # rembg 只能處理單張圖片，所以我們對 L (Frames) 進行迴圈
                for f in range(fg_video.shape[1]):
                    # 4.1 轉為 PIL Image (需要先轉到 CPU, 調整維度, 轉為 uint8 0-255)
                    # frame shape: (3, H, W) -> (H, W, 3)
                    frame_tensor = fg_video[:, f, :, :]
                    frame_np = frame_tensor.cpu().permute(1, 2, 0).numpy()
                    frame_np = (frame_np * 255).astype(np.uint8)
                    frame_pil = Image.fromarray(frame_np)
                    
                    # 4.2 使用 rembg 去背
                    # remove 會回傳 RGBA 圖片
                    output_pil = remove(frame_pil)
                    
                    # 4.3 提取 Alpha Channel
                    mask_pil = output_pil.split()[-1] # 取最後一個通道 (A)
                    
                    # 4.4 轉回 Tensor
                    # ToTensor() 會自動將 0-255 轉回 0.0-1.0，形狀變成 (1, H, W)
                    mask_tensor = transforms.ToTensor()(mask_pil)
                    alpha_masks_list.append(mask_tensor)
                
                # 4.5 堆疊回 Video Tensor (1, L, H, W)
                # stack on dim=1 (Time dimension)
                alpha_mask = torch.stack(alpha_masks_list, dim=1).to(fg_video.device)

                # 合成預覽: BG * (1-Alpha) + FG * Alpha
                bg_video = bg_sample[0]
                comp_video = bg_video * (1 - alpha_mask) + fg_video * alpha_mask
                
                # 最終 Sample 列表加入合成結果
                sample = comp_video.unsqueeze(0) 
                
                # 額外儲存 debug 網格
                debug_grid = torch.cat([bg_video, fg_video, alpha_mask.repeat(3,1,1,1), comp_video], dim=3) # 橫向拼接
                save_videos_grid(debug_grid.unsqueeze(0), f"{savedir}/sample/{sample_idx}-{fg_prompt}-layered.gif")

                # =========================================================
                # [新增] 單獨儲存各個組件 (Component Saving)
                # =========================================================
                print(f"  [Stage 4] Saving individual components...")
                base_name = f"{savedir}/sample/{sample_idx}-{fg_prompt}"
                
                # 1. Background
                save_videos_grid(bg_video.unsqueeze(0), f"{base_name}-1-bg.gif")
                
                # 2. Foreground (純前景，不帶透明，背景是黑的)
                save_videos_grid(fg_video.unsqueeze(0), f"{base_name}-2-fg.gif")
                
                # 3. Final Composite Output
                save_videos_grid(comp_video.unsqueeze(0), f"{base_name}-4-composite.gif")
                
                # 4. Foreground * Mask (Transparent representation)
                # 4a. 使用棋盤格背景 (Checkerboard)
                # 製作棋盤格背景, 原始形狀: (3, H, W)
                checkerboard = create_checkerboard_pattern(model_config.H, model_config.W).to(fg_video.device)
                
                # [FIX] 修正維度擴展邏輯
                # 目標形狀: (Channel, Frames, Height, Width) -> (3, L, H, W)
                # 1. unsqueeze(1) -> (3, 1, H, W)  (在 Channel 後面增加 Time 維度)
                # 2. repeat -> (3, L, H, W)       (複製 Time 維度)
                checkerboard = checkerboard.unsqueeze(1).repeat(1, model_config.L, 1, 1)
                
                # 合成: Checkerboard * (1 - Alpha) + FG * Alpha
                # 現在 checkerboard, fg_video, alpha_mask 的維度都對齊了 (C, L, H, W)
                fg_masked_grid = checkerboard * (1 - alpha_mask) + fg_video * alpha_mask
                
                save_videos_grid(fg_masked_grid.unsqueeze(0), f"{base_name}-3-fg_masked_grid.gif")
                
                # 4b. (Optional) 嘗試儲存帶有 RGBA 的 GIF 
                # (注意: save_videos_grid 預設可能只支援 RGB，所以這裡手動處理)
                try:
                    # 組合 RGBA: (3, L, H, W) + (1, L, H, W) -> (4, L, H, W)
                    fg_rgba = torch.cat([fg_video, alpha_mask], dim=0) # (4, L, H, W)
                    
                    # 轉為 List of PIL Images
                    fg_rgba = rearrange(fg_rgba, "c f h w -> f c h w")
                    pil_frames = []
                    for f in range(fg_rgba.shape[0]):
                        frame_np = fg_rgba[f].cpu().permute(1, 2, 0).numpy()
                        frame_np = (frame_np * 255).astype(np.uint8)
                        pil_frames.append(Image.fromarray(frame_np, mode="RGBA"))
                        
                    # 儲存 GIF (保留透明度)
                    pil_frames[0].save(
                        f"{base_name}-3-fg_masked_rgba.gif",
                        save_all=True,
                        append_images=pil_frames[1:],
                        duration=100, # 假設 10fps
                        loop=0,
                        disposal=2 # 清除上一幀，這對透明 GIF 很重要
                    )
                except Exception as e:
                    print(f"Failed to save RGBA GIF: {e}")

            else:
                # =========================================================
                # 一般生成模式 (Original Logic)
                # =========================================================
                print(f"sampling {prompt} ...")
                sample = pipeline(
                    prompt,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = model_config.W,
                    height              = model_config.H,
                    video_length        = model_config.L,

                    controlnet_images = controlnet_images,
                    controlnet_image_index = model_config.get("controlnet_image_indexs", [0]),
                ).videos
                
                prompt_safe = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt_safe}.gif")
                print(f"save to {savedir}/sample/{prompt_safe}.gif")

            samples.append(sample)
            sample_idx += 1

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