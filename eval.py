import torch
import numpy as np
from PIL import Image, ImageSequence
from transformers import CLIPProcessor, CLIPModel
import os

MODEL_NAME = "openai/clip-vit-large-patch14"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP model: {MODEL_NAME} on {device}...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def extract_frames_from_gif(gif_path, background_color=(0, 0, 0)):
    if not os.path.exists(gif_path):
        print(f"Error: File not found - {gif_path}")
        return []

    print(f"Processing: {gif_path} ...")
    gif = Image.open(gif_path)
    
    frames = []
    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert("RGBA")
        
        bg = Image.new("RGBA", frame.size, background_color + (255,))
        merged = Image.alpha_composite(bg, frame)
        
        frames.append(merged.convert("RGB"))
        
    print(f"  -> Extracted {len(frames)} frames.")
    return frames

def calculate_metrics(frames, prompt):
    if not frames:
        return 0.0, 0.0

    inputs = processor(text=[prompt], images=frames, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    image_embeds = outputs.image_embeds
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    
    text_embeds = outputs.text_embeds
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    text_similarities = (text_embeds @ image_embeds.T).squeeze()
    text_score = text_similarities.mean().item() * 100

    if len(frames) > 1:
        cosine_sim = torch.nn.functional.cosine_similarity(
            image_embeds[:-1], image_embeds[1:], dim=-1
        )
        consistency_score = cosine_sim.mean().item() * 100
    else:
        consistency_score = 100.0

    return text_score, consistency_score

if __name__ == "__main__":
    gif_1_path = "samples/layer1-2025-12-23T11-33-24/sample/1-1girl, looking at viewer, blurry background, upper body, contemporary, dress, best quality, masterpiece-4-composite.gif"      # 你的 GIF
    gif_2_path = "samples/layer1-2025-12-23T11-36-29/sample/0-1girl,-looking-at-viewer,-blurry-background,-upper-body,-contemporary,-dress,.gif"     # 對照組 GIF (如果沒有可以註解掉)
    
    target_prompt = "1girl, looking at viewer, blurry background, upper body, contemporary, dress, best quality, masterpiece, blurry background, garden" # 當時生成用的 Prompt

    frames_1 = extract_frames_from_gif(gif_1_path, background_color=(0,0,0))
    if frames_1:
        score_text_1, score_smooth_1 = calculate_metrics(frames_1, target_prompt)
        print(f"\nResults for {gif_1_path}:")
        print(f"  CLIP Text Score:       {score_text_1:.4f}")
        print(f"  CLIP Consistency:      {score_smooth_1:.4f}")

    if os.path.exists(gif_2_path):
        frames_2 = extract_frames_from_gif(gif_2_path, background_color=(0,0,0))
        if frames_2:
            score_text_2, score_smooth_2 = calculate_metrics(frames_2, target_prompt)
            print(f"\nResults for {gif_2_path}:")
            print(f"  CLIP Text Score:       {score_text_2:.4f}")
            print(f"  CLIP Consistency:      {score_smooth_2:.4f}")
            
            print("\n--- Comparison ---")
            print(f"Text Score Improvement: {(score_text_1 - score_text_2):.4f}")