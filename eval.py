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

DOMAIN_PROMPTS = [
    "anime illustration, 2D anime style",
    "photorealistic photo, real world",
    "cinematic film still, movie lighting",
    "cartoon, western animation style",
    "digital art, concept art",
    "oil painting, traditional painting",
    "watercolor painting",
    "3D render, CGI, octane render",
    "pixel art",
    "sketch, pencil drawing",
    "ink illustration, line art",
]

def _to_np_rgb(img: Image.Image) -> np.ndarray:
    # (H,W,3), uint8
    return np.array(img.convert("RGB"), dtype=np.uint8)

def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    # rgb: uint8 [0,255]
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def richness_edge_density(rgb: np.ndarray) -> float:
    gray = _rgb_to_gray(rgb)
    # finite difference
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    # pad to same size
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
    mag = np.sqrt(gx * gx + gy * gy)

    edge_mask = mag > 15.0
    return float(edge_mask.mean())

def richness_colorfulness(rgb: np.ndarray) -> float:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)

    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

def richness_entropy(rgb: np.ndarray) -> float:
    gray = _rgb_to_gray(rgb).astype(np.uint8)
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-8)
    p = p[p > 0]
    ent = -np.sum(p * np.log2(p))
    return float(ent)

def compute_richness_score(frames: list) -> dict:
    if not frames:
        return {
            "edge_density_mean": 0.0,
            "colorfulness_mean": 0.0,
            "entropy_mean": 0.0,
            "richness_score_0_100": 0.0,
        }

    edges = []
    colors = []
    ents = []

    for im in frames:
        rgb = _to_np_rgb(im)
        edges.append(richness_edge_density(rgb))
        colors.append(richness_colorfulness(rgb))
        ents.append(richness_entropy(rgb))

    edge_m = float(np.mean(edges))
    color_m = float(np.mean(colors))
    ent_m = float(np.mean(ents))

    edge_n = np.clip(edge_m / 0.25, 0.0, 1.0)

    color_n = np.clip(color_m / 40.0, 0.0, 1.0)

    ent_n = np.clip(ent_m / 8.0, 0.0, 1.0)

    richness = (0.4 * edge_n + 0.3 * color_n + 0.3 * ent_n) * 100.0

    return {
        "edge_density_mean": edge_m,
        "colorfulness_mean": color_m,
        "entropy_mean": ent_m,
        "richness_score_0_100": float(richness),
    }


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


def calculate_metrics(frames, prompt, domain_prompts=DOMAIN_PROMPTS):
    if not frames:
        return (
            0.0, 0.0,
            "N/A", 0.0, 0.0,
            {"edge_density_mean": 0.0, "colorfulness_mean": 0.0, "entropy_mean": 0.0, "richness_score_0_100": 0.0},
        )

    inputs = processor(text=[prompt], images=frames, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    image_embeds = outputs.image_embeds
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

    text_embeds = outputs.text_embeds
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    text_similarities = (text_embeds @ image_embeds.T).squeeze()  # (num_frames,)
    text_score = text_similarities.mean().item() * 100

    if len(frames) > 1:
        cosine_sim = torch.nn.functional.cosine_similarity(
            image_embeds[:-1], image_embeds[1:], dim=-1
        )
        consistency_score = cosine_sim.mean().item() * 100
    else:
        consistency_score = 100.0

    domain_inputs = processor(text=domain_prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        domain_text_embeds = model.get_text_features(**domain_inputs)
    domain_text_embeds = domain_text_embeds / domain_text_embeds.norm(p=2, dim=-1, keepdim=True)  # (D,dim)

    sim = image_embeds @ domain_text_embeds.T

    avg_sim = sim.mean(dim=0)  # (D,)
    top_idx = int(torch.argmax(avg_sim).item())
    domain_top1 = domain_prompts[top_idx]
    domain_score = float(avg_sim[top_idx].item() * 100)

    probs = torch.softmax(sim, dim=-1)  # (F,D)
    ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # (F,)
    ent_mean = float(ent.mean().item())

    max_ent = float(np.log(len(domain_prompts)))
    domain_consistency = (1.0 - (ent_mean / (max_ent + 1e-12))) * 100.0
    domain_consistency = float(np.clip(domain_consistency, 0.0, 100.0))

    richness = compute_richness_score(frames)

    return (
        text_score, consistency_score,
        domain_top1, domain_score, domain_consistency,
        richness,
    )

if __name__ == "__main__":
    gif_1_path = "your_GIF.gif"
    gif_2_path = "baseline_GIF.gif"

    target_prompt = "text prompt"
    
    frames_1 = extract_frames_from_gif(gif_1_path, background_color=(0, 0, 0))
    if frames_1:
        (
            score_text_1, score_smooth_1,
            domain_top1_1, domain_score_1, domain_cons_1,
            rich_1
        ) = calculate_metrics(frames_1, target_prompt)

        print(f"\nResults for {gif_1_path}:")
        print(f"  CLIP Text Score:            {score_text_1:.4f}")
        print(f"  CLIP Consistency:           {score_smooth_1:.4f}")
        print(f"  CLIP Domain Top-1:          {domain_top1_1}")
        print(f"  CLIP Domain Score:          {domain_score_1:.4f}")
        print(f"  CLIP Domain Consistency:    {domain_cons_1:.4f}")
        print(f"  Richness Score (0-100):     {rich_1['richness_score_0_100']:.4f}")
        print(f"    - Edge density mean:      {rich_1['edge_density_mean']:.6f}")
        print(f"    - Colorfulness mean:      {rich_1['colorfulness_mean']:.4f}")
        print(f"    - Entropy mean:           {rich_1['entropy_mean']:.4f}")

    if os.path.exists(gif_2_path):
        frames_2 = extract_frames_from_gif(gif_2_path, background_color=(0, 0, 0))
        if frames_2:
            (
                score_text_2, score_smooth_2,
                domain_top1_2, domain_score_2, domain_cons_2,
                rich_2
            ) = calculate_metrics(frames_2, target_prompt)

            print(f"\nResults for {gif_2_path}:")
            print(f"  CLIP Text Score:            {score_text_2:.4f}")
            print(f"  CLIP Consistency:           {score_smooth_2:.4f}")
            print(f"  CLIP Domain Top-1:          {domain_top1_2}")
            print(f"  CLIP Domain Score:          {domain_score_2:.4f}")
            print(f"  CLIP Domain Consistency:    {domain_cons_2:.4f}")
            print(f"  Richness Score (0-100):     {rich_2['richness_score_0_100']:.4f}")
            print(f"    - Edge density mean:      {rich_2['edge_density_mean']:.6f}")
            print(f"    - Colorfulness mean:      {rich_2['colorfulness_mean']:.4f}")
            print(f"    - Entropy mean:           {rich_2['entropy_mean']:.4f}")

            print("\n--- Comparison ---")
            print(f"Text Score Improvement:         {(score_text_1 - score_text_2):.4f}")
            print(f"Consistency Improvement:        {(score_smooth_1 - score_smooth_2):.4f}")
            print(f"Domain Score Improvement:       {(domain_score_1 - domain_score_2):.4f}")
            print(f"Domain Consistency Improvement: {(domain_cons_1 - domain_cons_2):.4f}")
            print(f"Richness Score Improvement:     {(rich_1['richness_score_0_100'] - rich_2['richness_score_0_100']):.4f}")