# animatediff/utils/mask_util.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from rembg import remove, new_session

class AttentionMaskController:
    """
    單例模式控制器，用於將 Mask 傳遞給 UNet 深處的 Attention 層
    """
    _instance = None
    
    def __init__(self):
        self.masks = None # Shape: [Batch, Frames, H, W]
        self.active = False
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_masks(self, masks):
        self.masks = masks
        self.active = True
        
    def get_mask_for_shape(self, shape):
        if not self.active or self.masks is None:
            return None
        b, l, h, w = self.masks.shape
        target_hw = shape 
        
        # Resize mask to match feature map
        current_mask = self.masks.view(-1, 1, h, w)
        current_mask = F.interpolate(current_mask, size=(target_hw, target_hw), mode='nearest')
        current_mask = current_mask.view(-1, target_hw * target_hw) 
        return current_mask

class AutoMaskGenerator:
    """
    使用 rembg 自動從影片 tensor 生成遮罩
    """
    def __init__(self):
        print("正在初始化 Background Remover (u2net)...")
        self.session = new_session("u2net") 

    def generate_mask_from_video_tensor(self, video_tensor):
        # video_tensor shape: [B, C, F, H, W], normalized to [-1, 1] or [0, 1]
        b, c, f, h, w = video_tensor.shape
        
        # Denormalize to [0, 255] for PIL
        # 假設輸入是 [-1, 1] (AnimateDiff 標準輸出)
        if video_tensor.min() < 0:
            video_tensor = (video_tensor + 1.0) / 2.0
            
        video_tensor = video_tensor.clamp(0, 1)
        frames = video_tensor[0].permute(1, 2, 3, 0).cpu().numpy()
        frames = (frames * 255).astype(np.uint8)
        
        masks = []
        print(f"正在生成 {f} 幀的遮罩...")
        
        for i in range(f):
            img_pil = Image.fromarray(frames[i])
            # rembg Magic
            mask_pil = remove(img_pil, session=self.session, only_mask=True)
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
            # 二值化
            mask_np = (mask_np > 0.5).astype(np.float32)
            masks.append(mask_np)
            
        mask_tensor = torch.from_numpy(np.stack(masks)).unsqueeze(0) # [1, F, H, W]
        return mask_tensor.to(video_tensor.device)