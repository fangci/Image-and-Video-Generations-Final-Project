from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat

# 引入 Mask Controller (我們仍需要它來做 Cross-Attention 的區域隔離)
from animatediff.utils.mask_util import AttentionMaskController

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

# [NEW] KV Cache Controller for KV-Edit
class KVCacheController:
    _instance = None
    
    def __init__(self):
        self.store_mode = False
        self.inject_mode = False
        # Cache 結構: { layer_id: [ (key_step0, value_step0), (key_step1, value_step1), ... ] }
        self.cache: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        # 用於在 Inject 模式下追蹤每個 layer 目前用到第幾個 step
        self.counters: Dict[str, int] = {} 

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def clear(self):
        self.cache = {}
        self.counters = {}
        self.store_mode = False
        self.inject_mode = False

    def reset_counters(self):
        self.counters = {}

    def store(self, layer_id, key, value):
        if layer_id not in self.cache:
            self.cache[layer_id] = []
        # 存到 CPU 以節省 VRAM，使用時再搬回 GPU
        self.cache[layer_id].append((key.detach().cpu(), value.detach().cpu()))

    def get(self, layer_id):
        if layer_id not in self.counters:
            self.counters[layer_id] = 0
        
        idx = self.counters[layer_id]
        
        if layer_id in self.cache and idx < len(self.cache[layer_id]):
            k, v = self.cache[layer_id][idx]
            self.counters[layer_id] += 1 # Increment counter for next call
            return k.cuda(), v.cuda()
        else:
            return None, None

class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
                for _ in range(num_layers)
            ]
        )

        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5"
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b f) n c", f=video_length)

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
            )

        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        if not return_dict:
            return (output,)
        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention

        # SC-Attn (attn1) - 這是我們要做 KV-Edit 的地方
        if unet_use_cross_frame_attention:
            self.attn1 = SparseCausalAttention2D(
                query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout,
                bias=attention_bias, cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = CrossAttention(
                query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout,
                bias=attention_bias, upcast_attention=upcast_attention,
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn (attn2)
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim, cross_attention_dim=cross_attention_dim, heads=num_attention_heads,
                dim_head=attention_head_dim, dropout=dropout, bias=attention_bias, upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None
        
        self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim) if cross_attention_dim is not None else None

        # FFN
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temporal
        if unet_use_temporal_attention:
            self.attn_temp = CrossAttention(
                query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout,
                bias=attention_bias, upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            raise ModuleNotFoundError("xformers not found")
        
        self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        if self.attn2 is not None:
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    # [KV-Edit] Manual Self-Attention with Injection
    def _manual_self_attention_kv(self, hidden_states, attention_mask, video_length, kv_controller):
        attn = self.attn1
        # 使用 id 作為 key，確保每個 block 存取對應的 cache
        layer_id = str(id(self)) 
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # 1. 計算 Query (永遠使用當前的輸入)
        query = attn.to_q(hidden_states)
        
        # 2. 處理 Key/Value (儲存或注入)
        if kv_controller.inject_mode:
            # Inject Mode: 嘗試從 Cache 讀取 K, V
            cached_k, cached_v = kv_controller.get(layer_id)
            if cached_k is not None:
                # 替換！這就是 KV-Edit 的核心
                key = cached_k
                value = cached_v
            else:
                # Fallback: 如果沒 Cache 到 (可能是第一次跑)，就正常算
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
        else:
            # Normal/Store Mode: 正常算
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            
            if kv_controller.store_mode:
                kv_controller.store(layer_id, key, value)

        # 3. 標準 Attention 計算 (手動執行以避開 xformers 的複雜性)
        inner_dim = query.shape[-1]
        heads = attn.heads
        head_dim = inner_dim // heads

        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = scores.softmax(dim=-1).to(value.dtype)
        hidden_states = torch.matmul(probs, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = attn.to_out[0](hidden_states)
        
        return hidden_states

    # [Region-Aware] Manual Cross-Attention
    def _manual_cross_attention(self, hidden_states, encoder_hidden_states, attention_mask):
        attn = self.attn2
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = query.shape[-1]
        heads = attn.heads
        head_dim = inner_dim // heads

        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        scores = scores + attention_mask

        probs = scores.softmax(dim=-1).to(value.dtype)
        hidden_states = torch.matmul(probs, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = attn.to_out[0](hidden_states)
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # 1. Self-Attn (KV-Edit 介入點)
        norm_hidden_states_1 = self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        
        kv_controller = KVCacheController.get_instance()
        
        # 判斷是否需要介入 Self-Attention
        if kv_controller.store_mode or kv_controller.inject_mode:
            # 使用手動 Attention 進行攔截
            hidden_states = self._manual_self_attention_kv(
                norm_hidden_states_1, attention_mask, video_length, kv_controller
            ) + hidden_states
        else:
            # 原始路徑 (為了速度，沒事不要進 manual)
            if self.unet_use_cross_frame_attention:
                hidden_states = self.attn1(norm_hidden_states_1, attention_mask=attention_mask, video_length=video_length) + hidden_states
            else:
                hidden_states = self.attn1(norm_hidden_states_1, attention_mask=attention_mask) + hidden_states

        # 2. Cross-Attn (Region-Aware 介入點)
        if self.attn2 is not None:
            norm_hidden_states_2 = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            
            mask_controller = AttentionMaskController.get_instance()
            if mask_controller.active and encoder_hidden_states.shape[1] > 77:
                n_pixels = norm_hidden_states_2.shape[1]
                spatial = int(n_pixels ** 0.5)
                
                if spatial * spatial == n_pixels:
                    region_mask = mask_controller.get_mask_for_shape(spatial)
                    if region_mask is not None:
                        region_mask = region_mask.to(device=hidden_states.device, dtype=norm_hidden_states_2.dtype)
                        batch_frames = norm_hidden_states_2.shape[0]
                        n_tokens = encoder_hidden_states.shape[1]
                        half = n_tokens // 2
                        
                        if region_mask.shape[0] != batch_frames:
                            if batch_frames == 2 * region_mask.shape[0]:
                                region_mask = torch.cat([region_mask, region_mask], dim=0)
                            else:
                                region_mask = None

                        if region_mask is not None:
                            attn_bias = torch.zeros((batch_frames, 1, n_pixels, n_tokens), device=hidden_states.device, dtype=norm_hidden_states_2.dtype)
                            mask_expanded = region_mask.unsqueeze(1).unsqueeze(-1)
                            
                            attn_bias[:, :, :, :half].masked_fill_(mask_expanded > 0.5, -10000.0)
                            attn_bias[:, :, :, half:].masked_fill_(mask_expanded < 0.5, -10000.0)
                            
                            try:
                                hidden_states = self._manual_cross_attention(norm_hidden_states_2, encoder_hidden_states, attn_bias) + hidden_states
                            except Exception:
                                hidden_states = self.attn2(norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask) + hidden_states
                        else:
                            hidden_states = self.attn2(norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask) + hidden_states
                    else:
                        hidden_states = self.attn2(norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask) + hidden_states
                else:
                    hidden_states = self.attn2(norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            else:
                hidden_states = self.attn2(norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask) + hidden_states

        # 3. FFN
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # 4. Temporal
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states_t = self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            hidden_states = self.attn_temp(norm_hidden_states_t) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states