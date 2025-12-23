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

# 引入 Mask Controller
from animatediff.utils.mask_util import AttentionMaskController

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class SharedAttentionState:
    _instance = None
    
    def __init__(self):
        self.mode = "normal"
        self.cache = {}
        self.counter = 0
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def reset(self):
        self.mode = "normal"
        self.cache = {}
        self.counter = 0
    
    def set_mode(self, mode):
        self.mode = mode
        self.counter = 0

class ReferenceCrossAttention(CrossAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        state = SharedAttentionState.get_instance()
        
        query = self.to_q(hidden_states)
        
        is_self_attn = (encoder_hidden_states is None)
        
        if is_self_attn:
            sequence_input = hidden_states
        else:
            sequence_input = encoder_hidden_states

        key = self.to_k(sequence_input)
        value = self.to_v(sequence_input)

        if is_self_attn and state.mode != "normal":
            current_id = state.counter
            
            if state.mode == "record":
                state.cache[current_id] = (key.detach(), value.detach())
                
            elif state.mode == "inject":
                if current_id in state.cache:
                    bg_key, bg_value = state.cache[current_id]
                    
                    if bg_key.device != key.device:
                        bg_key = bg_key.to(key.device)
                        bg_value = bg_value.to(value.device)

                    key = torch.cat([bg_key, key], dim=1)
                    value = torch.cat([bg_value, value], dim=1)
            
            state.counter += 1

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, query.shape[1], query.shape[-1], attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states


class SparseCausalAttention2D(CrossAttention):
    """Causal self-attention across frames with a simple sparse mask."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, **kwargs):
        # If this is cross-attention or no video length is provided, fall back to default behavior.
        if encoder_hidden_states is not None or video_length is None:
            return super().forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)

        b_f, seq_len, _ = hidden_states.shape
        if video_length <= 0 or b_f % video_length != 0:
            return super().forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)

        batch = b_f // video_length
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Merge frames into sequence so each token position has a frame id for causal masking.
        hidden_states_merged = rearrange(hidden_states, "(b f) t c -> b (f t) c", b=batch, f=video_length)
        total_tokens = hidden_states_merged.shape[1]

        frame_ids = torch.arange(video_length, device=device).repeat_interleave(seq_len)
        allow = frame_ids.unsqueeze(0) >= frame_ids.unsqueeze(1)
        mask_value = torch.finfo(dtype).min
        causal_mask = torch.where(allow, torch.zeros_like(allow, dtype=dtype), torch.full_like(allow, mask_value, dtype=dtype))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).repeat(batch, 1, 1, 1)

        attn_mask = causal_mask if attention_mask is None else attention_mask + causal_mask

        attended = super().forward(hidden_states_merged, encoder_hidden_states=None, attention_mask=attn_mask)
        attended = rearrange(attended, "b (f t) c -> (b f) t c", f=video_length)
        return attended


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
        attention_op_mode: str = "kvcache",
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
                    attention_op_mode=attention_op_mode,
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
        attention_op_mode: str = "kvcache",
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.attention_op_mode = attention_op_mode

        attn_cls = SparseCausalAttention2D if attention_op_mode == "mask" else ReferenceCrossAttention

        if unet_use_cross_frame_attention:
            self.attn1 = attn_cls(
                query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout,
                bias=attention_bias, cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = attn_cls(
                query_dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, dropout=dropout,
                bias=attention_bias, upcast_attention=upcast_attention,
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

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
        
        if hasattr(self.attn1, "_use_memory_efficient_attention_xformers"):
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        if self.attn2 is not None:
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # 1. Self-Attn
        norm_hidden_states_1 = self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        
        hidden_states = self.attn1(
            norm_hidden_states_1,
            attention_mask=attention_mask,
            video_length=video_length,
        ) + hidden_states

        # 2. Cross-Attn (Text)
        if self.attn2 is not None:
            norm_hidden_states_2 = self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            
            mask_controller = AttentionMaskController.get_instance()
            if mask_controller.active and encoder_hidden_states.shape[1] > 77:
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