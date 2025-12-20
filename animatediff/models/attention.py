
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat

# IMPORTANT: 這裡要跟 animate.py 用同一個 Controller 來源，不要再從 util import
from animatediff.utils.mask_util import AttentionMaskController

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


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
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."

        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        # encoder_hidden_states: [B, N, C] -> repeat to [B*F, N, C]
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

        # SC-Attn (attn1)
        assert unet_use_cross_frame_attention is not None
        if unet_use_cross_frame_attention:
            self.attn1 = SparseCausalAttention2D(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn (attn2)
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # FFN
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temporal Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            raise ModuleNotFoundError("xformers not found")
        if not torch.cuda.is_available():
            raise ValueError("xformers requires GPU")
        
        self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        if self.attn2 is not None:
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    # [NEW] Manual Attention Function to bypass diffusers/xformers shape issues
    def _manual_cross_attention(self, hidden_states, encoder_hidden_states, attention_mask):
        attn = self.attn2
        batch_size, sequence_length, _ = hidden_states.shape
        
        # 1. Projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 2. Reshape to Heads
        inner_dim = query.shape[-1]
        heads = attn.heads
        head_dim = inner_dim // heads

        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2) # [B, H, Pixels, D]
        key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)     # [B, H, Tokens, D]
        value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2) # [B, H, Tokens, D]

        # 3. Attention Scores
        # Q @ K^T -> [B, H, Pixels, Tokens] (This is where shapes are guaranteed to be correct)
        scale = head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # 4. Apply Mask [B, 1, Pixels, Tokens]
        # Broadcasting handles the Head dimension automatically
        scores = scores + attention_mask

        # 5. Softmax & Weighted Sum
        probs = scores.softmax(dim=-1)
        # cast back to value dtype if mixed precision
        probs = probs.to(value.dtype)
        
        hidden_states = torch.matmul(probs, value) # [B, H, Pixels, D]

        # 6. Output Projection
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = attn.to_out[0](hidden_states)
        
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # 1) attn1
        norm_hidden_states_1 = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        if self.unet_use_cross_frame_attention:
            hidden_states = self.attn1(norm_hidden_states_1, attention_mask=attention_mask, video_length=video_length) + hidden_states
        else:
            hidden_states = self.attn1(norm_hidden_states_1, attention_mask=attention_mask) + hidden_states

        # 2) attn2 (Cross-Attention)
        if self.attn2 is not None:
            norm_hidden_states_2 = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            # --- Region-Aware Logic ---
            controller = AttentionMaskController.get_instance()
            use_region_mask = bool(controller.active) and (encoder_hidden_states is not None) and (encoder_hidden_states.shape[1] > 77)

            if use_region_mask:
                n_pixels = norm_hidden_states_2.shape[1]
                spatial = int(n_pixels ** 0.5)

                if spatial * spatial == n_pixels:
                    region_mask = controller.get_mask_for_shape(spatial)
                    if region_mask is not None:
                        region_mask = region_mask.to(device=hidden_states.device, dtype=norm_hidden_states_2.dtype)
                        
                        batch_frames = norm_hidden_states_2.shape[0]
                        n_tokens = encoder_hidden_states.shape[1] # 154
                        half = n_tokens // 2

                        # Fix batch mismatch (Uncond + Cond)
                        if region_mask.shape[0] != batch_frames:
                            if batch_frames == 2 * region_mask.shape[0]:
                                region_mask = torch.cat([region_mask, region_mask], dim=0)
                            else:
                                region_mask = None # Give up

                        if region_mask is not None:
                            # Create Bias: [B, 1, Pixels, Tokens]
                            attn_bias = torch.zeros(
                                (batch_frames, 1, n_pixels, n_tokens),
                                device=hidden_states.device,
                                dtype=norm_hidden_states_2.dtype,
                            )
                            mask_expanded = region_mask.unsqueeze(1).unsqueeze(-1)
                            
                            # Apply masking
                            attn_bias[:, :, :, :half].masked_fill_(mask_expanded > 0.5, -10000.0)
                            attn_bias[:, :, :, half:].masked_fill_(mask_expanded < 0.5, -10000.0)

                            # [CRITICAL] Use Manual Attention to bypass Diffusers shape checks
                            try:
                                hidden_states = self._manual_cross_attention(
                                    norm_hidden_states_2,
                                    encoder_hidden_states,
                                    attn_bias
                                ) + hidden_states
                            except Exception as e:
                                print(f"Manual attn failed: {e}. Fallback to standard.")
                                hidden_states = self.attn2(
                                    norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                                ) + hidden_states
                        else:
                            hidden_states = self.attn2(
                                norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                            ) + hidden_states
                    else:
                        hidden_states = self.attn2(
                            norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                        ) + hidden_states
                else:
                    hidden_states = self.attn2(
                        norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                    ) + hidden_states
            else:
                hidden_states = self.attn2(
                    norm_hidden_states_2, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                ) + hidden_states

        # 3) FFN
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # 4) Temporal
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states_t = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states_t) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
    
    