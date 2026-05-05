"""
RAE (Reconstruction Autoencoder) head for image reconstruction using DINOv2 encoder and MAE decoder.
Also supports diffusion-based generation with stage2 DiT models.
"""

import math
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import ModelOutput

from cofai.backbone import Dinov2TimmBackbone
from cofai.engine.registry import instantiate_class
from einops import rearrange

# Add RAE models directory to path for importing stage2 models
_RAE_MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "RAE", "src")
if os.path.exists(_RAE_MODELS_PATH) and _RAE_MODELS_PATH not in sys.path:
    sys.path.insert(0, _RAE_MODELS_PATH)


# ==================== ViTMAE Config ====================
class ViTMAEConfig(PretrainedConfig):
    """Configuration class for ViT MAE decoder."""

    model_type = "vit_mae"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        mask_ratio=0.75,
        norm_pix_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss


# ==================== Helper Functions ====================
def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """Create 2D sin/cos positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Get 2D sin/cos positional embedding from grid."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Get 1D sin/cos positional embedding from grid."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ==================== ViTMAE Decoder Components ====================
@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    """Output class for ViTMAE decoder."""

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ViTMAESelfAttention(nn.Module):
    """Self-attention module for ViTMAE."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class ViTMAESelfOutput(nn.Module):
    """Self-output module for ViTMAE."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTMAEAttention(nn.Module):
    """Attention module for ViTMAE."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAESelfAttention(config)
        self.output = ViTMAESelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ViTMAEIntermediate(nn.Module):
    """Intermediate module for ViTMAE."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTMAEOutput(nn.Module):
    """Output module for ViTMAE."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTMAELayer(nn.Module):
    """Transformer layer for ViTMAE."""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs
        return outputs


# ==================== GeneralDecoder ====================
class GeneralDecoder(nn.Module):
    """General decoder for RAE using ViTMAE architecture.
    """

    def __init__(
        self,
        # Decoder architecture config (from decoder_config dict in YAML)
        decoder_config: dict,
        # Decoder input/output config
        patch_size: int = 16,
        image_size: int = 512,
        pretrained_path: Optional[str] = None,
        # Encoder info (needed for calculating num_patches and normalization)
        encoder_img_size: int = 512,
        encoder_patch_size: int = 16,
        encoder_hidden_size: int = 768,
        # Encoder normalization stats (required)
        encoder_mean: Union[list, torch.Tensor] = None,
        encoder_std: Union[list, torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()

        print("Building RAE decoder...")

        # Validate required parameters
        if encoder_mean is None:
            raise ValueError(
                "encoder_mean must be provided. "
                "Please specify encoder_mean in your configuration file."
            )
        if encoder_std is None:
            raise ValueError(
                "encoder_std must be provided. "
                "Please specify encoder_std in your configuration file."
            )

        # Convert encoder normalization stats to tensor format
        if not isinstance(encoder_mean, torch.Tensor):
            encoder_mean = torch.tensor(encoder_mean).view(1, 3, 1, 1)
        else:
            encoder_mean = (
                encoder_mean.view(1, 3, 1, 1)
                if encoder_mean.dim() == 1
                else encoder_mean
            )

        if not isinstance(encoder_std, torch.Tensor):
            encoder_std = torch.tensor(encoder_std).view(1, 3, 1, 1)
        else:
            encoder_std = (
                encoder_std.view(1, 3, 1, 1) if encoder_std.dim() == 1 else encoder_std
            )

        # Calculate num_patches
        num_patches = (encoder_img_size // encoder_patch_size) ** 2

        # Create ViTMAEConfig from decoder_config dict
        cfg_params = decoder_config.copy()
        cfg_params.update(
            {
                "hidden_size": encoder_hidden_size,  # Input dimension (from encoder)
                "patch_size": patch_size,
                "image_size": image_size,
            }
        )
        cfg = ViTMAEConfig(**cfg_params)

        # Build decoder layers
        self.decoder_embed = nn.Linear(
            cfg.hidden_size, cfg.decoder_hidden_size, bias=True
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, cfg.decoder_hidden_size),
            requires_grad=False,
        )

        _layer_cfg = deepcopy(cfg)
        _layer_cfg.hidden_size = cfg.decoder_hidden_size
        _layer_cfg.num_hidden_layers = cfg.decoder_num_hidden_layers
        _layer_cfg.num_attention_heads = cfg.decoder_num_attention_heads
        _layer_cfg.intermediate_size = cfg.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [
                ViTMAELayer(_layer_cfg)
                for _ in range(cfg.decoder_num_hidden_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(
            cfg.decoder_hidden_size, eps=cfg.layer_norm_eps
        )
        self.decoder_pred = nn.Linear(
            cfg.decoder_hidden_size,
            cfg.patch_size**2 * cfg.num_channels,
            bias=True,
        )
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_hidden_size))
        self.gradient_checkpointing = False
        self.config = cfg
        self.num_patches = num_patches
        self.initialize_weights(num_patches)
        self.decoder_config = _layer_cfg

        # Save encoder normalization stats
        self.encoder_mean = encoder_mean
        self.encoder_std = encoder_std

        # Load pretrained weights
        if pretrained_path:
            print(f"Loading pretrained decoder from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            keys = self.load_state_dict(state_dict, strict=False)

        # Move to device and set to eval mode
        if device is not None:
            self.to(device)
        self.eval()

    def interpolate_pos_encoding(self, token_res) -> torch.Tensor:
        """Interpolate position encodings for different resolutions."""
        class_pe = self.decoder_pos_embed[:, 0:1, :]
        patch_pe = self.decoder_pos_embed[:, 1:, :]
        patch_pe_2d = rearrange(patch_pe, "b (h w) c -> b c h w", h=32, w=32)

        patch_pe_2d = F.interpolate(
            patch_pe_2d,
            size=token_res,
            mode="bicubic",
            align_corners=False,
        )
        patch_pe = rearrange(patch_pe_2d, "b c h w -> b (h w) c")
        return torch.cat((class_pe, patch_pe), dim=1)

    def initialize_weights(self, num_patches):
        """Initialize decoder weights."""
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

    def unpatchify(self, feature, token_res):
        patch_size = self.config.patch_size
        x = rearrange(
            feature,
            "b (h w) (p q c) -> b c (h p) (w q)",
            h=token_res[0],
            w=token_res[1],
            p=patch_size,
            q=patch_size,
        )
        return x

    def forward(
        self,
        features: torch.Tensor,
        token_res: Optional[Tuple[int, int]] = None,
        token_format: str = "patch",
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns decoder logits.

        Args:
            features: Patch tokens, shape (B, N, C)
            token_res: Token resolution, format (H, W)
            token_format: Token format, default "patch"
            **kwargs: Other parameters (ignored)

        Returns:
            Decoder logits, shape (B, N, patch_size^2 * num_channels)
        """
        assert token_format in ["cls,patch", "patch"]
        x_ = self.decoder_embed(features)
        if token_format == "cls,patch":  # drop cls_token
            x_ = x_[:, 1:, :]
        decoder_pos_embed = self.decoder_pos_embed

        H, W = token_res
        if H * W != self.num_patches:
            # resize position encoding to token_res
            decoder_pos_embed = self.interpolate_pos_encoding(token_res)

        cls_token = self.trainable_cls_token.expand(x_.shape[0], -1, -1)
        x = torch.cat([cls_token, x_], dim=1)

        h = x + decoder_pos_embed

        # Forward through decoder layers
        for layer in self.decoder_layers:
            h = layer(
                h, head_mask=None, output_attentions=False
            )[0]

        h = self.decoder_norm(h)
        logits = self.decoder_pred(h)
        logits = logits[:, 1:, :] # drop cls token

        return logits

    def predict(
        self,
        features: torch.Tensor,
        token_res: Optional[Tuple[int, int]] = None,
        token_format: str = "patch",
        clamp: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Returns reconstructed image.

        Args:
            features: Patch tokens, shape (B, N, C)
            token_res: Token resolution, format (H, W)
            token_format: Token format, default "patch"
            clamp: Whether to clamp output to [0, 1] range
            **kwargs: Other parameters

        Returns:
            Decoded image, shape (B, C, H, W)
        """
        # Forward pass through decoder to get logits
        logits = self.forward(
            features,
            token_format=token_format,
            token_res=token_res,
        )
        x = self.unpatchify(logits, token_res)

        # Denormalize output
        if self.encoder_mean is not None and self.encoder_std is not None:
            x = x * self.encoder_std.to(x.device) + self.encoder_mean.to(
                x.device
            )

        if clamp:
            x = x.clamp(0, 1)

        return x


# ==================== RAE ====================
class RAE(nn.Module):
    """
    RAE (Reconstruction Autoencoder) using DINOv2 encoder and MAE decoder.

    This class implements a reconstruction autoencoder that:
    1. Encodes images using Dinov2TimmBackbone
    2. Decodes latent features using GeneralDecoder
    3. Supports normalization and reshaping of latent features
    """

    def __init__(
        self,
        # Nested configs (new format)
        encoder: Optional[dict] = None,
        decoder: Optional[dict] = None,
        # Encoder configs (legacy format, for backward compatibility)
        encoder_model_size: str = "base",
        encoder_img_size: int = 512,
        encoder_patch_size: int = 14,
        encoder_with_registers: bool = True,
        encoder_dynamic_size: bool = False,
        # Decoder configs (legacy format, for backward compatibility)
        decoder_config_path: str = "facebook/vit-mae-base",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # Normalization and reshaping (legacy format, for backward compatibility)
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()

        # Support both nested config format and legacy flat format
        if encoder is not None:
            # New nested format: instantiate encoder from config
            encoder_config = encoder.copy()
            encoder_type = encoder_config.get("type", "")

            # Handle parameter mapping for different encoder types
            # MAETimmBackbone uses model_name instead of model_size
            if "MAETimmBackbone" in encoder_type:
                if (
                    "model_size" in encoder_config
                    and "model_name" not in encoder_config
                ):
                    model_size = encoder_config.pop("model_size")
                    patch_size = encoder_config.get("patch_size", 16)
                    # Map model_size to model_name for MAE
                    model_name_map = {
                        "base": f"vit_base_patch{patch_size}_224.mae",
                        "small": f"vit_small_patch{patch_size}_224.mae",
                        "large": f"vit_large_patch{patch_size}_224.mae",
                    }
                    encoder_config["model_name"] = model_name_map.get(
                        model_size, f"vit_{model_size}_patch{patch_size}_224.mae"
                    )
                # Remove parameters that MAETimmBackbone doesn't accept
                encoder_config.pop("with_registers", None)
                encoder_config.pop("dynamic_size", None)
            elif "MAETransformersBackbone" in encoder_type:
                # MAETransformersBackbone uses model_name directly, no mapping needed
                # Remove parameters that MAETransformersBackbone doesn't accept
                encoder_config.pop("with_registers", None)
                encoder_config.pop("dynamic_size", None)
                encoder_config.pop("model_size", None)

            self.encoder = instantiate_class(encoder_config)

            # Extract encoder properties from the instantiated encoder object
            # Try to get from encoder object first, then fall back to config
            if hasattr(self.encoder, "patch_size"):
                self.encoder_patch_size = self.encoder.patch_size
            else:
                self.encoder_patch_size = encoder.get("patch_size", encoder_patch_size)

            if hasattr(self.encoder, "img_size"):
                self.encoder_input_size = self.encoder.img_size
            else:
                self.encoder_input_size = encoder.get("img_size", encoder_img_size)
        else:
            # Legacy format: create encoder with flat parameters
            self.encoder = Dinov2TimmBackbone(
                model_size=encoder_model_size,
                img_size=encoder_img_size,
                patch_size=encoder_patch_size,
                dynamic_size=encoder_dynamic_size,
                with_registers=encoder_with_registers,
            )
            self.encoder_input_size = encoder_img_size
            self.encoder_patch_size = encoder_patch_size

        # Get encoder properties
        # Try to get hidden_size from encoder object (works for all backbone types)
        if hasattr(self.encoder, "hidden_size"):
            self.latent_dim = self.encoder.hidden_size
        elif hasattr(self.encoder, "model") and hasattr(
            self.encoder.model, "embed_dim"
        ):
            # Fallback for timm-based encoders
            self.latent_dim = self.encoder.model.embed_dim
        else:
            raise ValueError(
                f"Cannot determine latent_dim from encoder {type(self.encoder)}. Encoder must have 'hidden_size' attribute or 'model.embed_dim' attribute."
            )
        assert self.encoder_input_size % self.encoder_patch_size == 0, (
            f"encoder_input_size {self.encoder_input_size} must be divisible by "
            f"encoder_patch_size {self.encoder_patch_size}"
        )
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        # Get encoder normalization stats
        # Try to get normalization from encoder object first, then fall back to AutoImageProcessor
        if hasattr(self.encoder, "processor"):
            # For transformers-based encoders (MAETransformersBackbone, etc.)
            proc = self.encoder.processor
            self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
            self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        elif hasattr(self.encoder, "input_transform"):
            # For timm-based encoders, extract from input_transform
            # MAE uses ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # DINOv2 uses: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # Both use the same ImageNet normalization
            self.encoder_mean = torch.tensor([0.4850, 0.4560, 0.4060]).view(1, 3, 1, 1)
            self.encoder_std = torch.tensor([0.2290, 0.2240, 0.2250]).view(1, 3, 1, 1)
        else:
            # Fallback: use DINOv2 AutoImageProcessor (for backward compatibility)
            if encoder is not None:
                encoder_model_size = encoder.get("model_size", encoder_model_size)
                encoder_with_registers = encoder.get(
                    "with_registers", encoder_with_registers
                )

            encoder_config_path = f"facebook/dinov2-{encoder_model_size}"
            if encoder_with_registers:
                encoder_config_path = (
                    f"facebook/dinov2-with-registers-{encoder_model_size}"
                )

            proc = AutoImageProcessor.from_pretrained(encoder_config_path)
            self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
            self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)

        # Initialize decoder
        decoder_image_size = None
        if decoder is not None:
            # New nested format: extract decoder parameters from config
            decoder_config_path = decoder.get("config_path", decoder_config_path)
            decoder_patch_size = decoder.get("patch_size", decoder_patch_size)
            decoder_image_size = decoder.get("image_size", None)
            pretrained_decoder_path = decoder.get(
                "pretrained_path", pretrained_decoder_path
            )
            # Extract normalization parameters from decoder config
            noise_tau = decoder.get("noise_tau", noise_tau)
            reshape_to_2d = decoder.get("reshape_to_2d", reshape_to_2d)
            normalization_stat_path = decoder.get(
                "normalization_stat_path", normalization_stat_path
            )
            eps = decoder.get("eps", eps)

        # Resolve decoder config path (support local paths like configs/decoder/ViTXL)
        decoder_config_path_resolved = decoder_config_path
        if not decoder_config_path.startswith(
            ("http://", "https://", "facebook/", "google/", "microsoft/")
        ):
            # Try to load as local path
            possible_paths = [
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "models",
                    "RAE",
                    decoder_config_path,
                ),
                os.path.join(os.getcwd(), "models", "RAE", decoder_config_path),
                decoder_config_path,
            ]
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    decoder_config_path_resolved = abs_path
                    print(f"Found decoder config at: {decoder_config_path_resolved}")
                    break

        decoder_config = AutoConfig.from_pretrained(decoder_config_path_resolved)
        # decoder_config.hidden_size is the input dimension for decoder_embed (should equal latent_dim)
        # decoder_config.decoder_hidden_size is the internal hidden size (should be preserved from config)
        decoder_config.hidden_size = self.latent_dim
        decoder_config.patch_size = decoder_patch_size
        # Use image_size from decoder config if provided, otherwise calculate from patches
        if decoder_image_size is not None:
            decoder_config.image_size = decoder_image_size
        else:
            decoder_config.image_size = int(
                decoder_patch_size * math.sqrt(self.base_patches)
            )
        # Ensure decoder_hidden_size exists (use default if not in config)
        if (
            not hasattr(decoder_config, "decoder_hidden_size")
            or decoder_config.decoder_hidden_size is None
        ):
            decoder_config.decoder_hidden_size = 512
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        # Load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(
                    f"Missing keys when loading pretrained decoder: {keys.missing_keys}"
                )

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d

        # Latent normalization
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            self.latent_mean = stats.get("mean", None)
            self.latent_var = stats.get("var", None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        """Add noise to latent features during training."""
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device
        )
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images to latent features."""
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = F.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )

        # Encode through DINOv2 backbone: encode part (blocks[:slot])
        # backbone.encode will handle normalization using input_transform
        h = self.encoder.encode(x)  # (B, N, C) after blocks[:slot]

        # Decode part: use decode_rae to process remaining blocks and extract patch tokens
        z = self.encoder.decode_rae(h)  # (B, N_patches, C) without prefix tokens

        # Add noise during training
        if self.training and self.noise_tau > 0:
            z = self.noising(z)

        # Reshape to 2D if needed (order: reshape_to_2d first, then normalization)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(math.sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)

        # Normalize latent features (after reshape_to_2d)
        if self.do_normalization:
            latent_mean = (
                self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            )
            latent_var = (
                self.latent_var.to(z.device) if self.latent_var is not None else 1
            )
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent features to reconstructed images."""
        # Denormalize latent features
        if self.do_normalization:
            latent_mean = (
                self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            )
            latent_var = (
                self.latent_var.to(z.device) if self.latent_var is not None else 1
            )
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean

        # Reshape from 2D to sequence if needed
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)  # (B, N, C)

        # Decode through GeneralDecoder
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)

        # Denormalize output
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(
            x_rec.device
        )

        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode and decode."""
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec


# ==================== RAE Diffusion Head ====================
class RAEDiffusionHead(nn.Module):
    """
    RAE Diffusion Head for image generation using RAE decoder and stage2 DiT diffusion model.
    
    This class integrates:
    1. RAE decoder (GeneralDecoder) for decoding latent features to images
    2. Stage2 DiT diffusion model for generating latent features
    3. Transport and Sampler for diffusion sampling
    4. Classifier-free guidance support
    """
    
    def __init__(
        self,
        # RAE decoder config (same as GeneralDecoder)
        decoder_config: dict,
        patch_size: int = 16,
        image_size: int = 256,
        pretrained_decoder_path: Optional[str] = None,
        encoder_img_size: int = 256,
        encoder_patch_size: int = 16,
        encoder_hidden_size: int = 768,
        encoder_mean: Union[list, torch.Tensor] = None,
        encoder_std: Union[list, torch.Tensor] = None,
        normalization_stat_path: Optional[str] = None,
        reshape_to_2d: bool = True,
        eps: float = 1e-5,
        # Stage2 diffusion model config
        stage2_config: Optional[dict] = None,
        stage2_ckpt: Optional[str] = None,
        # Transport config
        transport_config: Optional[dict] = None,
        # Sampler config
        sampler_config: Optional[dict] = None,
        # Guidance config
        guidance_config: Optional[dict] = None,
        # Misc config
        misc_config: Optional[dict] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        
        print("Building RAE Diffusion Head...")
        
        # Initialize RAE decoder (same as GeneralDecoder)
        self.decoder = GeneralDecoder(
            decoder_config=decoder_config,
            patch_size=patch_size,
            image_size=image_size,
            pretrained_path=pretrained_decoder_path,
            encoder_img_size=encoder_img_size,
            encoder_patch_size=encoder_patch_size,
            encoder_hidden_size=encoder_hidden_size,
            encoder_mean=encoder_mean,
            encoder_std=encoder_std,
            device=device,
        )
        
        # Initialize stage2 DiT model
        if stage2_config is None:
            raise ValueError("stage2_config must be provided for RAEDiffusionHead")
        
        try:
            # Import RAE modules (add to path if needed)
            import sys
            import os
            rae_src_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "RAE", "src")
            if os.path.exists(rae_src_path) and rae_src_path not in sys.path:
                sys.path.insert(0, rae_src_path)
            
            from stage2.models import Stage2ModelProtocol
            from utils.model_utils import instantiate_from_config as rae_instantiate_from_config
            
            # Create stage2 model config in RAE format
            rae_stage2_config = {
                "target": stage2_config.get("target", "stage2.models.DDT.DiTwDDTHead"),
                "params": stage2_config.get("params", {}),
                "ckpt": stage2_ckpt,
            }
            
            self.stage2_model = rae_instantiate_from_config(rae_stage2_config)
            if device is not None:
                self.stage2_model = self.stage2_model.to(device)
            self.stage2_model.eval()
            print(f"Loaded stage2 model: {stage2_config.get('target')}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to load stage2 model from RAE. Make sure models/RAE/src is accessible. Error: {e}"
            )
        
        # Initialize transport
        if transport_config is None:
            transport_config = {
                "path_type": "Linear",
                "prediction": "velocity",
                "time_dist_type": "uniform",
            }
        
        try:
            # Import RAE modules (add to path if needed)
            import sys
            import os
            rae_src_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "RAE", "src")
            if os.path.exists(rae_src_path) and rae_src_path not in sys.path:
                sys.path.insert(0, rae_src_path)
            
            from stage2.transport import create_transport
            
            # Calculate time_dist_shift from misc_config
            time_dist_shift = 1.0
            if misc_config:
                shift_dim = misc_config.get("time_dist_shift_dim", 768 * 16 * 16)
                shift_base = misc_config.get("time_dist_shift_base", 4096)
                time_dist_shift = math.sqrt(shift_dim / shift_base)
            
            self.transport = create_transport(
                **transport_config.get("params", {}),
                time_dist_shift=time_dist_shift,
            )
            print(f"Created transport: {transport_config.get('params', {})}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to create transport. Error: {e}")
        
        # Initialize sampler
        if sampler_config is None:
            sampler_config = {
                "mode": "ODE",
                "params": {
                    "sampling_method": "euler",
                    "num_steps": 50,
                    "atol": 1e-6,
                    "rtol": 1e-3,
                    "reverse": False,
                },
            }
        
        try:
            # Import RAE modules (add to path if needed)
            import sys
            import os
            rae_src_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "RAE", "src")
            if os.path.exists(rae_src_path) and rae_src_path not in sys.path:
                sys.path.insert(0, rae_src_path)
            
            from stage2.transport import Sampler
            
            self.sampler = Sampler(self.transport)
            self.sampler_config = sampler_config
            print(f"Created sampler: {sampler_config}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to create sampler. Error: {e}")
        
        # Guidance config
        if guidance_config is None:
            guidance_config = {
                "method": "cfg",
                "scale": 1.0,
                "t_min": 0.0,
                "t_max": 1.0,
            }
        self.guidance_config = guidance_config
        
        # Misc config
        if misc_config is None:
            misc_config = {
                "latent_size": [768, 16, 16],
                "num_classes": 1000,
            }
        self.misc_config = misc_config
        self.latent_size = misc_config.get("latent_size", [768, 16, 16])
        self.num_classes = misc_config.get("num_classes", 1000)
        
        # Move to device
        if device is not None:
            self.to(device)
        self.eval()
    
    def _setup_sampling_fn(self):
        """Setup sampling function based on sampler config."""
        mode = self.sampler_config.get("mode", "ODE")
        sampler_params = self.sampler_config.get("params", {})
        
        if mode == "ODE":
            sample_fn = self.sampler.sample_ode(**sampler_params)
        elif mode == "SDE":
            sample_fn = self.sampler.sample_sde(**sampler_params)
        else:
            raise NotImplementedError(f"Invalid sampling mode {mode}.")
        
        return sample_fn
    
    def _setup_guidance(self, y, device):
        """Setup classifier-free guidance."""
        guidance_scale = self.guidance_config.get("scale", 1.0)
        
        if guidance_scale > 1.0:
            t_min = self.guidance_config.get("t_min", 0.0)
            t_max = self.guidance_config.get("t_max", 1.0)
            model_kwargs = dict(
                y=y,
                cfg_scale=guidance_scale,
                cfg_interval=(t_min, t_max),
            )
            guidance_method = self.guidance_config.get("method", "cfg")
            if guidance_method == "autoguidance":
                # Autoguidance not implemented yet
                raise NotImplementedError("Autoguidance not implemented yet")
            else:
                model_fwd = self.stage2_model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fwd = self.stage2_model.forward
        
        return model_fwd, model_kwargs
    
    def forward(
        self,
        features: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate images using diffusion.
        
        Args:
            features: Not used for generation (generation starts from noise)
            class_labels: Class labels for conditional generation, shape (N,)
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            **kwargs: Other parameters
        
        Returns:
            Generated latent features, shape (N, C, H, W)
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        device = next(self.stage2_model.parameters()).device
        
        # Prepare class labels
        if class_labels is None:
            # Generate random class labels
            class_labels = torch.randint(0, self.num_classes, (num_samples,), device=device)
        else:
            if isinstance(class_labels, (list, tuple)):
                class_labels = torch.tensor(class_labels, device=device)
            num_samples = len(class_labels)
        
        # Setup sampling function
        sample_fn = self._setup_sampling_fn()
        
        # Setup guidance
        y = class_labels
        if self.guidance_config.get("scale", 1.0) > 1.0:
            # Duplicate for CFG
            y_null = torch.tensor([self.num_classes] * num_samples, device=device)
            y = torch.cat([y, y_null], 0)
        
        model_fwd, model_kwargs = self._setup_guidance(y, device)
        
        # Create sampling noise
        z = torch.randn(num_samples, *self.latent_size, device=device)
        if self.guidance_config.get("scale", 1.0) > 1.0:
            z = torch.cat([z, z], 0)
        
        # Sample latent features
        with torch.no_grad():
            samples = sample_fn(z, model_fwd, **model_kwargs)[-1]
        
        # Remove null class samples if using CFG
        if self.guidance_config.get("scale", 1.0) > 1.0:
            samples, _ = samples.chunk(2, dim=0)
        
        return samples
    
    def predict(
        self,
        features: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        seed: Optional[int] = None,
        clamp: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate and decode images.
        
        Args:
            features: Not used for generation
            class_labels: Class labels for conditional generation
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            clamp: Whether to clamp output to [0, 1] range
            **kwargs: Other parameters
        
        Returns:
            Generated images, shape (N, C, H, W)
        """
        # Generate latent features
        z = self.forward(
            features=features,
            class_labels=class_labels,
            num_samples=num_samples,
            seed=seed,
            **kwargs,
        )
        
        # Decode latent features to images using RAE decoder
        # Reshape from 2D to sequence if needed
        if len(z.shape) == 4:  # (B, C, H, W)
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)  # (B, N, C)
        
        # Calculate token resolution
        token_res = (h, w)
        
        # Decode through RAE decoder
        x_gen = self.decoder.predict(
            features=z,
            token_res=token_res,
            token_format="patch",
            clamp=clamp,
        )
        
        return x_gen
