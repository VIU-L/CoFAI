"""AI M2510 PCA-align codec for Qwen3-VL slot-9 visual tokens."""

import torch
from torch import nn
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
)
from compressai.layers import CheckerboardMaskedConv2d, sequential_channel_ramp
from compressai.models.base import CompressionModel
from compressai.models.utils import conv
from einops import rearrange

from cofai.latent_codecs.vit_feature_codec import ChannelGroupsLatentCodecContiguous
from cofai.layers.simple_depthwise_cnn import (
    DepthwiseSeparableConv,
    DepthwiseSeparableDeconvPixelShuffle,
)
from cofai.utils.tensor_ops import border_pad


class PCAAlignLatentCodec(CompressionModel):
    """Dual original/PCA gains around a lightweight hyperprior codec.

    Input and reconstructed tokens use ``(B, H*W, C)`` with ``token_res=(H,W)``.
    The proposal uses 1152 feature channels, 256 main-latent channels, 48 hyper
    channels and eight groups of 32 channels, with 65 learned QPs (0 through 64).
    Spatial grids must be even, as supplied by Qwen's spatial merge size of two.

    PCA statistics, gains and entropy tables are loaded by the common engine
    checkpoint loader. ``U_Transform`` stores the row-basis from the author's
    checkpoint; its transpose is the inverse projection. The published inference
    state includes the author's forced quantile/CDF update, so normal engine
    ``update()`` preserves the original entropy-coding behavior.
    """

    def __init__(self, h_dim=1152, y_dim=256, z_dim=48, groups=32, deduct_mean=True):
        super().__init__()
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.z_down_factor = 2
        self.groups = list(groups) if isinstance(groups, (list, tuple)) else [groups] * (y_dim // groups)
        if sum(self.groups) != y_dim:
            raise ValueError("Channel groups must sum to y_dim")
        self.deduct_mean = deduct_mean

        self.q_scale_pre = nn.Parameter(torch.ones(65, 1, h_dim))
        self.q_scale_post = nn.Parameter(torch.ones(65, 1, h_dim))
        self.q_scale_enc = nn.Parameter(torch.ones(65, y_dim, 1, 1))
        self.q_scale_dec = nn.Parameter(torch.ones(65, y_dim, 1, 1))
        self.u_scale_pre = nn.Parameter(torch.ones(65, 1, h_dim))
        self.u_scale_post = nn.Parameter(torch.ones(65, 1, h_dim))
        self.register_buffer("U_Transform", torch.eye(h_dim))
        self.register_buffer("mean_vector", torch.zeros(h_dim))

        self.f_a = nn.Sequential(
            DepthwiseSeparableConv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(y_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            DepthwiseSeparableDeconvPixelShuffle(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableDeconvPixelShuffle(y_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableDeconvPixelShuffle(y_dim, h_dim, kernel_size=3, stride=1),
        )
        h_a = nn.Sequential(
            DepthwiseSeparableConv(y_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(z_dim, z_dim, kernel_size=5, stride=2),
        )
        h_s = nn.Sequential(
            DepthwiseSeparableDeconvPixelShuffle(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableDeconvPixelShuffle(z_dim, z_dim * 2, kernel_size=3, stride=1),
        )
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(sum(self.groups[:k]), z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(z_dim, z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(z_dim, self.groups[k] * 2, kernel_size=5, stride=1),
            )
            for k in range(1, len(self.groups))
        }
        scctx = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={"y": GaussianConditionalLatentCodec(quantizer="noise")},
                context_prediction=CheckerboardMaskedConv2d(channels, channels * 2, kernel_size=5, stride=1, padding=2),
                entropy_parameters=sequential_channel_ramp(
                    channels * 2 + (k > 0) * channels * 2 + z_dim * 2,
                    channels * 2,
                    min_ch=z_dim * 2,
                    num_layers=3,
                    interp="linear",
                    make_layer=nn.Conv2d,
                    make_act=lambda: nn.ReLU(inplace=True),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            for k, channels in enumerate(self.groups)
        }
        self.y_lc = ChannelGroupsLatentCodecContiguous(
            groups=self.groups, channel_context=channel_context, latent_codec=scctx
        )
        self.hyper_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim), h_a=h_a, h_s=h_s, quantizer="noise"
        )

    def _apply_gain(self, h, original_gain, pca_gain, qp):
        if not 0 <= qp < 65:
            raise ValueError("QP must be between 0 and 64")
        if h.shape[-2] % 2 or h.shape[-1] % 2:
            raise ValueError("PCA-align requires an even Qwen spatial grid")
        original = h * original_gain[qp : qp + 1].transpose(1, 2).unsqueeze(-1)
        residual = h - self.mean_vector.view(1, -1, 1, 1)
        u = torch.einsum("dc,bchw->bdhw", self.U_Transform, residual)
        u = u * pca_gain[qp : qp + 1].transpose(1, 2).unsqueeze(-1)
        residual = torch.einsum("cd,bdhw->bchw", self.U_Transform.T, u)
        if not self.deduct_mean:
            residual = residual + self.mean_vector.view(1, -1, 1, 1)
        return original + residual

    def forward(self, h, token_res=None, qp=0, **kwargs):
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        h = self._apply_gain(h, self.q_scale_pre, self.u_scale_pre, qp)
        y = self.f_a(h) * self.q_scale_enc[qp : qp + 1]
        y_pad = border_pad(y, self.z_down_factor)
        hyper_out = self.hyper_lc(y_pad)
        y_out = self.y_lc(y_pad, hyper_out["params"])
        y_hat = y_out["y_hat"][:, :, : y.shape[2], : y.shape[3]] * self.q_scale_dec[qp : qp + 1]
        h_hat = self._apply_gain(self.f_s(y_hat), self.q_scale_post, self.u_scale_post, qp)
        return {
            "h_hat": rearrange(h_hat, "B C H W -> B (H W) C").contiguous(),
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, token_res=None, qp=0, **kwargs):
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        h = self._apply_gain(h, self.q_scale_pre, self.u_scale_pre, qp)
        y = self.f_a(h) * self.q_scale_enc[qp : qp + 1]
        y_pad = border_pad(y, self.z_down_factor)
        hyper_out = self.hyper_lc.compress(y_pad)
        y_out = self.y_lc.compress(y_pad, hyper_out["params"])
        return {
            "strings": {"y": y_out["strings"], "z": hyper_out["strings"]},
            "pstate": {
                "y_shape": y_out["shape"],
                "z_shape": hyper_out["shape"],
                "y_pad": tuple(y.shape[2:]),
                "qp": qp,
                "token_res": token_res,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        qp = pstate["qp"]
        y_h, y_w = pstate["y_pad"]
        hyper_out = self.hyper_lc.decompress(strings["z"], pstate["z_shape"])
        y_out = self.y_lc.decompress(strings["y"], pstate["y_shape"], hyper_out["params"])
        y_hat = y_out["y_hat"][:, :, :y_h, :y_w] * self.q_scale_dec[qp : qp + 1]
        h_hat = self._apply_gain(self.f_s(y_hat), self.q_scale_post, self.u_scale_post, qp)
        return {"h_hat": rearrange(h_hat, "B C H W -> B (H W) C").contiguous()}
