import torch
import torch.nn as nn
import torchvision.transforms as transforms
from einops import rearrange
from .base import parse_dtype, BackboneProtocol
from cofai.backbone.vqgan.vq_model import VQModel  # type: ignore


class VqganBackbone(nn.Module):
    """VQGAN-based backbone for image encoding and decoding.

    This backbone uses a VQGAN (Vector Quantized Generative Adversarial Network) model
    to encode images into discrete tokens and decode them back to images. The encoding
    process converts images to latent codes and quantizes them using a codebook.

    Args:
        vqgan_config (dict): Configuration dictionary for the VQModel initialization.
        **kwargs (dict): Unused keyword arguments for API compatibility.

    Attributes:
        vqgan (VQModel): The underlying VQGAN model.
        codebook_size (int): Size of the quantization codebook.
    """

    def __init__(self, **kwargs):
        super().__init__()
        if "type" in kwargs:
            kwargs.pop("type")
        self.vqgan = VQModel(**kwargs)
        self.codebook_size = self.vqgan.quantize.embedding.weight.size()[0]

    def forward(self, x):
        res = self.encode(x)
        x_hat = self.decode(res["z_q"])
        return x_hat

    def encode(self, x):
        """Encode input images into latent codes and tokens.

        The input images x are expected to be in the range [0, 1], which are then
        transformed to [-1, 1] for the VQGAN encoder. The encoder produces latent
        codes that are quantized using the codebook to produce discrete tokens.
        The quantization process produces z_q' = z + (z_q - z).detach(), which
        incurs a small MSE error (approximately 1e-18) between z_q' and z_q. We
        use z_q as the context for consistency.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W) in range [0, 1].

        Returns:
            vqgan_enc (dict): A dictionary containing:

                - "z" (torch.Tensor): Continuous latent codes before quantization.
                - "z_q" (torch.Tensor): Quantized latent codes of shape (B, C, H, W).
                - "tokens" (torch.Tensor): Discrete token indices of shape (B, H, W).
                - "shape" (tuple): Spatial dimensions (H, W) of the latent representation.
        """
        z = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
        z_q_prime, _, (_, _, idxs_1d) = self.vqgan.quantize(z)
        B, C, H, W = z_q_prime.shape
        tokens = idxs_1d.reshape(B, H, W)
        z_q = self.vqgan.quantize.embedding(idxs_1d)
        z_q = rearrange(z_q, "(B H W) C -> B C H W", B=B, H=H, W=W)
        return {"z": z, "z_q": z_q, "tokens": tokens, "shape": (H, W)}

    def tokens_to_features(self, tokens):
        """Convert discrete tokens to quantized latent features.

        Args:
            tokens (torch.Tensor): Discrete token indices of shape (B, H, W).

        Returns:
            z_q (torch.Tensor): Quantized latent features of shape (B, C, H, W).
        """
        B, H, W = tokens.shape
        z_q = self.vqgan.quantize.embedding(tokens.flatten())
        z_q = rearrange(z_q, "(B H W) C -> B C H W", B=B, H=H, W=W)
        return z_q

    def decode(self, z_q):
        """Decode quantized latent codes back to images.

        Args:
            z_q (torch.Tensor): Quantized latent codes of shape (B, C, H, W).

        Returns:
            x_hat (torch.Tensor): Reconstructed images of shape (B, 3, H, W) in range [0, 1].
        """
        x_hat = self.vqgan.decode(z_q)
        x_hat = (x_hat + 1) / 2
        return x_hat
