import torch
import torch.nn as nn
import torchvision.transforms as transforms
from einops import rearrange
from .base import parse_dtype, BackboneProtocol


class Dinov2OrgBackbone(nn.Module):
    """DINOv2 backbone using the original Facebook Research implementation.

    This class extends the original DINOv2 model to provide flexible feature extraction.
    The `slot` parameter determines the splitting point for dividing the ViT blocks into:

    * encode part: blocks[:slot],
    * decode part: blocks[slot:]

    Intermediate features are extracted after the encode part and before the decode part.

    Args:
        model_size (str): Model variant specification ('small', 'base', 'large', 'giant').
                         Defaults to 'small'.
        img_size (int): Base input image size. Defaults to 256.
        patch_size (int): Patch embedding size. Defaults to 16.
        dynamic_size (bool): Whether to support dynamically varying input sizes.
                            Defaults to False.
        slot (int): Block slicing position for feature extraction. Follows Python list
                   slicing conventions. -4 means the last 4th block. Defaults to -4.
        n_last_blocks (int): Number of final blocks to utilize for feature aggregation.
                            Defaults to 4.
        ckpt_path (str, optional): Path to pre-trained checkpoint for initialization.
                                  Defaults to None.
    """

    def __init__(
        self,
        model_size="small",
        img_size=256,
        patch_size=16,
        dynamic_size=False,
        slot=-4,  # cut position, -4 means the last 4th block
        n_last_blocks=4,  # number of last blocks to take
        ckpt_path=None,
    ):
        super().__init__()
        self.n_last_blocks = n_last_blocks
        assert model_size in ["small", "base", "large", "giant"]
        self.model_size = model_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.dynamic_size = dynamic_size
        self.slot = slot
        self.n_last_blocks = n_last_blocks
        self.ckpt_path = ckpt_path
        self.model = self.load_dinov2_model()
        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )

    def load_dinov2_model(self):
        from cofai.backbone.dinov2.hub.backbones import _make_dinov2_model

        if self.model_size == "small":
            model = _make_dinov2_model(
                arch_name="vit_small", pretrained=True, weights=self.ckpt_path
            )
        elif self.model_size == "base":
            model = _make_dinov2_model(
                arch_name="vit_base", pretrained=True, weights=self.ckpt_path
            )
        elif self.model_size == "large":
            model = _make_dinov2_model(
                arch_name="vit_large", pretrained=True, weights=self.ckpt_path
            )
        elif self.model_size == "giant":
            model = _make_dinov2_model(
                arch_name="vit_giant2",
                ffn_layer="swiglufused",
                pretrained=True,
                weights=self.ckpt_path,
            )
        model.eval()
        return model

    def load_dinov2_model_from_torch_hub(self):
        # relatively slow to fetch vit config from torch hub
        model_configs = {
            "small": "dinov2_vits14",
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14",
            "giant": "dinov2_vitg14",
        }
        model_name = model_configs[self.model_size]
        if self.ckpt_path:
            model = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=False
            )
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint)
        else:
            print("load checkpoint from torch hub, this may be slow...")
            model = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=True
            )
        model.eval()
        return model

    def forward(self, x, task="whole"):
        """Forward pass through the backbone.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
            task (str): Task type, one of ["whole", "cls", "seg"]. Defaults to "whole".

        Returns:
            feats (list[torch.Tensor, ...]): Output features, format depends on task.
        """
        assert task in ["whole", "cls", "seg"]
        with torch.inference_mode():
            h = self.encode(x)
            token_res = (x.size(2) // self.patch_size, x.size(3) // self.patch_size)
            h = self.decode(h, token_res=token_res, task=task)
            return h

    def encode(self, x):
        """Encode input images through the encoder part of the DINOv2 model.

        The encoding process applies input normalization, prepares tokens with masks,
        and processes the input through the first `slot` transformer blocks.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            h (torch.Tensor): Encoded features after the encoder blocks.
        """
        dino = self.model
        x = self.input_transform(x)
        x = dino.prepare_tokens_with_masks(x)
        for i, blk in enumerate(dino.blocks[: self.slot]):
            x = blk(x)
        return x

    def decode(self, h, token_res=None, task="whole"):
        """Decode encoded features through the decoder part of the DINOv2 model.

        Args:
            h (torch.Tensor): Encoded features from the encoder.
            token_res (tuple, optional): Token resolution (H, W) for reshaping patch tokens.
                                        Defaults to None.
            task (str): Decoding task type. Must be one of:

                - "whole": Return full token sequences from multiple layers.
                - "cls": Return class tokens and patch tokens separately.
                - "seg": Return patch tokens reshaped to 2D spatial format.
                Defaults to "whole".

        Returns:
            feats (list[torch.Tensor, ...]): Decoded features, format depends on task.
        """
        if task == "whole":
            return self.decode_whole(h, token_res=token_res)
        elif task == "cls":
            return self.decode_cls(h, token_res=token_res)
        elif task == "seg":
            return self.decode_seg(h, token_res=token_res)

    def _decode(
        self,
        x,
        slot=-4,
        n=4,  # Layers or n last layers to take
        norm=True,
        return_format="[whole]",
        token_res=None,
    ):
        allow_formats = ["[whole]", "[cls,patch]", "[cls]", "[patch]", "[patch2d]"]
        assert return_format in allow_formats, (
            f"return_format must be one of {allow_formats}"
        )
        dino = self.model

        # If n is an int, take the n last blocks. If it's a list, take them
        multi_outputs = []
        if slot is None:
            multi_outputs.append(x)  # x is just the last layer
        else:
            total_block_len = len(dino.blocks)
            if isinstance(n, int):
                blocks_to_take = range(total_block_len - n, total_block_len)
            else:
                blocks_to_take = n

            for i, blk in enumerate(dino.blocks[slot:]):
                x = blk(x)
                block_num = total_block_len + slot + i
                if block_num in blocks_to_take:
                    multi_outputs.append(x)
            assert len(multi_outputs) == len(blocks_to_take), (
                f"only {len(multi_outputs)} / {len(blocks_to_take)} blocks found"
            )

        if norm:
            multi_outputs = [dino.norm(out) for out in multi_outputs]

        if return_format == "[whole]":
            return multi_outputs

        multi_class_tokens = [out[:, 0] for out in multi_outputs]
        multi_patch_tokens = [
            out[:, dino.num_register_tokens + 1 :] for out in multi_outputs
        ]

        if return_format == "[cls,patch]":
            # feature list: [[cls token, patch tokens], ..., [cls token, patch tokens]]
            return tuple(zip(multi_class_tokens, multi_patch_tokens))
        elif return_format == "[cls]":
            return multi_class_tokens
        elif return_format == "[patch]":
            return multi_patch_tokens
        elif return_format == "[patch2d]":
            h, w = token_res
            multi_patch_tokens = [
                rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
                for out in multi_patch_tokens
            ]
            return multi_patch_tokens

    def decode_whole(self, h, token_res=None):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[whole]",
            token_res=token_res,
        )

    def decode_cls(self, h, token_res=None):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[cls,patch]",
            token_res=token_res,
        )

    def decode_seg(self, h, token_res):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[patch2d]",
            token_res=token_res,
        )

    def slide_encode(self, img, slide_window, slide_stride):
        """Encode images using sliding window approach.

        This method extracts features from overlapping image crops using a sliding
        window strategy. Useful for processing large images that don't fit in memory
        or for extracting features at multiple scales.

        Args:
            img (torch.Tensor): Input image of shape (B, 3, H_img, W_img).
            slide_window (tuple): Window size for cropping (h_crop, w_crop).
            slide_stride (tuple): Stride for sliding window (h_stride, w_stride).

        Returns:
            multi_crop_feats (list[torch.Tensor]): List of encoded features, one for each crop.
        """
        h_crop, w_crop = slide_window
        h_stride, w_stride = slide_stride
        _, _, h_img, w_img = img.shape

        multi_crop_features = []
        for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
            for w_idx in range(
                0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            ):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                # Crop and extract features
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_features = self.encode(crop_img)
                multi_crop_features.append([crop_features])

        return multi_crop_features

    def slide_decode_seg(self, feature_list, slide_res):
        """Decode features from sliding window encoding for segmentation task.

        Args:
            feature_list (list): List of encoded features from slide_encode.
            slide_res (tuple): Token resolution (H, W) for each crop.

        Returns:
            multi_crop_feats (list[list[torch.Tensor, ...]]): List of decoded segmentation features, one for each crop.
        """
        return [self.decode_seg(h[0], token_res=slide_res) for h in feature_list]
