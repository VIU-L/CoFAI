from .base import BackboneProtocol
from .tokenizer import VqganBackbone
from .timm import Dinov2TimmBackbone
from .dinov2_org import Dinov2OrgBackbone
from .vgg import VGGBackbone, setup_vgg_feature_extractor, extract_vgg_features

__all__ = [
    "BackboneProtocol",
    "VqganBackbone",
    "Dinov2TimmBackbone",
    "Dinov2OrgBackbone",
    "VGGBackbone",
    "setup_vgg_feature_extractor",
    "extract_vgg_features",
]

backbone_tools = {
    "setup_vgg_feature_extractor": setup_vgg_feature_extractor,
    "extract_vgg_features": extract_vgg_features,
}
