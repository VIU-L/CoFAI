"""PCA-align packet/checkpoint contracts; formal task results are separate."""

from pathlib import Path

import pytest
import torch
from torch import nn
from compressai.models.base import CompressionModel
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from cofai.engine.bitrate import bits_from_coded_data
from cofai.engine.checkpoint import load_checkpoint_into_model
from cofai.latent_codecs import PCAAlignLatentCodec
from cofai.utils.download import parse_manifest


class _Container(CompressionModel):
    def __init__(self):
        super().__init__()
        self.qwen = nn.Linear(1, 1)
        self.qwen_codec = PCAAlignLatentCodec(h_dim=32, y_dim=16, z_dim=8, groups=8)


@pytest.mark.parametrize("grid", [(6, 10), (8, 12)])
def test_packet_and_entropy_state_survive_common_checkpoint_loading(tmp_path, grid):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = _Container().eval()
    codec = model.qwen_codec
    with torch.no_grad():
        codec.U_Transform.copy_(torch.linalg.qr(torch.randn(32, 32)).Q)
        codec.mean_vector.copy_(torch.randn(32))
        codec.u_scale_pre[32].fill_(0.3)
        codec.u_scale_post[32].fill_(0.7)
    model.update(force=True, update_quantiles=True)
    checkpoint = tmp_path / "inference.pth"
    torch.save({"state_dict": {f"qwen_codec.{k}": v for k, v in codec.state_dict().items()}}, checkpoint)

    restored = _Container().eval()
    backbone_before = {k: v.clone() for k, v in restored.qwen.state_dict().items()}
    load_checkpoint_into_model(restored, {"path": str(checkpoint), "weights_only": True, "strict": False})
    restored.update()
    for key, value in codec.state_dict().items():
        assert torch.equal(value, restored.qwen_codec.state_dict()[key]), key
    for key, value in backbone_before.items():
        assert torch.equal(value, restored.qwen.state_dict()[key])

    h = torch.randn(1, grid[0] * grid[1], 32)
    with torch.inference_mode():
        coded = codec.compress(h, token_res=grid, qp=32)
        packet_path = tmp_path / "packet.pth"
        torch.save(coded, packet_path)
        packet = torch.load(packet_path, weights_only=True)
        a = codec.decompress(**coded)["h_hat"]
        b = restored.qwen_codec.decompress(**packet)["h_hat"]
        expected_forward = codec(h, token_res=grid, qp=32)
        loaded_forward = restored.qwen_codec(h, token_res=grid, qp=32)
    assert a.shape == h.shape and a.is_contiguous()
    assert torch.equal(a, b)
    assert torch.equal(expected_forward["h_hat"], loaded_forward["h_hat"])
    assert coded["pstate"]["token_res"] == grid and coded["pstate"]["qp"] == 32
    assert coded["pstate"]["y_pad"] == (grid[0] // 2, grid[1] // 2)
    assert set(bits_from_coded_data(coded)) == {"y", "z"}
    assert all(bits > 0 for bits in bits_from_coded_data(coded).values())


@pytest.mark.parametrize("qp", [-1, 65])
def test_reject_out_of_range_qp(qp):
    codec = PCAAlignLatentCodec(h_dim=32, y_dim=16, z_dim=8, groups=8)
    with pytest.raises(ValueError, match="QP"):
        codec(torch.zeros(1, 60, 32), token_res=(6, 10), qp=qp)


def test_reject_odd_spatial_grid():
    codec = PCAAlignLatentCodec(h_dim=32, y_dim=16, z_dim=8, groups=8)
    with pytest.raises(ValueError, match="even Qwen spatial grid"):
        codec(torch.zeros(1, 30, 32), token_res=(5, 6))


def test_example_inherits_ctc_and_matches_resource_manifest():
    repo = Path(__file__).resolve().parents[1]
    example = repo / "examples/pca_align/plan"
    stem = "mmstar__qwen3vl-8b-ins-slot9__pca-align__vqa"
    with initialize_config_dir(version_base=None, config_dir=str(example)):
        cfg = compose(
            config_name=stem,
            overrides=[f"hydra.searchpath=[file://{repo / 'conf'}]"],
        )
    with initialize_config_dir(version_base=None, config_dir=str(repo / "conf")):
        reference = compose(config_name="plan/mmstar__qwen3vl-8b-ins-slot9__Bypass__vqa")
    with open_dict(cfg):
        cfg.PROJECT_ROOT = "/tmp/cofai-pca-align-test"
    assert cfg.model.qwen_backbone.max_new_tokens == 512
    for field in ["model_path", "dtype", "min_pixels", "max_pixels", "slot", "do_sample"]:
        assert cfg.model.qwen_backbone[field] == reference.model.qwen_backbone[field]
    assert OmegaConf.to_container(cfg.dataset) == OmegaConf.to_container(reference.dataset)
    assert cfg.args.real and cfg.args.multi_run and cfg.args.max_samples is None
    assert list(cfg.multi_run) == [0, 8, 16, 32, 48, 64]
    assert cfg.load.weights_only and not cfg.load.strict
    entries = parse_manifest(repo / "examples/pca_align/pca-align.manifest.txt")
    assert len(entries) == 1
    assert str(cfg.load.path).endswith(entries[0].dest)
