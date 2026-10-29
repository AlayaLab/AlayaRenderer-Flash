from pathlib import Path

import numpy as np
import pytest

from alayarenderer_flash.cli import build_parser
from alayarenderer_flash.render import load_gbuffer_stream
from alayarenderer_flash.runtime import RuntimeConfig
from alayarenderer_flash.weights import resolve_base_model, resolve_checkpoint


def test_runtime_config_defaults(tmp_path):
    config = RuntimeConfig(base_model=tmp_path / "wan", checkpoint=tmp_path / "ckpt")
    assert config.small_encoder == tmp_path / "ckpt" / "gbuf_small_encoder.safetensors"
    assert config.tiny_decoder == tmp_path / "ckpt" / "tiny_decoder" / "taew2_1_ft.safetensors"
    assert (config.height, config.width, config.num_inference_steps) == (448, 832, 4)
    assert config.temporal_cadence.initial_pixel_frames == 13


def test_weights_local_override(tmp_path, monkeypatch):
    assert resolve_checkpoint(tmp_path) == tmp_path
    monkeypatch.setenv("ALAYA_BASE_MODEL", str(tmp_path))
    assert resolve_base_model() == Path(tmp_path)


def test_cli_render_args():
    args = build_parser().parse_args(["render", "a.npy", "b.npy", "-p", "snow", "--text-encoder-cpu"])
    assert args.gbuffers == [Path("a.npy"), Path("b.npy")]
    assert args.prompt == "snow" and args.text_encoder_cpu


def test_load_gbuffer_stream(tmp_path):
    for i in range(2):
        np.save(tmp_path / f"w{i}.npy", np.zeros((3, 5, 8, 8, 3), np.uint8))
    assert load_gbuffer_stream([tmp_path / "w0.npy", tmp_path / "w1.npy"]).shape == (6, 5, 8, 8, 3)
    np.save(tmp_path / "bad.npy", np.zeros((3, 4, 8, 8, 3), np.uint8))
    with pytest.raises(ValueError):
        load_gbuffer_stream([tmp_path / "bad.npy"])
