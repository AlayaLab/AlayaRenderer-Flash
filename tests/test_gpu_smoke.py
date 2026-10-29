"""End-to-end render with the released weights (needs CUDA and the weights).

Set ALAYA_CHECKPOINT / ALAYA_BASE_MODEL to local snapshots (or allow hub
downloads) and ALAYA_SAMPLE_WINDOWS to two fixture windows to enable it.
"""

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.skipif("ALAYA_SAMPLE_WINDOWS" not in os.environ, reason="set ALAYA_SAMPLE_WINDOWS=w0.npy:w1.npy")
def test_render_two_chunks():
    from alayarenderer_flash.render import OfflineRenderer, load_gbuffer_stream
    from alayarenderer_flash.runtime import RuntimeConfig
    from alayarenderer_flash.weights import resolve_base_model, resolve_checkpoint

    stream = load_gbuffer_stream(os.environ["ALAYA_SAMPLE_WINDOWS"].split(":"))
    config = RuntimeConfig(base_model=resolve_base_model(), checkpoint=resolve_checkpoint())
    frames = OfflineRenderer(config, text_encoder_on_cpu=True).render(stream[:29], "snowy night")
    assert frames.shape == (29, 448, 832, 3) and frames.dtype == np.uint8
    assert 5 < frames.mean() < 250
