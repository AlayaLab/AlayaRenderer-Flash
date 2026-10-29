"""Offline rendering of recorded G-buffer streams."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch

from .runtime import ModelRuntime, RuntimeConfig, SessionToken

__all__ = ["OfflineRenderer", "load_gbuffer_stream"]


def load_gbuffer_stream(paths: Sequence[str | Path]) -> np.ndarray:
    """Concatenate G-buffer windows into one ``(T, 5, H, W, 3)`` uint8 stream.

    Each ``.npy`` file holds frames in the order albedo, depth, metallic,
    normal, roughness, as exported by SuperTuxKart with G-buffer export.
    """
    stream = np.concatenate([np.load(path, mmap_mode="r") for path in paths], axis=0)
    if stream.ndim != 5 or stream.shape[1] != 5 or stream.shape[-1] != 3 or stream.dtype != np.uint8:
        raise ValueError(f"expected uint8 (T, 5, H, W, 3) G-buffers, got {stream.dtype} {stream.shape}")
    return stream


def _park_text_encoder_on_cpu(pipe) -> None:
    """Keep T5 on the CPU and move it to the GPU only while encoding a prompt.

    Prompt embeddings are cached by the runtime, so this costs one transfer per
    prompt. Running T5 itself on the CPU in bf16 drifts from the GPU result.
    """
    device = pipe._execution_device
    encoder = pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    forward = encoder.forward

    def gpu_forward(*args, **kwargs):
        encoder.to(device)
        try:
            return forward(*args, **kwargs)
        finally:
            encoder.to("cpu")
            torch.cuda.empty_cache()

    encoder.forward = gpu_forward
    type(pipe)._execution_device = property(lambda self: device)


class OfflineRenderer:
    """Render G-buffer streams causally with the live-demo runtime.

    The first chunk uses 13 frames and every later chunk the next 16, exactly
    like the live bridge. Frames that do not fill a final chunk are dropped.
    """

    def __init__(self, config: RuntimeConfig, *, text_encoder_on_cpu: bool = False) -> None:
        self.runtime = ModelRuntime(config)
        self.runtime.load()
        if text_encoder_on_cpu:
            _park_text_encoder_on_cpu(self.runtime.pipe)

    def chunks(self, stream: np.ndarray, prompt: str) -> Iterator[np.ndarray]:
        """Yield uint8 RGB frames ``(N, H, W, 3)`` for each rendered chunk."""
        runtime = self.runtime
        runtime.reset_session(SessionToken(0, 0, 0, 0, "offline", prompt, "offline", "offline"))
        first = runtime.cadence.initial_pixel_frames
        step = runtime.cadence.resumed_pixel_frames
        if stream.shape[0] < first:
            raise ValueError(f"need at least {first} G-buffer frames, got {stream.shape[0]}")
        starts = [0, *range(first, stream.shape[0] - step + 1, step)]
        for index, start in enumerate(starts):
            end = first if index == 0 else start + step
            video, _ = runtime._run_chunk(
                np.ascontiguousarray(stream[start:end]), prompt, index == 0, start - 1, lambda: None
            )
            yield runtime._video_to_rgb(video)

    def render(self, stream: np.ndarray, prompt: str) -> np.ndarray:
        """Render a whole stream and return uint8 RGB frames ``(N, H, W, 3)``."""
        return np.concatenate(list(self.chunks(stream, prompt)))
