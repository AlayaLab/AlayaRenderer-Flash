"""Chunk-by-chunk streaming wrapper around the stateful TAEHV decoder."""

from typing import List, Optional

import torch


class StreamingTinyDecoderCache:
    """Stateful per-chunk wrapper for the TAEHV tiny decoder.

    ``TAEHV.decode_video(..., parallel=False)`` carries MemBlock state inside a
    single call, but resets that state at the next call. This wrapper keeps the
    decoder-block memory alive across chunk calls, so decoding chunk k in call k
    matches a full sequential tiny decode of the concatenated latent stream.
    """

    def __init__(self, tiny_decoder):
        self.tiny_decoder = tiny_decoder
        self.dtype = next(tiny_decoder.parameters()).dtype
        self.device = next(tiny_decoder.parameters()).device
        self.reset()

    def reset(self):
        self._mem: List = [None] * len(self.tiny_decoder.decoder)
        self._is_first = True

    @torch.no_grad()
    def decode_chunk(
        self,
        chunk_latent_model_space: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Decode one model-space latent chunk to NCTHW RGB in [-1, 1]."""
        assert chunk_latent_model_space.shape[0] == 1, "streaming tiny decode requires batch=1"
        model = self.tiny_decoder.decoder
        x = chunk_latent_model_space.to(device=self.device, dtype=self.dtype)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # NCTHW -> NTCHW
        n, t, c, h, w = x.shape
        work_queue = [(xt, 0) for xt in x.reshape(n, t * c, h, w).chunk(t, dim=1)]
        out = []

        while work_queue:
            xt, i = work_queue.pop(0)
            if i == len(model):
                out.append(xt)
                continue

            block = model[i]
            block_name = block.__class__.__name__
            if block_name == "MemBlock":
                if self._mem[i] is None:
                    xt_new = block(xt, xt * 0)
                    self._mem[i] = xt.clone()
                else:
                    xt_new = block(xt, self._mem[i])
                    self._mem[i].copy_(xt)
                work_queue.insert(0, (xt_new, i + 1))
            elif block_name == "TGrow":
                xt = block(xt)
                _nt, c_grow, h_grow, w_grow = xt.shape
                for xt_next in reversed(xt.view(n, block.stride * c_grow, h_grow, w_grow).chunk(block.stride, 1)):
                    work_queue.insert(0, (xt_next, i + 1))
            else:
                work_queue.insert(0, (block(xt), i + 1))

        video = torch.stack(out, 1)  # [1, T_px, 3, H*8, W*8] in approximately [0, 1]
        if self._is_first:
            trim = int(getattr(self.tiny_decoder, "frames_to_trim", 0))
            if trim > 0:
                video = video[:, trim:]
            self._is_first = False

        video = (video.permute(0, 2, 1, 3, 4) * 2.0 - 1.0).clamp(-1.0, 1.0)
        if output_dtype is not None:
            video = video.to(output_dtype)
        return video
