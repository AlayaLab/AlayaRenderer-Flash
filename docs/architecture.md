# Architecture

AlayaRenderer-Flash renders a stream of engine G-buffers (albedo, depth,
metallic, normal, roughness) into RGB frames, conditioned on a text prompt.

```text
G-buffer frames ──► grouped G-buffer encoder ──► 5×16 latent channels ─┐
                                                                        ▼
text prompt ──► UMT5 ──► text embeddings ──► Helios DiT (1.3B, 4 Euler steps)
                                                 ▲            │
                    long/mid/short latent history ┘            ▼
                                                     stateful tiny decoder ──► RGB
```

## Package layout

| Module | Role |
|---|---|
| `alayarenderer_flash.models.transformer` | `HeliosTransformer3DModel`: Wan 2.1 DiT with multi-term memory patches (history 8/2/1), first-frame anchor, text-sink attention, and 96 input channels (16 RGB + 80 G-buffer) |
| `alayarenderer_flash.models.kernels` | Triton RMSNorm/LayerNorm/RoPE kernels and the attention dispatcher (FlashAttention 3 on Hopper, FlashAttention 2, PyTorch SDPA) |
| `alayarenderer_flash.models.gbuffer_encoder` | Distilled grouped G-buffer encoder that replaces the Wan VAE encoder |
| `alayarenderer_flash.models.loading` | Loads `transformer_partial` weights and checks that G-buffer and text-sink weights are present |
| `alayarenderer_flash.pipeline.helios` | Autoregressive streaming sampler: chunked denoising with the latent history carried across calls |
| `alayarenderer_flash.decoders` | TAEHV tiny decoder (MIT) and the chunk-streaming wrapper that keeps decoder memory across chunks |
| `alayarenderer_flash.gbuffer` | Causal window contract: 13 frames for the first chunk, then 16 new frames plus 1 boundary frame |
| `alayarenderer_flash.runtime` | `ModelRuntime`: loads everything, keeps the session state on the GPU, renders one window at a time |
| `alayarenderer_flash.render` | `OfflineRenderer` for recorded G-buffer streams |
| `alayarenderer_flash.demo` | Live services: bridge, worker, web preview, and their shell supervisors |

## Streaming contract

The first chunk encodes 13 G-buffer frames into 4 latents. Each later chunk
takes 16 new frames plus the previous boundary frame (17 inputs), encodes them
to 5 latents, and drops the first latent. Anything that breaks continuity resets
the history: a scene change, a restarted race, or a new prompt.

## Live demo processes

1. **stk**: a SuperTuxKart build with G-buffer export writes G-buffers into shared memory.
2. **bridge** (`demo/bridge.py`): assembles causal windows and publishes them
   through a double-buffered shared-memory transport (`transport.py`).
3. **worker** (`demo/worker.py`): runs `ModelRuntime` with `torch.compile` and
   publishes frames at an adaptive rate.
4. **preview** (`demo/preview.py`): serves the web UI, the video stream, style
   prompts, and player input.

`demo/scripts/alaya_stk_checkpoint_service.sh` supervises all four; it backs
`alayarender-flash serve`.
