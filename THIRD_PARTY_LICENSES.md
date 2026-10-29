# Third-party licenses

AlayaRenderer-Flash is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`).
This file lists third-party code, models, runtime downloads, dependencies and
media that the project contains or uses. Licenses follow each upstream's own
declaration at the versions below; check upstream for the authoritative text.

## Code included in this repository

| Component | Upstream | License | Location | License text |
| --- | --- | --- | --- | --- |
| Helios (modified) | https://github.com/PKU-YuanGroup/Helios | Apache-2.0 | `src/alayarenderer_flash/models/transformer.py`, `pipeline/helios.py`, `models/kernels/` | `LICENSE` (same license); per-file headers mark modifications |
| TAEHV (vendored) | https://github.com/madebyollin/taehv | MIT | `src/alayarenderer_flash/decoders/taehv.py` | `LICENSES/TAEHV-MIT.txt` |

## Models and weights

| Component | Upstream | License | How it is used |
| --- | --- | --- | --- |
| Wan2.1-T2V-1.3B-Diffusers | https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers | Apache-2.0 | Base model; text encoder, tokenizer, VAE config downloaded by the user, not redistributed |
| TAEW2.1 decoder weights | https://github.com/madebyollin/taehv | MIT | Fine-tuned into `tiny_decoder/taew2_1_ft.safetensors` in the weights repository |
| AlayaRenderer-Flash checkpoint | https://huggingface.co/AlayaLab/AlayaRenderer-Flash | See the model repository's `LICENSE` | Downloaded at runtime |

## Downloaded at runtime (not redistributed)

| Component | Source | License | Notes |
| --- | --- | --- | --- |
| FlashAttention 3 / 2 kernels | `kernels-community/flash-attn3`, `kernels-community/flash-attn2` on the Hugging Face Hub | BSD-3-Clause (Dao-AILab/flash-attention); confirm against the Hub repository | Fetched by `kernels.get_kernel` with a pinned kernel version; `ALAYA_ATTN_BACKEND=sdpa` avoids the download |

## Python dependencies (versions from `uv.lock`)

| Package | Locked version | License |
| --- | --- | --- |
| torch | 2.11.0+cu128 | BSD-3-Clause |
| torchvision | 0.26.0+cu128 | BSD-3-Clause |
| NVIDIA CUDA runtime wheels (`nvidia-*`, pulled in by torch) | per `uv.lock` | NVIDIA proprietary EULAs |
| diffusers | 0.42.0 | Apache-2.0 |
| transformers | 5.20.2 | Apache-2.0 |
| tokenizers | 0.23.3 | Apache-2.0 |
| accelerate | 1.15.0 | Apache-2.0 |
| huggingface-hub | 2.3.0 | Apache-2.0 |
| safetensors | 0.8.0 | Apache-2.0 |
| kernels | 0.17.2 | Apache-2.0 |
| numpy | 2.4.6 | BSD-3-Clause |
| pillow | 13.0.0 | MIT-CMU |
| einops | 0.8.2 | MIT |
| imageio | 2.38.1 | BSD-2-Clause |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause; its bundled FFmpeg binary is LGPL/GPL depending on the build |
| opencv-python-headless (`demo` extra) | 5.0.0.93 | Apache-2.0; bundled FFmpeg is LGPL |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| ftfy | 6.3.1 | Apache-2.0 |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| sentencepiece | 0.2.2 | Apache-2.0 |

Transitive dependencies are pinned in `uv.lock`; none is redistributed by this
repository.

## External programs

| Program | License | Notes |
| --- | --- | --- |
| SuperTuxKart (engine and game assets) | Engine GPL-3.0-or-later; assets mostly CC-BY-SA, varying per asset | Not distributed; the demo talks to a user-provided build over shared memory |

## Media in this repository

| File | What it shows | Origin | License status |
| --- | --- | --- | --- |
| `assets/teaser.png` | Project overview illustration | Designed by Alaya Lab and generated with OpenAI GPT Image | Output owned by Alaya Lab under OpenAI's terms of use; released with this repository under Apache-2.0 |
| `src/alayarenderer_flash/demo/assets/alaya_renderer_mark.png`, `alaya-renderer-mark-white.png` | Alaya Renderer logo | Alaya Lab | Alaya Lab trademark/logo; not covered by Apache-2.0 trademark grant |
