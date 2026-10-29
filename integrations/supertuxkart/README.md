# SuperTuxKart for the live demo

The demo needs a SuperTuxKart build that exports G-buffers and accepts browser
input. That build is a modified version of upstream
[stk-code](https://github.com/supertuxkart/stk-code) and is **not distributed
with this repository**. The offline renderer (`alayarender-flash render`) does
not need it.

A compatible build provides:

- G-buffer export (albedo, depth, metallic, normal, roughness) to a shared-memory
  capture directory (`STK_GBUFFER_CAPTURE_DIR`);
- browser input with AI fallback and input-event watermarks (`STK_BROWSER_INPUT_FILE`);
- optional capture pacing in profile mode (`ALAYA_STK_CAPTURE_FPS`).

**License:** SuperTuxKart is GPL-3.0-or-later. Its game assets are licensed
separately (mostly CC-BY-SA). Neither the binary nor the assets are distributed here.

## Running

The demo renders headless through EGL (`SDL_VIDEODRIVER=offscreen`) and needs the
NVIDIA EGL vendor file (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`).

Point `ALAYA_STK_BINARY` in your `deploy.env` at the `supertuxkart` binary. The
game data ([stk-assets](https://svn.code.sf.net/p/supertuxkart/code/stk-assets))
must be next to the binary as upstream expects.

The demo copies a minimal STK user profile
(`src/alayarenderer_flash/demo/stk_home_template/`) into each run's private home
directory, so the first-run setup screens are skipped.
