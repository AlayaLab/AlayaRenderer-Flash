# Live SuperTuxKart demo

## Requirements

- Linux, an NVIDIA GPU with an EGL-capable driver, and `/dev/shm`.
- Real-time rates need a Hopper-class GPU (H100/H200/H20). The model alone runs
  at about 31 FPS on one H200.
- `tmux` and `curl`, which the service supervisor uses.

## 1. Install the demo extra

```bash
uv sync --extra demo   # OpenCV; without it the bridge caps output near 20 FPS
```

## 2. Get a SuperTuxKart build with G-buffer export

This build is not distributed here. See
[integrations/supertuxkart/README.md](../integrations/supertuxkart/README.md)
for what it must provide.

## 3. Download the weights

```bash
uv run alayarender-flash download      # prints the cached snapshot paths
uv run python -c "from kernels import get_kernel; get_kernel('kernels-community/flash-attn3')"
```

The worker runs with `HF_HUB_OFFLINE=1`, so the FlashAttention kernels have to
be in the Hugging Face cache before you start it.

## 4. Configure

```bash
cp examples/deploy.env.example deploy.env
# set ALAYA_STK_BINARY, ALAYA_BASE_MODEL and ALAYA_MODEL_CHECKPOINT
```

## 5. Run

```bash
uv run alayarender-flash serve --config deploy.env doctor --json   # preflight checks
uv run alayarender-flash serve --config deploy.env start
uv run alayarender-flash serve --config deploy.env status --json
uv run alayarender-flash serve --config deploy.env logs worker --follow
uv run alayarender-flash serve --config deploy.env stop
```

The first start compiles the model, which takes a few minutes; the compile
cache is reused afterwards. Open `http://127.0.0.1:8095`, through an SSH
tunnel for a remote host:

```bash
ssh -N -L 8095:127.0.0.1:8095 user@gpu-host
```

Logs live under `$ALAYA_STATE_DIR/logs/<runtime id>/`, one file per service.
A `logcap` window in the service's tmux session rotates each log in place once
it passes `ALAYA_LOG_MAX_BYTES` (default 50 MB), keeping `ALAYA_LOG_BACKUPS`
copies (default 3), and keeps only the newest 20 worker hang dumps.

## Landing page media

The landing page shows an optional demo video and poster. Neither ships with
this repository because both are rendered from SuperTuxKart scenes; place your
own files at `src/alayarenderer_flash/demo/assets/landing-demo-poster.png` and
`landing-demo-first-60s.mp4` to enable them. Without them the page still works
and PLAY opens the live demo.

## Access control

The preview server binds to `127.0.0.1` by default. Put it behind a reverse
proxy rather than binding it to a public interface.

| Endpoint | Who | Notes |
| --- | --- | --- |
| `GET` pages, frames, `/state`, `/ws/stream` | visitors | read-only, except driving input over the socket |
| `POST /input`, `/api/prompt`, `/api/track` | visitors | 64 KiB JSON-object bodies, 10 s read timeout, 16 concurrent body reads; prompts have a per-IP cooldown |
| `POST /record/start`, `/record/stop` | operators | disabled unless `ALAYA_PREVIEW_ADMIN_TOKEN_FILE` (or `ALAYA_PREVIEW_ADMIN_TOKEN`) is set; requests must send `X-Alaya-Admin-Token` |

Browser `POST` and WebSocket requests must come from the server's own host or
an origin listed in `ALAYA_PREVIEW_ALLOWED_ORIGINS` (comma-separated, e.g. the
public demo URL); other origins get `403`. With `--read-only-observer`, input,
prompt, track and recording requests are all rejected.

## Controls

- **Auto**: an AI driver steers; the UI shows its inputs.
- **User Input**: press a direction key to take over; switch back to Auto to release.
- **Style**: pick a preset or type a prompt and press Apply. The model history
  resets and the last frame stays on screen until the new style renders.

## Troubleshooting

- **Stuck on loading**: check `logs worker`. Usual causes are the first compile,
  missing kernels in the HF cache, a wrong weight path, or GPU memory.
- **Low FPS**: make sure nothing else is using the GPU, or set
  `ALAYA_WORKER_GPU` to a dedicated card.
- **Attention kernel errors on new GPUs**: set `ALAYA_ATTN_BACKEND=sdpa`.
