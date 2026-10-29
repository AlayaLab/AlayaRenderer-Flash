"""Command-line interface: ``alayarender-flash {render,serve,download}``."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

DEFAULT_PROMPT = "natural daylight forest rally, balanced colors, realistic game render"


def _add_weight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="local checkpoint dir (default: download from the Hub)"
    )
    parser.add_argument(
        "--base-model", type=Path, default=None, help="local Wan2.1-T2V-1.3B-Diffusers dir (default: download)"
    )


def _render(args: argparse.Namespace) -> int:
    import imageio.v2 as imageio
    import numpy as np

    from .render import OfflineRenderer, load_gbuffer_stream
    from .runtime import RuntimeConfig
    from .weights import resolve_base_model, resolve_checkpoint

    config = RuntimeConfig(
        base_model=resolve_base_model(args.base_model),
        checkpoint=resolve_checkpoint(args.checkpoint),
        device=args.device,
        seed=args.seed,
        compile_transformer=args.compile,
        compile_small_encoder=args.compile,
        compile_tiny_decoder=args.compile,
    )
    stream = load_gbuffer_stream(args.gbuffers)
    renderer = OfflineRenderer(config, text_encoder_on_cpu=args.text_encoder_cpu)
    frames = []
    for index, chunk in enumerate(renderer.chunks(stream, args.prompt)):
        frames.append(chunk)
        print(f"chunk {index}: {chunk.shape[0]} frames", file=sys.stderr)
    rgb = np.concatenate(frames)
    if args.save_frames:
        np.save(args.save_frames, rgb)
    imageio.mimwrite(args.output, list(rgb), fps=args.fps, quality=8)
    print(f"wrote {rgb.shape[0]} frames to {args.output}", file=sys.stderr)
    return 0


def _download(args: argparse.Namespace) -> int:
    from .weights import resolve_base_model, resolve_checkpoint

    print(f"checkpoint: {resolve_checkpoint()}")
    print(f"base model: {resolve_base_model()}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    script = resources.files("alayarenderer_flash.demo") / "scripts" / "alaya_stk_checkpoint_service.sh"
    env = dict(os.environ)
    env.setdefault("ALAYA_PYTHON_BIN", sys.executable)
    return subprocess.call(["bash", str(script), "--config", str(args.config), *args.service_args], env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alayarender-flash", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render recorded G-buffer windows to a video")
    render.add_argument("gbuffers", nargs="+", type=Path, help="G-buffer .npy windows, (T, 5, H, W, 3) uint8")
    render.add_argument("-o", "--output", type=Path, default=Path("render.mp4"))
    render.add_argument("-p", "--prompt", default=DEFAULT_PROMPT)
    render.add_argument("--device", default="cuda:0")
    render.add_argument("--seed", type=int, default=0)
    render.add_argument("--fps", type=int, default=30)
    render.add_argument("--compile", action="store_true", help="enable torch.compile (slow first run)")
    render.add_argument(
        "--text-encoder-cpu", action="store_true", help="keep T5 off the GPU between prompts (<24 GB GPUs)"
    )
    render.add_argument("--save-frames", type=Path, default=None, help="also save RGB frames as .npy")
    _add_weight_args(render)
    render.set_defaults(func=_render)

    download = sub.add_parser("download", help="prefetch the checkpoint and base model into the HF cache")
    download.set_defaults(func=_download)

    serve = sub.add_parser("serve", help="run the live SuperTuxKart demo services")
    serve.add_argument("--config", type=Path, required=True, help="deploy.env profile (see docs/demo.md)")
    serve.add_argument("service_args", nargs=argparse.REMAINDER, help="doctor | start | status | stop | logs ...")
    serve.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
