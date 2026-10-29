"""Download two recorded SuperTuxKart G-buffer windows for a quick offline render.

uv run python examples/fetch_samples.py
uv run alayarender-flash render samples/window-00.npy samples/window-01.npy -o demo.mp4
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

from alayarenderer_flash.weights import CHECKPOINT_REVISION, DEFAULT_REPO_ID


def main() -> None:
    out = Path("samples")
    out.mkdir(exist_ok=True)
    for name in ("window-00.npy", "window-01.npy"):
        path = hf_hub_download(DEFAULT_REPO_ID, f"samples/{name}", revision=CHECKPOINT_REVISION, local_dir=".")
        print(path)


if __name__ == "__main__":
    main()
