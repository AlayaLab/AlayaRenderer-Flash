"""Resolve model weights from local paths or the Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPO_ID = "AlayaLab/AlayaRenderer-Flash"
BASE_MODEL_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
# Downloads are pinned to these commits; $ALAYA_CHECKPOINT_REVISION and
# $ALAYA_BASE_MODEL_REVISION override them.
CHECKPOINT_REVISION = "aed45cdc6ddbd28aa7432ecbc8589772a750ec0c"
BASE_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"

# Only what the runtime loads: skip the sample inputs in our repo and the base
# transformer weights, which ours replace.
_CHECKPOINT_PATTERNS = [
    "transformer/*",
    "transformer_partial.safetensors",
    "gbuf_small_encoder.safetensors",
    "tiny_decoder/*",
]
_BASE_MODEL_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "vae/*",
    "transformer/config.json",
]


def _snapshot(repo_id: str, revision: str, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns))


def resolve_checkpoint(path: str | os.PathLike[str] | None = None) -> Path:
    """Return a local AlayaRenderer-Flash checkpoint directory.

    Order: explicit ``path``, ``$ALAYA_CHECKPOINT``, then a download of
    :data:`DEFAULT_REPO_ID` into the Hugging Face cache.
    """
    path = path or os.environ.get("ALAYA_CHECKPOINT")
    if path:
        return Path(path).expanduser()
    revision = os.environ.get("ALAYA_CHECKPOINT_REVISION", CHECKPOINT_REVISION)
    return _snapshot(DEFAULT_REPO_ID, revision, _CHECKPOINT_PATTERNS)


def resolve_base_model(path: str | os.PathLike[str] | None = None) -> Path:
    """Return a local Wan 2.1 T2V 1.3B Diffusers directory.

    Order: explicit ``path``, ``$ALAYA_BASE_MODEL``, then a download of the
    needed components of :data:`BASE_MODEL_REPO_ID`.
    """
    path = path or os.environ.get("ALAYA_BASE_MODEL")
    if path:
        return Path(path).expanduser()
    revision = os.environ.get("ALAYA_BASE_MODEL_REVISION", BASE_MODEL_REVISION)
    return _snapshot(BASE_MODEL_REPO_ID, revision, _BASE_MODEL_PATTERNS)
