"""Resident model runtime shared by the live worker and the offline renderer.

``ModelRuntime`` keeps the autoregressive history, the grouped G-buffer encoder
phase, the stateful tiny-decoder cache and the RNG on the GPU, and renders one
causal G-buffer window at a time.
"""

from __future__ import annotations

import copy
import hashlib
import os
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.models import AutoencoderKLWan

from .decoders.streaming import StreamingTinyDecoderCache
from .decoders.taehv import TAEHV
from .gbuffer import MODALITIES, gbuffer_pixel_window_for_latent_chunk
from .models.gbuffer_encoder import GBufferSmallEncoder, load_gbuffer_encoder_weights
from .models.kernels import replace_all_norms_with_flash_norms, replace_rmsnorm_with_fp32, replace_rope_with_flash_rope
from .models.loading import assert_gbuffer_patch_embedding_ready, assert_text_sink_ready, load_partial_weights
from .models.transformer import HeliosTransformer3DModel, expand_patch_embedding_for_gbuffer
from .pipeline.helios import HeliosPipeline
from .transport import (
    TRANSPORT_NAME as SHM_SLOT_TRANSPORT,
)
from .transport import (
    SlotChanged,
    SlotReference,
    state_matches_reference,
)
from .transport import (
    _read_state_snapshot as read_slot_state_snapshot,
)

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)


CONTROL_COLUMNS = ("steer", "accel", "brake", "speed", "position")

SUPPORTED_LATENT_WINDOWS = (2, 4)

MODEL_RUN_WINDOW_WARNING_SECONDS = 0.5


def load_stk_grouped_small_encoder(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    encoder = GBufferSmallEncoder(
        num_gbuffers=len(MODALITIES),
        hidden_dim=128,
        out_channels=len(MODALITIES) * 16,
        mode="grouped",
    )
    loaded, missing, unexpected = load_gbuffer_encoder_weights(encoder, str(path))
    if loaded <= 0 or missing or unexpected:
        raise RuntimeError(
            "STK reference grouped small encoder checkpoint mismatch: "
            f"loaded={loaded} missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    return encoder.to(device=device, dtype=dtype).eval()


@dataclass(frozen=True)
class StkCausalWindowSpec:
    phase: str
    bridge_pixel_count: int
    encoder_input_pixel_count: int
    latent_offset: int
    encoded_latent_count: int
    selected_latent_count: int


def stk_causal_window_spec(
    is_initial: bool,
    latent_window_size: int,
    temporal_factor: int = 4,
) -> StkCausalWindowSpec:
    """Resolve the exact causal window contract used by STK training/validation."""
    local_pixel_start, encoder_input_pixel_count, latent_offset = gbuffer_pixel_window_for_latent_chunk(
        0 if is_initial else 1,
        latent_window_size=latent_window_size,
        temporal_factor=temporal_factor,
    )
    expected_local_start = 0 if is_initial else (latent_window_size - 1) * temporal_factor
    if local_pixel_start != expected_local_start:
        raise RuntimeError(
            f"STK reference causal window start changed: got={local_pixel_start} expected={expected_local_start}"
        )
    bridge_pixel_count = encoder_input_pixel_count if is_initial else encoder_input_pixel_count - 1
    encoded_latent_count = (encoder_input_pixel_count - 1) // temporal_factor + 1
    if encoded_latent_count != latent_offset + latent_window_size:
        raise RuntimeError(
            "STK reference causal latent contract changed: "
            f"encoded={encoded_latent_count} offset={latent_offset} "
            f"selected={latent_window_size}"
        )
    return StkCausalWindowSpec(
        phase="initial" if is_initial else "resumed",
        bridge_pixel_count=bridge_pixel_count,
        encoder_input_pixel_count=encoder_input_pixel_count,
        latent_offset=latent_offset,
        encoded_latent_count=encoded_latent_count,
        selected_latent_count=latent_window_size,
    )


@dataclass(frozen=True)
class CausalEncodingTrace:
    phase: str
    bridge_pixel_count: int
    input_pixel_count: int
    latent_offset: int
    encoded_latent_count: int
    selected_latent_count: int
    boundary_source_seq: int

    def metadata(self) -> dict[str, Any]:
        return {
            "causal_encoding_phase": self.phase,
            "causal_bridge_pixel_count": self.bridge_pixel_count,
            "causal_input_pixel_count": self.input_pixel_count,
            "causal_latent_offset": self.latent_offset,
            "causal_encoded_latent_count": self.encoded_latent_count,
            "causal_selected_latent_count": self.selected_latent_count,
            "causal_boundary_source_seq": self.boundary_source_seq,
        }


class InputNotReady(RuntimeError):
    """The producer has not committed the next compatible immutable window yet."""


class ContinuityError(RuntimeError):
    """The retained immutable windows cannot reconstruct a continuous stream."""


class FenceChanged(RuntimeError):
    """Generation, prompt, track, or source session changed during work."""


@dataclass(frozen=True)
class TemporalCadence:
    latent_window: int
    initial_pixel_frames: int
    resumed_pixel_frames: int


def cadence_for_latent_window(latent_window: int) -> TemporalCadence:
    if latent_window not in SUPPORTED_LATENT_WINDOWS:
        supported = ", ".join(str(value) for value in SUPPORTED_LATENT_WINDOWS)
        raise ValueError(f"latent window must be one of: {supported}")
    return TemporalCadence(
        latent_window=latent_window,
        initial_pixel_frames=4 * latent_window - 3,
        resumed_pixel_frames=4 * latent_window,
    )


@dataclass
class RuntimeConfig:
    """Model and schedule settings for :class:`ModelRuntime`.

    The defaults match the public SuperTuxKart deployment. ``base_model`` is a
    local Wan 2.1 T2V 1.3B Diffusers snapshot and ``checkpoint`` a local
    AlayaRenderer-Flash snapshot (see :mod:`alayarenderer_flash.weights`).
    """

    base_model: Path
    checkpoint: Path
    small_encoder: Path | None = None
    tiny_decoder: Path | None = None
    device: str = "cuda:0"
    height: int = 448
    width: int = 832
    latent_window: int = 4
    num_inference_steps: int = 4
    few_step_shift: float = 3.0
    guidance_scale: float = 1.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    seed: int = 0
    window_transport: str = "npz"
    warmup_chunks: int = 0
    compile_transformer: bool = False
    compile_small_encoder: bool = False
    compile_tiny_decoder: bool = False
    small_encoder_mode: str = "grouped"
    small_encoder_dim: int = 128
    temporal_cadence: TemporalCadence = field(init=False)

    def __post_init__(self) -> None:
        self.base_model = Path(self.base_model)
        self.checkpoint = Path(self.checkpoint)
        self.small_encoder = Path(self.small_encoder or self.checkpoint / "gbuf_small_encoder.safetensors")
        self.tiny_decoder = Path(self.tiny_decoder or self.checkpoint / "tiny_decoder" / "taew2_1_ft.safetensors")
        self.temporal_cadence = cadence_for_latent_window(self.latent_window)


def clean_input_event_id(value: Any) -> str:
    return "".join(char for char in str(value or "") if char.isprintable())[:160]


@dataclass(frozen=True)
class SessionToken:
    reset_epoch: int
    source_reset_epoch: int
    request_id: int
    prompt_revision: int
    prompt_hash: str
    prompt: str
    track_id: str
    source_session_id: str

    @property
    def generation(self) -> int:
        """Compatibility alias for the externally controlled reset epoch."""
        return self.reset_epoch

    def key(self) -> tuple[Any, ...]:
        return (
            self.reset_epoch,
            self.source_reset_epoch,
            self.request_id,
            self.prompt_revision,
            self.prompt_hash,
            self.track_id,
            self.source_session_id,
        )


@dataclass(frozen=True)
class WindowHeader:
    path: Path
    window_transport: str
    gbuffers_file: str
    gbuffers_shape: tuple[int, ...]
    gbuffers_dtype: str
    gbuffers_payload_nbytes: int
    gbuffers_file_size_bytes: int
    schema_version: int
    reset_epoch: int
    source_reset_epoch: int
    render_request_id: int
    window_id: int
    previous_window_id: int
    previous_last_seq: int
    is_initial: bool
    sample_start: int
    sample_end: int
    frame_count: int
    first_seq: int
    last_seq: int
    prompt_revision: int
    track_id: str
    source_session_id: str
    gbuffers_slot_index: int = -1
    gbuffers_slot_window_id: int = -1
    gbuffers_slot_writer_epoch: int = -1
    gbuffers_slot_commit_token: str = ""
    gbuffers_slot_state_file: str = ""
    gbuffers_data_offset_bytes: int = 0

    @property
    def generation(self) -> int:
        return self.reset_epoch


@dataclass
class SharedSlotReadLease:
    """Non-mutating private slot mapping revalidated after CUDA H2D."""

    reference: SlotReference
    state_path: Path
    slot_path: Path
    before_state: dict[str, Any]
    before_state_identity: tuple[int, int, int, int]
    before_slot_identity: tuple[int, int, int, int]
    fd: int
    array: np.memmap
    closed: bool = False

    @staticmethod
    def _slot_identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    @classmethod
    def open(
        cls,
        reference: SlotReference,
        slot_path: Path,
        state_path: Path,
    ) -> SharedSlotReadLease:
        before_state, before_state_identity = read_slot_state_snapshot(state_path)
        if not state_matches_reference(before_state, reference):
            raise SlotChanged(f"slot {reference.slot_index} no longer owns window {reference.window_id}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(slot_path, flags)
        array: np.memmap | None = None
        try:
            stat_before = os.fstat(fd)
            if not stat.S_ISREG(stat_before.st_mode):
                raise SlotChanged(f"slot file {slot_path.name} is not regular")
            if stat_before.st_size != reference.file_size_bytes:
                raise SlotChanged(f"slot file has {stat_before.st_size} bytes, expected {reference.file_size_bytes}")
            with os.fdopen(os.dup(fd), "rb") as handle:
                array = np.memmap(
                    handle,
                    dtype=np.uint8,
                    mode="c",
                    offset=0,
                    shape=reference.shape,
                    order="C",
                )
            stat_after_map = os.fstat(fd)
            before_slot_identity = cls._slot_identity(stat_before)
            if cls._slot_identity(stat_after_map) != before_slot_identity:
                raise SlotChanged(f"slot file {slot_path.name} changed while mapping")
            if not array.flags.c_contiguous:
                raise SlotChanged(f"slot file {slot_path.name} is not C-contiguous")
            return cls(
                reference=reference,
                state_path=state_path,
                slot_path=slot_path,
                before_state=before_state,
                before_state_identity=before_state_identity,
                before_slot_identity=before_slot_identity,
                fd=fd,
                array=array,
            )
        except BaseException:
            if array is not None:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            os.close(fd)
            raise

    def verify_after_h2d(self) -> None:
        if self.closed:
            raise SlotChanged("shared slot lease closed before verification")
        stat_after = os.fstat(self.fd)
        path_stat_after = os.stat(self.slot_path, follow_symlinks=False)
        after_state, after_state_identity = read_slot_state_snapshot(self.state_path)
        if self.before_state != after_state:
            raise SlotChanged(
                f"slot {self.reference.slot_index} state changed during H2D for window {self.reference.window_id}"
            )
        if not state_matches_reference(after_state, self.reference):
            raise SlotChanged(f"slot {self.reference.slot_index} lost token/window ownership during H2D")
        if self.before_state_identity != after_state_identity:
            raise SlotChanged("shared slot state inode/identity changed during H2D")
        if self._slot_identity(stat_after) != self.before_slot_identity:
            raise SlotChanged("shared slot file inode/identity changed during H2D")
        if (
            not stat.S_ISREG(path_stat_after.st_mode)
            or self._slot_identity(path_stat_after) != self.before_slot_identity
        ):
            raise SlotChanged("shared slot path inode/identity changed during H2D")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            mapping = getattr(self.array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        finally:
            os.close(self.fd)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass
class WindowData:
    header: WindowHeader
    gbuffers: np.ndarray | None
    frame_seqs: np.ndarray
    input_event_ids: np.ndarray
    source_frames: np.ndarray
    sample_indices: np.ndarray
    controls: np.ndarray
    kart_ids: np.ndarray
    prompt: str
    shared_slot_lease: SharedSlotReadLease | None = None
    gbuffers_gpu: torch.Tensor | None = None
    last_gbuffer_pixel_frame: np.ndarray | None = None


@dataclass
class OutputFrame:
    rgb: np.ndarray
    source_seq: int
    source_frame: int
    sample_index: int
    control: dict[str, Any]
    kart_id: int
    window_offset: int
    window_frame_count: int
    input_event_id: str = ""
    causal_trace: CausalEncodingTrace | None = None


def clone_state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_state_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_state_value(item) for item in value)
    if isinstance(value, list):
        return [clone_state_value(item) for item in value]
    return copy.deepcopy(value)


@dataclass
class RuntimeSnapshot:
    stream_state: dict[str, Any]
    tiny_mem: list[Any]
    tiny_is_first: bool
    generator_state: torch.Tensor
    encoder_first: bool
    last_gbuffer_pixel_frame: np.ndarray | None
    last_window_id: int | None
    last_sample_index: int | None
    last_source_seq: int | None
    last_causal_trace: CausalEncodingTrace | None


class ModelRuntime:
    def __init__(self, args: RuntimeConfig):
        self.args = args
        self.cadence: TemporalCadence = args.temporal_cadence
        self.device = torch.device(args.device)
        self.weight_dtype = torch.bfloat16
        self.pipe: Any = None
        self.encoder: Any = None
        self.tiny_decoder: Any = None
        self.tiny_cache: Any = None
        self.encoder_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.transformer_compiled = False
        self.small_encoder_compiled = False
        self.tiny_decoder_compiled = False
        self._tiny_decoder_original_forwards: list[tuple[Any, Callable[..., Any]]] = []
        self.compile_fallback_reason = ""
        self.warmed = args.warmup_chunks == 0
        self.token: SessionToken | None = None
        self.stream_state: dict[str, Any] = {}
        self.generator = torch.Generator(device=self.device)
        self.encoder_first = True
        self.last_gbuffer_pixel_frame: np.ndarray | None = None
        self.last_window_id: int | None = None
        self.last_sample_index: int | None = None
        self.last_source_seq: int | None = None
        self.last_causal_trace: CausalEncodingTrace | None = None
        self._reset_model_timing_telemetry()

    @property
    def compile_mode(self) -> str:
        modes = []
        if self.transformer_compiled:
            modes.append("transformer")
        if self.small_encoder_compiled:
            modes.append("small_encoder")
        if self.tiny_decoder_compiled:
            modes.append("tiny_decoder")
        if not modes:
            return "eager"
        return "+".join(modes)

    def load(self) -> None:
        for path in (self.args.base_model, self.args.checkpoint, self.args.small_encoder, self.args.tiny_decoder):
            if not path.exists():
                raise FileNotFoundError(path)
        transformer = HeliosTransformer3DModel.from_pretrained(
            str(self.args.checkpoint),
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
            transformer_additional_kwargs={
                "has_multi_term_memory_patch": True,
                "zero_history_timestep": True,
                "restrict_self_attn": False,
                "guidance_cross_attn": True,
                "use_text_sink": True,
            },
        )
        if not bool(getattr(transformer.config, "use_text_sink", False)):
            raise RuntimeError("STK transformer must be loaded with text_sink enabled")
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
        expand_patch_embedding_for_gbuffer(transformer, len(MODALITIES))
        expected_channels = 16 + len(MODALITIES) * 16
        if transformer.patch_embedding.weight.shape[1] != expected_channels:
            raise RuntimeError("checkpoint patch_embedding channel mismatch")
        for module_name in ("patch_short", "patch_mid", "patch_long", "norm_out"):
            if not hasattr(transformer, module_name):
                continue
            module = getattr(transformer, module_name)
            if any(parameter.is_meta for parameter in module.parameters()):
                module = module.to_empty(device="cpu")
                setattr(transformer, module_name, module)
                for parameter in module.parameters():
                    torch.nn.init.zeros_(parameter)

        vae = AutoencoderKLWan.from_pretrained(str(self.args.base_model), subfolder="vae", torch_dtype=torch.float32)
        scheduler = UniPCMultistepScheduler.from_pretrained(str(self.args.base_model), subfolder="scheduler")
        self.pipe = HeliosPipeline.from_pretrained(
            str(self.args.base_model),
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=self.weight_dtype,
        )
        partial_path = self.args.checkpoint / "transformer_partial.safetensors"
        if not partial_path.exists():
            partial_path = self.args.checkpoint / "transformer_partial.pth"
        if not partial_path.exists():
            raise FileNotFoundError(partial_path)
        load_partial_weights(self.pipe.transformer, partial_path)
        assert_gbuffer_patch_embedding_ready(self.pipe.transformer)
        assert_text_sink_ready(self.pipe.transformer)
        self.pipe = self.pipe.to(self.device)
        self.pipe.vae.to("cpu")
        self._install_prompt_embed_cache()

        if self.args.small_encoder_mode != "grouped" or self.args.small_encoder_dim != 128:
            raise RuntimeError("the released checkpoint uses the grouped small encoder with dim 128")
        self.encoder = load_stk_grouped_small_encoder(
            self.args.small_encoder,
            device=self.device,
            dtype=self.weight_dtype,
        )
        self.tiny_decoder = TAEHV(str(self.args.tiny_decoder)).to(device=self.device, dtype=torch.float32).eval()
        torch.cuda.empty_cache()
        self._configure_compile()

    def _install_prompt_embed_cache(self) -> None:
        """Memoize T5 text encoding across windows.

        The live loop calls ``pipe()`` once per window, but the prompt only
        changes when the player edits it. ``pipeline.encode_prompt`` re-runs the
        T5 encoder on every call (its ``prompt_embeds`` bypass is unusable — when
        embeds are passed it leaves ``prompt_attention_mask`` unbound), so here we
        cache the full return tuple keyed by prompt text and only re-encode when
        the prompt (or negative prompt / CFG flag) actually changes.
        """
        original_encode = self.pipe.encode_prompt
        cache: dict = {}
        self._prompt_cache = cache
        self._prompt_encode_misses = 0

        def cached_encode_prompt(
            prompt,
            negative_prompt=None,
            do_classifier_free_guidance=True,
            **kwargs,
        ):
            key = (
                tuple(prompt) if isinstance(prompt, (list, tuple)) else prompt,
                tuple(negative_prompt) if isinstance(negative_prompt, (list, tuple)) else negative_prompt,
                bool(do_classifier_free_guidance),
            )
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = original_encode(
                prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                **kwargs,
            )
            # A session realistically cycles a handful of prompts; bound the cache.
            if len(cache) >= 32:
                cache.clear()
            cache[key] = result
            self._prompt_encode_misses += 1
            key_prompt = key[0]
            phash = hashlib.sha1(key_prompt.encode("utf-8")).hexdigest()[:12] if isinstance(key_prompt, str) else "list"
            print(
                f"[worker] text-encode (miss #{self._prompt_encode_misses}) prompt_hash={phash}",
                flush=True,
            )
            return result

        self.pipe.encode_prompt = cached_encode_prompt

    def _configure_compile(self) -> None:
        if self.args.compile_transformer:
            try:
                self.pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
                self.transformer_compiled = True
            except Exception as error:
                self.compile_fallback_reason = f"transformer compile setup failed: {error}"
                self._disable_transformer_compile()
        if self.args.compile_small_encoder:
            try:
                encoder = self.encoder
                self.encoder_fn = torch.compile(
                    lambda tensor: encoder(tensor),
                    mode="max-autotune-no-cudagraphs",
                    dynamic=False,
                )
                self.small_encoder_compiled = True
            except Exception as error:
                self.compile_fallback_reason = f"small encoder compile setup failed: {error}"
                self._disable_small_encoder_compile()
        if self.encoder_fn is None:
            self._disable_small_encoder_compile()
        if self.args.compile_tiny_decoder:
            try:
                originals: list[tuple[Any, Callable[..., Any]]] = []
                for block in self.tiny_decoder.decoder:
                    original_forward = block.forward
                    block.forward = torch.compile(
                        original_forward,
                        mode="max-autotune-no-cudagraphs",
                        dynamic=False,
                    )
                    originals.append((block, original_forward))
                self._tiny_decoder_original_forwards = originals
                self.tiny_decoder_compiled = True
            except Exception as error:
                for block, original_forward in reversed(locals().get("originals", [])):
                    block.forward = original_forward
                self._tiny_decoder_original_forwards = []
                self.tiny_decoder_compiled = False
                self.compile_fallback_reason = f"tiny decoder compile setup failed: {error}"

    def _disable_transformer_compile(self) -> None:
        if self.pipe is not None and hasattr(self.pipe.transformer, "_compiled_call_impl"):
            self.pipe.transformer._compiled_call_impl = None
        self.transformer_compiled = False

    def _disable_small_encoder_compile(self) -> None:
        self.encoder_fn = self.encoder
        self.small_encoder_compiled = False

    def _disable_tiny_decoder_compile(self) -> None:
        for block, original_forward in reversed(self._tiny_decoder_original_forwards):
            block.forward = original_forward
        self._tiny_decoder_original_forwards = []
        self.tiny_decoder_compiled = False

    def fallback_to_eager(self, reason: str) -> bool:
        if not (self.transformer_compiled or self.small_encoder_compiled or self.tiny_decoder_compiled):
            return False
        self.compile_fallback_reason = reason[:1000]
        self._disable_transformer_compile()
        self._disable_small_encoder_compile()
        self._disable_tiny_decoder_compile()
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        torch.cuda.empty_cache()
        return True

    def reset_session(self, token: SessionToken) -> None:
        self.token = token
        self.stream_state = {}
        self.tiny_cache = StreamingTinyDecoderCache(self.tiny_decoder)
        self.generator = torch.Generator(device=self.device).manual_seed(self.args.seed)
        self.encoder_first = True
        self.last_gbuffer_pixel_frame = None
        self.last_window_id = None
        self.last_sample_index = None
        self.last_source_seq = None
        self.last_causal_trace = None
        self._reset_model_timing_telemetry()

    def _reset_model_timing_telemetry(self) -> None:
        self.model_timing_window_id = -1
        self.model_run_window_seconds = 0.0
        self.model_run_window_max_seconds = 0.0
        self.model_run_chunk_return_seconds = 0.0
        self.model_run_chunk_return_max_seconds = 0.0
        self.model_pixels_to_tensor_seconds = 0.0
        self.model_boundary_h2d_seconds = 0.0
        self.model_window_h2d_seconds = 0.0
        self.model_gpu_temporal_concat_seconds = 0.0
        self.model_shared_slot_direct_h2d_seconds = 0.0
        self.model_shared_slot_boundary_copy_seconds = 0.0
        self.model_shared_slot_verify_seconds = 0.0
        self.model_shared_slot_direct_total_seconds = 0.0
        self.model_shared_slot_direct_attempt_count = 0
        self.model_shared_slot_direct_success_count = 0
        self.model_shared_slot_changed_count = 0
        self.model_small_encoder_seconds = 0.0
        self.model_diffusion_seconds = 0.0
        self.model_tiny_decoder_seconds = 0.0
        self.model_video_to_rgb_cpu_seconds = 0.0
        self.model_video_to_rgb_cpu_max_seconds = 0.0
        self.model_snapshot_seconds = 0.0
        self.model_snapshot_max_seconds = 0.0
        self.model_window_cycle_seconds = 0.0
        self.model_window_cycle_max_seconds = 0.0

    def record_model_snapshot(self, duration: float) -> None:
        self.model_snapshot_seconds = max(0.0, duration)
        self.model_snapshot_max_seconds = max(
            self.model_snapshot_max_seconds,
            self.model_snapshot_seconds,
        )

    def record_model_window_cycle(self, window_id: int, duration: float) -> None:
        self.model_timing_window_id = window_id
        self.model_window_cycle_seconds = max(0.0, duration)
        self.model_window_cycle_max_seconds = max(
            self.model_window_cycle_max_seconds,
            self.model_window_cycle_seconds,
        )
        if (
            self.model_window_cycle_seconds <= MODEL_RUN_WINDOW_WARNING_SECONDS
            or self.model_run_window_seconds > MODEL_RUN_WINDOW_WARNING_SECONDS
        ):
            return
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            print(
                f"{timestamp} WARNING slow model window cycle "
                f"window_id={window_id} "
                f"total_ms={self.model_window_cycle_seconds * 1000.0:.2f} "
                f"snapshot_ms={self.model_snapshot_seconds * 1000.0:.2f} "
                f"run_window_ms={self.model_run_window_seconds * 1000.0:.2f} "
                f"run_chunk_return_ms="
                f"{self.model_run_chunk_return_seconds * 1000.0:.2f} "
                f"video_to_rgb_cpu_ms="
                f"{self.model_video_to_rgb_cpu_seconds * 1000.0:.2f}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            pass

    def model_timing_telemetry(self) -> dict[str, Any]:
        return {
            "model_timing_window_id": self.model_timing_window_id,
            "model_run_window_ms": round(self.model_run_window_seconds * 1000.0, 2),
            "model_run_window_max_ms": round(self.model_run_window_max_seconds * 1000.0, 2),
            "model_run_chunk_return_ms": round(self.model_run_chunk_return_seconds * 1000.0, 2),
            "model_run_chunk_return_max_ms": round(self.model_run_chunk_return_max_seconds * 1000.0, 2),
            "model_pixels_to_tensor_ms": round(self.model_pixels_to_tensor_seconds * 1000.0, 2),
            "model_boundary_h2d_ms": round(self.model_boundary_h2d_seconds * 1000.0, 2),
            "model_window_h2d_ms": round(self.model_window_h2d_seconds * 1000.0, 2),
            "model_gpu_temporal_concat_ms": round(self.model_gpu_temporal_concat_seconds * 1000.0, 2),
            "model_shared_slot_direct_h2d_ms": round(self.model_shared_slot_direct_h2d_seconds * 1000.0, 2),
            "model_shared_slot_boundary_copy_ms": round(self.model_shared_slot_boundary_copy_seconds * 1000.0, 2),
            "model_shared_slot_verify_ms": round(self.model_shared_slot_verify_seconds * 1000.0, 2),
            "model_shared_slot_direct_total_ms": round(self.model_shared_slot_direct_total_seconds * 1000.0, 2),
            "model_shared_slot_direct_attempt_count": (self.model_shared_slot_direct_attempt_count),
            "model_shared_slot_direct_success_count": (self.model_shared_slot_direct_success_count),
            "model_shared_slot_changed_count": self.model_shared_slot_changed_count,
            "model_small_encoder_ms": round(self.model_small_encoder_seconds * 1000.0, 2),
            "model_diffusion_ms": round(self.model_diffusion_seconds * 1000.0, 2),
            "model_tiny_decoder_ms": round(self.model_tiny_decoder_seconds * 1000.0, 2),
            "model_video_to_rgb_cpu_ms": round(self.model_video_to_rgb_cpu_seconds * 1000.0, 2),
            "model_video_to_rgb_cpu_max_ms": round(self.model_video_to_rgb_cpu_max_seconds * 1000.0, 2),
            "model_snapshot_ms": round(self.model_snapshot_seconds * 1000.0, 2),
            "model_snapshot_max_ms": round(self.model_snapshot_max_seconds * 1000.0, 2),
            "model_window_cycle_ms": round(self.model_window_cycle_seconds * 1000.0, 2),
            "model_window_cycle_max_ms": round(self.model_window_cycle_max_seconds * 1000.0, 2),
        }

    def causal_trace_telemetry(self) -> dict[str, Any]:
        if self.last_causal_trace is None:
            return {}
        return self.last_causal_trace.metadata()

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            stream_state=clone_state_value(self.stream_state),
            tiny_mem=clone_state_value(self.tiny_cache._mem),
            tiny_is_first=bool(self.tiny_cache._is_first),
            generator_state=self.generator.get_state().clone(),
            encoder_first=self.encoder_first,
            last_gbuffer_pixel_frame=(
                None if self.last_gbuffer_pixel_frame is None else self.last_gbuffer_pixel_frame.copy()
            ),
            last_window_id=self.last_window_id,
            last_sample_index=self.last_sample_index,
            last_source_seq=self.last_source_seq,
            last_causal_trace=self.last_causal_trace,
        )

    def restore(self, snapshot: RuntimeSnapshot) -> None:
        self.stream_state = snapshot.stream_state
        self.tiny_cache = StreamingTinyDecoderCache(self.tiny_decoder)
        self.tiny_cache._mem = snapshot.tiny_mem
        self.tiny_cache._is_first = snapshot.tiny_is_first
        self.generator.set_state(snapshot.generator_state)
        self.encoder_first = snapshot.encoder_first
        self.last_gbuffer_pixel_frame = (
            None if snapshot.last_gbuffer_pixel_frame is None else snapshot.last_gbuffer_pixel_frame.copy()
        )
        self.last_window_id = snapshot.last_window_id
        self.last_sample_index = snapshot.last_sample_index
        self.last_source_seq = snapshot.last_source_seq
        self.last_causal_trace = snapshot.last_causal_trace

    def validate_continuity(self, window: WindowData) -> None:
        header = window.header
        if self.last_window_id is None:
            if not header.is_initial or not self.encoder_first:
                raise ContinuityError(
                    f"a reset session must begin with an initial {self.cadence.initial_pixel_frames}-frame window"
                )
            return
        if header.is_initial:
            raise ContinuityError("received a second initial window without a session reset")
        if header.previous_window_id != self.last_window_id:
            raise ContinuityError(
                f"window {header.window_id} points to {header.previous_window_id}, expected {self.last_window_id}"
            )
        if header.previous_last_seq != self.last_source_seq:
            raise ContinuityError(
                f"window {header.window_id} previous_last_seq={header.previous_last_seq}, "
                f"expected {self.last_source_seq}"
            )
        if header.sample_start != (self.last_sample_index or 0) + 1:
            raise ContinuityError(
                f"window {header.window_id} sample_start={header.sample_start}, "
                f"expected {(self.last_sample_index or 0) + 1}"
            )
        if header.first_seq <= (self.last_source_seq or -1):
            raise ContinuityError("source frame sequence did not advance")

    def _pixels_to_tensor(self, pixels: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(pixels))
        if self.args.window_transport in {"npy", SHM_SLOT_TRANSPORT}:
            tensor = tensor.to(device=self.device)
            tensor = tensor.permute(1, 4, 0, 2, 3).reshape(
                1,
                len(MODALITIES) * 3,
                pixels.shape[0],
                self.args.height,
                self.args.width,
            )
            return tensor.to(dtype=self.weight_dtype).div_(127.5).sub_(1.0)
        tensor = tensor.permute(1, 4, 0, 2, 3).reshape(
            1, len(MODALITIES) * 3, pixels.shape[0], self.args.height, self.args.width
        )
        return tensor.to(device=self.device, dtype=self.weight_dtype).div_(127.5).sub_(1.0)

    def stage_shared_slot_window(
        self,
        window: WindowData,
        fence: Callable[[], None],
    ) -> None:
        """Copy a mapped slot to CUDA, then fail closed unless ownership stayed stable."""
        lease = window.shared_slot_lease
        if lease is None:
            return
        if window.gbuffers is None or window.gbuffers_gpu is not None:
            lease.close()
            window.shared_slot_lease = None
            raise RuntimeError("shared slot window is not in a stageable state")

        self.model_shared_slot_direct_attempt_count += 1
        self.model_shared_slot_direct_h2d_seconds = 0.0
        self.model_shared_slot_boundary_copy_seconds = 0.0
        self.model_shared_slot_verify_seconds = 0.0
        self.model_shared_slot_direct_total_seconds = 0.0
        total_started_at = time.monotonic()
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        window_tensor: torch.Tensor | None = None
        try:
            fence()
            h2d_start.record()
            window_tensor = self._pixels_to_tensor(window.gbuffers)
            h2d_end.record()
            # Ownership must be checked only after every source byte reached CUDA.
            torch.cuda.synchronize(self.device)
            self.model_shared_slot_direct_h2d_seconds = h2d_start.elapsed_time(h2d_end) / 1000.0

            boundary_copy_started_at = time.monotonic()
            last_pixel_frame = np.array(window.gbuffers[-1], dtype=np.uint8, copy=True, order="C")
            self.model_shared_slot_boundary_copy_seconds = time.monotonic() - boundary_copy_started_at

            verify_started_at = time.monotonic()
            lease.verify_after_h2d()
            self.model_shared_slot_verify_seconds = time.monotonic() - verify_started_at
            fence()

            window.gbuffers_gpu = window_tensor
            window.last_gbuffer_pixel_frame = last_pixel_frame
            window_tensor = None
            self.model_shared_slot_direct_success_count += 1
        except SlotChanged as exc:
            self.model_shared_slot_changed_count += 1
            raise InputNotReady(str(exc)) from exc
        finally:
            if window_tensor is not None:
                del window_tensor
            window.gbuffers = None
            window.shared_slot_lease = None
            try:
                lease.close()
            finally:
                self.model_shared_slot_direct_total_seconds = time.monotonic() - total_started_at

    def _run_chunk(
        self,
        pixels: np.ndarray | None,
        prompt: str,
        is_initial: bool,
        boundary_source_seq: int,
        fence: Callable[[], None],
        prepared_window_tensor: torch.Tensor | None = None,
        prepared_last_pixel_frame: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, CausalEncodingTrace]:
        started_at = time.monotonic()
        spec = stk_causal_window_spec(
            is_initial,
            latent_window_size=self.cadence.latent_window,
            temporal_factor=4,
        )
        expected_frames = spec.bridge_pixel_count
        if prepared_window_tensor is None:
            if pixels is None:
                raise ValueError("G-buffer window has neither CPU pixels nor a staged CUDA tensor")
            bridge_pixel_count = int(pixels.shape[0])
        else:
            if pixels is not None:
                raise ValueError("direct shared-slot window retained an unsafe CPU mapping")
            expected_tensor_shape = (
                1,
                len(MODALITIES) * 3,
                expected_frames,
                self.args.height,
                self.args.width,
            )
            if tuple(prepared_window_tensor.shape) != expected_tensor_shape:
                raise ValueError(
                    "staged shared-slot tensor shape mismatch: "
                    f"got={tuple(prepared_window_tensor.shape)} "
                    f"expected={expected_tensor_shape}"
                )
            if prepared_window_tensor.device != self.device:
                raise ValueError("staged shared-slot tensor is on the wrong CUDA device")
            if prepared_window_tensor.dtype != self.weight_dtype:
                raise ValueError("staged shared-slot tensor has the wrong dtype")
            if prepared_last_pixel_frame is None:
                raise ValueError("staged shared-slot tensor is missing its final pixel frame")
            bridge_pixel_count = int(prepared_window_tensor.shape[2])
        if bridge_pixel_count != expected_frames:
            raise ValueError(
                f"small encoder {spec.phase} chunk needs {expected_frames} frames, got {bridge_pixel_count}"
            )
        if is_initial and boundary_source_seq != -1:
            raise ContinuityError("initial causal window cannot declare a boundary source seq")
        if not is_initial and boundary_source_seq < 0:
            raise ContinuityError("resumed causal window is missing its boundary source seq")
        fence()
        pixel_start = torch.cuda.Event(enable_timing=True)
        pixel_end = torch.cuda.Event(enable_timing=True)
        boundary_h2d_start = torch.cuda.Event(enable_timing=True)
        boundary_h2d_end = torch.cuda.Event(enable_timing=True)
        window_h2d_start = torch.cuda.Event(enable_timing=True)
        window_h2d_end = torch.cuda.Event(enable_timing=True)
        temporal_concat_start = torch.cuda.Event(enable_timing=True)
        temporal_concat_end = torch.cuda.Event(enable_timing=True)
        encoder_start = torch.cuda.Event(enable_timing=True)
        encoder_end = torch.cuda.Event(enable_timing=True)
        diffusion_start = torch.cuda.Event(enable_timing=True)
        diffusion_end = torch.cuda.Event(enable_timing=True)
        decoder_start = torch.cuda.Event(enable_timing=True)
        decoder_end = torch.cuda.Event(enable_timing=True)
        pixel_start.record()
        if prepared_window_tensor is not None:
            window_tensor = prepared_window_tensor
        else:
            window_h2d_start.record()
            window_tensor = self._pixels_to_tensor(pixels)
            window_h2d_end.record()
        if is_initial:
            pixel_tensor = window_tensor
        else:
            if self.last_gbuffer_pixel_frame is None:
                raise ContinuityError("grouped causal encoder is missing its boundary frame")
            boundary_h2d_start.record()
            boundary_tensor = self._pixels_to_tensor(self.last_gbuffer_pixel_frame[None, ...])
            boundary_h2d_end.record()
            temporal_concat_start.record()
            pixel_tensor = torch.cat((boundary_tensor, window_tensor), dim=2)
            temporal_concat_end.record()
        if pixel_tensor.shape[2] != spec.encoder_input_pixel_count:
            raise ContinuityError(
                "STK causal G-buffer window mismatch: "
                f"got={pixel_tensor.shape[2]} expected={spec.encoder_input_pixel_count}"
            )
        pixel_end.record()
        encoder_start.record()
        with torch.no_grad():
            encoded_gbuffer_latents = self.encoder_fn(pixel_tensor)
        encoder_end.record()
        encoded_latent_count = int(encoded_gbuffer_latents.shape[2])
        if encoded_latent_count != spec.encoded_latent_count:
            raise RuntimeError(
                "small encoder causal latent count mismatch: "
                f"got={encoded_latent_count} expected={spec.encoded_latent_count}"
            )
        gbuffer_latents = encoded_gbuffer_latents[
            :, :, spec.latent_offset : spec.latent_offset + spec.selected_latent_count
        ].contiguous()
        selected_latent_count = int(gbuffer_latents.shape[2])
        if selected_latent_count != spec.selected_latent_count:
            raise RuntimeError(
                f"small encoder selected {selected_latent_count} latent frames, expected {spec.selected_latent_count}"
            )
        causal_trace = CausalEncodingTrace(
            phase=spec.phase,
            bridge_pixel_count=bridge_pixel_count,
            input_pixel_count=int(pixel_tensor.shape[2]),
            latent_offset=spec.latent_offset,
            encoded_latent_count=encoded_latent_count,
            selected_latent_count=selected_latent_count,
            boundary_source_seq=boundary_source_seq,
        )
        fence()
        diffusion_start.record()
        with torch.no_grad():
            latent_chunk = self.pipe(
                prompt=prompt,
                negative_prompt=self.args.negative_prompt,
                height=self.args.height,
                width=self.args.width,
                num_frames=self.cadence.initial_pixel_frames,
                num_inference_steps=self.args.num_inference_steps,
                euler_few_step=True,
                few_step_shift=self.args.few_step_shift,
                guidance_scale=self.args.guidance_scale,
                generator=self.generator,
                history_sizes=[8, 2, 1],
                latent_window_size=self.cadence.latent_window,
                is_keep_x0=True,
                vae_decode_type="causal",
                overlap_decode=False,
                tiny_decoder=None,
                use_zero_init=False,
                zero_steps=1,
                use_interpolate_prompt=False,
                gbuffer_latents_all=gbuffer_latents,
                teacher_forcing_latents=None,
                is_enable_stage2=False,
                use_dmd=False,
                per_chunk_save_dir=None,
                per_chunk_save_inputs=False,
                long_history_dropout=0.0,
                mid_history_dropout=0.0,
                short_history_dropout=0.0,
                short_history_scale=1.0,
                adaptive_short_style_mean=True,
                short_style_anchor_frames=4,
                short_style_drift_threshold=0.05,
                short_style_drift_gain=1.0,
                short_style_alpha_max=0.50,
                short_style_correction_momentum=0.80,
                profile_pipeline=False,
                stream_session_state=self.stream_state,
                output_type="latent",
            ).frames
        diffusion_end.record()
        decoder_start.record()
        with torch.no_grad():
            video = self.tiny_cache.decode_chunk(latent_chunk, output_dtype=torch.float32)
        decoder_end.record()
        torch.cuda.synchronize(self.device)
        current_input_prep_seconds = pixel_start.elapsed_time(pixel_end) / 1000.0
        self.model_pixels_to_tensor_seconds = current_input_prep_seconds + (
            self.model_shared_slot_direct_h2d_seconds if prepared_window_tensor is not None else 0.0
        )
        self.model_boundary_h2d_seconds = (
            0.0 if is_initial else boundary_h2d_start.elapsed_time(boundary_h2d_end) / 1000.0
        )
        self.model_window_h2d_seconds = (
            self.model_shared_slot_direct_h2d_seconds
            if prepared_window_tensor is not None
            else window_h2d_start.elapsed_time(window_h2d_end) / 1000.0
        )
        self.model_gpu_temporal_concat_seconds = (
            0.0 if is_initial else temporal_concat_start.elapsed_time(temporal_concat_end) / 1000.0
        )
        self.model_small_encoder_seconds = encoder_start.elapsed_time(encoder_end) / 1000.0
        self.model_diffusion_seconds = diffusion_start.elapsed_time(diffusion_end) / 1000.0
        self.model_tiny_decoder_seconds = decoder_start.elapsed_time(decoder_end) / 1000.0
        fence()
        if video.shape[2] != expected_frames:
            raise RuntimeError(f"tiny decoder produced {video.shape[2]} frames for a {expected_frames}-frame input")
        self.encoder_first = False
        self.last_gbuffer_pixel_frame = (
            prepared_last_pixel_frame if prepared_window_tensor is not None else np.ascontiguousarray(pixels[-1]).copy()
        )
        self.last_causal_trace = causal_trace
        duration = time.monotonic() - started_at
        self.model_run_chunk_return_seconds = duration
        self.model_run_chunk_return_max_seconds = max(
            self.model_run_chunk_return_max_seconds,
            duration,
        )
        return video, causal_trace

    def _video_to_rgb(self, video: torch.Tensor) -> np.ndarray:
        rgb_tensor = (video[0].permute(1, 2, 3, 0).add(1.0).mul(127.5)).round().clamp_(0, 255).to(torch.uint8)
        transfer_started_at = time.monotonic()
        rgb = rgb_tensor.cpu().numpy()
        transfer_duration = time.monotonic() - transfer_started_at
        self.model_video_to_rgb_cpu_seconds = transfer_duration
        self.model_video_to_rgb_cpu_max_seconds = max(
            self.model_video_to_rgb_cpu_max_seconds,
            transfer_duration,
        )
        return np.ascontiguousarray(rgb)

    def warmup(self, window: WindowData, token: SessionToken, fence: Callable[[], None]) -> None:
        if self.warmed:
            return
        if window.header.frame_count < self.cadence.initial_pixel_frames:
            raise ContinuityError(f"two-chunk warmup needs at least {self.cadence.initial_pixel_frames} input frames")
        self.reset_session(token)
        first = window.gbuffers[: self.cadence.initial_pixel_frames]
        second = np.repeat(window.gbuffers[-1:], self.cadence.resumed_pixel_frames, axis=0)
        self._run_chunk(first, token.prompt, True, -1, fence)
        self._run_chunk(
            second,
            token.prompt,
            False,
            int(window.frame_seqs[self.cadence.initial_pixel_frames - 1]),
            fence,
        )
        torch.cuda.synchronize(self.device)
        self.warmed = True
        self.reset_session(token)

    def run_window(
        self,
        window: WindowData,
        token: SessionToken,
        fence: Callable[[], None],
    ) -> list[OutputFrame]:
        started_at = time.monotonic()
        self.model_timing_window_id = window.header.window_id
        self.model_run_chunk_return_seconds = 0.0
        self.model_video_to_rgb_cpu_seconds = 0.0
        try:
            if self.token is None or self.token.key() != token.key():
                raise FenceChanged("runtime token does not match active authority")
            self.validate_continuity(window)
            is_initial = self.encoder_first
            boundary_source_seq = -1 if is_initial else int(self.last_source_seq)
            video, causal_trace = self._run_chunk(
                window.gbuffers,
                token.prompt,
                is_initial,
                boundary_source_seq,
                fence,
                prepared_window_tensor=window.gbuffers_gpu,
                prepared_last_pixel_frame=window.last_gbuffer_pixel_frame,
            )
            rgb = self._video_to_rgb(video)
            frames: list[OutputFrame] = []
            for index in range(window.header.frame_count):
                control_values = window.controls[index]
                control = {name: float(control_values[column]) for column, name in enumerate(CONTROL_COLUMNS)}
                control["brake"] = bool(control_values[2] >= 0.5)
                control["position"] = int(round(float(control_values[4])))
                frames.append(
                    OutputFrame(
                        rgb=rgb[index],
                        source_seq=int(window.frame_seqs[index]),
                        input_event_id=clean_input_event_id(window.input_event_ids[index]),
                        source_frame=int(window.source_frames[index]),
                        sample_index=int(window.sample_indices[index]),
                        control=control,
                        kart_id=int(window.kart_ids[index]),
                        window_offset=index,
                        window_frame_count=window.header.frame_count,
                        causal_trace=causal_trace,
                    )
                )
            self.last_window_id = window.header.window_id
            self.last_sample_index = window.header.sample_end
            self.last_source_seq = window.header.last_seq
            return frames
        finally:
            duration = time.monotonic() - started_at
            self.model_run_window_seconds = duration
            self.model_run_window_max_seconds = max(
                self.model_run_window_max_seconds,
                duration,
            )
            if duration > MODEL_RUN_WINDOW_WARNING_SECONDS:
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                try:
                    print(
                        f"{timestamp} WARNING slow model run_window "
                        f"window_id={window.header.window_id} "
                        f"total_ms={duration * 1000.0:.2f} "
                        f"snapshot_ms={self.model_snapshot_seconds * 1000.0:.2f} "
                        f"run_chunk_return_ms="
                        f"{self.model_run_chunk_return_seconds * 1000.0:.2f} "
                        f"video_to_rgb_cpu_ms="
                        f"{self.model_video_to_rgb_cpu_seconds * 1000.0:.2f}",
                        file=sys.stderr,
                        flush=True,
                    )
                except (OSError, ValueError):
                    pass
