# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import math
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import regex as re
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel

from ..models.transformer import HeliosTransformer3DModel
from .output import HeliosPipelineOutput
from .utils import add_noise, convert_flow_pred_to_x0

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, HeliosPipeline

        >>> # Available models: BestWishYsh/Helios-Base, BestWishYsh/Helios-Mid, BestWishYsh/Helios-Distilled
        >>> model_id = "BestWishYsh/Helios-Base"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = HeliosPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=384,
        ...     width=640,
        ...     num_frames=132,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=24)
        ```
"""


@torch.amp.autocast("cuda", dtype=torch.float32)
def optimized_scale(positive_flat, negative_flat):
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class VAEDecodeType(str, Enum):
    DEFAULT = "default"
    DEFAULT_BATCH = "default_batch"
    # Streaming causal decode (mirrors advanced_models/minWM `cached_decode`): the VAE
    # temporal feat-cache is carried across chunks so chunk boundaries stay continuous,
    # instead of `default`'s per-chunk `vae.decode()` which resets the cache + treats every
    # chunk as a fresh video start. NOTE: changes output frame count — only the genuine first
    # latent frame is `first_chunk=True` (-> 1 px frame), every later latent frame -> 4 px
    # frames, so total = (sum_latent_frames - 1) * 4 + 1 (vs `default`'s num_chunks * 13).
    CAUSAL = "causal"


class StreamingVAEDecodeCache:
    """Causal feat_cache that survives across per-chunk VAE decodes.

    Per-chunk condition encode plus causal decode. Wan VAE is a
    causal 3D-conv autoencoder; a one-shot ``vae.decode(full_latents)`` is connected
    because each conv's ``feat_cache`` carries temporal state from frame i to i+1.
    Helios's per-chunk ``vae.decode(chunk)`` defeats this — the ``decode`` wrapper
    calls ``clear_cache()`` on entry/exit, so each chunk starts with empty state and
    the boundary frames lose the prior chunk's context -> visible seams.

    We hold our OWN ``_feat_map`` / ``_conv_idx`` (NOT ``vae._feat_map``) because any
    ``vae.encode()`` between our decode calls (e.g. g-buffer / condition encode) also
    calls ``clear_cache`` and would wipe vae-internal state mid-stream.

    Output is numerically equivalent (modulo float-accum order) to one-shot decode of
    the concatenated latents: chunk-0 first latent frame is ``first_chunk=True`` (-> 1
    px frame), every later latent frame -> 4 px frames, total = (sum_lat - 1) * 4 + 1.
    Assumes batch size 1 (validation default).
    """

    def __init__(self, vae):
        self.vae = vae
        vae.clear_cache()
        # Snapshot a fresh [None] * decoder_conv_count list of the correct length.
        self._feat_map: List = list(vae._feat_map)
        self._conv_idx: List[int] = [0]
        self._is_first = True

    @torch.no_grad()
    def decode_chunk(self, raw_latents: torch.Tensor) -> torch.Tensor:
        """raw_latents: (1, C, T, H, W) in VAE input scale (post-denorm).
        Returns: (1, 3, T_pixel, H, W) float in [-1, 1]."""
        assert raw_latents.shape[0] == 1, "streaming causal decode requires batch=1"
        vae = self.vae
        x = vae.post_quant_conv(raw_latents)  # WanCausalConv3d kernel=1 -> no temporal cache
        outs = []
        for i in range(x.shape[2]):
            self._conv_idx[0] = 0
            first = self._is_first and (i == 0)
            out_i = vae.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=first,
            )
            outs.append(out_i)
        self._is_first = False
        out = torch.cat(outs, dim=2)
        if getattr(vae.config, "patch_size", None) is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            out = unpatchify(out, patch_size=vae.config.patch_size)
        return torch.clamp(out, min=-1.0, max=1.0)


class HeliosPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video / image-to-video / video-to-video generation using Helios.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`HeliosTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: UniPCMultistepScheduler,
        transformer: HeliosTransformer3DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, text_inputs.attention_mask.bool()

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            len(prompt)
        if prompt_embeds is None:
            (prompt_embeds, prompt_attention_mask) = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        negative_prompt_attention_mask = None
        return (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask)

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if hasattr(self.transformer, "patch_long"):
            height // self.vae_scale_factor_spatial
            width // self.vae_scale_factor_spatial
            (patch_long_h, patch_long_w) = self.transformer.patch_long.kernel_size[-2:]
        pass

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 448,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def stage1_sample(
        self,
        latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        guidance_scale: Optional[float] = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        device: Optional[torch.device] = None,
        transformer_dtype: torch.dtype = None,
        generator: Optional[torch.Generator] = None,
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        use_dmd: bool = False,
        dmd_sigmas: torch.Tensor = None,
        dmd_timesteps: torch.Tensor = None,
        euler_few_step: bool = False,
        gbuffer_latents: Optional[torch.Tensor] = None,
        callback_on_step_end: Optional[callable] = None,
        callback_on_step_end_tensor_inputs: list = None,
        progress_bar=None,
    ):
        batch_size = latents.shape[0]

        for i, t in enumerate(timesteps):
            is_first_step = i == 0

            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0])

            latent_model_input = latents.to(transformer_dtype)
            # Concat G-buffer latents (clean conditioning) with noisy latent
            if gbuffer_latents is not None:
                latent_model_input = torch.cat([latent_model_input, gbuffer_latents.to(transformer_dtype)], dim=1)
            else:
                # Pad with zeros if patch_embedding expects extra GBuffer channels
                expected_channels = self.transformer.patch_embedding.weight.shape[1]
                if latent_model_input.shape[1] < expected_channels:
                    pad = torch.zeros(
                        *latent_model_input.shape[:1],
                        expected_channels - latent_model_input.shape[1],
                        *latent_model_input.shape[2:],
                        device=latent_model_input.device,
                        dtype=latent_model_input.dtype,
                    )
                    latent_model_input = torch.cat([latent_model_input, pad], dim=1)
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short.to(transformer_dtype),
                    latents_history_mid=latents_history_mid.to(transformer_dtype),
                    latents_history_long=latents_history_long.to(transformer_dtype),
                    is_first_denoising_step=is_first_step,
                    return_dict=False,
                )[0]

            if self.do_classifier_free_guidance and not use_dmd:
                with self.transformer.cache_context("uncond"):
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short.to(transformer_dtype),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                        is_first_denoising_step=is_first_step,
                        return_dict=False,
                    )[0]

                if use_cfg_zero_star:
                    noise_pred_text = noise_pred
                    positive_flat = noise_pred_text.view(batch_size, -1)
                    negative_flat = noise_uncond.view(batch_size, -1)

                    alpha = optimized_scale(positive_flat, negative_flat)
                    alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                    alpha = alpha.to(noise_pred_text.dtype)

                    if (i <= zero_steps) and use_zero_init:
                        noise_pred = noise_pred_text * 0.0
                    else:
                        noise_pred = noise_uncond * alpha + guidance_scale * (noise_pred_text - noise_uncond * alpha)
                else:
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            if euler_few_step:
                # [DUO-VSR eq.5 deploy] pure first-order Euler ALONG the ODE (no renoise), EXACTLY what
                # the progressive student was trained to do: z_{i+1} = z_i + (σ_{i+1} − σ_i)·v(z_i, t_i).
                # dmd_sigmas is the fixed K+1 grid (descending, last==0); last step lands at σ=0 (clean).
                sig_cur = dmd_sigmas[i].to(torch.float32)
                sig_next = dmd_sigmas[i + 1].to(torch.float32)
                latents = (latents.float() + (sig_next - sig_cur) * noise_pred.float()).to(latents.dtype)
            elif use_dmd:
                # xt must be 16ch (matches noise_pred); latent_model_input may be 96ch (gbuffer-concat)
                if latents.shape != noise_pred.shape:
                    raise RuntimeError(
                        f"DMD x0 conversion expects xt/noise_pred shape match, got "
                        f"latents={tuple(latents.shape)} noise_pred={tuple(noise_pred.shape)}"
                    )
                pred_image_or_video = convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=latents,
                    timestep=t * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                    sigmas=dmd_sigmas,
                    timesteps=dmd_timesteps,
                )
                if i < len(timesteps) - 1:
                    latents = add_noise(
                        pred_image_or_video,
                        randn_tensor(pred_image_or_video.shape, generator=generator, device=device),
                        timesteps[i + 1] * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                        sigmas=dmd_sigmas,
                        timesteps=dmd_timesteps,
                    )
                else:
                    latents = pred_image_or_video
            else:
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 384,
        width: int = 640,
        num_frames: int = 73,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        image_latents: Optional[torch.Tensor] = None,
        video: Optional[PipelineImageInput] = None,
        use_interpolate_prompt: bool = False,
        history_sizes: list = [8, 2, 1],
        latent_window_size: int = 9,
        is_keep_x0: bool = True,
        euler_few_step: bool = False,
        few_step_shift: float = 3.0,
        is_enable_stage2: bool = False,
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        use_dmd: bool = False,
        is_skip_first_section: bool = False,
        vae_decode_type: VAEDecodeType = "default",
        overlap_decode: bool = False,
        tiny_decoder=None,
        per_chunk_save_dir: Optional[str] = None,
        per_chunk_save_inputs: bool = False,
        gbuffer_latents_all: Optional[torch.Tensor] = None,
        teacher_forcing_latents: Optional[torch.Tensor] = None,
        long_history_dropout: float = 0.0,
        mid_history_dropout: float = 0.0,
        short_history_dropout: float = 0.0,
        short_history_scale: float = 1.0,
        adaptive_short_style_mean: bool = False,
        short_style_anchor_frames: int = 8,
        short_style_drift_threshold: float = 0.1,
        short_style_drift_gain: float = 0.5,
        short_style_alpha_max: float = 0.35,
        short_style_correction_momentum: float = 0.8,
        profile_pipeline: bool = False,
        stream_session_state: Optional[dict] = None,
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `384`):
                The height in pixels of the generated image.
            width (`int`, defaults to `640`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `73`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HeliosPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~HeliosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HeliosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        history_sizes = sorted(history_sizes, reverse=True)
        torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )
        1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        num_frames = max(num_frames, 1)
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        profile_stats = {
            "prompt_encode": 0.0,
            "denoise": 0.0,
            "vae_decode": 0.0,
            "tiny_decode": 0.0,
            "per_chunk_postprocess": 0.0,
            "per_chunk_export": 0.0,
            "input_decode_export": 0.0,
            "final_batch_decode": 0.0,
            "final_postprocess": 0.0,
            "total": 0.0,
        }

        def _profile_sync():
            if profile_pipeline and torch.cuda.is_available() and (device.type == "cuda"):
                torch.cuda.synchronize(device)

        def _profile_tic():
            if not profile_pipeline:
                return None
            _profile_sync()
            return time.perf_counter()

        def _profile_toc(name, start):
            if start is None:
                return
            _profile_sync()
            profile_stats[name] += time.perf_counter() - start

        _profile_total_start = _profile_tic()
        if use_interpolate_prompt or (prompt is not None and isinstance(prompt, str)):
            batch_size = 1
        _profile_prompt_start = _profile_tic()
        (all_prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask) = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
            )
        )
        _profile_toc("prompt_encode", _profile_prompt_start)
        transformer_dtype = self.transformer.dtype
        all_prompt_embeds = all_prompt_embeds.to(transformer_dtype)
        num_channels_latents = 16
        window_num_frames = (latent_window_size - 1) * self.vae_scale_factor_temporal + 1
        num_latent_frames_total = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        num_latent_sections = max(1, math.ceil(num_latent_frames_total / latent_window_size))
        _stream_resume = bool(stream_session_state and stream_session_state.get("initialized", False))
        total_generated_latent_frames = (
            int(stream_session_state.get("total_generated_latent_frames", 0)) if _stream_resume else 0
        )
        _stream_chunk_offset = int(stream_session_state.get("chunk_index", 0)) if _stream_resume else 0
        StreamingVAEDecodeCache(self.vae) if vae_decode_type == "causal" and tiny_decoder is None else None
        tiny_stream_decode_cache = None
        _overlap_decode = bool(
            overlap_decode
            and vae_decode_type == "causal"
            and (tiny_decoder is None)
            and (per_chunk_save_dir is None)
            and (not per_chunk_save_inputs)
            and (teacher_forcing_latents is None)
            and torch.cuda.is_available()
        )
        _decode_stream = torch.cuda.Stream() if _overlap_decode else None
        _overlap_chunk_videos: List[torch.Tensor] = []
        num_input_gbuffers = (
            gbuffer_latents_all.shape[1] // num_channels_latents
            if per_chunk_save_inputs and gbuffer_latents_all is not None
            else 0
        )
        [
            StreamingVAEDecodeCache(self.vae) for _ in range(num_input_gbuffers)
        ] if per_chunk_save_inputs and vae_decode_type == "causal" else None
        if _stream_resume:
            history_latents = stream_session_state["history_latents"].to(device=device, dtype=torch.float32)
            image_latents = stream_session_state.get("image_latents", image_latents)
            if image_latents is not None:
                image_latents = image_latents.to(device=device, dtype=torch.float32)
        else:
            history_latents = torch.zeros(
                batch_size,
                num_channels_latents,
                sum(history_sizes),
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
                device=device,
                dtype=torch.float32,
            )
        (height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial)
        num_latent_sections * latent_window_size
        if gbuffer_latents_all is not None:
            self.transformer.patch_embedding.weight.shape[1] - num_channels_latents
        anchor_history_stats: Optional[tuple] = (
            stream_session_state.get("anchor_history_stats") if _stream_resume else None
        )
        stream_session_state.get("style_sink_latents") if _stream_resume else None
        list(stream_session_state.get("style_sink_pool", [])) if _stream_resume else []
        short_style_anchor_stats: Optional[tuple] = (
            stream_session_state.get("short_style_anchor_stats") if _stream_resume else None
        )
        short_style_anchor_pool: list = (
            list(stream_session_state.get("short_style_anchor_pool", [])) if _stream_resume else []
        )
        short_style_correction_ema: Optional[torch.Tensor] = (
            stream_session_state.get("short_style_correction_ema") if _stream_resume else None
        )
        stream_session_state.get("chunk_stats_ema") if _stream_resume else None
        for k in range(num_latent_sections):
            current_guidance_scale = guidance_scale
            prompt_embeds = all_prompt_embeds
            _global_chunk_index = _stream_chunk_offset + k
            is_first_section = _global_chunk_index == 0
            is_second_section = _global_chunk_index == 1
            if is_keep_x0:
                if is_first_section:
                    history_sizes_first_section = [1] + history_sizes.copy()
                    history_latents_first_section = torch.zeros(
                        batch_size,
                        num_channels_latents,
                        sum(history_sizes_first_section),
                        height // self.vae_scale_factor_spatial,
                        width // self.vae_scale_factor_spatial,
                        device=device,
                        dtype=torch.float32,
                    )
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
                    (latents_prefix, latents_history_long, latents_history_mid, latents_history_1x) = (
                        history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                            history_sizes_first_section, dim=2
                        )
                    )
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                else:
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
                    latents_prefix = image_latents
                    (latents_history_long, latents_history_mid, latents_history_1x) = history_latents[
                        :, :, -sum(history_sizes) :
                    ].split(history_sizes, dim=2)
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)

            def _drop_history_frames(x, probability, lane):
                probability = float(probability)
                if probability <= 0.0 or x.shape[2] == 0:
                    return x
                if probability >= 1.0:
                    return torch.zeros_like(x)
                dropout_rng = torch.Generator(device="cpu")
                dropout_rng.manual_seed((int(_global_chunk_index) + 1) * 1000003 + int(lane) * 9176)
                keep = (
                    (torch.rand(x.shape[2], generator=dropout_rng) >= probability)
                    .to(device=x.device, dtype=x.dtype)
                    .view(1, 1, -1, 1, 1)
                )
                return x * keep

            latents_history_long = _drop_history_frames(latents_history_long, long_history_dropout, 1)
            latents_history_mid = _drop_history_frames(latents_history_mid, mid_history_dropout, 2)
            if is_keep_x0 and latents_history_short.shape[2] > 1:
                latents_history_short = torch.cat(
                    [
                        latents_history_short[:, :, :1],
                        _drop_history_frames(latents_history_short[:, :, 1:], short_history_dropout, 3),
                    ],
                    dim=2,
                )
            if is_keep_x0 and latents_history_short.shape[2] > 1:
                latents_history_short = torch.cat(
                    [latents_history_short[:, :, :1], latents_history_short[:, :, 1:] * float(short_history_scale)],
                    dim=2,
                )
            if (
                adaptive_short_style_mean
                and short_style_anchor_stats is not None
                and is_keep_x0
                and (latents_history_short.shape[2] > 1)
            ):
                rolling_short = latents_history_short[:, :, 1:].to(torch.float32)
                short_mean = rolling_short.mean(dim=(2, 3, 4), keepdim=True)
                (anchor_mean, anchor_std) = short_style_anchor_stats
                anchor_mean = anchor_mean.to(device=rolling_short.device)
                anchor_std = anchor_std.to(device=rolling_short.device)
                normalized_drift = (short_mean - anchor_mean).abs() / anchor_std.clamp_min(1e-06)
                adaptive_alpha = (
                    (normalized_drift - float(short_style_drift_threshold)) * float(short_style_drift_gain)
                ).clamp(0.0, float(short_style_alpha_max))
                desired_correction = adaptive_alpha * (anchor_mean - short_mean)
                momentum = min(0.999, max(0.0, float(short_style_correction_momentum)))
                if short_style_correction_ema is None or short_style_correction_ema.shape != desired_correction.shape:
                    short_style_correction_ema = torch.zeros_like(desired_correction)
                short_style_correction_ema = (
                    momentum * short_style_correction_ema + (1.0 - momentum) * desired_correction
                )
                if stream_session_state is not None:
                    stream_session_state["short_style_correction_ema"] = short_style_correction_ema.detach()
                corrected_short = (rolling_short + short_style_correction_ema).to(latents_history_short.dtype)
                latents_history_short = torch.cat([latents_history_short[:, :, :1], corrected_short], dim=2)
            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                window_num_frames,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )
            if not is_enable_stage2:
                self.scheduler.set_timesteps(num_inference_steps, mu=1, device=device)
                if euler_few_step:
                    base = torch.linspace(
                        0.999, 0.0, steps=num_inference_steps + 1, dtype=torch.float32, device=device
                    )[:-1]
                    s = float(few_step_shift)
                    shifted = s * base / (1 + (s - 1) * base) if s > 1 else base
                    timesteps = (shifted * 1000.0).to(device)
                    sigmas = torch.cat([shifted, torch.zeros(1, device=shifted.device, dtype=shifted.dtype)])
                    self.scheduler.timesteps = timesteps
                    self.scheduler.sigmas = sigmas
                timesteps = self.scheduler.timesteps
                dmd_sigmas = None
                dmd_timesteps = None
                if use_dmd or euler_few_step:
                    dmd_sigmas = self.scheduler.sigmas.to(self.transformer.device)
                    dmd_timesteps = self.scheduler.timesteps.to(self.transformer.device)
                self._num_timesteps = len(timesteps)
            chunk_gbuffer_latents = None
            if gbuffer_latents_all is not None:
                chunk_gbuffer_latents = gbuffer_latents_all[:, :, k * latent_window_size : (k + 1) * latent_window_size]
            _profile_denoise_start = _profile_tic()
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                latents = self.stage1_sample(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    timesteps=timesteps,
                    guidance_scale=current_guidance_scale,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    device=device,
                    transformer_dtype=transformer_dtype,
                    generator=generator,
                    use_cfg_zero_star=use_cfg_zero_star,
                    use_zero_init=use_zero_init,
                    zero_steps=zero_steps,
                    use_dmd=use_dmd,
                    dmd_sigmas=dmd_sigmas,
                    dmd_timesteps=dmd_timesteps,
                    euler_few_step=euler_few_step,
                    gbuffer_latents=chunk_gbuffer_latents,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    progress_bar=progress_bar,
                )
            _profile_toc("denoise", _profile_denoise_start)
            if is_keep_x0 and (
                is_first_section and image_latents is None or (is_skip_first_section and is_second_section)
            ):
                image_latents = latents[:, :, 0:1, :, :]
            chunk_to_decode = latents
            latents_for_history = latents
            total_generated_latent_frames += latents_for_history.shape[2]
            history_latents = torch.cat([history_latents, latents_for_history], dim=2)
            if stream_session_state is not None:
                history_latents = history_latents[:, :, -sum(history_sizes) :].contiguous()
            if adaptive_short_style_mean and short_style_anchor_stats is None:
                short_style_anchor_pool.append(latents_for_history.detach().to(torch.float32))
                pooled_style = torch.cat(short_style_anchor_pool, dim=2)
                anchor_frame_count = max(1, int(short_style_anchor_frames))
                if pooled_style.shape[2] >= anchor_frame_count:
                    anchor_frames = pooled_style[:, :, :anchor_frame_count]
                    anchor_mean = anchor_frames.mean(dim=(2, 3, 4), keepdim=True)
                    anchor_std = anchor_frames.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-06)
                    short_style_anchor_stats = (anchor_mean, anchor_std)
                    short_style_anchor_pool = []
                    if stream_session_state is not None:
                        stream_session_state["short_style_anchor_stats"] = (anchor_mean, anchor_std)
                        stream_session_state.pop("short_style_anchor_pool", None)
                    print(f"[short-style-anchor] captured frames={anchor_frame_count}")
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
            (slice(None), slice(None), slice(-latent_window_size, None))
            if output_type == "latent" and per_chunk_save_dir is None and (not per_chunk_save_inputs):
                continue
        if vae_decode_type == "causal":
            self.vae.clear_cache()
        self._current_timestep = None
        if stream_session_state is not None:
            stream_session_state.update(
                initialized=True,
                chunk_index=_stream_chunk_offset + num_latent_sections,
                total_generated_latent_frames=total_generated_latent_frames,
                history_latents=history_latents.detach(),
                image_latents=image_latents.detach() if image_latents is not None else None,
                anchor_history_stats=anchor_history_stats,
                tiny_stream_decode_cache=tiny_stream_decode_cache,
            )
        video = chunk_to_decode if stream_session_state is not None else real_history_latents
        self.maybe_free_model_hooks()
        _profile_toc("total", _profile_total_start)
        return HeliosPipelineOutput(frames=video)
