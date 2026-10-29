# Copyright 2025 The Helios Team. All rights reserved.
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
#
# Modified by Alaya Lab, 2026: exports trimmed to the kernels used here (tiled linear removed).
#
# Adapted from Helios (https://github.com/PKU-YuanGroup/Helios), helios/modules/helios_kernels/__init__.py

"""Fused attention, normalization and RoPE kernels for the Helios transformer."""

from .attention_dispatch import attn_varlen_func, create_navit_attention_masks
from .fp32_rmsnorm import replace_rmsnorm_with_fp32
from .triton_norm import replace_all_norms_with_flash_norms
from .triton_rope import replace_rope_with_flash_rope

__all__ = [
    "attn_varlen_func",
    "create_navit_attention_masks",
    "replace_all_norms_with_flash_norms",
    "replace_rmsnorm_with_fp32",
    "replace_rope_with_flash_rope",
]
