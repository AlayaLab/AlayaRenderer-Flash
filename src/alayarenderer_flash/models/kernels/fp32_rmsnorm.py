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
# Adapted from Helios (https://github.com/PKU-YuanGroup/Helios), helios/modules/helios_kernels/fp32_rmsnorm.py

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm
from diffusers.utils import is_torch_npu_available, is_torch_version

# ------------------------------- replace funtion -------------------------------


def replace_rmsnorm_with_fp32(model):
    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.RMSNorm, RMSNorm)):

            def new_forward(self, x):
                return FP32RMSNorm.forward(self, x)

            module.forward = new_forward.__get__(module, module.__class__)
            patched_count += 1
    print(f"Patched {patched_count} FP32_RMSNorm modules\n")
    return model


# ------------------------------- Tiled MLP -------------------------------


class FP32RMSNorm(RMSNorm):
    def forward(self, hidden_states):
        if is_torch_npu_available():
            raise ValueError("FP32RMSNorm is not available on NPU")

        if not is_torch_version(">=", "2.4"):
            raise ValueError("FP32RMSNorm is only available in PyTorch 2.4 or higher")

        original_dtype = hidden_states.dtype
        hidden_states = nn.functional.rms_norm(
            hidden_states.float(),
            normalized_shape=(hidden_states.shape[-1],),
            weight=self.weight.float(),
            eps=self.eps,
        )

        bias = getattr(self, "bias", None)
        if bias is not None:
            hidden_states = hidden_states + bias.float()

        return hidden_states.to(original_dtype)
