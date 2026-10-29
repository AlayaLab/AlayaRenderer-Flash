"""Distilled grouped G-buffer encoder that replaces the Wan VAE encoder on the hot path."""

import torch
import torch.nn as nn


class ResBlock3d(nn.Module):
    def __init__(self, dim, conv_groups=1, norm_groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups, dim)
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=conv_groups)
        self.norm2 = nn.GroupNorm(norm_groups, dim)
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=conv_groups)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class GBufferSmallEncoder(nn.Module):
    def __init__(self, num_gbuffers=5, hidden_dim=128, out_channels=80, num_res_blocks=2, mode="separate"):
        super().__init__()
        assert mode in ("separate", "grouped", "joint")
        if mode in ("separate", "grouped"):
            assert out_channels % num_gbuffers == 0, (
                f"{mode} mode needs out_channels divisible by num_gbuffers, got {out_channels}/{num_gbuffers}"
            )
            assert hidden_dim % 32 == 0, f"{mode} mode expects hidden_dim divisible by 32, got {hidden_dim}"
        self.num_gbuffers = num_gbuffers
        self.mode = mode
        self.in_channels = num_gbuffers * 3
        self.out_channels = out_channels
        if mode == "separate":
            conv_in = 3
            conv_hidden = hidden_dim
            conv_out = out_channels // num_gbuffers
            conv_groups = 1
            norm_groups = 32
        elif mode == "grouped":
            conv_in = self.in_channels
            conv_hidden = hidden_dim * num_gbuffers
            conv_out = out_channels
            conv_groups = num_gbuffers
            norm_groups = 32 * num_gbuffers
        else:
            conv_in = self.in_channels
            conv_hidden = hidden_dim
            conv_out = out_channels
            conv_groups = 1
            norm_groups = 32

        self.patchify = nn.Conv3d(
            conv_in,
            conv_hidden,
            kernel_size=(4, 8, 8),
            stride=(4, 8, 8),
            groups=conv_groups,
        )
        self.blocks = nn.ModuleList(
            [ResBlock3d(conv_hidden, conv_groups=conv_groups, norm_groups=norm_groups) for _ in range(num_res_blocks)]
        )
        self.proj_out = nn.Conv3d(conv_hidden, conv_out, kernel_size=1, groups=conv_groups)

    def _encode(self, x):
        # replicate-pad the first frame by 3 to mirror the Wan VAE causal first-frame
        # handling: T=4k+1 pixel frames -> k+1 latent frames
        x = torch.cat([x[:, :, :1].expand(-1, -1, 3, -1, -1), x], dim=2)
        x = self.patchify(x)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert C == self.in_channels, f"expected {self.in_channels} channels, got {C}"
        if self.mode in ("joint", "grouped"):
            return self._encode(x)
        x = x.reshape(B * self.num_gbuffers, 3, T, H, W)
        z = self._encode(x)
        return z.reshape(B, self.out_channels, *z.shape[2:])


def load_gbuffer_encoder_weights(model, checkpoint_path):
    """Load weights from either a raw state dict (.pth) or a DiT checkpoint
    (.safetensors / .pth) that contains the encoder under a `gbuffer_embedder.` prefix."""
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        sd = load_file(checkpoint_path)
    else:
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
    prefixed = {k[len("gbuffer_embedder.") :]: v for k, v in sd.items() if k.startswith("gbuffer_embedder.")}
    if prefixed:
        sd = prefixed
    target = model.state_dict()
    converted = {}
    skipped = []
    for key, value in sd.items():
        if key in target and tuple(value.shape) != tuple(target[key].shape):
            if (
                getattr(model, "mode", "") == "grouped"
                and value.ndim >= 1
                and target[key].shape[0] == value.shape[0] * model.num_gbuffers
                and tuple(target[key].shape[1:]) == tuple(value.shape[1:])
            ):
                value = value.repeat((model.num_gbuffers,) + (1,) * (value.ndim - 1))
            else:
                skipped.append(key)
                continue
        converted[key] = value
    missing, unexpected_loaded = model.load_state_dict(converted, strict=False)
    loaded = len(converted) - len(unexpected_loaded)
    unexpected = list(unexpected_loaded) + skipped
    if loaded == 0:
        raise ValueError(f"no GBufferSmallEncoder weights matched in {checkpoint_path}")
    return loaded, missing, unexpected
