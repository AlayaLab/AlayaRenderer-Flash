"""Checkpoint loading helpers for the AlayaRenderer-Flash transformer."""

import torch


def load_state_dict_file(path):
    """Load a tensor-only state dict from ``.safetensors`` or a legacy ``.pth`` file."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _raise_on_partial_load_mismatch(load_info, component_name):
    missing = list(getattr(load_info, "missing_keys", []) or [])
    unexpected = list(getattr(load_info, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RuntimeError(
            f"[INFER LOAD FAILED] {component_name} partial checkpoint mismatch: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )


def load_partial_weights(t, partial_path):
    """transformer_partial.pth: patch_embedding gbuffer channels, MTMP patches,
    norm_out.scale_shift_table, text_sink — same load logic as
    the reference inference script (no-LoRA path)."""
    print(f"[load] partial weights {partial_path} ...", flush=True)
    partial_sd = load_state_dict_file(partial_path)

    pe_keys = {k.replace("patch_embedding.", ""): v for k, v in partial_sd.items() if k.startswith("patch_embedding.")}
    if pe_keys:
        info = t.patch_embedding.load_state_dict(pe_keys, strict=False)
        _raise_on_partial_load_mismatch(info, "patch_embedding")
        print(f"  patch_embedding: {len(pe_keys)} keys", flush=True)

    for module_name in ["patch_short", "patch_mid", "patch_long"]:
        keys = {k.replace(f"{module_name}.", ""): v for k, v in partial_sd.items() if k.startswith(f"{module_name}.")}
        if keys and hasattr(t, module_name):
            info = getattr(t, module_name).load_state_dict(keys, strict=False)
            _raise_on_partial_load_mismatch(info, module_name)
            print(f"  {module_name}: {len(keys)} keys", flush=True)

    sst_key = "norm_out.scale_shift_table"
    if sst_key in partial_sd:
        t.norm_out.scale_shift_table.data.copy_(partial_sd[sst_key])
        print("  norm_out.scale_shift_table: loaded", flush=True)

    if bool(getattr(t.config, "use_text_sink", False)):
        ts_loaded = 0
        for block_idx, block in enumerate(t.blocks):
            for sub in ("text_sink_k", "text_sink_v", "text_sink_norm_k"):
                if not hasattr(block.attn1, sub):
                    continue
                prefix = f"blocks.{block_idx}.attn1.{sub}."
                sub_state = {k.replace(prefix, ""): v for k, v in partial_sd.items() if k.startswith(prefix)}
                if sub_state:
                    info = getattr(block.attn1, sub).load_state_dict(sub_state, strict=False)
                    ts_loaded += len(sub_state)
                    _raise_on_partial_load_mismatch(info, f"blocks.{block_idx}.attn1.{sub}")
        expected = len(t.blocks) * 5
        if ts_loaded != expected:
            raise RuntimeError(f"text_sink loaded {ts_loaded}, expected {expected}")
        print(f"  text_sink: {ts_loaded} keys across {len(t.blocks)} blocks", flush=True)


def _patch_embedding_base_weight(patch_embedding):
    if hasattr(patch_embedding, "base_layer"):
        return patch_embedding.base_layer.weight
    return patch_embedding.weight


def assert_gbuffer_patch_embedding_ready(transformer):
    pe = transformer.patch_embedding
    base_weight = _patch_embedding_base_weight(pe).data
    in_ch = base_weight.shape[1]
    base_gb_abs = base_weight[:, 16:in_ch].abs().sum().item() if in_ch > 16 else 0.0
    pe_lora_params = {n: p for n, p in pe.named_parameters() if "lora_" in n}
    lora_b_abs = sum(p.data.abs().sum().item() for n, p in pe_lora_params.items() if "lora_B" in n)
    if not (base_gb_abs > 0 or lora_b_abs > 0):
        raise RuntimeError(
            "[INFER LOAD FAILED] patch_embedding has no effective G-Buffer weights after loading "
            f"LoRA/partial: base_gbuffer_abs={base_gb_abs:.4f}, lora_B_abs={lora_b_abs:.4f}. "
            "Use a gbuffer-trained --transformer_path, a partial checkpoint containing patch_embedding, "
            "or a LoRA checkpoint that trains patch_embedding."
        )
    print(
        "  patch_embedding G-Buffer verified after trained-weight load: "
        f"base_gbuffer_abs={base_gb_abs:.4f}, lora_B_abs={lora_b_abs:.4f}"
    )


def assert_text_sink_ready(transformer):
    missing = []
    zero_v = []
    for block_idx, block in enumerate(transformer.blocks):
        for sub in ("text_sink_k", "text_sink_v", "text_sink_norm_k"):
            if not hasattr(block.attn1, sub):
                missing.append(f"blocks.{block_idx}.attn1.{sub}")
        if hasattr(block.attn1, "text_sink_v"):
            v_weight = block.attn1.text_sink_v.weight
            if v_weight.detach().float().abs().sum().item() == 0.0:
                zero_v.append(f"blocks.{block_idx}.attn1.text_sink_v.weight")
    if missing:
        raise RuntimeError(
            f"[INFER LOAD FAILED] use_text_sink=true but {len(missing)} text_sink submodules "
            f"are missing (first few: {missing[:5]})."
        )
    if zero_v:
        raise RuntimeError(
            f"[INFER LOAD FAILED] use_text_sink=true but text_sink_v is zero-initialized in "
            f"{len(zero_v)} blocks (first few: {zero_v[:5]}). Load a checkpoint with trained "
            "text_sink weights or pass --use_text_sink false for a non-text-sink checkpoint."
        )
