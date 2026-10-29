"""Causal G-buffer window contract shared by the bridge, worker and offline renderer."""

MODALITIES = ("albedo", "depth", "metallic", "normal", "roughness")


def gbuffer_pixel_window_for_latent_chunk(choice_idx, latent_window_size, temporal_factor=4):
    """Return ``(local_pixel_start, pixel_count, latent_offset)`` for one latent chunk.

    The first chunk follows the causal first-frame convention and needs only
    ``(lw - 1) * factor + 1`` pixels. Later stateless encoder calls include the
    preceding boundary pixel, produce one boundary latent, and discard it via
    ``latent_offset=1``; they therefore consume ``lw * factor + 1`` pixels while
    adding only ``lw * factor`` new video frames.
    """
    choice_idx = int(choice_idx)
    latent_window_size = int(latent_window_size)
    temporal_factor = int(temporal_factor)
    if choice_idx < 0:
        raise ValueError(f"choice_idx must be non-negative, got {choice_idx}")
    if latent_window_size < 1:
        raise ValueError(f"latent_window_size must be positive, got {latent_window_size}")
    if temporal_factor < 1:
        raise ValueError(f"temporal_factor must be positive, got {temporal_factor}")

    latent_start = choice_idx * latent_window_size
    if latent_start == 0:
        return 0, (latent_window_size - 1) * temporal_factor + 1, 0
    return (
        latent_start * temporal_factor - temporal_factor,
        latent_window_size * temporal_factor + 1,
        1,
    )
