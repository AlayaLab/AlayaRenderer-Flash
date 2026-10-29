import pytest

from alayarenderer_flash.gbuffer import MODALITIES, gbuffer_pixel_window_for_latent_chunk
from alayarenderer_flash.runtime import cadence_for_latent_window, stk_causal_window_spec


def test_modalities_order():
    assert MODALITIES == ("albedo", "depth", "metallic", "normal", "roughness")


@pytest.mark.parametrize(("window", "initial", "resumed"), [(4, 13, 16), (2, 5, 8)])
def test_cadence(window, initial, resumed):
    cadence = cadence_for_latent_window(window)
    assert (cadence.initial_pixel_frames, cadence.resumed_pixel_frames) == (initial, resumed)


def test_causal_window_contract():
    assert gbuffer_pixel_window_for_latent_chunk(0, 4) == (0, 13, 0)
    assert gbuffer_pixel_window_for_latent_chunk(1, 4) == (12, 17, 1)
    spec = stk_causal_window_spec(is_initial=False, latent_window_size=4)
    assert (spec.encoder_input_pixel_count, spec.latent_offset, spec.selected_latent_count) == (17, 1, 4)


def test_causal_window_rejects_bad_input():
    with pytest.raises(ValueError):
        gbuffer_pixel_window_for_latent_chunk(-1, 4)


def test_stale_prompt_boundary_is_rejected():
    from alayarenderer_flash.demo.bridge import prompt_boundary_is_current

    assert prompt_boundary_is_current(None, 10)
    assert prompt_boundary_is_current(120, 100)  # slightly ahead of the bridge: same session
    # Boundary carried over from a previous capture session (incident 2026-10-26).
    assert not prompt_boundary_is_current(2_889_050, 412)
