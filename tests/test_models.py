import torch

from alayarenderer_flash.decoders.streaming import StreamingTinyDecoderCache
from alayarenderer_flash.decoders.taehv import TAEHV
from alayarenderer_flash.models.gbuffer_encoder import GBufferSmallEncoder


def test_grouped_encoder_shapes():
    encoder = GBufferSmallEncoder(num_gbuffers=5, hidden_dim=32, out_channels=80, mode="grouped").eval()
    with torch.no_grad():
        out = encoder(torch.randn(1, 15, 13, 32, 32))
    assert out.shape == (1, 80, 4, 4, 4)


def test_streaming_tiny_decoder_matches_full_decode():
    torch.manual_seed(0)
    decoder = TAEHV(checkpoint_path=None).eval()
    latents = torch.randn(1, 16, 8, 4, 4)
    stream = StreamingTinyDecoderCache(decoder)
    chunked = torch.cat([stream.decode_chunk(latents[:, :, :4]), stream.decode_chunk(latents[:, :, 4:])], dim=2)
    stream.reset()
    full = stream.decode_chunk(latents)
    assert chunked.shape[1] == 3
    torch.testing.assert_close(chunked, full)
