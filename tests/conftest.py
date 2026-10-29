import os

# Keep unit tests offline and off the hub FlashAttention kernels.
os.environ.setdefault("ALAYA_ATTN_BACKEND", "sdpa")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
