# Stub so Ocean-OCR's sequence_parallel_utils can be imported when flash-attn is not installed.
# Ocean uses use_flash=False when seqlens is None (typical inference); then this is never called.
import torch
import torch.nn.functional as F


def flash_attn_varlen_func(q, k, v, *args, causal=True, **kwargs):
    """Fallback when flash_attn is not installed: use PyTorch SDPA (slower, more VRAM)."""
    # q,k,v: [total_q, num_heads, head_dim]; output: [total_q, num_heads, head_dim] -> reshape in caller
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.squeeze(0)
