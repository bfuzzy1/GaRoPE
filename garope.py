from __future__ import annotations
import torch
from torch import nn
from typing import Tuple

__all__ = ["GaRoPERotaryEmbedding", "apply_rotary_pos_emb"]

PHI = (1.0 + 5 ** 0.5) / 2.0
BASE = 10_000.0
SCALE = PHI

class GaRoPERotaryEmbedding(nn.Module):
    """Golden Angle Rotary Position Embedding (GaRoPE).

    GaRoPE is a rotary embedding that combines three phase generators
    (linear, logarithmic, and square root) with golden ratio based weights.
    This construction ensures the rotations never alias, regardless of sequence length.

    Only (seq_len x n) tables for cos/sin are stored, not (seq_len x dim).
    Optionally, the lowest frequency channel can be skipped (zeroed).

    Args:
        dim (int): Head dimension (must be even).
        skip_lowest (bool, optional): If True, zero out the lowest frequency plane (last channel).
            This treats it as an absolute bias. Defaults to True.
        device (torch.device or None, optional): Device for buffer initialization.
    """

    def __init__(self, dim: int, *, skip_lowest: bool = True,
                 device: torch.device | None = None):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("head dim must be even")
        self.dim = dim
        self.skip_lowest = skip_lowest

        n = dim // 2
        inv_freq = BASE ** (-torch.arange(0, n, dtype=torch.float32, device=device) / n)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        g = PHI ** (-torch.arange(0, n + 2, dtype=torch.float32, device=device))
        self.register_buffer("g", g, persistent=False)

    def forward(self, seq_len: int, *,
                device: torch.device | None = None,
                dtype: torch.dtype = torch.float32
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the GaRoPE cos/sin tables for a given sequence length.

        Args:
            seq_len (int): Sequence length.
            device (torch.device or None, optional): Device for computation. Defaults to buffer device.
            dtype (torch.dtype, optional): Output dtype for cos/sin tables. Defaults to torch.float32.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine tables of shape (seq_len, n),
            where n = head_dim // 2.
        """
        device = device or self.inv_freq.device
        p = torch.arange(seq_len, dtype=torch.float32, device=device)

        f_lin = p[:, None]
        f_log = torch.log1p(p)[:, None]
        f_sqrt = torch.sqrt(p)[:, None]

        g0 = self.g[:self.inv_freq.size(0)]
        g1 = self.g[1:self.inv_freq.size(0) + 1]
        g2 = self.g[2:self.inv_freq.size(0) + 2]

        angles = SCALE * self.inv_freq * (
            g0 * f_lin + g1 * f_log + g2 * f_sqrt
        )

        if angles.shape[1] > 2:
            angles[:, 1:-1] *= PHI ** 0.5

        angles[:, 0] = 0.0
        if self.skip_lowest:
            angles[:, -1] = 0.0

        cos = torch.cos(angles).to(dtype)
        sin = torch.sin(angles).to(dtype)
        return cos, sin

def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to an input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, seq_len, n_heads, head_dim).
        cos (torch.Tensor): Cosine table of shape (seq_len, n), where n = head_dim // 2.
        sin (torch.Tensor): Sine table of shape (seq_len, n), where n = head_dim // 2.

    Returns:
        torch.Tensor: Output tensor of the same shape as x, with rotary embedding applied.
    """
    b, L, h, d = x.shape
    n = d // 2
    x = x.view(b, L, h, n, 2)
    x_even, x_odd = x[..., 0], x[..., 1]

    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos

    return torch.stack((rot_even, rot_odd), dim=-1).view(b, L, h, d)

if __name__ == "__main__":
    torch.manual_seed(0)
    dim, seq, heads, batch = 128, 64, 4, 2
    pe = GaRoPERotaryEmbedding(dim)
    cos, sin = pe(seq)
    z = torch.randn(batch, seq, heads, dim)
    z_rot = apply_rotary_pos_emb(z.clone(), cos, sin)
    assert z_rot.shape == z.shape
    assert torch.allclose(z_rot.norm(dim=-1), z.norm(dim=-1), atol=1e-5)
    print("[OK] GaRoPE lookup + rotation sanity check passed.")
