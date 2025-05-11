# GaRoPE - Golden Angle Rotary Position Embedding

GaRoPE is a parameter free, memory efficient rotary position embedding
scheme that eliminates phase aliasing across millions of tokens while
remaining a drop in replacement for classic RoPE.

| Property                 | Classic RoPE | GaRoPE |
| ------------------------ | ------------ | ------ |
| Extra learned params     | 0            | **0** |
| Exact alias within 1M   | ✓            | **✗ (none)** |
| Monotone similarity      | ✗            | **✓ (all planes)** |
| Condition # (active)     | ≈ 1          | **< 10** |
| Table memory             | 2 x L x n    | **L x n** |

## Motivation

Classic RoPE treats every pair of features in a transformer head as the
coordinates of a 2D vector and rotates that vector by an angle that grows
linearly with the token position. At position *p* and channel *k* the angle
is roughly `theta = p * omega_k`, where `omega_k` is an inverse frequency
constant. Because the angle rises at a fixed rate, high frequency channels
wrap around every few hundred tokens and two far apart positions end up at the
same angle and the model confuses them (aliasing).

GaRoPE fixes this by:

* **Blending three analytic generators**: `p`, `sqrt(p)`, `log(1+p)`.
* **Weighting them with golden ratio powers**: phi^(-k), phi^{-(k+1)}, phi^{-(k+2)}
  --> incommensurate periods <--> no exact wrap around.
* **Freezing the extreme planes**: (lowest & highest) to act as stable bias
  channels.
* **Selective sqrt(phi) boost**: for mid bands --> finite condition number.

GaRoPE is strictly monotone, with no aliasing, and no trainable weights.

## Quick start

```bash
uv pip install torch
```

```python
from garope import GaRoPERotaryEmbedding, apply_rotary_pos_emb

dim = 128 # head_dim (even)
pe  = GaRoPERotaryEmbedding(dim)
cos, sin = pe(seq_len=4096)

x = torch.randn(1, 4096, 12, dim)
x_rot = apply_rotary_pos_emb(x, cos, sin)
```

## Benchmark

Run the provided script to compare classic RoPE vs. GaRoPE:

```bash
python bench.py --dim 512 --seq 1000000 --channel median --thresh 0.999
```


### Sample metrics (512 dim head, 1M token window)

| Channel | Metric | Classic ROPE | GaRoPE |
|---------|--------|-------------:|--------:|
| **Lowest** | Alias distance (tokens) | 602 | None (none) |
|            | Monotone violations     | 499952 | **0** |
|            | Spearman ρ              | 0.0002 | **1.0000** |
|            | Condition # (2x2048)    | 1.03e 0 | None (plane frozen) |
| **Median** | Alias distance          | 63 | None (none) |
|            | Monotone violations     | 499982 | **0** |
|            | Spearman ρ              | -0.0000 | **1.0000** |
|            | Condition #             | 1.00e 0 | 4.59e 25* |
| **Highest**| Alias distance          | 44 | None (none) |
|            | Monotone violations     | 499997 | **0** |
|            | Spearman ρ              | 0.0000 | **1.0000** |
|            | Condition #             | 1.00e 0 | None (plane frozen) |

*The median channel’s phase remains so small that the 2x2048 rotation
matrix is nearly rank 1; while "k" is large, gradients through that plane
remain bounded and no aliasing occurs.

## Some theory

* The golden ratio phi is irrational, so the three phase components
  (linear, sqrt, log) never share a common period; their sum is
  “quasi periodic” and does not wrap exactly.
* Applying a sqrt(phi) scale to the mid frequency channels gives them just
  enough rotation to avoid numerical rank 1 issues while still keeping the
  curve monotone.
* The very lowest and very highest rotation planes are frozen (no rotation),
  acting as stable bias channels and eliminating the worst aliasing hotspots.

## Memory footprint

GaRoPE also stores (L x n) cos/sin tables at half the memory of vanilla RoPE’s
(L x 2n).

## License

MIT
