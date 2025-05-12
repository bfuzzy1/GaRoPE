import argparse
import textwrap
import torch
from garope import GaRoPERotaryEmbedding

# baseline 
class RoPERotaryEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len, *, device=None, dtype=torch.float32):
        device = device or self.inv_freq.device
        p = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None]
        angles = p * self.inv_freq[None, :]
        cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1).to(dtype)
        sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1).to(dtype)
        return cos, sin

def alias(cos_vec, thresh=0.999):
    vec = cos_vec[1:]
    drop = (vec < thresh).nonzero(as_tuple=True)[0]
    if not drop.numel():
        return None
    rebound = (vec[drop[0]+1:] >= thresh).nonzero(as_tuple=True)[0]
    return int(rebound[0] + drop[0] + 2) if rebound.numel() else None

def mono_viol(cos_vec):
    return int(((cos_vec[2:] - cos_vec[1:-1]) > 0).sum().item())

def spearman(cos_vec):
    sim = -cos_vec[1:]
    dist = torch.arange(1, sim.numel() + 1, device=sim.device, dtype=sim.dtype)
    sim_rank = torch.zeros_like(sim).scatter_(0, sim.argsort(), torch.arange(sim.numel(), device=sim.device, dtype=sim.dtype))
    dist_rank = torch.zeros_like(dist).scatter_(0, dist.argsort(), torch.arange(dist.numel(), device=sim.device, dtype=dist.dtype))
    sim_rank -= sim_rank.mean()
    dist_rank -= dist_rank.mean()
    return float((sim_rank*dist_rank).mean()/(sim_rank.std()*dist_rank.std()+1e-9))

def cond(cos_row, sin_row):
    s = torch.linalg.svdvals(torch.vstack([cos_row, sin_row]).float())
    return float((s.max()/s.min()).item())

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--seq", type=int, default=1_000_000)
    p.add_argument("--channel", choices=["lowest","median","highest"], default="median")
    p.add_argument("--thresh", type=float, default=0.999)
    args = p.parse_args()
    torch.set_grad_enabled(False)

    n = args.dim//2
    cid = {"lowest": n-1, "median": n//2, "highest": 0}[args.channel]

    rope   = RoPERotaryEmbedding(args.dim)
    garope = GaRoPERotaryEmbedding(args.dim)

    cos_r, sin_r = rope(args.seq)
    cos_g, sin_g = garope(args.seq)

    v_r, v_g = cos_r[:, cid], cos_g[:, cid]

    report = f"""
GaRoPE vs. RoPE — channel={args.channel}  dim={args.dim}  seq={args.seq}
──────────────────────────────────────────────────────────────────────────
• Alias distance (cos ≥ {args.thresh}, after first drop)
    RoPE   : {alias(v_r,args.thresh)}
    GaRoPE : {alias(v_g,args.thresh)}

• Monotonicity violations
    RoPE   : {mono_viol(v_r)}
    GaRoPE : {mono_viol(v_g)}

• Spearman ρ (ideal = 1)
    RoPE   : {spearman(v_r):.4f}
    GaRoPE : {spearman(v_g):.4f}

• Condition number (2×2048 sub matrix, float64)
    RoPE   : {cond(cos_r[:2048, cid], sin_r[:2048, cid]):.2e}
    GaRoPE : {cond(cos_g[:2048, cid], sin_g[:2048, cid]):.2e}
"""
    print(textwrap.dedent(report))
