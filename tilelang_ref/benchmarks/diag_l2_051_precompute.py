import importlib.util

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "m", "examples/elementwise/example_level2_051_precompute_apply.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BS, IN, OUT = 128, 1024, 1024
torch.manual_seed(0)
x = torch.rand(BS, IN, dtype=torch.float32).npu()
w = torch.randn(OUT, IN, dtype=torch.float32).npu()
bias = torch.randn(OUT, dtype=torch.float32).npu()
sub = torch.randn(OUT, dtype=torch.float32).npu()
col = torch.sum(w, dim=0).contiguous()
off = (torch.sum(bias) - torch.sum(sub)).reshape(1).contiguous()

full = F.linear(x, w, bias)
full = full - sub
full = torch.mean(full, dim=1, keepdim=True)
full = torch.logsumexp(full, dim=1, keepdim=True)
full = F.gelu(full) + x

rew = ((x * col).sum(dim=1, keepdim=True) + off) / OUT
rew = F.gelu(rew) + x

row_fn = m.level2_051_row_gelu_scalar(BS, IN, OUT, block_N=256)
row_out = row_fn(x, col, off)
add_fn = m.level2_051_add_residual(BS, IN, block_M=16, block_N=256)
out = add_fn(x, row_out)
torch.npu.synchronize()

for name, val in [("rewrite", rew), ("tile", out)]:
    d = (val - full).abs()
    rel = d / (full.abs() + 1e-6)
    print(
        name,
        "max",
        d.max().item(),
        "mean",
        d.mean().item(),
        "relmax",
        rel.max().item(),
        "allclose",
        torch.allclose(val.cpu(), full.cpu(), rtol=1e-2, atol=1e-2),
    )

d = (out - rew).abs()
print("tile-vs-rewrite", "max", d.max().item(), "mean", d.mean().item())
expected_row = F.gelu(((x * col).sum(dim=1) + off[0]) / OUT)
d = (row_out - expected_row).abs()
print("row", "max", d.max().item(), "mean", d.mean().item(), "row0", row_out[:4].cpu(), expected_row[:4].cpu())
