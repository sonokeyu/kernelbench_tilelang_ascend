from example_level2_033_gemm_scale_batchnorm import level2_033_gemm_scale_batchnorm

import torch
import torch.nn.functional as F


# KernelBench Level 2 ID 39 has the same operator chain as ID 33 with a different benchmark shape:
# Linear -> learnable scale -> BatchNorm1d(default affine).
def level2_039_gemm_scale_batchnorm(BS, IN, OUT, eps=1e-5, dtype="float"):
    return level2_033_gemm_scale_batchnorm(BS, IN, OUT, eps, dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IN, OUT = 4, 3, 5
    func = level2_039_gemm_scale_batchnorm(BS, IN, OUT)
    x = torch.randn(BS, IN, dtype=torch.float32).npu()
    w = torch.randn(OUT, IN, dtype=torch.float32).npu()
    bias = torch.randn(OUT, dtype=torch.float32).npu()
    scale = torch.randn(OUT, dtype=torch.float32).npu()
    out = func(x, w, bias, scale)
    torch.npu.synchronize()
    ref = F.linear(x.cpu(), w.cpu(), bias.cpu()) * scale.cpu()
    ref = torch.nn.BatchNorm1d(OUT)(ref)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level2_039_gemm_scale_batchnorm passed")
