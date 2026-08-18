"""Next real-kernel upgrade batch: #26/#58/#59/#68/#100.

Every entry is a separate static writer kernel. The producer output is materialized
at the documented boundary; no weights, aliases, zero domains, or host-side
precomputation are used.
"""
import tilelang
import tilelang.language as T
import torch

PC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _grid(M, N, block_M, block_N):
    return T.ceildiv(M, block_M), T.ceildiv(N, block_N), block_M // 2


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_059_swish_scale(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = _grid(M, N, block_M, block_N)
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            s = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.sigmoid(s, x)
            T.tile.mul(x, x, s)
            T.tile.mul(x, x, 2.0)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])
    return main


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_068_min_subtract(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = _grid(M, N, block_M, block_N)
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.min(x, x, 2.0)
            T.tile.sub(x, x, 2.0)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])
    return main


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_100_clamp_divide(M, N, block_M=16, block_N=1024, dtype="float"):
    mn, nn, sub = _grid(M, N, block_M, block_N)
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.tile.max(x, x, -1.0)
            T.tile.div(x, x, 2.0)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])
    return main


@tilelang.jit(out_idx=[2], pass_configs=PC)
def level2_026_add_hardswish_product(M, N, block_M=16, block_N=1024,
                                     dtype="float"):
    mn, nn, sub = _grid(M, N, block_M, block_N)
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), AddInput: T.Tensor((M, N), dtype),
             Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            h = T.alloc_shared((sub, block_N), dtype)
            t = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.copy(AddInput[rs:rs + sub, cs:cs + block_N], h)
            T.tile.add(x, x, h)
            T.tile.add(t, x, 3.0)
            T.tile.max(t, t, 0.0)
            T.tile.min(t, t, 6.0)
            T.tile.mul(t, t, 1.0 / 6.0)
            T.tile.mul(h, x, t)
            T.tile.mul(x, x, h)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])
    return main


@tilelang.jit(out_idx=[2], pass_configs=PC)
def level2_058_gate_sub_clamp(M, N, block_M=16, block_N=1024,
                              dtype="float"):
    mn, nn, sub = _grid(M, N, block_M, block_N)
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Bias: T.Tensor((1,), dtype),
             Out: T.Tensor((M, N), dtype)):
        with T.Kernel(mn * nn, is_npu=True) as (cid, vid):
            bm, bn = cid // nn, cid % nn
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), dtype)
            gate = T.alloc_shared((sub, block_N), dtype)
            bias = T.alloc_shared((1,), dtype)
            bias_tile = T.alloc_shared((sub, block_N), dtype)
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            T.copy(Bias[0:1], bias)
            T.tile.broadcast(bias_tile, bias)
            T.tile.add(gate, x, 3.0)
            T.tile.sigmoid(gate, gate)
            T.tile.mul(gate, x, gate)
            T.tile.div(gate, gate, 6.0)
            T.tile.sub(gate, gate, bias_tile)
            T.tile.max(gate, gate, -1.0)
            T.tile.min(gate, gate, 1.0)
            T.copy(gate, Out[rs:rs + sub, cs:cs + block_N])
    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    M, N = 64, 2048
    x = torch.randn(M, N, dtype=torch.float32).npu()
    add = torch.randn(M, N, dtype=torch.float32).npu()
    bias = torch.randn(1, dtype=torch.float32).npu()
    cases = [
        ("59", level2_059_swish_scale(M, N), (x,), x * torch.sigmoid(x) * 2.0),
        ("68", level2_068_min_subtract(M, N), (x,), torch.minimum(x, torch.tensor(2.0, device=x.device)) - 2.0),
        ("100", level2_100_clamp_divide(M, N), (x,), torch.clamp(x, min=-1.0) / 2.0),
        ("26", level2_026_add_hardswish_product(M, N), (x, add), (x + add) * torch.nn.functional.hardswish(x + add)),
        # #58 is not run here: the current backend cannot broadcast runtime Bias[1]
        # to the 2D tile without changing the original scalar-parameter boundary.
    ]
    for kid, fn, args, ref in cases:
        out = fn(*args)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
        print(f"level2_{kid} passed")
