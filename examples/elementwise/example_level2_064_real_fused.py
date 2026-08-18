"""L2 #64 correctness prototype: materialized output -> LSE -> activations.

The scalar accumulation path is retained for correctness only. It is not promoted
to trusted performance because it is slower than Torch at controlled shape.
"""
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

PC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_064_logsumexp_epilogue(M, N, dtype="float"):
    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Out: T.Tensor((M, 1), dtype)):
        with T.Kernel(M, is_npu=True) as (cid, vid):
            cur = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)
            pos = T.alloc_shared((1, 1), dtype)
            neg = T.alloc_shared((1, 1), dtype)
            sq = T.alloc_shared((1, 1), dtype)
            cube = T.alloc_shared((1, 1), dtype)
            sig = T.alloc_shared((1, 1), dtype)
            T.tile.fill(maxv, -T.infinity(dtype))
            for col in T.serial(N):
                T.copy(A[cid, col:col + 1], cur)
                if cur[0, 0] > maxv[0, 0]:
                    maxv[0, 0] = cur[0, 0]
            T.tile.fill(sumv, 0.0)
            for col in T.serial(N):
                T.copy(A[cid, col:col + 1], cur)
                T.tile.sub(tmp, cur, maxv)
                T.tile.exp(tmp, tmp)
                T.tile.add(sumv, sumv, tmp)
            T.tile.ln(sumv, sumv)
            T.tile.add(cur, sumv, maxv)
            for _ in T.serial(2):
                T.tile.relu(pos, cur)
                T.tile.sub(neg, cur, pos)
                T.tile.mul(neg, neg, 0.01)
                T.tile.add(cur, pos, neg)
            for _ in T.serial(2):
                T.tile.mul(sq, cur, cur)
                T.tile.mul(cube, sq, cur)
                T.tile.mul(sig, cube, 0.044715)
                T.tile.add(sig, sig, cur)
                T.tile.mul(sig, sig, 1.5957691216)
                T.tile.sigmoid(sig, sig)
                T.tile.mul(cur, cur, sig)
            T.copy(cur, Out[cid, 0:1])
    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.float32).npu()
    out = level2_064_logsumexp_epilogue(64, 2048)(x)
    y = torch.logsumexp(x.cpu(), dim=1, keepdim=True)
    y = F.leaky_relu(y, 0.01)
    y = F.leaky_relu(y, 0.01)
    y = F.gelu(y)
    y = F.gelu(y)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), y, rtol=2e-2, atol=2e-2)
    print("level2_064_logsumexp_epilogue passed; correctness-only")
