"""Dedicated arbitrary-input materialized epilogues with static JIT shapes."""
import tilelang
import tilelang.language as T
import torch

PC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def _grid(M, N, block_M):
    return T.ceildiv(M, block_M) * T.ceildiv(N, 1024)


def _body(M, N, block_M, block_N, body):
    m_num, n_num = T.ceildiv(M, block_M), T.ceildiv(N, block_N)
    sub = block_M // 2

    @T.prim_func
    def main(A: T.Tensor((M, N), "float"), Out: T.Tensor((M, N), "float")):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bm, bn = cid // n_num, cid % n_num
            rs, cs = bm * block_M + vid * sub, bn * block_N
            x = T.alloc_shared((sub, block_N), "float")
            t = T.alloc_shared((sub, block_N), "float")
            T.copy(A[rs:rs + sub, cs:cs + block_N], x)
            body(x, t)
            T.copy(x, Out[rs:rs + sub, cs:cs + block_N])
    return main


def _relu_div_body(x, t):
    T.tile.relu(x, x)
    T.tile.div(x, x, 2.0)


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_063_relu_div(M, N, block_M=16, block_N=1024):
    return _body(M, N, block_M, block_N, _relu_div_body)


def _relu_hardswish_body(x, t):
    T.tile.relu(x, x)
    T.tile.add(t, x, 3.0)
    T.tile.max(t, t, 0.0)
    T.tile.min(t, t, 6.0)
    T.tile.mul(x, x, t)
    T.tile.div(x, x, 6.0)


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_057_relu_hardswish(M, N, block_M=16, block_N=1024):
    return _body(M, N, block_M, block_N, _relu_hardswish_body)


def _sigmoid_scale_add_body(x, t):
    T.tile.sigmoid(t, x)
    T.tile.mul(t, t, 2.0)
    T.tile.add(x, x, t)


@tilelang.jit(out_idx=[1], pass_configs=PC)
def level2_070_sigmoid_scale_add(M, N, block_M=16, block_N=1024):
    return _body(M, N, block_M, block_N, _sigmoid_scale_add_body)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 2048, dtype=torch.float32).npu()
    for name, fn, ref in [
        ("063", level2_063_relu_div, torch.relu(x) / 2.0),
        ("057", level2_057_relu_hardswish, torch.relu(x) * torch.clamp((torch.relu(x) + 3.0) / 6.0, 0, 1)),
        ("070", level2_070_sigmoid_scale_add, x + 2.0 * torch.sigmoid(x)),
    ]:
        out = fn(64, 2048)(x)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
        print(f"{name} passed")
