import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[1, 2], pass_configs=pass_configs)
def layer_norm_partials(B, C, H, W, block_N, dtype="float"):
    size = C * H * W
    n_num = T.ceildiv(size, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((B, C, H, W), dtype),
        SumPartials: T.Tensor((B, n_num), dtype),
        SqPartials: T.Tensor((B, n_num), dtype),
    ):
        with T.Kernel(B * n_num, is_npu=True) as (cid, vid):
            b = cid // n_num
            bn = cid % n_num
            x = T.alloc_shared((1, 1), dtype)
            sq = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            sumsq = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(sumv, 0.0)
                T.tile.fill(sumsq, 0.0)
                for i in T.serial(block_N):
                    idx = bn * block_N + i
                    if idx < size:
                        c = idx // (H * W)
                        rem = idx % (H * W)
                        h = rem // W
                        w = rem % W
                        T.copy(A[b, c, h : h + 1, w : w + 1], x)
                        T.tile.add(sumv, sumv, x)
                        T.tile.mul(sq, x, x)
                        T.tile.add(sumsq, sumsq, sq)
                T.copy(sumv, SumPartials[b : b + 1, bn : bn + 1])
                T.copy(sumsq, SqPartials[b : b + 1, bn : bn + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def layer_norm_stats(B, C, H, W, block_N, eps=1e-5, dtype="float"):
    size = C * H * W
    n_num = T.ceildiv(size, block_N)

    @T.prim_func
    def main(
        SumPartials: T.Tensor((B, n_num), dtype),
        SqPartials: T.Tensor((B, n_num), dtype),
        Stats: T.Tensor((B, 2), dtype),
    ):
        with T.Kernel(B, is_npu=True) as (cid, vid):
            b = cid
            cur = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            sumsq = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            invstd = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(sumv, 0.0)
                T.tile.fill(sumsq, 0.0)
                for bn in T.serial(n_num):
                    T.copy(SumPartials[b : b + 1, bn : bn + 1], cur)
                    T.tile.add(sumv, sumv, cur)
                    T.copy(SqPartials[b : b + 1, bn : bn + 1], cur)
                    T.tile.add(sumsq, sumsq, cur)
                T.tile.div(mean, sumv, size)
                T.tile.div(var, sumsq, size)
                T.tile.mul(cur, mean, mean)
                T.tile.sub(var, var, cur)
                T.tile.add(invstd, var, eps)
                T.tile.rsqrt(invstd, invstd)
                T.copy(mean, Stats[b : b + 1, 0:1])
                T.copy(invstd, Stats[b : b + 1, 1:2])

    return main


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def layer_norm_mean(B, C, H, W, block_N, dtype="float"):
    size = C * H * W
    n_num = T.ceildiv(size, block_N)

    @T.prim_func
    def main(SumPartials: T.Tensor((B, n_num), dtype), Stats: T.Tensor((B, 2), dtype)):
        with T.Kernel(B, is_npu=True) as (cid, vid):
            b = cid
            cur = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            zero = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(sumv, 0.0)
                for bn in T.serial(n_num):
                    T.copy(SumPartials[b : b + 1, bn : bn + 1], cur)
                    T.tile.add(sumv, sumv, cur)
                T.tile.div(mean, sumv, size)
                T.tile.fill(zero, 0.0)
                T.copy(mean, Stats[b : b + 1, 0:1])
                T.copy(zero, Stats[b : b + 1, 1:2])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def layer_norm_var_partials(B, C, H, W, block_N, dtype="float"):
    size = C * H * W
    n_num = T.ceildiv(size, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((B, C, H, W), dtype),
        Stats: T.Tensor((B, 2), dtype),
        VarPartials: T.Tensor((B, n_num), dtype),
    ):
        with T.Kernel(B * n_num, is_npu=True) as (cid, vid):
            b = cid // n_num
            bn = cid % n_num
            x = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            diff = T.alloc_shared((1, 1), dtype)
            var_sum = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Stats[b : b + 1, 0:1], mean)
                T.tile.fill(var_sum, 0.0)
                for i in T.serial(block_N):
                    idx = bn * block_N + i
                    if idx < size:
                        c = idx // (H * W)
                        rem = idx % (H * W)
                        h = rem // W
                        w = rem % W
                        T.copy(A[b, c, h : h + 1, w : w + 1], x)
                        T.tile.sub(diff, x, mean)
                        T.tile.mul(diff, diff, diff)
                        T.tile.add(var_sum, var_sum, diff)
                T.copy(var_sum, VarPartials[b : b + 1, bn : bn + 1])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def layer_norm_finalize_invstd(B, C, H, W, block_N, eps=1e-5, dtype="float"):
    size = C * H * W
    n_num = T.ceildiv(size, block_N)

    @T.prim_func
    def main(
        StatsIn: T.Tensor((B, 2), dtype),
        VarPartials: T.Tensor((B, n_num), dtype),
        StatsOut: T.Tensor((B, 2), dtype),
    ):
        with T.Kernel(B, is_npu=True) as (cid, vid):
            b = cid
            cur = T.alloc_shared((1, 1), dtype)
            var = T.alloc_shared((1, 1), dtype)
            invstd = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(StatsIn[b : b + 1, 0:1], mean)
                T.tile.fill(var, 0.0)
                for bn in T.serial(n_num):
                    T.copy(VarPartials[b : b + 1, bn : bn + 1], cur)
                    T.tile.add(var, var, cur)
                T.tile.div(invstd, var, size)
                T.tile.add(invstd, invstd, eps)
                T.tile.rsqrt(invstd, invstd)
                T.copy(mean, StatsOut[b : b + 1, 0:1])
                T.copy(invstd, StatsOut[b : b + 1, 1:2])

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def layer_norm_apply(B, C, H, W, block_N, dtype="float"):
    size = C * H * W
    n_num = T.ceildiv(size, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((B, C, H, W), dtype),
        Stats: T.Tensor((B, 2), dtype),
        Out: T.Tensor((B, C, H, W), dtype),
    ):
        with T.Kernel(B * n_num, is_npu=True) as (cid, vid):
            b = cid // n_num
            bn = cid % n_num
            x = T.alloc_shared((1, 1), dtype)
            mean = T.alloc_shared((1, 1), dtype)
            invstd = T.alloc_shared((1, 1), dtype)
            out = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Stats[b : b + 1, 0:1], mean)
                T.copy(Stats[b : b + 1, 1:2], invstd)
                for i in T.serial(block_N):
                    idx = bn * block_N + i
                    if idx < size:
                        c = idx // (H * W)
                        rem = idx % (H * W)
                        h = rem // W
                        w = rem % W
                        T.copy(A[b, c, h : h + 1, w : w + 1], x)
                        T.tile.sub(out, x, mean)
                        T.tile.mul(out, out, invstd)
                        T.copy(out, Out[b, c, h : h + 1, w : w + 1])

    return main


def layer_norm_staged(B, C, H, W, block_N, eps=1e-5, dtype="float"):
    stage1 = layer_norm_partials(B, C, H, W, block_N, dtype=dtype)
    stage2 = layer_norm_mean(B, C, H, W, block_N, dtype=dtype)
    stage3 = layer_norm_var_partials(B, C, H, W, block_N, dtype=dtype)
    stage4 = layer_norm_finalize_invstd(B, C, H, W, block_N, eps=eps, dtype=dtype)
    stage5 = layer_norm_apply(B, C, H, W, block_N, dtype=dtype)

    def run(x):
        sum_partials, _ = stage1(x)
        mean_stats = stage2(sum_partials)
        var_partials = stage3(x, mean_stats)
        stats = stage4(mean_stats, var_partials)
        return stage5(x, stats)

    return run


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 4, 7
    func = layer_norm_staged(B, C, H, W, 32)
    x = torch.randn(B, C, H, W, dtype=torch.float32).npu()
    out = func(x)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.nn.LayerNorm(x.shape[1:])(x.cpu()), rtol=1e-2, atol=1e-2)
    print("layer_norm_staged passed")
