import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def conv2d_relu_maxpool2d(BS, IC, OC, H, W, K, pool_k=2, dtype="float"):
    OH = H - K + 1
    OW = W - K + 1
    POH = OH // pool_k
    POW = OW // pool_k

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OC, IC, K, K), dtype),
        Bias: T.Tensor((OC,), dtype),
        Y: T.Tensor((BS, OC, POH, POW), dtype),
    ):
        with T.Kernel(BS * OC * POH * POW, is_npu=True) as (cid, vid):
            pw = cid % POW
            rem0 = cid // POW
            ph = rem0 % POH
            rem1 = rem0 // POH
            oc = rem1 % OC
            b = rem1 // OC

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            best = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(best, -T.infinity(dtype))
                for py in T.serial(pool_k):
                    oh = ph * pool_k + py
                    for px in T.serial(pool_k):
                        ow = pw * pool_k + px
                        T.copy(Bias[oc : oc + 1], acc)
                        for ic in T.serial(IC):
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    T.copy(X[b, ic, oh + kh, ow + kw : ow + kw + 1], x)
                                    T.copy(Weight[oc, ic, kh, kw : kw + 1], w)
                                    T.tile.mul(prod, x, w)
                                    T.tile.add(acc, acc, prod)
                        T.tile.relu(acc, acc)
                        if acc[0, 0] > best[0, 0]:
                            best[0, 0] = acc[0, 0]
                T.copy(best, Y[b, oc, ph, pw : pw + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear4d_relu(BS, IC, H, W, OUT, dtype="float"):
    FLAT = IC * H * W

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        Weight: T.Tensor((OUT, FLAT), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for c in T.serial(IC):
                    for yy in T.serial(H):
                        for xx in T.serial(W):
                            flat = c * H * W + yy * W + xx
                            T.copy(X[b, c, yy, xx : xx + 1], x)
                            T.copy(Weight[o, flat : flat + 1], w)
                            T.tile.mul(prod, x, w)
                            T.tile.add(acc, acc, prod)
                T.tile.relu(acc, acc)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear_relu(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        Weight: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(Weight[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.tile.relu(acc, acc)
                T.copy(acc, Y[b, o : o + 1])

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def linear(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(
        X: T.Tensor((BS, IN), dtype),
        Weight: T.Tensor((OUT, IN), dtype),
        Bias: T.Tensor((OUT,), dtype),
        Y: T.Tensor((BS, OUT), dtype),
    ):
        with T.Kernel(BS * OUT, is_npu=True) as (cid, vid):
            o = cid % OUT
            b = cid // OUT

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.copy(Bias[o : o + 1], acc)
                for i in T.serial(IN):
                    T.copy(X[b, i : i + 1], x)
                    T.copy(Weight[o, i : i + 1], w)
                    T.tile.mul(prod, x, w)
                    T.tile.add(acc, acc, prod)
                T.copy(acc, Y[b, o : o + 1])

    return main


def run_lenet5(x, params):
    conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b = params
    c1 = conv2d_relu_maxpool2d(x.shape[0], x.shape[1], conv1_w.shape[0], x.shape[2], x.shape[3], conv1_w.shape[2])(
        x, conv1_w, conv1_b
    )
    torch.npu.synchronize()
    c2 = conv2d_relu_maxpool2d(c1.shape[0], c1.shape[1], conv2_w.shape[0], c1.shape[2], c1.shape[3], conv2_w.shape[2])(
        c1, conv2_w, conv2_b
    )
    torch.npu.synchronize()
    h1 = linear4d_relu(c2.shape[0], c2.shape[1], c2.shape[2], c2.shape[3], fc1_w.shape[0])(c2, fc1_w, fc1_b)
    torch.npu.synchronize()
    h2 = linear_relu(h1.shape[0], h1.shape[1], fc2_w.shape[0])(h1, fc2_w, fc2_b)
    torch.npu.synchronize()
    out = linear(h2.shape[0], h2.shape[1], fc3_w.shape[0])(h2, fc3_w, fc3_b)
    torch.npu.synchronize()
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, NUM_CLASSES = 2, 7
    x = torch.randn(BS, 1, 32, 32, dtype=torch.float32).npu()

    params = (
        torch.randn(6, 1, 5, 5, dtype=torch.float32).npu(),
        torch.randn(6, dtype=torch.float32).npu(),
        torch.randn(16, 6, 5, 5, dtype=torch.float32).npu(),
        torch.randn(16, dtype=torch.float32).npu(),
        torch.randn(120, 16 * 5 * 5, dtype=torch.float32).npu(),
        torch.randn(120, dtype=torch.float32).npu(),
        torch.randn(84, 120, dtype=torch.float32).npu(),
        torch.randn(84, dtype=torch.float32).npu(),
        torch.randn(NUM_CLASSES, 84, dtype=torch.float32).npu(),
        torch.randn(NUM_CLASSES, dtype=torch.float32).npu(),
    )

    out = run_lenet5(x, params)

    cpu_params = [p.cpu() for p in params]
    ref = F.max_pool2d(F.relu(F.conv2d(x.cpu(), cpu_params[0], cpu_params[1])), kernel_size=2, stride=2)
    ref = F.max_pool2d(F.relu(F.conv2d(ref, cpu_params[2], cpu_params[3])), kernel_size=2, stride=2)
    ref = ref.view(BS, -1)
    ref = F.relu(F.linear(ref, cpu_params[4], cpu_params[5]))
    ref = F.relu(F.linear(ref, cpu_params[6], cpu_params[7]))
    ref = F.linear(ref, cpu_params[8], cpu_params[9])
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level3_004_lenet5 passed")
