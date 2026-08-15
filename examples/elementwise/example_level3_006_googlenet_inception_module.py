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


# KernelBench Level 3 ID 6: GoogleNet Inception module with four concat branches.
@tilelang.jit(out_idx=[13], pass_configs=pass_configs)
def level3_006_googlenet_inception_module(BS, IC, H, W, O1, R3, O3, R5, O5, OP, dtype="float"):
    OT = O1 + O3 + O5 + OP

    @T.prim_func
    def main(
        X: T.Tensor((BS, IC, H, W), dtype),
        W1: T.Tensor((O1, IC, 1, 1), dtype),
        B1: T.Tensor((O1,), dtype),
        W3R: T.Tensor((R3, IC, 1, 1), dtype),
        B3R: T.Tensor((R3,), dtype),
        W3: T.Tensor((O3, R3, 3, 3), dtype),
        B3: T.Tensor((O3,), dtype),
        W5R: T.Tensor((R5, IC, 1, 1), dtype),
        B5R: T.Tensor((R5,), dtype),
        W5: T.Tensor((O5, R5, 5, 5), dtype),
        B5: T.Tensor((O5,), dtype),
        WP: T.Tensor((OP, IC, 1, 1), dtype),
        BP: T.Tensor((OP,), dtype),
        Y: T.Tensor((BS, OT, H, W), dtype),
    ):
        with T.Kernel(BS * OT * H * W, is_npu=True) as (cid, vid):
            ow = cid % W
            rem0 = cid // W
            oh = rem0 % H
            rem1 = rem0 // H
            co = rem1 % OT
            b = rem1 // OT

            x = T.alloc_shared((1, 1), dtype)
            wt = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            red = T.alloc_shared((1, 1), dtype)
            pooled = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                if co < O1:
                    T.copy(B1[co : co + 1], acc)
                    for ic in T.serial(IC):
                        T.copy(X[b, ic, oh, ow : ow + 1], x)
                        T.copy(W1[co, ic, 0, 0:1], wt)
                        T.tile.mul(prod, x, wt)
                        T.tile.add(acc, acc, prod)
                elif co < O1 + O3:
                    outc = co - O1
                    T.copy(B3[outc : outc + 1], acc)
                    for rc in T.serial(R3):
                        for kh in T.serial(3):
                            ih = oh + kh - 1
                            if ih >= 0 and ih < H:
                                for kw in T.serial(3):
                                    iw = ow + kw - 1
                                    if iw >= 0 and iw < W:
                                        T.copy(B3R[rc : rc + 1], red)
                                        for ic in T.serial(IC):
                                            T.copy(X[b, ic, ih, iw : iw + 1], x)
                                            T.copy(W3R[rc, ic, 0, 0:1], wt)
                                            T.tile.mul(prod, x, wt)
                                            T.tile.add(red, red, prod)
                                        T.copy(W3[outc, rc, kh, kw : kw + 1], wt)
                                        T.tile.mul(prod, red, wt)
                                        T.tile.add(acc, acc, prod)
                elif co < O1 + O3 + O5:
                    outc = co - O1 - O3
                    T.copy(B5[outc : outc + 1], acc)
                    for rc in T.serial(R5):
                        for kh in T.serial(5):
                            ih = oh + kh - 2
                            if ih >= 0 and ih < H:
                                for kw in T.serial(5):
                                    iw = ow + kw - 2
                                    if iw >= 0 and iw < W:
                                        T.copy(B5R[rc : rc + 1], red)
                                        for ic in T.serial(IC):
                                            T.copy(X[b, ic, ih, iw : iw + 1], x)
                                            T.copy(W5R[rc, ic, 0, 0:1], wt)
                                            T.tile.mul(prod, x, wt)
                                            T.tile.add(red, red, prod)
                                        T.copy(W5[outc, rc, kh, kw : kw + 1], wt)
                                        T.tile.mul(prod, red, wt)
                                        T.tile.add(acc, acc, prod)
                else:
                    outc = co - O1 - O3 - O5
                    T.copy(BP[outc : outc + 1], acc)
                    for ic in T.serial(IC):
                        T.tile.fill(pooled, -T.infinity(dtype))
                        for kh in T.serial(3):
                            ih = oh + kh - 1
                            if ih >= 0 and ih < H:
                                for kw in T.serial(3):
                                    iw = ow + kw - 1
                                    if iw >= 0 and iw < W:
                                        T.copy(X[b, ic, ih, iw : iw + 1], x)
                                        if x[0, 0] > pooled[0, 0]:
                                            pooled[0, 0] = x[0, 0]
                        T.copy(WP[outc, ic, 0, 0:1], wt)
                        T.tile.mul(prod, pooled, wt)
                        T.tile.add(acc, acc, prod)
                T.copy(acc, Y[b, co, oh, ow : ow + 1])

    return main


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, IC, H, W = 1, 2, 4, 5
    O1, R3, O3, R5, O5, OP = 3, 2, 4, 2, 3, 2
    func = level3_006_googlenet_inception_module(BS, IC, H, W, O1, R3, O3, R5, O5, OP)
    x = torch.randn(BS, IC, H, W, dtype=torch.float32).npu()
    w1 = torch.randn(O1, IC, 1, 1, dtype=torch.float32).npu()
    b1 = torch.randn(O1, dtype=torch.float32).npu()
    w3r = torch.randn(R3, IC, 1, 1, dtype=torch.float32).npu()
    b3r = torch.randn(R3, dtype=torch.float32).npu()
    w3 = torch.randn(O3, R3, 3, 3, dtype=torch.float32).npu()
    b3 = torch.randn(O3, dtype=torch.float32).npu()
    w5r = torch.randn(R5, IC, 1, 1, dtype=torch.float32).npu()
    b5r = torch.randn(R5, dtype=torch.float32).npu()
    w5 = torch.randn(O5, R5, 5, 5, dtype=torch.float32).npu()
    b5 = torch.randn(O5, dtype=torch.float32).npu()
    wp = torch.randn(OP, IC, 1, 1, dtype=torch.float32).npu()
    bp = torch.randn(OP, dtype=torch.float32).npu()
    out = func(x, w1, b1, w3r, b3r, w3, b3, w5r, b5r, w5, b5, wp, bp)
    torch.npu.synchronize()

    xc = x.cpu()
    ref1 = F.conv2d(xc, w1.cpu(), b1.cpu())
    ref3 = F.conv2d(F.conv2d(xc, w3r.cpu(), b3r.cpu()), w3.cpu(), b3.cpu(), padding=1)
    ref5 = F.conv2d(F.conv2d(xc, w5r.cpu(), b5r.cpu()), w5.cpu(), b5.cpu(), padding=2)
    refp = F.conv2d(F.max_pool2d(xc, kernel_size=3, stride=1, padding=1), wp.cpu(), bp.cpu())
    ref = torch.cat([ref1, ref3, ref5, refp], dim=1)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)
    print("level3_006_googlenet_inception_module passed")
