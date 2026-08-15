"""TileLang Level 3 #5: AlexNet – grid per spatial position, all OC per block.
conv+relu+maxpool x3, conv+relu x2, flatten, fc+relu x2, fc. Dropout(p=0) skip."""
import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

tilelang.cache.clear_cache()
PASS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], pass_configs=PASS)
def conv2d_relu_maxpool2d_kernel(BS, IC, OC, IH, IW, K, stride, pool_k, pool_stride, dtype="float"):
    """Output grid = BS * OH * OW; each block processes all OC and pool window.
    Caller pre-pads input; no padding inside kernel."""
    OH = (IH - K) // stride + 1
    OW = (IW - K) // stride + 1
    POH = (OH - pool_k) // pool_stride + 1
    POW = (OW - pool_k) // pool_stride + 1

    @T.prim_func
    def main(X: T.Tensor((BS, IC, IH, IW), dtype),
             Weight: T.Tensor((OC, IC, K, K), dtype),
             Bias: T.Tensor((OC,), dtype),
             Y: T.Tensor((BS, OC, POH, POW), dtype)):
        with T.Kernel(BS * POH * POW, is_npu=True) as (cid, vid):
            pw_spatial = cid % POW
            ph_spatial = (cid // POW) % POH
            b = cid // (POW * POH)

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)
            best_val = T.alloc_shared((1, 1), dtype)

            for oc in T.serial(OC):
                if vid == 0:
                    T.tile.fill(best_val, -T.infinity(dtype))
                    for py in T.serial(pool_k):
                        oh = ph_spatial * pool_stride + py
                        ow_base = pw_spatial * pool_stride
                        for px in T.serial(pool_k):
                            ow = ow_base + px
                            T.copy(Bias[oc:oc+1], acc)
                            for ic in T.serial(IC):
                                for kh in T.serial(K):
                                    ih = oh * stride + kh
                                    for kw in T.serial(K):
                                        iw = ow * stride + kw
                                        T.copy(X[b, ic, ih, iw:iw+1], x)
                                        T.copy(Weight[oc, ic, kh, kw:kw+1], w)
                                        T.tile.mul(prod, x, w)
                                        T.tile.add(acc, acc, prod)
                            T.tile.relu(acc, acc)
                            if acc[0, 0] > best_val[0, 0]:
                                best_val[0, 0] = acc[0, 0]
                    T.copy(best_val, Y[b, oc, ph_spatial, pw_spatial:pw_spatial+1])
    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS)
def conv2d_relu_kernel(BS, IC, OC, IH, IW, K, stride, dtype="float"):
    """Output grid = BS * OH * OW; each block processes all OC."""
    OH = (IH - K) // stride + 1
    OW = (IW - K) // stride + 1

    @T.prim_func
    def main(X: T.Tensor((BS, IC, IH, IW), dtype),
             Weight: T.Tensor((OC, IC, K, K), dtype),
             Bias: T.Tensor((OC,), dtype),
             Y: T.Tensor((BS, OC, OH, OW), dtype)):
        with T.Kernel(BS * OH * OW, is_npu=True) as (cid, vid):
            ow = cid % OW; oh = (cid // OW) % OH; b = cid // (OW * OH)

            x = T.alloc_shared((1, 1), dtype)
            w = T.alloc_shared((1, 1), dtype)
            prod = T.alloc_shared((1, 1), dtype)
            acc = T.alloc_shared((1, 1), dtype)

            for oc in T.serial(OC):
                if vid == 0:
                    T.copy(Bias[oc:oc+1], acc)
                    for ic in T.serial(IC):
                        for kh in T.serial(K):
                            ih = oh * stride + kh
                            for kw in T.serial(K):
                                iw = ow * stride + kw
                                T.copy(X[b, ic, ih, iw:iw+1], x)
                                T.copy(Weight[oc, ic, kh, kw:kw+1], w)
                                T.tile.mul(prod, x, w)
                                T.tile.add(acc, acc, prod)
                    T.tile.relu(acc, acc)
                    T.copy(acc, Y[b, oc, oh, ow:ow+1])
    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS)
def linear4d_relu(BS, IC, IH, IW, OUT, dtype="float"):
    FLAT = IC * IH * IW
    @T.prim_func
    def main(X: T.Tensor((BS, IC, IH, IW), dtype),
             Weight: T.Tensor((OUT, FLAT), dtype),
             Bias: T.Tensor((OUT,), dtype),
             Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid
            x = T.alloc_shared((1, 1), dtype); w = T.alloc_shared((1, 1), dtype)
            pr = T.alloc_shared((1, 1), dtype); ac = T.alloc_shared((1, 1), dtype)
            for o in T.serial(OUT):
                if vid == 0:
                    T.copy(Bias[o:o+1], ac)
                    for c in T.serial(IC):
                        for yy in T.serial(IH):
                            for xx in T.serial(IW):
                                T.copy(X[b,c,yy,xx:xx+1], x)
                                T.copy(Weight[o, c*IH*IW+yy*IW+xx: c*IH*IW+yy*IW+xx+1], w)
                                T.tile.mul(pr, x, w); T.tile.add(ac, ac, pr)
                    T.tile.relu(ac, ac); T.copy(ac, Y[b, o:o+1])
    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS)
def linear_relu(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype),
             Weight: T.Tensor((OUT, IN), dtype),
             Bias: T.Tensor((OUT,), dtype),
             Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid
            x = T.alloc_shared((1, 1), dtype); w = T.alloc_shared((1, 1), dtype)
            pr = T.alloc_shared((1, 1), dtype); ac = T.alloc_shared((1, 1), dtype)
            for o in T.serial(OUT):
                if vid == 0:
                    T.copy(Bias[o:o+1], ac)
                    for i in T.serial(IN):
                        T.copy(X[b,i:i+1], x); T.copy(Weight[o,i:i+1], w)
                        T.tile.mul(pr, x, w); T.tile.add(ac, ac, pr)
                    T.tile.relu(ac, ac); T.copy(ac, Y[b, o:o+1])
    return main


@tilelang.jit(out_idx=[3], pass_configs=PASS)
def linear(BS, IN, OUT, dtype="float"):
    @T.prim_func
    def main(X: T.Tensor((BS, IN), dtype),
             Weight: T.Tensor((OUT, IN), dtype),
             Bias: T.Tensor((OUT,), dtype),
             Y: T.Tensor((BS, OUT), dtype)):
        with T.Kernel(BS, is_npu=True) as (cid, vid):
            b = cid
            x = T.alloc_shared((1, 1), dtype); w = T.alloc_shared((1, 1), dtype)
            pr = T.alloc_shared((1, 1), dtype); ac = T.alloc_shared((1, 1), dtype)
            for o in T.serial(OUT):
                if vid == 0:
                    T.copy(Bias[o:o+1], ac)
                    for i in T.serial(IN):
                        T.copy(X[b,i:i+1], x); T.copy(Weight[o,i:i+1], w)
                        T.tile.mul(pr, x, w); T.tile.add(ac, ac, pr)
                    T.copy(ac, Y[b, o:o+1])
    return main


def run_alexnet(x, params):
    BS = x.shape[0]
    (c1_w,c1_b,c2_w,c2_b,c3_w,c3_b,c4_w,c4_b,c5_w,c5_b,
     fc1_w,fc1_b,fc2_w,fc2_b,fc3_w,fc3_b) = params

    x1 = F.pad(x, (2,2,2,2), mode='constant', value=0)
    c1 = conv2d_relu_maxpool2d_kernel(BS,3,96,x1.shape[2],x1.shape[3],11,4,3,2)(x1,c1_w,c1_b)
    torch.npu.synchronize()

    c2_in = F.pad(c1, (2,2,2,2), mode='constant', value=0)
    h2 = conv2d_relu_maxpool2d_kernel(BS,96,256,c2_in.shape[2],c2_in.shape[3],5,1,3,2)(c2_in,c2_w,c2_b)
    torch.npu.synchronize()

    h3_in = F.pad(h2, (1,1,1,1), mode='constant', value=0)
    h3 = conv2d_relu_kernel(BS,256,384,h3_in.shape[2],h3_in.shape[3],3,1)(h3_in,c3_w,c3_b)
    torch.npu.synchronize()

    h4_in = F.pad(h3, (1,1,1,1), mode='constant', value=0)
    h4 = conv2d_relu_kernel(BS,384,384,h4_in.shape[2],h4_in.shape[3],3,1)(h4_in,c4_w,c4_b)
    torch.npu.synchronize()

    h5_in = F.pad(h4, (1,1,1,1), mode='constant', value=0)
    h5 = conv2d_relu_maxpool2d_kernel(BS,384,256,h5_in.shape[2],h5_in.shape[3],3,1,3,2)(h5_in,c5_w,c5_b)
    torch.npu.synchronize()

    h6 = linear4d_relu(BS,256,6,6,4096)(h5,fc1_w,fc1_b); torch.npu.synchronize()
    h7 = linear_relu(BS,4096,4096)(h6,fc2_w,fc2_b); torch.npu.synchronize()
    out = linear(BS,4096,fc3_w.shape[0])(h7,fc3_w,fc3_b); torch.npu.synchronize()
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    BS, NC = 2, 10
    x = torch.randn(BS, 3, 224, 224, dtype=torch.float32).npu()
    P = (
        torch.randn(96, 3, 11, 11, dtype=torch.float32).npu(),
        torch.randn(96, dtype=torch.float32).npu(),
        torch.randn(256, 96, 5, 5, dtype=torch.float32).npu(),
        torch.randn(256, dtype=torch.float32).npu(),
        torch.randn(384, 256, 3, 3, dtype=torch.float32).npu(),
        torch.randn(384, dtype=torch.float32).npu(),
        torch.randn(384, 384, 3, 3, dtype=torch.float32).npu(),
        torch.randn(384, dtype=torch.float32).npu(),
        torch.randn(256, 384, 3, 3, dtype=torch.float32).npu(),
        torch.randn(256, dtype=torch.float32).npu(),
        torch.randn(4096, 256 * 6 * 6, dtype=torch.float32).npu(),
        torch.randn(4096, dtype=torch.float32).npu(),
        torch.randn(4096, 4096, dtype=torch.float32).npu(),
        torch.randn(4096, dtype=torch.float32).npu(),
        torch.randn(NC, 4096, dtype=torch.float32).npu(),
        torch.randn(NC, dtype=torch.float32).npu(),
    )
    out = run_alexnet(x, P)
    cpu = [p.float().cpu() for p in P]; xc = x.float().cpu()
    ref = F.max_pool2d(F.relu(F.conv2d(xc, cpu[0], cpu[1], stride=4, padding=2)), 3, stride=2)
    ref = F.max_pool2d(F.relu(F.conv2d(ref, cpu[2], cpu[3], padding=2)), 3, stride=2)
    ref = F.relu(F.conv2d(ref, cpu[4], cpu[5], padding=1))
    ref = F.relu(F.conv2d(ref, cpu[6], cpu[7], padding=1))
    ref = F.max_pool2d(F.relu(F.conv2d(ref, cpu[8], cpu[9], padding=1)), 3, stride=2)
    ref = ref.reshape(BS, -1)
    ref = F.relu(F.linear(ref, cpu[10], cpu[11]))
    ref = F.relu(F.linear(ref, cpu[12], cpu[13]))
    ref = F.linear(ref, cpu[14], cpu[15])
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-2, atol=1e-2)
    print("level3_005_alexnet passed")
