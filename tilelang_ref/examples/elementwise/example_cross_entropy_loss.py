import tilelang
import tilelang.language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def cross_entropy_loss(B, C, dtype="float", target_dtype="int32"):
    @T.prim_func
    def main(
        Pred: T.Tensor((B, C), dtype),
        Target: T.Tensor((B,), target_dtype),
        Out: T.Tensor((1, 1), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            cur = T.alloc_shared((1, 1), dtype)
            maxv = T.alloc_shared((1, 1), dtype)
            sumv = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)
            logsum = T.alloc_shared((1, 1), dtype)
            loss = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(loss, 0.0)
                for b in T.serial(B):
                    T.tile.fill(maxv, -T.infinity(dtype))
                    for c in T.serial(C):
                        T.copy(Pred[b, c : c + 1], cur)
                        if cur[0, 0] > maxv[0, 0]:
                            maxv[0, 0] = cur[0, 0]

                    T.tile.fill(sumv, 0.0)
                    for c in T.serial(C):
                        T.copy(Pred[b, c : c + 1], cur)
                        T.tile.sub(tmp, cur, maxv)
                        T.tile.exp(tmp, tmp)
                        T.tile.add(sumv, sumv, tmp)

                    T.tile.ln(logsum, sumv)
                    T.tile.add(logsum, logsum, maxv)
                    T.copy(Pred[b, Target[b] : Target[b] + 1], cur)
                    T.tile.sub(tmp, logsum, cur)
                    T.tile.add(loss, loss, tmp)

                T.tile.div(loss, loss, B)
                T.copy(loss, Out)

    return main


def ref_program(predictions, targets):
    return torch.nn.functional.cross_entropy(predictions, targets.long())


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C = 7, 13
    func = cross_entropy_loss(B, C)
    predictions = torch.randn(B, C, dtype=torch.float32).npu()
    targets = torch.randint(0, C, (B,), dtype=torch.int32).npu()
    out = func(predictions, targets)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu().reshape(()), ref_program(predictions.cpu(), targets.cpu()), rtol=1e-3, atol=1e-3)
    print("cross_entropy_loss passed")
