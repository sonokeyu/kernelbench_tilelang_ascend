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
def kl_div_loss(B, N, dtype="float"):
    @T.prim_func
    def main(Pred: T.Tensor((B, N), dtype), Target: T.Tensor((B, N), dtype), Out: T.Tensor((1, 1), dtype)):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            p = T.alloc_shared((1, 1), dtype)
            t = T.alloc_shared((1, 1), dtype)
            tmp = T.alloc_shared((1, 1), dtype)
            total = T.alloc_shared((1, 1), dtype)

            if vid == 0:
                T.tile.fill(total, 0.0)
                for b in T.serial(B):
                    for n in T.serial(N):
                        T.copy(Pred[b, n : n + 1], p)
                        T.copy(Target[b, n : n + 1], t)
                        T.tile.ln(p, p)
                        T.tile.ln(tmp, t)
                        T.tile.sub(tmp, tmp, p)
                        T.tile.mul(tmp, tmp, t)
                        T.tile.add(total, total, tmp)

                T.tile.div(total, total, B)
                T.copy(total, Out)

    return main


def ref_program(predictions, targets):
    return torch.nn.functional.kl_div(torch.log(predictions), targets, reduction="batchmean")


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 5, 17
    func = kl_div_loss(B, N)
    predictions = torch.rand(B, N, dtype=torch.float32).softmax(dim=-1).npu()
    targets = torch.rand(B, N, dtype=torch.float32).softmax(dim=-1).npu()
    out = func(predictions, targets)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu().reshape(()), ref_program(predictions.cpu(), targets.cpu()), rtol=1e-3, atol=1e-3)
    print("kl_div_loss passed")
