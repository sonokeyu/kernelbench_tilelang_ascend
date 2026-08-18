import argparse
import csv
import importlib.util
import os
import time

import torch
import torch.nn.functional as F


def event_ms(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="53,13,89")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    channel_sub = torch.randn(a.M, dtype=torch.float32).npu()
    spec = importlib.util.spec_from_file_location(
        "p2", "/workspace/tilelang-ascend/examples/elementwise/example_level2_real_p2_batch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = {
        "53": (mod.level2_053_scale_hardtanh, lambda: F.hardtanh(x * 0.5, -2, 2), (x,), "Gemm_Scaling_Hardtanh_GELU"),
        "13": (mod.level2_013_postsoftmax_tanh_scale, lambda: torch.tanh(x) * 2, (x,), "ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling"),
        "89": (mod.level2_089_postsoftmax_sub_swish, lambda: (x - channel_sub[:, None]) * torch.sigmoid(x - channel_sub[:, None]), (x, channel_sub), "ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max"),
    }
    rows = []
    for kid in a.ids.split(","):
        factory, ref, call, operator = cases[kid]
        expected = ref()
        torch_ms = event_ms(ref, (), a.warmup, a.iters)
        t0 = time.perf_counter()
        fn = factory(a.M, a.N)
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000
        actual = fn(*call)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
        tile_ms = event_ms(fn, call, a.warmup, a.iters)
        row = {"id": int(kid), "operator": operator, "shape": f"{a.M},{a.N}", "block_M": 16, "block_N": 1024, "warmup": a.warmup, "iters": a.iters, "torch_mean_ms": torch_ms, "tilelang_mean_ms": tile_ms, "compile_ms": compile_ms, "speedup_mean_torch_over_tilelang": torch_ms / tile_ms, "tilelang_passed": True, "variant": "arbitrary_input_real_fused_epilogue"}
        rows.append(row)
        print(row, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
