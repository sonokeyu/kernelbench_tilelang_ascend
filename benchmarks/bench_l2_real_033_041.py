import argparse
import csv
import importlib.util
import os
import time

import torch
import torch.nn.functional as F


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    p.add_argument("--ids", default="33,41")
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    rows = []
    for kid in a.ids.split(","):
        if kid == "33":
            scale = torch.randn(a.N, dtype=torch.float32).npu()
            ref = lambda: F.batch_norm(x * scale[None, :], None, None, training=True, eps=1e-5)
            path = "/workspace/tilelang-ascend/examples/elementwise/example_level2_033_real_fused.py"
            mod = load(path, "case33")
            expected = ref()
            torch_ms = event_ms(ref, (), a.warmup, a.iters)
            t0 = time.perf_counter(); fn = mod.level2_033_scale_batchnorm(a.M, a.N); torch.npu.synchronize(); compile_ms = (time.perf_counter() - t0) * 1000
            actual = fn(x, scale); args = (x, scale)
            operator = "Gemm_Scale_BatchNorm"
        else:
            path = "/workspace/tilelang-ascend/examples/elementwise/example_level2_041_real_fused.py"
            mod = load(path, "case41")
            ref = lambda: torch.relu(F.gelu(F.batch_norm(x, None, None, training=True, eps=1e-5), approximate="tanh"))
            expected = ref()
            torch_ms = event_ms(ref, (), a.warmup, a.iters)
            t0 = time.perf_counter(); fn = mod.level2_041_batchnorm_gelu_relu(a.M, a.N); torch.npu.synchronize(); compile_ms = (time.perf_counter() - t0) * 1000
            actual = fn(x); args = (x,)
            operator = "Gemm_BatchNorm_GELU_ReLU"
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
        tile_ms = event_ms(fn, args, a.warmup, a.iters)
        rows.append({"id": int(kid), "operator": operator, "shape": f"{a.M},{a.N}", "block_N": 1024, "warmup": a.warmup, "iters": a.iters, "torch_mean_ms": torch_ms, "tilelang_mean_ms": tile_ms, "compile_ms": compile_ms, "speedup_mean_torch_over_tilelang": torch_ms / tile_ms, "tilelang_passed": True, "variant": "arbitrary_input_real_fused_training_bn"})
        print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
