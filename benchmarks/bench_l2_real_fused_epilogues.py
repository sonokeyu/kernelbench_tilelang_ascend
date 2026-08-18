"""Benchmark the L2 real fused epilogue kernels (#7, #48, #90, #97).

Follows TILELANG_REAL_KERNEL_OPTIMIZATION_GUIDE.md:
  1. same input / dtype / device / semantics on both sides
  2. correctness before timing
  3. compile time recorded separately from run time
  4. warmup then NPU-Event timed hot loop
  5. limited tile sweep first, long retest on the best config
"""

import argparse
import csv
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F


EX_DIR = "/workspace/tilelang-ascend/examples/elementwise"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npu_ms(fn, args, warmup, iters):
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


def timed_build(factory):
    t0 = time.perf_counter()
    fn = factory()
    torch.npu.synchronize()
    return fn, (time.perf_counter() - t0) * 1000.0


# ---------------- case definitions ----------------

def case_007(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    row_bias = torch.randn(M, dtype=torch.float32).npu()

    def torch_ref(a, rb):
        a = torch.relu(a)
        a = F.leaky_relu(a, negative_slope=0.01)
        a = F.gelu(a)
        a = torch.sigmoid(a)
        return a + rb[:, None]

    mod = load_module(
        "ex007", os.path.join(EX_DIR, "example_level2_007_epilogue_fused.py")
    )
    return {
        "id": "7",
        "operator": "Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd_Epilogue",
        "stages": "relu,gelu,sigmoid,bias_add (leaky_relu range-eliminated)",
        "args": (x, row_bias),
        "torch_ref": torch_ref,
        "factory": lambda bm, bn: mod.level2_007_epilogue_fused(M, N, bm, bn),
    }


def case_048(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    row_scale = torch.randn(M, dtype=torch.float32).npu()
    row_bias = torch.randn(M, dtype=torch.float32).npu()

    def torch_ref(a, rs, rb):
        a = a * rs[:, None]
        a = torch.tanh(a)
        a = a * rb[:, None]
        return torch.sigmoid(a)

    mod = load_module(
        "ex048", os.path.join(EX_DIR, "example_level2_048_epilogue_fused.py")
    )
    return {
        "id": "48",
        "operator": "Conv3d_Scaling_Tanh_Multiply_Sigmoid_Epilogue",
        "stages": "scale,tanh,multiply,sigmoid",
        "args": (x, row_scale, row_bias),
        "torch_ref": torch_ref,
        "factory": lambda bm, bn: mod.level2_048_epilogue_fused(M, N, bm, bn),
    }


def case_090(M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float32).npu()
    row_sum = torch.randn(M, dtype=torch.float32).npu()

    def torch_ref(a, rs):
        a = F.leaky_relu(a, negative_slope=0.2)
        a = a + rs[:, None]
        a = torch.clamp(a, min=-1.0, max=1.0)
        return F.gelu(a)

    mod = load_module(
        "ex090", os.path.join(EX_DIR, "example_level2_090_epilogue_fused.py")
    )
    return {
        "id": "90",
        "operator": "Conv3d_LeakyReLU_Sum_Clamp_GELU_Epilogue",
        "stages": "leaky_relu,sum_add,clamp,gelu",
        "args": (x, row_sum),
        "torch_ref": torch_ref,
        "factory": lambda bm, bn: mod.level2_090_epilogue_fused(M, N, bm, bn),
    }


def case_097(M, N):
    torch.manual_seed(0)
    eps, bias_scalar, divide_value = 1e-5, 0.3, 2.0
    x = torch.randn(M, N, dtype=torch.float32).npu()
    rm = torch.randn(N, dtype=torch.float32).npu()
    rv = torch.rand(N, dtype=torch.float32).npu() + 0.5
    w = torch.randn(N, dtype=torch.float32).npu()
    b = torch.randn(N, dtype=torch.float32).npu()

    def torch_ref(a, m, v, ww, bb):
        y = F.batch_norm(a, m, v, ww, bb, training=False, eps=eps)
        y = y + bias_scalar
        y = y / divide_value
        return y * torch.sigmoid(y)

    mod = load_module(
        "ex097", os.path.join(EX_DIR, "example_level2_097_epilogue_fused.py")
    )
    return {
        "id": "97",
        "operator": "Matmul_BatchNorm_BiasAdd_Divide_Swish_Epilogue",
        "stages": "bn_eval(rsqrt in kernel),bias_add,divide,swish",
        "args": (x, rm, rv, w, b),
        "torch_ref": torch_ref,
        "factory": lambda bm, bn: mod.level2_097_epilogue_fused(
            M, N, eps, bias_scalar, divide_value, bm, bn
        ),
    }


CASES = {"7": case_007, "48": case_048, "90": case_090, "97": case_097}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="7,48,90,97")
    parser.add_argument("--M", type=int, default=4096)
    parser.add_argument("--N", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, EX_DIR)
    rows = []
    for kid in [k.strip() for k in args.ids.split(",") if k.strip()]:
        spec = CASES[kid](args.M, args.N)
        call_args = spec["args"]
        expected = spec["torch_ref"](*call_args)
        torch.npu.synchronize()
        torch_ms = npu_ms(spec["torch_ref"], call_args, args.warmup, args.iters)
        print(f"[id={kid}] torch_mean={torch_ms:.6f} ms", flush=True)

        for block_n in args.block_n:
            try:
                fn, compile_ms = timed_build(
                    lambda bn=block_n: spec["factory"](args.block_m, bn)
                )
                out = fn(*call_args)
                torch.npu.synchronize()
                torch.testing.assert_close(
                    out.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2
                )
                tile_ms = npu_ms(fn, call_args, args.warmup, args.iters)
                passed, err = "True", ""
            except Exception as exc:  # noqa: BLE001
                tile_ms, compile_ms = float("nan"), float("nan")
                passed, err = "False", str(exc).splitlines()[0][:160]

            speedup = torch_ms / tile_ms if tile_ms == tile_ms and tile_ms > 0 else float("nan")
            row = {
                "id": kid,
                "operator": spec["operator"],
                "stages": spec["stages"],
                "shape": f"{args.M},{args.N}",
                "variant": "tilelang_real_fused_epilogue",
                "block_M": args.block_m,
                "block_N": block_n,
                "warmup": args.warmup,
                "iters": args.iters,
                "torch_mean_ms": f"{torch_ms:.9f}",
                "tilelang_mean_ms": f"{tile_ms:.9f}",
                "compile_ms": f"{compile_ms:.6f}",
                "speedup_mean_torch_over_tilelang": f"{speedup:.9f}",
                "tilelang_passed": passed,
                "error": err,
            }
            rows.append(row)
            print(
                f"  block_N={block_n} tile={tile_ms:.6f} ms "
                f"speedup={speedup:.3f}x passed={passed} {err}",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
