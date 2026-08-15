import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import tilelang


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bench_events(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def check(out, ref, rtol=1e-2, atol=1e-2):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=rtol, atol=atol)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def run_case(mod, shape, warmup, repeat):
    bs, ic, oc, d, h, w, k = shape
    torch.manual_seed(0)
    x = torch.randn(bs, ic, d, h, w, dtype=torch.float32).npu()
    weight = torch.randn(ic, oc, k, k, k, dtype=torch.float32).npu()
    bias = torch.randn(oc, dtype=torch.float32).npu()

    bn = torch.nn.BatchNorm3d(oc, eps=1e-5).npu()

    def torch_fn():
        y = F.conv_transpose3d(x, weight, bias) * 2.0
        y = bn(y)
        return F.adaptive_avg_pool3d(y, (1, 1, 1))

    torch.npu.synchronize()
    t0 = time.perf_counter()
    torch_out = torch_fn()
    torch.npu.synchronize()
    torch_first_ms = (time.perf_counter() - t0) * 1000
    torch_stats = bench_events(torch_fn, warmup, repeat)

    tilelang.cache.clear_cache()
    t0 = time.perf_counter()
    tile_func = mod.level2_077_convtranspose3d_scale_batchnorm_globalavgpool(bs, ic, oc, d, h, w, k)
    torch.npu.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1000
    torch.npu.synchronize()
    t0 = time.perf_counter()
    tile_out = tile_func(x, weight, bias)
    torch.npu.synchronize()
    tile_first_ms = (time.perf_counter() - t0) * 1000
    correct, error = check(tile_out, torch_out)
    tile_stats = bench_events(lambda: tile_func(x, weight, bias), warmup, repeat)

    return {
        "id": 77,
        "operator": "ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool",
        "shape": str(list(shape)),
        "BS": bs,
        "IC": ic,
        "OC": oc,
        "D": d,
        "H": h,
        "W": w,
        "K": k,
        "torch_first_ms": torch_first_ms,
        "torch_mean_ms": torch_stats["mean_ms"],
        "torch_median_ms": torch_stats["median_ms"],
        "torch_min_ms": torch_stats["min_ms"],
        "torch_max_ms": torch_stats["max_ms"],
        "tilelang_compile_ms": compile_ms,
        "tilelang_first_ms": tile_first_ms,
        "tilelang_mean_ms": tile_stats["mean_ms"],
        "tilelang_median_ms": tile_stats["median_ms"],
        "tilelang_min_ms": tile_stats["min_ms"],
        "tilelang_max_ms": tile_stats["max_ms"],
        "speedup_mean_torch_over_tilelang": torch_stats["mean_ms"] / tile_stats["mean_ms"],
        "tilelang_passed": correct,
        "error": error,
        "notes": "Shape sweep for scalar fused ConvTranspose3d+BatchNorm+GlobalAvgPool template.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results/l2_077_shape_sweep.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_077_convtranspose3d_scale_batchnorm_globalavgpool.py"),
        "l2_077_mod",
    )
    shapes = [
        (2, 1, 3, 2, 2, 2, 2),
        (2, 1, 3, 3, 3, 3, 2),
        (4, 1, 4, 2, 2, 2, 2),
        (2, 2, 4, 2, 2, 2, 2),
    ]
    rows = []
    for shape in shapes:
        print(f"RUN shape={shape}", flush=True)
        row = run_case(mod, shape, args.warmup, args.repeat)
        rows.append(row)
        print(
            f"  pass={row['tilelang_passed']} torch={row['torch_mean_ms']:.6f} "
            f"tile={row['tilelang_mean_ms']:.6f} speedup={row['speedup_mean_torch_over_tilelang']:.3f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
