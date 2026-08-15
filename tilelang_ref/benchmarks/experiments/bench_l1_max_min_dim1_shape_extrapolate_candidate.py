import argparse
import csv
import importlib.util
import os
import statistics
import sys
import time

import torch
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
    return statistics.mean(times), statistics.median(times), min(times), max(times)


def check(out, ref):
    try:
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-3, atol=1e-3)
        return True, ""
    except Exception as exc:
        return False, str(exc).splitlines()[0][:240]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--K", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--block-B", type=int, default=16)
    parser.add_argument("--block-N", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--skip-torch-bench", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--out", default="benchmarks/results/l1_max_min_dim1_shape_extrapolate.csv")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    max_mod = load_module(os.path.join(base, "examples/elementwise/example_max_dim1.py"), "max_dim1_shape")
    min_mod = load_module(os.path.join(base, "examples/elementwise/example_min_dim1.py"), "min_dim1_shape")

    rows = []
    for K in args.K:
        print(f"shape_begin B={args.B} K={K} N={args.N}", flush=True)
        torch.manual_seed(0)
        x = torch.rand(args.B, K, args.N, dtype=torch.float32).npu()
        for op_name, op_id, factory, ref_fn in [
            ("Max reduction over dim", 49, max_mod.max_dim1, lambda t: torch.max(t, dim=1)[0]),
            ("Min reduction over dim", 53, min_mod.min_dim1, lambda t: torch.min(t, dim=1)[0]),
        ]:
            ref = None
            if args.skip_correctness:
                torch_stats = (None, None, None, None)
                print(f"{op_name} torch_ref_skipped", flush=True)
            else:
                print(f"{op_name} torch_ref_begin", flush=True)
                ref = ref_fn(x)
                torch.npu.synchronize()
                if args.skip_torch_bench:
                    torch_stats = (None, None, None, None)
                    print(f"{op_name} torch_bench_skipped", flush=True)
                else:
                    torch_stats = bench_events(lambda fn=ref_fn: fn(x), args.warmup, args.repeat)
                    print(f"{op_name} torch_bench_done mean={torch_stats[0]:.6f}", flush=True)

            print(f"{op_name} compile_begin", flush=True)
            tilelang.cache.clear_cache()
            t0 = time.perf_counter()
            func = factory(args.B, K, args.N, args.block_B, args.block_N)
            torch.npu.synchronize()
            compile_ms = (time.perf_counter() - t0) * 1000
            print(f"{op_name} compile_done ms={compile_ms:.3f}", flush=True)

            print(f"{op_name} first_call_begin", flush=True)
            out = func(x)
            torch.npu.synchronize()
            if args.skip_correctness:
                correct, error = None, "skipped"
            else:
                correct, error = check(out, ref)
            print(f"{op_name} first_call_done correct={correct} error={error}", flush=True)

            print(f"{op_name} tile_bench_begin", flush=True)
            tile_stats = bench_events(lambda: func(x), args.warmup, args.repeat)
            print(f"{op_name} tile_bench_done mean={tile_stats[0]:.6f}", flush=True)

            rows.append({
                "id": op_id,
                "operator": op_name,
                "B": args.B,
                "K": K,
                "N": args.N,
                "block_B": args.block_B,
                "block_N": args.block_N,
                "torch_mean_ms": torch_stats[0],
                "torch_median_ms": torch_stats[1],
                "torch_min_ms": torch_stats[2],
                "torch_max_ms": torch_stats[3],
                "tilelang_compile_ms": compile_ms,
                "tilelang_mean_ms": tile_stats[0],
                "tilelang_median_ms": tile_stats[1],
                "tilelang_min_ms": tile_stats[2],
                "tilelang_max_ms": tile_stats[3],
                "speedup_mean_torch_over_tilelang": torch_stats[0] / tile_stats[0] if torch_stats[0] else None,
                "correct": correct,
                "error": error,
            })

            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"checkpoint {args.out}", flush=True)


if __name__ == "__main__":
    main()
