import argparse
import csv
import importlib.util
import os
import sys
import time

import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def npu_ms(fn, args, warmup=3, iters=10):
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


def assert_close(name, actual, expected):
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3, msg=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=128)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--block-b", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=1024)
    parser.add_argument("--block-k", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--out", default="benchmarks/results/l1_max_min_dim1_kparallel_ab.csv")
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    max_orig_mod = load_module("example_max_dim1", os.path.join(root, "examples", "elementwise", "example_max_dim1.py"))
    min_orig_mod = load_module("example_min_dim1", os.path.join(root, "examples", "elementwise", "example_min_dim1.py"))
    opt_mod = load_module("example_max_min_dim1_kparallel", os.path.join(root, "examples", "elementwise", "example_max_min_dim1_kparallel.py"))

    torch.manual_seed(0)
    x = torch.randn(args.B, args.K, args.N, dtype=torch.float32).npu()

    rows = []
    for op_name, torch_fn, orig_mod, orig_factory, opt_factory in [
        ("max", lambda a: torch.max(a, dim=1)[0], max_orig_mod, "max_dim1", "max_dim1_kparallel"),
        ("min", lambda a: torch.min(a, dim=1)[0], min_orig_mod, "min_dim1", "min_dim1_kparallel"),
    ]:
        expected = torch_fn(x)
        torch_ms = npu_ms(torch_fn, (x,), warmup=args.warmup, iters=args.iters)
        rows.append({"op": op_name, "variant": "torch", "B": args.B, "K": args.K, "N": args.N, "block_B": "", "block_K": "", "block_N": "", "compile_ms": "", "mean_ms": f"{torch_ms:.9f}", "speedup_vs_torch": "1.000000000", "correct": "true", "error": ""})

        orig_fn, orig_compile = timed_build(lambda m=orig_mod, f=orig_factory: getattr(m, f)(args.B, args.K, args.N, args.block_b, args.block_n))
        try:
            assert_close(f"{op_name} original", orig_fn(x), expected)
            orig_correct, orig_error = "true", ""
        except Exception as exc:
            orig_correct, orig_error = "false", str(exc).splitlines()[0][:240]
        orig_ms = npu_ms(orig_fn, (x,), warmup=args.warmup, iters=args.iters)
        rows.append({"op": op_name, "variant": "tilelang_original", "B": args.B, "K": args.K, "N": args.N, "block_B": args.block_b, "block_K": "", "block_N": args.block_n, "compile_ms": f"{orig_compile:.6f}", "mean_ms": f"{orig_ms:.9f}", "speedup_vs_torch": f"{torch_ms / orig_ms:.9f}", "correct": orig_correct, "error": orig_error})

        for block_k in args.block_k:
            try:
                opt_fn, opt_compile = timed_build(lambda f=opt_factory, bk=block_k: getattr(opt_mod, f)(args.B, args.K, args.N, args.block_b, bk, args.block_n))
                assert_close(f"{op_name} kparallel block_k={block_k}", opt_fn(x), expected)
                opt_ms = npu_ms(opt_fn, (x,), warmup=args.warmup, iters=args.iters)
                correct, error = "true", ""
            except Exception as exc:
                opt_compile = ""
                opt_ms = float("nan")
                correct, error = "false", str(exc).splitlines()[0][:240]
            rows.append({"op": op_name, "variant": f"tilelang_kparallel_bk{block_k}", "B": args.B, "K": args.K, "N": args.N, "block_B": args.block_b, "block_K": block_k, "block_N": args.block_n, "compile_ms": f"{opt_compile}" if opt_compile != "" else "", "mean_ms": f"{opt_ms:.9f}" if opt_ms == opt_ms else "", "speedup_vs_torch": f"{torch_ms / opt_ms:.9f}" if opt_ms == opt_ms else "", "correct": correct, "error": error})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row, flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
