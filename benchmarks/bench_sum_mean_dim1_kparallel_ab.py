import argparse
import csv
import importlib.util
import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(os.path.dirname(THIS_DIR), "examples", "elementwise")
if os.path.isdir(EXAMPLES_DIR):
    sys.path.insert(0, EXAMPLES_DIR)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def npu_ms(fn, args, warmup=10, iters=50):
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
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--block-b", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=1024)
    parser.add_argument("--block-k", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        default="/data/chenkeyu/tilelang_ref/benchmarks/results/l1_sum_mean_dim1_kparallel_ab_shape128x256x4096.csv",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.B, args.K, args.N, dtype=torch.float32).npu()

    sum_ref = lambda a: torch.sum(a, dim=1, keepdim=True)
    mean_ref = lambda a: torch.mean(a, dim=1)
    expected_sum = sum_ref(x)
    expected_mean = mean_ref(x)

    root = "/workspace/tilelang-ascend"
    sum_orig_mod = load_module(
        "example_sum_dim1",
        os.path.join(root, "examples", "elementwise", "example_sum_dim1.py"),
    )
    mean_orig_mod = load_module(
        "example_mean_dim1",
        os.path.join(root, "examples", "elementwise", "example_mean_dim1.py"),
    )
    opt_mod = load_module(
        "example_sum_mean_dim1_kparallel",
        os.path.join(root, "examples", "elementwise", "example_sum_mean_dim1_kparallel.py"),
    )

    rows = []
    for op_name, torch_fn, expected, orig_factory_name, opt_factory_name in [
        ("sum", sum_ref, expected_sum, "sum_dim1", "sum_dim1_kparallel"),
        ("mean", mean_ref, expected_mean, "mean_dim1", "mean_dim1_kparallel"),
    ]:
        torch_ms = npu_ms(torch_fn, (x,), warmup=args.warmup, iters=args.iters)
        rows.append(
            {
                "op": op_name,
                "variant": "torch",
                "B": args.B,
                "K": args.K,
                "N": args.N,
                "block_B": "",
                "block_K": "",
                "block_N": "",
                "compile_ms": "",
                "mean_ms": f"{torch_ms:.9f}",
                "speedup_vs_torch": "1.000000000",
                "correct": "true",
            }
        )

        orig_mod = sum_orig_mod if op_name == "sum" else mean_orig_mod
        orig_fn, orig_compile = timed_build(
            lambda m=orig_mod, f=orig_factory_name: getattr(m, f)(
                args.B, args.K, args.N, args.block_b, args.block_n
            )
        )
        orig_out = orig_fn(x)
        assert_close(f"{op_name} original", orig_out, expected)
        orig_ms = npu_ms(orig_fn, (x,), warmup=args.warmup, iters=args.iters)
        rows.append(
            {
                "op": op_name,
                "variant": "tilelang_original",
                "B": args.B,
                "K": args.K,
                "N": args.N,
                "block_B": args.block_b,
                "block_K": "",
                "block_N": args.block_n,
                "compile_ms": f"{orig_compile:.6f}",
                "mean_ms": f"{orig_ms:.9f}",
                "speedup_vs_torch": f"{torch_ms / orig_ms:.9f}",
                "correct": "true",
            }
        )

        for block_k in args.block_k:
            opt_fn, opt_compile = timed_build(
                lambda f=opt_factory_name, bk=block_k: getattr(opt_mod, f)(
                    args.B, args.K, args.N, args.block_b, bk, args.block_n
                )
            )
            opt_out = opt_fn(x)
            assert_close(f"{op_name} kparallel block_k={block_k}", opt_out, expected)
            opt_ms = npu_ms(opt_fn, (x,), warmup=args.warmup, iters=args.iters)
            rows.append(
                {
                    "op": op_name,
                    "variant": f"tilelang_kparallel_bk{block_k}",
                    "B": args.B,
                    "K": args.K,
                    "N": args.N,
                    "block_B": args.block_b,
                    "block_K": block_k,
                    "block_N": args.block_n,
                    "compile_ms": f"{opt_compile:.6f}",
                    "mean_ms": f"{opt_ms:.9f}",
                    "speedup_vs_torch": f"{torch_ms / opt_ms:.9f}",
                    "correct": "true",
                }
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
