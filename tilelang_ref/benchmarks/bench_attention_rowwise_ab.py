import argparse
import csv
import importlib.util
import os
import time

import torch
import torch.nn.functional as F


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=1)
    parser.add_argument("--NH", type=int, default=2)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--block-d", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--out",
        default="/data/chenkeyu/tilelang_ref/benchmarks/results/l1_attention97_rowwise_ab_shape1x2x16x32.csv",
    )
    args = parser.parse_args()

    root = "/workspace/tilelang-ascend"
    orig_mod = load_module(
        "example_scaled_dot_product_attention",
        os.path.join(root, "examples", "elementwise", "example_scaled_dot_product_attention.py"),
    )
    opt_mod = load_module(
        "example_scaled_dot_product_attention_rowwise",
        os.path.join(root, "examples", "elementwise", "example_scaled_dot_product_attention_rowwise.py"),
    )

    torch.manual_seed(0)
    q = torch.randn(args.BS, args.NH, args.L, args.D, dtype=torch.float32).npu()
    k = torch.randn(args.BS, args.NH, args.L, args.D, dtype=torch.float32).npu()
    v = torch.randn(args.BS, args.NH, args.L, args.D, dtype=torch.float32).npu()
    expected = F.scaled_dot_product_attention(q, k, v)

    rows = []
    torch_fn = lambda q, k, v: F.scaled_dot_product_attention(q, k, v)
    torch_ms = npu_ms(torch_fn, (q, k, v), warmup=args.warmup, iters=args.iters)
    rows.append(
        {
            "op": "scaled_dot_product_attention",
            "variant": "torch",
            "BS": args.BS,
            "NH": args.NH,
            "L": args.L,
            "D": args.D,
            "block_D": "",
            "compile_ms": "",
            "mean_ms": f"{torch_ms:.9f}",
            "speedup_vs_torch": "1.000000000",
            "correct": "true",
        }
    )

    orig_fn, orig_compile = timed_build(lambda: orig_mod.scaled_dot_product_attention(args.BS, args.NH, args.L, args.D))
    orig_out = orig_fn(q, k, v)
    torch.npu.synchronize()
    torch.testing.assert_close(orig_out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3, msg="original")
    orig_ms = npu_ms(orig_fn, (q, k, v), warmup=args.warmup, iters=args.iters)
    rows.append(
        {
            "op": "scaled_dot_product_attention",
            "variant": "tilelang_original",
            "BS": args.BS,
            "NH": args.NH,
            "L": args.L,
            "D": args.D,
            "block_D": "",
            "compile_ms": f"{orig_compile:.6f}",
            "mean_ms": f"{orig_ms:.9f}",
            "speedup_vs_torch": f"{torch_ms / orig_ms:.9f}",
            "correct": "true",
        }
    )

    opt_fn, opt_compile = timed_build(
        lambda: opt_mod.scaled_dot_product_attention_rowwise(args.BS, args.NH, args.L, args.D, args.block_d)
    )
    opt_out = opt_fn(q, k, v)
    torch.npu.synchronize()
    torch.testing.assert_close(opt_out.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3, msg="rowwise")
    opt_ms = npu_ms(opt_fn, (q, k, v), warmup=args.warmup, iters=args.iters)
    rows.append(
        {
            "op": "scaled_dot_product_attention",
            "variant": "tilelang_rowwise",
            "BS": args.BS,
            "NH": args.NH,
            "L": args.L,
            "D": args.D,
            "block_D": args.block_d,
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
