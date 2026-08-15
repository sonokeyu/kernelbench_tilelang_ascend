import argparse
import importlib.util
import os
import sys
import time

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sync():
    torch.npu.synchronize()


def timed(label, fn):
    print(f"START {label}", flush=True)
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    ms = (time.perf_counter() - t0) * 1000
    print(f"DONE {label} {ms:.3f} ms", flush=True)
    return out, ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--BS", type=int, default=128)
    parser.add_argument("--IN", type=int, default=1024)
    parser.add_argument("--OUT", type=int, default=1024)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--skip-torch-npu", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mod = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_014_gemm_divide_sum_scale_precompute.py"),
        "l2_014_diag_mod",
    )

    torch.manual_seed(0)
    print("START allocate inputs", flush=True)
    x = torch.rand(args.BS, args.IN, dtype=torch.float32).npu()
    w = torch.randn(args.OUT, args.IN, dtype=torch.float32).npu()
    sync()
    print("DONE allocate inputs", flush=True)

    def torch_fn():
        y = torch.matmul(x, w.T)
        y = y / 2.0
        y = torch.sum(y, dim=1, keepdim=True)
        return y * 1.5

    if args.skip_torch_npu:
        print("START cpu_ref", flush=True)
        ref = torch.matmul(x.cpu(), w.cpu().T)
        ref = torch.sum(ref / 2.0, dim=1, keepdim=True) * 1.5
        print("DONE cpu_ref", flush=True)
    else:
        ref, _ = timed("torch_first", torch_fn)

    print("START clear cache", flush=True)
    mod.tilelang.cache.clear_cache()
    print("DONE clear cache", flush=True)

    _, pre_factory_ms = timed(
        "factory_precompute_colsum",
        lambda: mod.precompute_level2_014_colsum(args.IN, args.OUT, block_n=args.block_n),
    )
    pre_cols = mod.precompute_level2_014_colsum(args.IN, args.OUT, block_n=args.block_n)
    print(f"FACTORY precompute repeated object {pre_cols}", flush=True)

    _, apply_factory_ms = timed(
        "factory_apply",
        lambda: mod.apply_level2_014_summary(args.BS, args.IN, block_n=args.block_n),
    )
    apply = mod.apply_level2_014_summary(args.BS, args.IN, block_n=args.block_n)
    print(f"FACTORY apply repeated object {apply}", flush=True)

    colsum, _ = timed("precompute_first", lambda: pre_cols(w))
    out, _ = timed("apply_first", lambda: apply(x, colsum))

    print("START correctness", flush=True)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
    print("DONE correctness PASS", flush=True)

    for i in range(args.repeat):
        _, ms = timed(f"apply_repeat_{i}", lambda: apply(x, colsum))
        print(f"RESULT apply_repeat_{i} {ms:.3f} ms", flush=True)

    print(f"SUMMARY factory_precompute_ms={pre_factory_ms:.3f} factory_apply_ms={apply_factory_ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
