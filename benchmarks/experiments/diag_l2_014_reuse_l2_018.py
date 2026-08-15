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
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    mod18 = load_module(
        os.path.join(base_dir, "examples/elementwise/example_level2_018_matmul_sum_max_avgpool_logsumexp_logsumexp.py"),
        "l2_018_for_014_diag",
    )

    torch.manual_seed(0)
    x = torch.rand(args.BS, args.IN, dtype=torch.float32).npu()
    w = torch.randn(args.OUT, args.IN, dtype=torch.float32).npu()
    ref = torch.matmul(x.cpu(), w.cpu().T)
    ref = torch.sum(ref / 2.0, dim=1, keepdim=True) * 1.5

    _, scale_ms = timed("make_scaled_weight", lambda: w * 0.75)
    w_scaled = w * 0.75
    bias_sum = torch.zeros(1, dtype=torch.float32).npu()

    mod18.tilelang.cache.clear_cache()
    pre_cols = timed(
        "factory_018_precompute_colsum",
        lambda: mod18.precompute_level2_018_colsum(args.IN, args.OUT, block_n=args.block_n),
    )[0]
    apply = timed(
        "factory_018_apply",
        lambda: mod18.apply_level2_018_summary(args.BS, args.IN, block_n=args.block_n),
    )[0]

    colsum, _ = timed("018_precompute_first", lambda: pre_cols(w_scaled))
    out, _ = timed("018_apply_first", lambda: apply(x, colsum, bias_sum))
    print("START correctness", flush=True)
    torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-2, atol=1e-2)
    print("DONE correctness PASS", flush=True)

    for i in range(args.repeat):
        _, ms = timed(f"018_apply_repeat_{i}", lambda: apply(x, colsum, bias_sum))
        print(f"RESULT apply_repeat_{i} {ms:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
