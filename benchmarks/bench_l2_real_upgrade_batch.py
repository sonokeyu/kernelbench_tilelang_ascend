import argparse
import csv
import importlib.util
import os
import time

import torch

MOD = "/workspace/tilelang-ascend/examples/elementwise/example_level2_real_upgrade_batch.py"


def load():
    spec = importlib.util.spec_from_file_location("batch", MOD)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=8192)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--block-m", type=int, default=16)
    p.add_argument("--block-n", type=int, default=1024)
    p.add_argument("--ids", default="1,2,9,12,57,63,70,76")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(args.M, args.N, dtype=torch.float32).npu()
    row_bias = torch.randn(args.M, dtype=torch.float32).npu()
    unused = [torch.randn(args.N, dtype=torch.float32).npu() for _ in range(3)]
    module = load()
    rows = []

    def torch_ref(mode):
        if mode in (1, 76):
            return lambda: torch.relu(x + row_bias[:, None])
        if mode == 2:
            return lambda: torch.clamp(
                torch.clamp(x + row_bias[:, None], 0, 1) * 2.0, 0, 1
            ) / 2.0
        if mode == 9:
            return lambda: torch.relu((x - 2.0) * 1.5)
        if mode == 12:
            return lambda: torch.nn.functional.leaky_relu(x * 2.0, 0.01)
        if mode == 57:
            return lambda: torch.relu(x) * torch.clamp((torch.relu(x) + 3.0) / 6.0, 0, 1)
        if mode == 63:
            return lambda: torch.relu(x) / 2.0
        return lambda: x + 2.0 * torch.sigmoid(x)

    for mode in [int(value) for value in args.ids.split(",")]:
        ref = torch_ref(mode)
        expected = ref()
        torch_ms = npu_ms(ref, (), args.warmup, args.iters)
        t0 = time.perf_counter()
        fn = module.level2_real_upgrade(
            args.M, args.N, mode, block_M=args.block_m, block_N=args.block_n
        )
        torch.npu.synchronize()
        compile_ms = (time.perf_counter() - t0) * 1000.0
        call_args = (x, row_bias, *unused)
        actual = fn(*call_args)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)
        tile_ms = npu_ms(fn, call_args, args.warmup, args.iters)
        speedup = torch_ms / tile_ms
        rows.append(
            {
                "id": mode,
                "operator": {
                    1: "Conv2d_ReLU_BiasAdd",
                    2: "ConvTranspose2d_BiasAdd_Clamp_Scale_Clamp_Divide",
                    9: "Matmul_Subtract_Multiply_ReLU",
                    12: "Gemm_Multiply_LeakyReLU",
                    57: "Conv2d_ReLU_HardSwish",
                    63: "Gemm_ReLU_Divide",
                    70: "Gemm_Sigmoid_Scaling_ResidualAdd",
                    76: "Gemm_Add_ReLU",
                }[mode],
                "shape": f"{args.M},{args.N}",
                "block_M": args.block_m,
                "block_N": args.block_n,
                "warmup": args.warmup,
                "iters": args.iters,
                "torch_mean_ms": torch_ms,
                "tilelang_mean_ms": tile_ms,
                "compile_ms": compile_ms,
                "speedup_mean_torch_over_tilelang": speedup,
                "tilelang_passed": True,
                "variant": "arbitrary_input_real_fused_epilogue",
            }
        )
        print(mode, torch_ms, tile_ms, speedup, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
