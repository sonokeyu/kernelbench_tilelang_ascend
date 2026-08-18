"""Controlled benchmark for five independent arbitrary-input epilogue kernels."""
import argparse
import csv
import importlib.util
import os
import time

import torch
import torch.nn.functional as F

ROOT = "/workspace/tilelang-ascend/examples/elementwise"
META = {
    "21": ("example_level2_021_real_fused.py", "level2_021_bias_scale_sigmoid", "Conv2d_Add_Scale_Sigmoid"),
    "31": ("example_level2_031_real_fused.py", "level2_031_min_bias_scale", "Conv2d_Min_Add_Multiply"),
    "46": ("example_level2_046_real_fused.py", "level2_046_sub_tanh_sub", "Conv2d_Subtract_Tanh_Subtract"),
    "82": ("example_level2_082_real_fused.py", "level2_082_tanh_scale_bias", "Conv2d_Tanh_Scaling_BiasAdd"),
    "92": ("example_level2_092_real_fused.py", "level2_092_tanh_hardswish_residual", "Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd"),
}


def load(kid):
    file, _, _ = META[kid]
    spec = importlib.util.spec_from_file_location("case_" + kid, os.path.join(ROOT, file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def event_ms(fn, args, warmup, iters):
    for _ in range(warmup): fn(*args)
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True); end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(*args)
    end.record(); torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default="82,21,46,31,92")
    p.add_argument("--M", type=int, default=4096); p.add_argument("--N", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=10); p.add_argument("--iters", type=int, default=30)
    p.add_argument("--out", required=True); a = p.parse_args()
    torch.manual_seed(0)
    x = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    bias = torch.randn(a.M, dtype=torch.float32).npu()
    scale = torch.randn(a.M, dtype=torch.float32).npu()
    residual = torch.randn(a.M, a.N, dtype=torch.float32).npu()
    rows = []
    for kid in a.ids.split(","):
        mod = load(kid); _, symbol, operator = META[kid]
        if kid == "82": ref=lambda: torch.tanh(x)*2.0+bias[:,None]; call=(x,bias)
        elif kid == "21": ref=lambda: torch.sigmoid((x+bias[:,None])*scale[:,None]); call=(x,bias,scale)
        elif kid == "46": ref=lambda: torch.tanh(x-0.5)-0.2; call=(x,)
        elif kid == "31": ref=lambda: (torch.minimum(x,torch.tensor(0.5,device=x.device))+bias[:,None])*2.0; call=(x,bias)
        else:
            ref=lambda: F.hardswish(torch.tanh(x))+residual; call=(x,residual)
        expected=ref(); torch_ms=event_ms(ref,(),a.warmup,a.iters)
        t0=time.perf_counter(); fn=getattr(mod,symbol)(a.M,a.N); torch.npu.synchronize(); compile_ms=(time.perf_counter()-t0)*1000
        actual=fn(*call); torch.npu.synchronize(); torch.testing.assert_close(actual.cpu(),expected.cpu(),rtol=1e-2,atol=1e-2)
        tile_ms=event_ms(fn,call,a.warmup,a.iters)
        row={"id":int(kid),"operator":operator,"shape":f"{a.M},{a.N}","block_M":16,"block_N":1024,"warmup":a.warmup,"iters":a.iters,"torch_mean_ms":torch_ms,"tilelang_mean_ms":tile_ms,"compile_ms":compile_ms,"speedup_mean_torch_over_tilelang":torch_ms/tile_ms,"tilelang_passed":True,"variant":"arbitrary_input_real_fused_epilogue"}
        rows.append(row); print(row,flush=True)
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    with open(a.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__": main()
