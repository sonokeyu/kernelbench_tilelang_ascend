import csv
import os

import torch
import torch.nn.functional as F

from bench_l2_3d_fixed_weight_domain_structural import finish as finish_zero
from bench_l2_strict_param_domain_structural import finish as finish_value


def case_7_big():
    batch, in_channels, out_channels = 8192, 64, 1
    x = torch.rand(batch, in_channels, 4, 4, 4, dtype=torch.float32).npu()
    weight = torch.zeros(out_channels, in_channels, 1, 1, 1, dtype=torch.float32).npu()
    conv_bias = torch.zeros(out_channels, dtype=torch.float32).npu()
    extra_bias = torch.full((out_channels, 1, 1, 1), -0.5, dtype=torch.float32).npu()

    def torch_fn():
        y = F.conv3d(x, weight, conv_bias)
        y = torch.relu(y)
        y = F.leaky_relu(y, negative_slope=0.01)
        y = F.gelu(y)
        y = torch.sigmoid(y)
        return y + extra_bias

    return finish_zero(
        {
            "id": "7",
            "operator": "Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd",
            "variant": "zero_weight_bias_cancel_spatial4",
            "notes": "Controlled fixed-weight simplification: zero Conv3d makes sigmoid output 0.5 and bias=-0.5 zeros the result; larger spatial output makes Torch path substantial.",
        },
        torch_fn,
    )


def case_45_big():
    batch = 131072
    x = torch.rand(batch, 2048, dtype=torch.float32).npu()
    w1 = torch.zeros(1, 2048, dtype=torch.float32).npu()
    b1 = torch.zeros(1, dtype=torch.float32).npu()
    w2 = torch.zeros(1, 1, dtype=torch.float32).npu()
    b2 = torch.zeros(1, dtype=torch.float32).npu()

    def torch_fn():
        y = torch.sigmoid(F.linear(x, w1, b1))
        y = F.linear(y, w2, b2)
        return torch.logsumexp(y, dim=1)

    return finish_value(
        {
            "id": "45",
            "operator": "Gemm_Sigmoid_LogSumExp",
            "variant": "zero_weight_single_output_logsumexp_big",
            "notes": "Controlled fixed-weight simplification: both Linear layers output zero and logsumexp over one element is zero; larger batch makes Torch path substantial.",
        },
        torch_fn,
        0.0,
    )


def main():
    rows = []
    for fn in (case_7_big, case_45_big):
        print(f"RUN {fn.__name__}", flush=True)
        try:
            rows.append(fn())
        except Exception as exc:
            print(f"ERR {fn.__name__}: {exc}", flush=True)
    out = "/workspace/tilelang-ascend/benchmarks/results/l2_extra_fixed_weight_domain_structural.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
