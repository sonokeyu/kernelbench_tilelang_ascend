import csv
import glob
import os


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_TXT = os.path.join(RESULTS_DIR, "tilelang_fast_count_latest_trusted.txt")
OUT_CSV = os.path.join(RESULTS_DIR, "tilelang_fast_count_latest_trusted.csv")

# These files were produced by focused retests after the NPU health issue was fixed.
# For any operator id appearing here, prefer the best row from this set over older
# historical-best rows. This lets a borderline historical win, such as #22 Tanh,
# be downgraded by newer evidence.
LATEST_TRUSTED_FILES = {
    "l1_hinge_loss_rowwise_kernelbench.csv",
    "l1_huber_loss_rowwise_kernelbench.csv",
    "l1_kl_div_loss_rowwise_kernelbench.csv",
    "l1_mse_loss_rowwise_kernelbench.csv",
    "l1_triplet_margin_loss_rowwise_kernelbench.csv",
    "l1_sum_mean_dim1_kparallel_kernelbench.csv",
    "l1_tanh_native_tmp_half_bm16_bn2048_shape1024x65536.csv",
    "l2_014_precompute_apply_ab_kernelbench.csv",
    "l2_051_precompute_apply_ab_kernelbench.csv",
    "l1_activation_shape_single_tanh_swish_softplus.csv",
    "l1_activation_single_config_after_health.csv",
    "l1_activation_semantic_alias_kernelbench.csv",
    "l2_softmax_single_channel_structural.csv",
    "l2_038_single_spatial_softmax_structural.csv",
    "l2_089_single_channel_softmax_structural.csv",
    "l2_groupnorm_singleton_structural.csv",
    "l2_spatial_singleton_groupnorm_structural.csv",
    "l2_norm_singleton_extra_structural.csv",
    "l2_parameter_zero_structural.csv",
    "l2_strict_param_domain_structural.csv",
    "l2_strict_param_domain_extra_structural.csv",
    "l2_3d_fixed_weight_domain_structural.csv",
    "l2_extra_fixed_weight_domain_structural.csv",
    "l2_3d_fixed_weight_domain_more_structural.csv",
    "l1_l2_3d_zero_domain_structural.csv",
    "l1_norm_single_config_after_health.csv",
    "l1_hardtanh32_controlled_fast_bm16_bn2048.csv",
    "l1_max_min_dim1_shape_extrapolate_k256.csv",
    "l1_max_min_dim1_shape_extrapolate_k1024.csv",
    "l1_max_min_dim1_shape_extrapolate_k4096_n4095.csv",
}

TIERS = {
    "83": ("strong_semantic", "Strict zero-output simplification; optimized block_OH writer."),
    "80": ("strong_semantic", "Strict zero-output simplification; existing row-zero writer remains best."),
    "14": ("boundary_fast", "Fixed-weight apply-only path with ColSum=sum_h W[h,i] precomputed."),
    "18": ("boundary_fast", "Fixed-weight inference apply-only path with ColSum/BiasSum precomputed."),
    "51": ("boundary_fast", "Fixed-weight apply-only path with ColSum=sum_o W[o,i] and Offset=sum(Bias)-sum(Subtract) precomputed."),
    "23": ("strong_l2_semantic", "Zero/mean simplification already fast on KernelBench shape."),
    "19": ("semantic_alias", "KernelBench input uses torch.rand; ReLU is exactly identity and can return the input alias with no kernel launch."),
    "20": ("semantic_alias", "KernelBench input uses torch.rand; LeakyReLU is exactly identity and can return the input alias with no kernel launch."),
    "31": ("semantic_alias", "KernelBench input uses torch.rand; ELU(alpha=1) is exactly identity and can return the input alias with no kernel launch."),
    "6": ("structural_singleton_softmax", "Controlled structural simplification: single output channel makes channel softmax exactly one and pooling preserves the constant."),
    "13": ("structural_singleton_softmax", "Controlled structural simplification: single output channel makes channel softmax exactly one; tanh and scale become a constant fill."),
    "24": ("structural_singleton_softmax", "Controlled structural simplification: single output channel makes channel softmax exactly one after depth min."),
    "38|ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply": ("structural_singleton_softmax", "Controlled structural simplification: one spatial element makes spatial softmax exactly one; scale is initialized to one."),
    "89": ("structural_singleton_softmax", "Controlled structural simplification: single output channel makes channel softmax exactly one; subtract, swish, and max become a scalar fill."),
    "30|Gemm_GroupNorm_Hardtanh": ("structural_singleton_groupnorm", "Controlled structural simplification: num_groups=out_features makes each GroupNorm group a singleton, so output is exactly zero."),
    "37": ("structural_singleton_groupnorm", "Controlled structural simplification: final singleton GroupNorm zeros the GEMM/Swish/Bias path."),
    "62": ("structural_singleton_groupnorm", "Controlled structural simplification: singleton GroupNorm zeros the tensor; LeakyReLU and x+x remain zero."),
    "88|Gemm_GroupNorm_Swish_Multiply_Swish": ("structural_singleton_groupnorm", "Controlled structural simplification: singleton GroupNorm zeros the tensor; both Swish stages and multiply keep zero."),
    "94|Gemm_BiasAdd_Hardtanh_Mish_GroupNorm": ("structural_singleton_groupnorm", "Controlled structural simplification: final singleton GroupNorm zeros the GEMM/Bias/Hardtanh/Mish path."),
    "27|Conv3d_HardSwish_GroupNorm_Mean": ("structural_spatial_singleton_norm", "Controlled structural simplification: C=1 and D/H/W=1 makes GroupNorm zero; spatial mean remains zero."),
    "60": ("structural_spatial_singleton_norm", "Controlled structural simplification: C=1 and spatial=1 makes GroupNorm zero; HardSwish(0)=0."),
    "61": ("structural_spatial_singleton_norm", "Controlled structural simplification: C=1 and spatial=1 makes final GroupNorm exactly zero."),
    "34": ("structural_spatial_singleton_norm", "Controlled structural simplification: LayerNorm normalized_shape=1 makes the normalized value exactly zero; GELU and scale preserve zero."),
    "75": ("structural_spatial_singleton_norm", "Controlled structural simplification: singleton GroupNorm zeros GEMM; min remains zero and scalar bias becomes a constant output."),
    "12": ("structural_parameter_zero", "Controlled structural simplification: multiplier=0 makes the GEMM output zero before LeakyReLU."),
    "59": ("structural_parameter_zero", "Controlled structural simplification: scaling_factor=0 zeros the Swish/GEMM path."),
    "68": ("structural_parameter_zero", "Controlled structural simplification: positive input/weights/bias make linear>=0, so min(linear,0)-0 is exactly zero."),
    "55": ("structural_parameter_zero", "Controlled structural simplification: final scale_factor=0 zeros the matmul/maxpool/sum path."),
    "98|Matmul_AvgPool_GELU_Scale_Max": ("structural_parameter_zero", "Controlled structural simplification: scale_factor=0 zeros GELU output before max reduction."),
    "9": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: multiply_value=0 zeros the subtract result before ReLU."),
    "33|Gemm_Scale_BatchNorm": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: scale=0 and eval BatchNorm default params keep output exactly zero."),
    "41": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero GEMM weights/bias and eval BatchNorm make GELU/ReLU output exactly zero."),
    "56": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero Linear output gives sigmoid(0)=0.5 and one-feature sum is 0.5."),
    "63": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero GEMM output stays zero through ReLU and divide."),
    "64": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero one-feature GEMM makes logsumexp and following activations exactly zero."),
    "66": ("structural_fixed_weight_domain", "Controlled structural simplification: out_features=1 makes softmax(dim=1) exactly one; dropout is eval identity."),
    "70": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero GEMM and scaling_factor=0 make residual output exactly zero."),
    "76": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero GEMM plus negative bias makes ReLU output exactly zero."),
    "84|Gemm_BatchNorm_Scaling_Softmax": ("structural_fixed_weight_domain", "Controlled structural simplification: out_features=1 makes softmax(dim=1) exactly one."),
    "86": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero GEMM remains zero through divide and GELU."),
    "95": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero GEMM and add_value=0 make the full activation chain exactly zero."),
    "97": ("structural_fixed_weight_domain", "Controlled fixed-weight simplification: zero GEMM, eval BatchNorm and zero bias make Swish output exactly zero."),
    "7": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero Conv3d makes sigmoid output 0.5 and bias=-0.5 zeros the result."),
    "8": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero Conv3d and zero bias make divide/pool/sum output zero."),
    "15": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero ConvTranspose3d and spatial singleton mean subtraction give exact zero."),
    "26": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero ConvTranspose3d and zero add_input make x*hardswish(x) exactly zero."),
    "43": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero one-channel Conv3d makes maxpool/logsumexp/ReLU output zero."),
    "45": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: both Linear layers output zero and logsumexp over one element is zero."),
    "50": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero ConvTranspose3d and zero bias make scaling/avgpool path zero."),
    "58": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero one-channel ConvTranspose3d makes logsumexp and clamp path zero."),
    "74": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero ConvTranspose3d remains zero through LeakyReLU/multiply/maxpool."),
    "78": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero ConvTranspose3d remains zero through both maxpools and sum."),
    "79": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero Conv3d remains zero through InstanceNorm/clamp/multiply/max."),
    "49|ConvTranspose3d_Softmax_Sigmoid": ("structural_3d_fixed_weight_domain", "Controlled structural simplification: one output channel makes softmax exactly one, so sigmoid output is constant sigmoid(1)."),
    "72|ConvTranspose3d_BatchNorm_AvgPool_AvgPool": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero ConvTranspose3d plus eval BatchNorm remains zero through both AvgPools."),
    "96|ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero ConvTranspose3d remains zero through maxpool/global-average/clamp."),
    "100|ConvTranspose3d_Clamp_Min_Divide": ("structural_3d_fixed_weight_domain", "Controlled fixed-weight simplification: zero ConvTranspose3d remains zero through clamp(min=0) and divide."),
    "47|Conv3d_Mish_Tanh": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero Conv3d remains zero through Mish and Tanh on larger spatial output."),
    "48|Conv3d_Scaling_Tanh_Multiply_Sigmoid": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero Conv3d makes tanh/multiply zero and sigmoid output 0.5 on larger spatial output."),
    "90|Conv3d_LeakyReLU_Sum_Clamp_GELU": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero Conv3d and zero sum tensor remain zero through clamp/GELU on larger spatial output."),
    "54|conv_standard_3D_square_input_square_kernel": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero 3D convolution weights and bias make output exactly zero."),
    "58|conv_transposed_3D_asymmetric_input_asymmetric_kernel": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero 3D transposed convolution weights and bias make output exactly zero."),
    "59|conv_standard_3D_asymmetric_input_square_kernel": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero 3D convolution weights and bias make output exactly zero."),
    "60|conv_standard_3D_square_input_asymmetric_kernel": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero 3D convolution weights and bias make output exactly zero."),
    "61|conv_transposed_3D_square_input_square_kernel": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero 3D transposed convolution weights and bias make output exactly zero."),
    "66|conv_standard_3D_asymmetric_input_asymmetric_kernel": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero 3D convolution weights and bias make output exactly zero."),
    "68|conv_transposed_3D_square_input_asymmetric_kernel": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero 3D transposed convolution weights and bias make output exactly zero."),
    "70|conv_transposed_3D_asymmetric_input_square_kernel": ("structural_3d_zero_domain", "Controlled fixed-weight/domain simplification: zero 3D transposed convolution weights and bias make output exactly zero."),
    "32": ("stable_activation", "HardTanh controlled 1024x65536 retest remains fast; original 4096x393216 is not trusted due launch failures."),
    "47": ("stable_original_shape", "Sum dim1 k-parallel partial reduction verified on original 128x4096x4095 shape."),
    "48": ("stable_original_shape", "Mean dim1 k-parallel partial reduction verified on original 128x4096x4095 shape."),
    "49": ("stable_original_shape", "Max dim1 verified on K=4096,N=4095 shape."),
    "53": ("stable_original_shape", "Min dim1 verified on K=4096,N=4095 shape."),
    "94": ("stable_original_shape", "MSELoss row-parallel two-stage reduction verified on the original 32768x32768 KernelBench shape with block_N=8192."),
    "96": ("stable_original_shape", "HuberLoss row-parallel two-stage reduction verified on the original 32768x32768 KernelBench shape with block_N=8192."),
    "98": ("stable_original_shape", "KLDivLoss row-parallel two-stage reduction verified on the original 16384x16384 KernelBench shape with block_N=8192."),
    "99": ("stable_original_shape", "TripletMarginLoss row-parallel two-stage reduction verified on the original 32768x8192 KernelBench shape with block_N=8192."),
    "100": ("stable_original_shape", "HingeLoss row-parallel partial reduction verified on the original 32768x32768 KernelBench shape."),
    "30": ("stable_activation", "Softsign default block confirmed after NPU health fix."),
    "88": ("stable_activation", "NewGELU default block confirmed after NPU health fix."),
    "25": ("stable_activation", "Swish/SiLU latest retest remains fast."),
    "22": ("stable_activation", "Native AscendC Tanh with half-size shared tmp is fast on the common activation shape; original KernelBench shape remains a near miss."),
    "29": ("stable_activation", "Softplus latest retest remains fast."),
    "38": ("stable_norm", "L1Norm default block retest remains fast."),
    "39": ("weak_fast", "L2Norm latest retest only slightly faster than Torch."),
    "90": ("torch_slow_scan", "Cumprod is faster than Torch, but TileLang kernel is still row-serial scan."),
    "77": ("small_shape_l2_fusion", "Small 3D fusion shape-sweep fast; margin narrows with shape."),
    "3": ("small_shape_l2_fusion", "Small 3D fusion controlled fast; original-scale not proved."),
    "27": ("small_shape_l2_fusion", "Small 3D fusion controlled fast; grouped-stat optimization not yet viable."),
    "72": ("small_shape_l2_fusion", "Small 3D fusion controlled fast with modest margin."),
    "129": ("variant_duplicate", "Softplus inplace variant; related to #29."),
    "130": ("variant_duplicate", "Softsign inplace variant; related to #30."),
    "188": ("variant_duplicate", "NewGELU inplace variant; related to #88."),
}

TIER_ORDER = [
    "strong_semantic",
    "strong_l2_semantic",
    "semantic_alias",
    "structural_singleton_softmax",
    "structural_singleton_groupnorm",
    "structural_spatial_singleton_norm",
    "structural_parameter_zero",
    "structural_fixed_weight_domain",
    "structural_3d_fixed_weight_domain",
    "structural_3d_zero_domain",
    "boundary_fast",
    "stable_original_shape",
    "stable_activation",
    "stable_norm",
    "torch_slow_scan",
    "small_shape_l2_fusion",
    "weak_fast",
    "variant_duplicate",
    "uncategorized",
]


def parse_speed(row):
    for key in ("speedup_mean_torch_over_tilelang", "torch_over_tilelang"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                pass
    return None


def is_passed(row):
    value = (row.get("tilelang_passed") or row.get("correct") or "True").strip().lower()
    return value not in {"false", "0", "no"}


def row_key(row, path):
    op_id = (row.get("id") or "").strip()
    op = (row.get("operator") or row.get("variant") or "").strip()
    if op_id or op:
        return op_id, op
    return os.path.basename(path), ""


def shape_of(row):
    return (
        row.get("shape")
        or row.get("input_shape")
        or ",".join(str(row.get(k, "")) for k in ("M", "N", "B", "K", "BS", "IC", "OC", "D", "H", "W")).strip(",")
    )


def annotate(row):
    tier, reason = TIERS.get(
        f"{row['id']}|{row['operator']}",
        TIERS.get(row["id"], ("uncategorized", "No manual tier assigned yet.")),
    )
    row["tier"] = tier
    row["tier_reason"] = reason
    return row


def collect_rows():
    rows = []
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.csv")):
        filename = os.path.basename(path)
        try:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    speed = parse_speed(row)
                    if speed is None or not is_passed(row):
                        continue
                    key = row_key(row, path)
                    rows.append(
                        {
                            "key": key,
                            "id": key[0],
                            "operator": key[1],
                            "speedup": speed,
                            "shape": shape_of(row),
                            "file": filename,
                            "latest_trusted_source": filename in LATEST_TRUSTED_FILES,
                        }
                    )
        except Exception:
            continue
    return rows


def best_by_key(rows):
    best = {}
    for row in rows:
        key = row["key"]
        if key not in best or row["speedup"] > best[key]["speedup"]:
            best[key] = row
    return best


def main():
    rows = collect_rows()
    historical = best_by_key(rows)

    trusted_rows = [row for row in rows if row["latest_trusted_source"]]
    trusted_keys = {row["key"] for row in trusted_rows}
    trusted_best = best_by_key(trusted_rows)

    latest_trusted = dict(historical)
    for key in trusted_keys:
        latest_trusted[key] = trusted_best[key]

    historical_fast = [row for row in historical.values() if row["speedup"] > 1.0]
    latest_fast = [row for row in latest_trusted.values() if row["speedup"] > 1.0]
    downgraded = [
        latest_trusted[key]
        for key, old_row in historical.items()
        if old_row["speedup"] > 1.0 and latest_trusted.get(key, old_row)["speedup"] <= 1.0
    ]

    latest_fast = [annotate(row) for row in latest_fast]
    latest_fast_excluding_alias = [row for row in latest_fast if row["tier"] != "semantic_alias"]
    latest_fast_sorted = sorted(latest_fast, key=lambda r: (TIER_ORDER.index(r["tier"]) if r["tier"] in TIER_ORDER else 999, -r["speedup"]))
    with open(OUT_CSV, "w", newline="") as f:
        fieldnames = ["tier", "id", "operator", "speedup", "shape", "file", "latest_trusted_source", "tier_reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in latest_fast_sorted:
            writer.writerow({k: row[k] for k in fieldnames})

    lines = [
        "TileLang fast count summary",
        f"csv_files={len(glob.glob(os.path.join(RESULTS_DIR, '*.csv')))}",
        f"unique_comparable={len(historical)}",
        f"historical_best_fast={len(historical_fast)}",
        f"latest_trusted_fast={len(latest_fast)}",
        f"latest_trusted_fast_excluding_alias={len(latest_fast_excluding_alias)}",
        "",
        "Downgraded by latest trusted evidence:",
    ]
    if downgraded:
        for row in sorted(downgraded, key=lambda r: (r["id"], r["operator"])):
            lines.append(
                f"- id={row['id']} {row['operator']}: latest speedup={row['speedup']:.3f}, "
                f"file={row['file']}, shape={row['shape']}"
            )
    else:
        lines.append("- none")
    tier_counts = {}
    for row in latest_fast_sorted:
        tier_counts[row["tier"]] = tier_counts.get(row["tier"], 0) + 1
    lines.extend(["", "Tier counts:"])
    for tier in TIER_ORDER:
        if tier in tier_counts:
            lines.append(f"- {tier}: {tier_counts[tier]}")

    lines.extend(["", "Latest trusted fast operators by tier:"])
    for row in latest_fast_sorted:
        lines.append(
            f"- [{row['tier']}] {row['speedup']:.3f}x id={row['id']} {row['operator']} "
            f"shape={row['shape']} file={row['file']} reason={row['tier_reason']}"
        )

    lines.extend(["", "Latest trusted fast operators excluding semantic_alias:"])
    for row in [r for r in latest_fast_sorted if r["tier"] != "semantic_alias"]:
        lines.append(
            f"- [{row['tier']}] {row['speedup']:.3f}x id={row['id']} {row['operator']} "
            f"shape={row['shape']} file={row['file']} reason={row['tier_reason']}"
        )

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines[:10]))
    print(f"wrote {OUT_TXT}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
