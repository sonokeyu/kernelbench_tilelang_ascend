#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path


RESULTS = Path("/data/chenkeyu/tilelang_ref/benchmarks/results")
OUT = RESULTS / "tilelang_overall_fast_slow_dedup_stats.txt"


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def level_of(path: str) -> str:
    base = os.path.basename(path)
    if base.startswith("l1_"):
        return "L1"
    if base.startswith("l2_"):
        return "L2"
    return "unknown"


paths = glob.glob(str(RESULTS / "**" / "*.csv"), recursive=True)
comparisons = []
tile_only = []
failed = []

for path in paths:
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                passed = str(row.get("tilelang_passed", row.get("passed", ""))).lower()
                if passed in ("false", "0", "fail"):
                    row["_path"] = path
                    failed.append(row)
                    continue

                tile_value = as_float(
                    row.get("tilelang_mean_ms")
                    or row.get("tile_mean_ms")
                    or row.get("tilelang_ms")
                )
                torch_value = as_float(
                    row.get("torch_mean_ms")
                    or row.get("torch_ms")
                    or row.get("torch_mean")
                )

                if tile_value is not None:
                    row["_path"] = path
                    if torch_value is None:
                        tile_only.append(row)
                        continue
                    if torch_value > 0 and tile_value > 0:
                        row["_torch"] = torch_value
                        row["_tile"] = tile_value
                        row["_speed"] = torch_value / tile_value
                        comparisons.append(row)
                    continue

                # Specialized A/B CSVs store one Torch row followed by one or
                # more TileLang variant rows using mean_ms/speedup_vs_torch.
                variant = row.get("variant", "")
                mean_value = as_float(row.get("mean_ms"))
                speed = as_float(row.get("speedup_vs_torch"))
                if variant and variant != "torch" and mean_value and speed and speed > 0:
                    row["_path"] = path
                    row["_tile"] = mean_value
                    row["_speed"] = speed
                    row["_torch"] = mean_value * speed
                    comparisons.append(row)
    except (OSError, csv.Error):
        continue

fast = [row for row in comparisons if row["_speed"] > 1.0]
slow = [row for row in comparisons if row["_speed"] <= 1.0]

best_by_op = {}
for row in comparisons:
    key = (level_of(row["_path"]), row.get("id"))
    previous = best_by_op.get(key)
    if previous is None or row["_speed"] > previous["_speed"]:
        best_by_op[key] = row

best_fast = [row for row in best_by_op.values() if row["_speed"] > 1.0]
best_slow = [row for row in best_by_op.values() if row["_speed"] <= 1.0]

lines = [
    f"csv_files {len(paths)}",
    f"comparison_rows {len(comparisons) + len(tile_only) + len(failed)}",
    f"usable_torch_vs_tile {len(comparisons)}",
    f"tile_only {len(tile_only)}",
    f"failed {len(failed)}",
    f"fast {len(fast)}",
    f"slow_or_equal {len(slow)}",
    "fast_rows",
]
for row in sorted(fast, key=lambda item: item["_speed"], reverse=True):
    lines.append(
        "{} {} torch={:.6f} tile={:.6f} speedup={:.3f} {}".format(
            row.get("id"),
            row.get("operator") or row.get("op") or row.get("name"),
            row["_torch"],
            row["_tile"],
            row["_speed"],
            os.path.basename(row["_path"]),
        )
    )

lines.extend(
    [
        f"unique_torch_vs_tile_ops {len(best_by_op)}",
        f"unique_fast_ops {len(best_fast)}",
        f"unique_slow_or_equal_ops {len(best_slow)}",
    ]
)
for level in ("L1", "L2", "unknown"):
    part = [row for key, row in best_by_op.items() if key[0] == level]
    part_fast = [row for row in part if row["_speed"] > 1.0]
    lines.append(
        f"{level}_unique_torch_vs_tile_ops {len(part)} "
        f"fast {len(part_fast)} slow_or_equal {len(part) - len(part_fast)}"
    )

lines.append("unique_fast_best_rows")
for row in sorted(best_fast, key=lambda item: (level_of(item["_path"]), -item["_speed"])):
    lines.append(
        "{} {} {} torch={:.6f} tile={:.6f} speedup={:.3f} {}".format(
            level_of(row["_path"]),
            row.get("id"),
            row.get("operator") or row.get("op") or row.get("name"),
            row["_torch"],
            row["_tile"],
            row["_speed"],
            os.path.basename(row["_path"]),
        )
    )

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

