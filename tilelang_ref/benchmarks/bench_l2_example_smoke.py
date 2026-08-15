#!/usr/bin/env python3
"""Run TileLang L2 example __main__ smoke tests in isolated subprocesses."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path


if Path("/data/chenkeyu/tilelang_ref/examples/elementwise").exists():
    ROOT = Path("/data/chenkeyu")
    EXAMPLES = ROOT / "tilelang_ref" / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "tilelang_ref" / "benchmarks" / "results" / "l2_example_smoke.csv"
else:
    ROOT = Path("/workspace/tilelang-ascend")
    EXAMPLES = ROOT / "examples" / "elementwise"
    DEFAULT_OUT = ROOT / "benchmarks" / "results" / "l2_example_smoke.csv"


def discover_examples() -> dict[int, Path]:
    examples: dict[int, Path] = {}
    for path in EXAMPLES.glob("example_level2_*.py"):
        match = re.match(r"example_level2_(\d+)_", path.name)
        if match:
            examples[int(match.group(1))] = path
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    examples = discover_examples()
    if args.ids.strip():
        wanted = [int(part) for part in args.ids.split(",") if part.strip()]
    else:
        wanted = sorted(examples)

    rows = []
    for kid in wanted:
        path = examples.get(kid)
        row = {
            "id": kid,
            "example_file": str(path) if path else "",
            "status": "MISSING",
            "elapsed_s": 0.0,
            "returncode": "",
            "last_stdout": "",
            "last_stderr": "",
        }
        if path is None:
            rows.append(row)
            print(f"ID {kid}: MISSING", flush=True)
            continue

        print(f"RUN id={kid} file={path.name}", flush=True)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                ["python3", "-u", str(path)],
                cwd=str(ROOT if (ROOT / "tilelang_ref").exists() else ROOT),
                text=True,
                capture_output=True,
                timeout=args.timeout,
            )
            elapsed = time.perf_counter() - t0
            row["elapsed_s"] = elapsed
            row["returncode"] = proc.returncode
            row["last_stdout"] = "\\n".join(proc.stdout.strip().splitlines()[-5:])
            row["last_stderr"] = "\\n".join(proc.stderr.strip().splitlines()[-8:])
            row["status"] = "PASS" if proc.returncode == 0 else "FAIL"
            print(f"  {row['status']} elapsed={elapsed:.1f}s rc={proc.returncode}", flush=True)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - t0
            row["elapsed_s"] = elapsed
            row["status"] = "TIMEOUT"
            row["returncode"] = "timeout"
            row["last_stdout"] = "\\n".join((exc.stdout or "").strip().splitlines()[-5:])
            row["last_stderr"] = "\\n".join((exc.stderr or "").strip().splitlines()[-8:])
            print(f"  TIMEOUT elapsed={elapsed:.1f}s", flush=True)
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
