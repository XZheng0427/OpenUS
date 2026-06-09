"""Aggregate cardiac-detection multi-seed run results.
"""

import argparse
import json
import math
import os
import sys
from typing import List

import numpy as np


# Keys match what FocusCTRMetric emits — note the ``focus/`` prefix added
# by mmengine from the metric's ``default_prefix``.
_DEFAULT_METRICS = (
    "focus/AP50_thorax", "focus/AP50_cardiac", "focus/ap50_mean",
    "focus/mAP",
    "focus/ctr_mae_valid", "focus/ctr_missing_rate", "focus/ctr_valid_rate",
    "focus/ctr_acc_0.03", "focus/ctr_acc_0.05", "focus/ctr_acc_0.1",
)


def _read_final_test(log_path):
    if not os.path.isfile(log_path):
        raise FileNotFoundError(log_path)
    found = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("phase") == "final_test" or "final_test" in d:
                found = d
    if found is None:
        raise ValueError(f"no 'final_test' record in {log_path}")
    return found["final_test"]


def _mean_ci95(xs: List[float]):
    arr = np.asarray([x for x in xs if x is not None and not math.isnan(x)],
                     dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    m = float(arr.mean())
    if arr.size < 2:
        return m, float("nan")
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(arr.size)
    # 95% CI half-width using normal approx (n=5 -> use t-table 2.776 if
    # you want stricter; for in-repo dashboards 1.96 is fine and matches
    # the landmark/LVEF aggregators).
    return m, 1.96 * se


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", required=True,
                   help="parent dir containing per-seed run dirs")
    p.add_argument("--pattern", required=True,
                   help="run-dir name template with literal '{seed}' placeholder")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--metrics", nargs="*", default=list(_DEFAULT_METRICS))
    args = p.parse_args()

    rows = []
    missing = []
    for seed in args.seeds:
        run_dir = os.path.join(args.runs_root, args.pattern.format(seed=seed))
        log_path = os.path.join(run_dir, "log.txt")
        try:
            final = _read_final_test(log_path)
        except (FileNotFoundError, ValueError) as e:
            missing.append((seed, str(e)))
            continue
        rows.append({"seed": seed, **final})

    if missing:
        print("WARNING: missing runs:", file=sys.stderr)
        for seed, err in missing:
            print(f"  seed={seed}: {err}", file=sys.stderr)
        print(file=sys.stderr)

    if not rows:
        raise SystemExit("no rows to aggregate")

    # ---------- per-seed table ----------
    print("========== cardiac-detection multi-seed results (per seed) ==========")
    header_cells = ["seed"] + list(args.metrics)
    widths = [6] + [max(14, len(m) + 2) for m in args.metrics]
    print("  " + " ".join(f"{h:>{w}s}" for h, w in zip(header_cells, widths)))
    for r in rows:
        cells = [f"{r['seed']:>{widths[0]}d}"]
        for w, m in zip(widths[1:], args.metrics):
            v = r.get(m)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append(f"{'-':>{w}s}")
            else:
                cells.append(f"{float(v):>{w}.5f}")
        print("  " + " ".join(cells))

    # ---------- aggregate ----------
    print()
    print(f"========== aggregate (mean ± 95% CI across {len(rows)} seeds) ==========")
    for m in args.metrics:
        vals = [r.get(m) for r in rows]
        mean, ci = _mean_ci95(vals)
        if math.isnan(mean):
            print(f"  {m:<28s}    -")
        else:
            ci_str = f"± {ci:.5f}" if not math.isnan(ci) else "(single seed)"
            print(f"  {m:<28s} {mean:.5f}  {ci_str}")


if __name__ == "__main__":
    main()
