"""Build the USenhance paired-image manifest.
"""

import argparse
import json
import os
import random
from pathlib import Path

ORGANS = ("breast", "carotid", "kidney", "liver", "thyroid")


def _list_organ_pairs(data_root: str, organ: str):
    organ_dir = os.path.join(data_root, "train_datasets", organ)
    lo_dir = os.path.join(organ_dir, "low_quality")
    hi_dir = os.path.join(organ_dir, "high_quality")
    lo = {f for f in os.listdir(lo_dir) if f.lower().endswith(".png")}
    hi = {f for f in os.listdir(hi_dir) if f.lower().endswith(".png")}
    lo_only = sorted(lo - hi)
    hi_only = sorted(hi - lo)
    if lo_only or hi_only:
        raise SystemExit(
            f"organ={organ!r}: unmatched filenames between low/high\n"
            f"  low-only ({len(lo_only)}): {lo_only[:5]}\n"
            f"  high-only ({len(hi_only)}): {hi_only[:5]}"
        )
    return sorted(lo & hi)


def _stratified_split(filenames, organ, train_frac, rng):
    """Per-organ shuffle + take first train_frac as train, rest as test."""
    items = list(filenames)
    rng.shuffle(items)
    n_train = int(round(len(items) * train_frac))
    train = items[:n_train]
    test = items[n_train:]
    return [(f, organ, "train") for f in train] + [(f, organ, "test") for f in test]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, type=str,
                   help="path to OpenUS_datasets/image_enhancement")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--train_frac", default=0.8, type=float)
    p.add_argument("--out", required=True, type=str,
                   help="path to write the enhance_manifest_seed<seed>.json")
    p.add_argument("--holdout_out", default=None, type=str,
                   help="path to write enhance_holdout.json (defaults to "
                        "alongside --out)")
    args = p.parse_args()

    rng = random.Random(args.seed)

    records = []
    for organ in ORGANS:
        pairs = _list_organ_pairs(args.data_root, organ)
        split = _stratified_split(pairs, organ, args.train_frac, rng)
        records.extend(split)

    out_records = []
    for fname, organ, split in records:
        out_records.append({
            "lq":    f"train_datasets/{organ}/low_quality/{fname}",
            "hq":    f"train_datasets/{organ}/high_quality/{fname}",
            "organ": organ,
            "split": split,
        })

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_records, f, indent=2)

    n_train = sum(r["split"] == "train" for r in out_records)
    n_test  = sum(r["split"] == "test"  for r in out_records)
    print(f"wrote {args.out} ({len(out_records)} records: "
          f"train={n_train}  test={n_test})")
    for organ in ORGANS:
        n_tr = sum(r["split"] == "train" and r["organ"] == organ for r in out_records)
        n_te = sum(r["split"] == "test"  and r["organ"] == organ for r in out_records)
        ratio = n_tr / max(n_tr + n_te, 1)
        print(f"  {organ:8s}  train={n_tr:4d}  test={n_te:4d}  train_frac={ratio:.3f}")

    holdout_dir = os.path.join(args.data_root, "low_quality_images")
    holdout = []
    if os.path.isdir(holdout_dir):
        for fname in sorted(os.listdir(holdout_dir)):
            if fname.lower().endswith(".png"):
                holdout.append({"lq": f"low_quality_images/{fname}"})
        holdout_out = args.holdout_out or os.path.join(
            os.path.dirname(args.out), "enhance_holdout.json"
        )
        with open(holdout_out, "w") as f:
            json.dump(holdout, f, indent=2)
        print(f"wrote {holdout_out} ({len(holdout)} holdout images)")


if __name__ == "__main__":
    main()
