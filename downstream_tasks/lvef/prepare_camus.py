"""Build the per-seed CAMUS LVEF manifest.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict


def _load_annotation(annotation_path: str):
    """Load annotation_CAMUS_v2.json and group entries by (patient, view).

    The JSON is flat: one entry per frame, keyed by PNG filename. We collapse
    it to one entry per (patient, view) since the LVEF / ED / ES fields are
    invariant across frames of the same view.
    """
    with open(annotation_path) as f:
        ann = json.load(f)

    by_pv = {}
    for fname, row in ann.items():
        # CAMUS__{patient}__{view}__{frame}.png
        parts = fname.split("__")
        if len(parts) != 4 or not parts[3].endswith(".png"):
            print(f"warning: skipping unparseable annotation key {fname!r}", file=sys.stderr)
            continue
        patient, view = parts[1], parts[2]
        key = (patient, view)
        # Keep one row per (patient, view); they all agree on ED / ES / EF.
        if key not in by_pv:
            by_pv[key] = row
    return by_pv


def _resolve_frame(images_root: str, patient: str, view: str, frame_1idx: int) -> str:
    """Return the relative path 'images/CAMUS__{patient}__{view}__{frame:05d}.png'.

    ``frame_1idx`` is the 1-indexed value from End_diastole / End_systole in
    the annotation JSON. The PNGs are 0-indexed (Slice_ID matches the {frame}
    suffix), so we subtract 1.

    Raises FileNotFoundError if the file does not exist on disk.
    """
    frame_0idx = frame_1idx - 1
    rel = os.path.join("images", f"CAMUS__{patient}__{view}__{frame_0idx:05d}.png")
    abs_path = os.path.join(images_root, rel)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(abs_path)
    return rel


def _split_patients(patients, n_train: int, n_val: int, n_test: int, seed: int):
    """Patient-level split. Deterministic given the seed.

    Returns dict[patient] -> 'train' | 'val' | 'test'.
    """
    if n_train + n_val + n_test != len(patients):
        raise ValueError(
            f"split sizes {n_train}+{n_val}+{n_test}={n_train+n_val+n_test} "
            f"do not sum to len(patients)={len(patients)}"
        )
    rng = random.Random(seed)
    shuffled = list(patients)
    rng.shuffle(shuffled)
    out = {}
    for p in shuffled[:n_train]:
        out[p] = "train"
    for p in shuffled[n_train:n_train + n_val]:
        out[p] = "val"
    for p in shuffled[n_train + n_val:]:
        out[p] = "test"
    return out


def _ef_stats(records, split):
    efs = [r["ef"] for r in records if r["split"] == split]
    if not efs:
        return None
    n = len(efs)
    mean = sum(efs) / n
    var = sum((e - mean) ** 2 for e in efs) / n
    return {"n": n, "min": min(efs), "max": max(efs), "mean": mean, "std": var ** 0.5}


def build_manifest(args):
    annotation_path = os.path.join(args.data_root, "annotation_CAMUS_v2.json")
    by_pv = _load_annotation(annotation_path)

    # Patient inventory: must have both CH2 and CH4.
    by_patient = defaultdict(dict)
    for (patient, view), row in by_pv.items():
        by_patient[patient][view] = row
    patients = sorted(p for p, views in by_patient.items()
                      if "CH2" in views and "CH4" in views)
    missing = sorted(p for p, views in by_patient.items()
                     if not ("CH2" in views and "CH4" in views))
    if missing:
        print(f"warning: skipping {len(missing)} patient(s) without both views: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", file=sys.stderr)

    splits = _split_patients(
        patients,
        n_train=args.train, n_val=args.val, n_test=args.test,
        seed=args.split_seed,
    )

    records = []
    skipped = 0
    for patient in patients:
        ch2 = by_patient[patient]["CH2"]
        ch4 = by_patient[patient]["CH4"]
        # CH2 and CH4 may disagree on EF in theory; in CAMUS_v2 they're
        # identical. Take the CH4 value (clinical default for LVEF).
        ef = float(ch4["Left_ventricular_ejection_fraction"])

        try:
            ch2_ed = _resolve_frame(args.data_root, patient, "CH2", ch2["End_diastole"])
            ch2_es = _resolve_frame(args.data_root, patient, "CH2", ch2["End_systole"])
            ch4_ed = _resolve_frame(args.data_root, patient, "CH4", ch4["End_diastole"])
            ch4_es = _resolve_frame(args.data_root, patient, "CH4", ch4["End_systole"])
        except FileNotFoundError as e:
            print(f"warning: skipping patient {patient}; missing file: {e}", file=sys.stderr)
            skipped += 1
            continue

        records.append({
            "patient": patient,
            "ch2_ed": ch2_ed,
            "ch2_es": ch2_es,
            "ch4_ed": ch4_ed,
            "ch4_es": ch4_es,
            "ef": ef,
            "split": splits[patient],
        })

    # Sanity: no patient appears in two splits.
    seen = {}
    for r in records:
        if r["patient"] in seen and seen[r["patient"]] != r["split"]:
            raise RuntimeError(f"patient {r['patient']} in two splits: "
                               f"{seen[r['patient']]} and {r['split']}")
        seen[r["patient"]] = r["split"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)

    print(f"wrote {len(records)} records ({skipped} skipped) to {args.out}")
    for split in ("train", "val", "test"):
        s = _ef_stats(records, split)
        if s is None:
            print(f"  {split:<5s}  (no records)")
        else:
            print(f"  {split:<5s}  n={s['n']:<4d}  "
                  f"ef: min={s['min']:.1f}  max={s['max']:.1f}  "
                  f"mean={s['mean']:.2f}  std={s['std']:.2f}")


def build_argparser():
    p = argparse.ArgumentParser(
        "Build per-seed CAMUS LVEF manifest "
        "(one record per patient, 4 frames: CH2 ED/ES + CH4 ED/ES)"
    )
    p.add_argument("--data_root", required=True, type=str,
                   help="prepared CAMUS_2 directory (contains images/ and annotation_CAMUS_v2.json)")
    p.add_argument("--out", required=True, type=str,
                   help="output manifest path, e.g. camus_lvef_manifest_seed0.json")
    p.add_argument("--split_seed", default=0, type=int)
    p.add_argument("--train", default=400, type=int)
    p.add_argument("--val",   default=50,  type=int)
    p.add_argument("--test",  default=50,  type=int)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    build_manifest(args)
