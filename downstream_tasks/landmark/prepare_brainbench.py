"""Build the consolidated BrainBenchmark landmark manifest.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from downstream_tasks.landmark import landmark_schema as schema


# Filename pattern: "{subject}_registered_{structure}.csv"
# Subject IDs look like "10", "11.2", "12.1", ...
_CSV_RE = re.compile(r"^(?P<subject>.+?)_registered_(?P<structure>[a-z]+)\.csv$")


# ---------- structure-level colours for the overlay grid -------------------

_STRUCTURE_COLOURS = {
    "skull":      (255,   0,   0),   # red
    "thalami":    (  0, 255,   0),   # green
    "cerebellum": (  0, 100, 255),   # blue
    "cavum":      (255, 200,   0),   # yellow
    "sylvius":    (255,   0, 255),   # magenta
    "midline":    (  0, 255, 255),   # cyan
}


# ---------- IO --------------------------------------------------------------

def _read_csv_points(path: str):
    """Read a 2-column CSV with floats, no header. Returns list of (x, y) floats."""
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(f"{path}: row {line!r} has {len(parts)} cols, expected 2")
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _group_subjects(landmark_dir: str):
    """Walk landmark_dir and group CSV files by subject ID.

    Returns: dict[subject_id] -> dict[structure] -> path_to_csv
    """
    groups = defaultdict(dict)
    for fname in sorted(os.listdir(landmark_dir)):
        if not fname.endswith(".csv"):
            continue
        m = _CSV_RE.match(fname)
        if not m:
            print(f"warning: skipping unparseable CSV name {fname!r}", file=sys.stderr)
            continue
        subj = m.group("subject")
        struct = m.group("structure")
        groups[subj][struct] = os.path.join(landmark_dir, fname)
    return dict(groups)


def _resolve_image_path(images_dir: str, subject: str):
    """Return the path to the registered image for ``subject`` if it exists."""
    candidate = os.path.join(images_dir, f"{subject}_registered.jpeg")
    return candidate if os.path.isfile(candidate) else None


# ---------- validation gates -----------------------------------------------

def _validate_subject(subject: str, structure_files: dict, image_path: str | None):
    """Raise ValueError on any anomaly for one subject."""
    if image_path is None:
        raise ValueError(f"[{subject}] no matching image at images_registered/")

    missing = set(schema.STRUCTURE_ORDER) - set(structure_files)
    extra   = set(structure_files) - set(schema.STRUCTURE_ORDER)
    if missing:
        raise ValueError(f"[{subject}] missing structures: {sorted(missing)}")
    if extra:
        raise ValueError(f"[{subject}] unexpected structures: {sorted(extra)}")


def _validate_points(subject: str, structure: str, pts: list, expected: int):
    if len(pts) != expected:
        raise ValueError(
            f"[{subject}/{structure}] expected {expected} points, found {len(pts)}"
        )


# ---------- splitting -------------------------------------------------------

def _split_subjects(subjects: list, n_test: int, n_val: int, seed=None):
    """Split subjects into train/val/test.

    When ``seed`` is ``None``, uses deterministic sorted-ID order (legacy
    behaviour from v1-v3). With ``seed`` set, uses a seeded random shuffle.
    The sorted-ID order is adversarial — it puts low-id subjects (mostly 1-33)
    into the test set, which is on the harder end of 1000 random splits even
    for a trivial template predictor. Multi-seed experiments should always
    pass an explicit seed.

    Sorting key for the no-seed path: split subject 'X.Y' into (X_int, Y_int)
    so "2.1" < "10".
    """
    if seed is None:
        def key(s):
            parts = s.split(".")
            try:
                return tuple(int(p) for p in parts)
            except ValueError:
                return (10**9, s)  # push odd IDs to the end
        ordered = sorted(subjects, key=key)
    else:
        rng = random.Random(seed)
        ordered = list(subjects)
        rng.shuffle(ordered)

    test  = ordered[:n_test]
    val   = ordered[n_test : n_test + n_val]
    train = ordered[n_test + n_val :]
    return train, val, test


# ---------- overlay debug grid ---------------------------------------------

def _draw_overlay(image_path: str, landmarks_24: list, radius: int = 5):
    """Open image, draw a colour-coded dot per landmark, return RGB PIL image."""
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    idx = 0
    for structure in schema.STRUCTURE_ORDER:
        c = _STRUCTURE_COLOURS[structure]
        count = schema.STRUCTURE_COUNTS[structure]
        for _ in range(count):
            x, y = landmarks_24[idx]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         outline=c, width=2)
            idx += 1
    return im


def _build_overlay_grid(records: list, images_dir: str, out_path: str,
                        n_samples: int = 8, cols: int = 4):
    if not records:
        return
    chosen = records[:n_samples] if len(records) >= n_samples else records
    overlays = [
        _draw_overlay(os.path.join(images_dir, os.path.basename(r["image"])),
                      r["landmarks"])
        for r in chosen
    ]
    rows = (len(overlays) + cols - 1) // cols
    w, h = overlays[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for i, im in enumerate(overlays):
        gx, gy = (i % cols) * w, (i // cols) * h
        grid.paste(im, (gx, gy))
    grid.save(out_path)


# ---------- main ------------------------------------------------------------

def build_manifest(
    data_root: str,
    out_path: str,
    n_test: int = 24,
    n_val: int = 15,
    overlay_path: str | None = None,
    overlay_n: int = 8,
    strict_24: bool = True,
    split_seed=None,
):
    landmark_dir = os.path.join(data_root, "landmarks", "reg_landmarks")
    images_dir   = os.path.join(data_root, "images_registered")
    if not os.path.isdir(landmark_dir):
        raise FileNotFoundError(f"missing dir: {landmark_dir}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"missing dir: {images_dir}")

    print(f"landmark_dir = {landmark_dir}")
    print(f"images_dir   = {images_dir}")
    print(f"strict_24    = {strict_24}")
    print()

    groups = _group_subjects(landmark_dir)
    print(f"discovered {len(groups)} subjects with at least one CSV")

    records = []
    dropped = []        # (subject, reason) — non-fatal in strict_24 mode
    failures = []       # (subject, reason) — always fatal (missing image / structures / bad coords)

    for subject in sorted(groups):
        structure_files = groups[subject]
        image_path = _resolve_image_path(images_dir, subject)
        try:
            _validate_subject(subject, structure_files, image_path)

            with Image.open(image_path) as im:
                W, H = im.size

            all_points = []
            non_conforming = []
            for structure in schema.STRUCTURE_ORDER:
                pts = _read_csv_points(structure_files[structure])
                expected = schema.STRUCTURE_COUNTS[structure]
                if len(pts) != expected:
                    non_conforming.append(
                        f"{structure}={len(pts)} (expected {expected})"
                    )
                all_points.extend(pts)

            if non_conforming:
                # Real BrainBenchmark variation: ~10% of subjects have an extra
                # perimeter / anatomical point or a missing one. The paper's
                # "24 landmarks" assumes the conforming subset.
                if strict_24:
                    dropped.append((subject, "; ".join(non_conforming)))
                    continue
                else:
                    # Loose mode: truncate too-long, pad too-short with NaN.
                    # NOT recommended — heatmap loss can't handle NaN targets.
                    raise ValueError(
                        f"loose mode not implemented; use --strict_24 True. "
                        f"non-conforming: {non_conforming}"
                    )

            if len(all_points) != 24:
                raise ValueError(
                    f"expected 24 points after concat, got {len(all_points)}"
                )

            arr = np.array(all_points, dtype=np.float64)
            if (arr[:, 0] < 0).any() or (arr[:, 0] >= W).any():
                raise ValueError(
                    f"x outside [0, {W}); min={arr[:, 0].min():.1f}, "
                    f"max={arr[:, 0].max():.1f}"
                )
            if (arr[:, 1] < 0).any() or (arr[:, 1] >= H).any():
                raise ValueError(
                    f"y outside [0, {H}); min={arr[:, 1].min():.1f}, "
                    f"max={arr[:, 1].max():.1f}"
                )

            records.append({
                "subject":   subject,
                "image":     f"images_registered/{subject}_registered.jpeg",
                "image_hw":  [H, W],
                "landmarks": [[float(x), float(y)] for x, y in all_points],
            })
        except Exception as e:
            failures.append((subject, str(e)))

    if failures:
        print(f"\nFAILED ({len(failures)}):", file=sys.stderr)
        for subj, msg in failures:
            print(f"  {subj}: {msg}", file=sys.stderr)
        raise SystemExit(2)

    if dropped:
        print(f"\nDROPPED (non-conforming, strict_24={strict_24}): {len(dropped)} subjects")
        for subj, reason in dropped:
            print(f"  {subj}: {reason}")
        print()

    n_total = len(records)
    print(f"validated {n_total} subjects, 0 missing images, 0 missing structures")
    print(f"24 landmarks per subject "
          f"(skull=4 thalami=3 cerebellum=8 cavum=4 sylvius=3 midline=2)")

    if n_total < n_test + n_val:
        raise SystemExit(
            f"need at least {n_test + n_val} subjects for the requested test+val "
            f"split sizes, only have {n_total}"
        )

    subjects = [r["subject"] for r in records]
    train, val, test = _split_subjects(subjects, n_test=n_test, n_val=n_val,
                                       seed=split_seed)
    print(f"split_seed = {split_seed!r}  ({'seeded shuffle' if split_seed is not None else 'sorted-ID (legacy)'})")
    split_map = {**{s: "train" for s in train},
                 **{s: "val"   for s in val},
                 **{s: "test"  for s in test}}
    for r in records:
        r["split"] = split_map[r["subject"]]

    print(f"split: train={len(train)} val={len(val)} test={len(test)}  "
          f"(total={n_total})")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {out_path}")

    if overlay_path is not None:
        _build_overlay_grid(records, images_dir, overlay_path, n_samples=overlay_n)
        print(f"overlay debug grid -> {overlay_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="BrainBenchmark root (contains images_registered/ and landmarks/reg_landmarks/)")
    p.add_argument("--out", required=True,
                   help="path to landmark_manifest.json (NOT manifest.json)")
    p.add_argument("--n_test", type=int, default=24)
    p.add_argument("--n_val",  type=int, default=15)
    p.add_argument("--overlay_path", default=None,
                   help="optional path to landmark_debug_overlay.png; if unset, "
                        "writes next to --out")
    p.add_argument("--overlay_n", type=int, default=8)
    p.add_argument("--strict_24", action="store_true", default=True,
                   help="(default True) drop subjects whose per-structure point counts deviate from the canonical 24-landmark schema")
    p.add_argument("--no-strict_24", dest="strict_24", action="store_false")
    p.add_argument("--split_seed", type=int, default=None,
                   help="Seed for random train/val/test split. When omitted, "
                        "uses the legacy sorted-ID deterministic order (v1-v3 "
                        "behaviour). Multi-seed experiments should always pass "
                        "an explicit value; the sorted-ID split is adversarial.")
    args = p.parse_args()

    overlay_path = args.overlay_path
    if overlay_path is None:
        overlay_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                    "landmark_debug_overlay.png")

    if os.path.basename(args.out) == "manifest.json":
        raise SystemExit(
            "refuse: --out is 'manifest.json', which collides with the "
            "downloader's metadata file. Use 'landmark_manifest.json'."
        )

    build_manifest(
        data_root=args.data_root,
        out_path=args.out,
        n_test=args.n_test,
        n_val=args.n_val,
        overlay_path=overlay_path,
        overlay_n=args.overlay_n,
        strict_24=args.strict_24,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
