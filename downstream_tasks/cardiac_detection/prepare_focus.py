"""One-shot FOCUS dataset preparation for the cardiac-detection task.
"""

import argparse
import json
import os
import shutil
import struct
import sys
from collections import Counter
from pathlib import Path


# ---------- PNG size reader (no PIL dep) ------------------------------------

def _png_dimensions(path):
    """Return (width, height) for a PNG by reading the IHDR chunk.

    Avoids a Pillow dependency for the prepare step (the conda env that runs
    prepare may not have Pillow even though the training env does).
    """
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"not a PNG: {path}")
        f.read(4)               # IHDR length
        if f.read(4) != b"IHDR":
            raise ValueError(f"first chunk not IHDR in {path}")
        w, h = struct.unpack(">II", f.read(8))
        return int(w), int(h)


# ---------- annotation parsers ----------------------------------------------

def _parse_ellipses(path):
    """Read FOCUS ``annfiles_ellipse/<stem>.txt`` -> list of dicts.

    Line format: ``cx cy a b theta_deg class`` (whitespace-separated).
    """
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 6:
                raise ValueError(f"malformed ellipse line in {path}: {line!r}")
            cx, cy, a, b, theta_deg = (float(x) for x in parts[:5])
            label = parts[5]
            out.append({
                "class":     label,
                "cx":        cx,
                "cy":        cy,
                "a":         a,
                "b":         b,
                "theta_deg": theta_deg,
            })
    return out


def _parse_rectangles(path):
    """Read FOCUS ``annfiles_rectangle/<stem>.txt`` -> list of dicts.

    DOTA-style: ``x1 y1 x2 y2 x3 y3 x4 y4 class difficulty``.
    """
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 10:
                raise ValueError(f"malformed rectangle line in {path}: {line!r}")
            coords = [float(x) for x in parts[:8]]
            corners = [[coords[2*i], coords[2*i+1]] for i in range(4)]
            out.append({
                "class":      parts[8],
                "corners":    corners,
                "difficulty": int(parts[9]),
            })
    return out


# ---------- main prepare loop -----------------------------------------------

def _collect_split(split_dir, source_tag, dst_images, dst_annfiles, name_prefix):
    """Copy one FOCUS split (training / validation / testing) into the
    prepared layout and return per-image meta records.

    ``name_prefix`` lets us avoid filename collisions when training/+validation/
    are merged into a single trainval/ directory (FOCUS uses 001.png in both).
    """
    src_images = split_dir / "images"
    src_ann_e  = split_dir / "annfiles_ellipse"
    src_ann_r  = split_dir / "annfiles_rectangle"
    if not src_images.is_dir():
        raise FileNotFoundError(f"missing {src_images}")
    image_files = sorted(p for p in src_images.iterdir()
                         if p.suffix.lower() == ".png")
    records = {}
    for img_path in image_files:
        stem = img_path.stem
        new_stem = f"{name_prefix}_{stem}" if name_prefix else stem

        ell_path = src_ann_e / f"{stem}.txt"
        rec_path = src_ann_r / f"{stem}.txt"
        if not ell_path.is_file():
            raise FileNotFoundError(f"missing ellipse ann: {ell_path}")
        if not rec_path.is_file():
            raise FileNotFoundError(f"missing rectangle ann: {rec_path}")

        ellipses   = _parse_ellipses(ell_path)
        rectangles = _parse_rectangles(rec_path)
        w, h       = _png_dimensions(img_path)

        # Copy image + annotation under the new name.
        shutil.copy2(img_path, dst_images / f"{new_stem}.png")
        # Mirror the source annotation file so mmrotate's DOTADataset can read it.
        shutil.copy2(rec_path, dst_annfiles / f"{new_stem}.txt")

        records[new_stem] = {
            "file":       f"{new_stem}.png",
            "orig_w":     w,
            "orig_h":     h,
            "source":     source_tag,
            "ellipses":   ellipses,
            "rectangles": rectangles,
        }
    return records


def _print_summary(meta, label):
    sizes = Counter((m["orig_w"], m["orig_h"]) for m in meta.values())
    class_counts = Counter()
    for m in meta.values():
        for r in m["rectangles"]:
            class_counts[r["class"]] += 1
    print(f"---- {label}: {len(meta)} images ----")
    for cls, n in sorted(class_counts.items()):
        print(f"  {cls:<10s} boxes: {n}")
    print(f"  dimension histogram (W x H):")
    for (w, h), n in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>4d}  {w} x {h}")


def main():
    p = argparse.ArgumentParser("Prepare FOCUS dataset for cardiac detection")
    p.add_argument(
        "--data_root", required=True,
        help="path to the FOCUS-dataset directory containing "
             "training/ validation/ testing/",
    )
    p.add_argument(
        "--prepared_dir", required=True,
        help="output directory (will be created if missing; will be skipped "
             "if a .done marker already exists)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="ignore an existing .done marker and re-prepare from scratch",
    )
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    prepared  = Path(args.prepared_dir).resolve()
    done_marker = prepared / ".done"

    if done_marker.is_file() and not args.force:
        print(f"prepare_focus: {done_marker} already exists -> skip "
              f"(re-run with --force to rebuild)")
        return 0

    tmp = prepared.with_name(prepared.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "trainval" / "images").mkdir(parents=True)
    (tmp / "trainval" / "annfiles").mkdir(parents=True)
    (tmp / "test"     / "images").mkdir(parents=True)
    (tmp / "test"     / "annfiles").mkdir(parents=True)

    train_recs = _collect_split(
        data_root / "training", "training",
        tmp / "trainval" / "images", tmp / "trainval" / "annfiles",
        name_prefix="train",
    )
    val_recs = _collect_split(
        data_root / "validation", "validation",
        tmp / "trainval" / "images", tmp / "trainval" / "annfiles",
        name_prefix="val",
    )
    test_recs = _collect_split(
        data_root / "testing", "testing",
        tmp / "test" / "images", tmp / "test" / "annfiles",
        name_prefix="",
    )
    trainval_meta = {**train_recs, **val_recs}
    test_meta     = test_recs

    with open(tmp / "trainval" / "meta.json", "w") as f:
        json.dump(trainval_meta, f, indent=2)
    with open(tmp / "test" / "meta.json", "w") as f:
        json.dump(test_meta, f, indent=2)

    _print_summary(trainval_meta, "trainval (training + validation)")
    _print_summary(test_meta,     "test (testing)")

    # Sanity gates per the FOCUS paper: 250 trainval images / 50 test.
    if len(trainval_meta) != 250:
        print(f"WARNING: trainval has {len(trainval_meta)} images "
              f"(expected 250 per paper)", file=sys.stderr)
    if len(test_meta) != 50:
        print(f"WARNING: test has {len(test_meta)} images "
              f"(expected 50 per paper)", file=sys.stderr)

    # Atomic flip: remove any stale prepared/ then rename tmp -> prepared.
    if prepared.exists():
        shutil.rmtree(prepared)
    os.rename(tmp, prepared)
    # Marker written last so partial runs do not look complete.
    done_marker.touch()
    print(f"prepare_focus: wrote {prepared} (marker {done_marker})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
