"""Canonical 24-landmark schema for the BrainBenchmark task.
"""

# Structure names match the on-disk CSV filename suffixes in
#   OpenUS_datasets/brain_benchmark/landmarks/reg_landmarks/{subject}_registered_{structure}.csv
# Counts verified by inspecting the CSVs (row count per file).
#
# IMPORTANT: these names differ from the research spec (01_fetal_brain_landmark.md):
#   spec "thalamic" -> on-disk "thalami"
#   spec "cerebellar" -> on-disk "cerebellum"
#   spec "sylvian" -> on-disk "sylvius"
# Always use the on-disk names; the structure-iteration order below is what
# defines the concatenation in landmark_manifest.json.

STRUCTURE_ORDER = ["skull", "thalami", "cerebellum", "cavum", "sylvius", "midline"]
STRUCTURE_COUNTS = {
    "skull":      4,
    "thalami":    3,
    "cerebellum": 8,
    "cavum":      4,
    "sylvius":    3,
    "midline":    2,
}
assert [STRUCTURE_COUNTS[s] for s in STRUCTURE_ORDER] == [4, 3, 8, 4, 3, 2]

# Numeric placeholder names — the CSVs do not name individual points, only
# their (x, y) coordinates in image space ordered around each structure. The
# canonical order below is therefore: concatenate every CSV's rows in the
# order defined by STRUCTURE_ORDER, in their original on-disk row order.
LANDMARK_ORDER = [
    # skull (4)
    "skull_0", "skull_1", "skull_2", "skull_3",
    # thalami (3)
    "thalami_0", "thalami_1", "thalami_2",
    # cerebellum perimeter (8) — ordered around the contour as the CSV writes them
    "cerebellum_0", "cerebellum_1", "cerebellum_2", "cerebellum_3",
    "cerebellum_4", "cerebellum_5", "cerebellum_6", "cerebellum_7",
    # cavum (4)
    "cavum_0", "cavum_1", "cavum_2", "cavum_3",
    # sylvius (3)
    "sylvius_0", "sylvius_1", "sylvius_2",
    # midline (2)
    "midline_0", "midline_1",
]
assert len(LANDMARK_ORDER) == 24, (
    f"LANDMARK_ORDER must have exactly 24 entries, got {len(LANDMARK_ORDER)}"
)
assert sum(STRUCTURE_COUNTS.values()) == 24

# Channel-permutation arrays applied alongside coordinate flips.
#
#   perm[k] = the channel that ends up at position k after the flip.
#
# Identity arrays below mean "do not swap channels" — only the coordinates
# move. This is the SAFE default until we have visually verified which
# structures are L/R symmetric in the registered images and which need
# semantic re-labelling on a horizontal flip.
#
# When updating these, the unit tests in test_landmark.py will verify:
#   - perm is a valid permutation of 0..23
#   - applying perm twice gives the identity (flip is involutive)
HFLIP_PERM = list(range(24))
VFLIP_PERM = list(range(24))
assert len(HFLIP_PERM) == 24 and len(VFLIP_PERM) == 24
assert sorted(HFLIP_PERM) == list(range(24)), "HFLIP_PERM must be a permutation"
assert sorted(VFLIP_PERM) == list(range(24)), "VFLIP_PERM must be a permutation"

# vmamba_small backbone produces 4 feature maps with these channel counts.
VSSM_SMALL_DIMS = (96, 192, 384, 768)
