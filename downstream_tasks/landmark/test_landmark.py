"""Unit tests for landmark transforms and metrics.

Run with:
    pytest downstream_tasks/landmark/test_landmark.py -q

These tests do NOT need the BrainBenchmark data or a GPU — they validate the
coord-aware augmentation invariants and heatmap round-trip with synthetic
data only.
"""

import torch
from PIL import Image

from downstream_tasks.landmark import landmark_schema as schema
from downstream_tasks.landmark.transforms_landmark import (
    coords_to_heatmaps,
    hflip_with_coords,
    resize_with_coords,
    vflip_with_coords,
)
from downstream_tasks.landmark.metrics_landmark import soft_argmax_2d


# ---------- schema sanity ----------------------------------------------------

def test_landmark_order_size():
    assert len(schema.LANDMARK_ORDER) == 24


def test_perms_are_valid_permutations():
    assert sorted(schema.HFLIP_PERM) == list(range(24))
    assert sorted(schema.VFLIP_PERM) == list(range(24))


def test_perms_are_involutive():
    """Applying perm twice must yield the identity, so flip(flip(x)) == x."""
    for name, perm in [("HFLIP_PERM", schema.HFLIP_PERM), ("VFLIP_PERM", schema.VFLIP_PERM)]:
        twice = [perm[perm[i]] for i in range(24)]
        assert twice == list(range(24)), f"{name} is not involutive: {twice}"


# ---------- coord-aware ops --------------------------------------------------

def _make_dummy_pil(W=320, H=240):
    return Image.new("RGB", (W, H), color=0)


def test_resize_with_coords_scales_correctly():
    img = _make_dummy_pil(W=320, H=240)
    coords = torch.tensor([[160.0, 120.0], [0.0, 0.0], [319.0, 239.0]])
    img_r, coords_r, scale = resize_with_coords(img, coords, img_size=224)
    assert img_r.size == (224, 224)
    # sx = 224/320 = 0.7, sy = 224/240 ≈ 0.933
    sx, sy = 224 / 320, 224 / 240
    expected = coords * torch.tensor([sx, sy])
    assert torch.allclose(coords_r, expected, atol=1e-5)
    assert torch.allclose(scale, torch.tensor([sx, sy]))


def test_hflip_involutive_with_identity_perm():
    img = _make_dummy_pil(W=224, H=224)
    coords = torch.tensor([[10.0, 50.0], [200.0, 100.0]])
    perm = [0, 1]
    img1, c1 = hflip_with_coords(img, coords, img_size=224, channel_perm=perm)
    img2, c2 = hflip_with_coords(img1, c1, img_size=224, channel_perm=perm)
    assert torch.allclose(c2, coords)


def test_vflip_involutive_with_identity_perm():
    img = _make_dummy_pil(W=224, H=224)
    coords = torch.tensor([[10.0, 50.0], [200.0, 100.0]])
    perm = [0, 1]
    img1, c1 = vflip_with_coords(img, coords, img_size=224, channel_perm=perm)
    img2, c2 = vflip_with_coords(img1, c1, img_size=224, channel_perm=perm)
    assert torch.allclose(c2, coords)


def test_hflip_then_vflip_equals_180_rotation():
    img = _make_dummy_pil(W=224, H=224)
    coords = torch.tensor([[10.0, 50.0], [200.0, 100.0]])
    perm = [0, 1]
    _, c1 = hflip_with_coords(img, coords, img_size=224, channel_perm=perm)
    _, c2 = vflip_with_coords(img, c1, img_size=224, channel_perm=perm)
    expected = torch.tensor([
        [223.0 - 10.0, 223.0 - 50.0],
        [223.0 - 200.0, 223.0 - 100.0],
    ])
    assert torch.allclose(c2, expected)


def test_full_schema_perms_involutive():
    """Schema HFLIP_PERM and VFLIP_PERM must round-trip the *full* 24-channel
    coord stack, not just two channels."""
    img = _make_dummy_pil(W=224, H=224)
    coords = torch.rand(24, 2) * 223.0
    img_h, c_h = hflip_with_coords(img, coords, 224, schema.HFLIP_PERM)
    _,    c_hh = hflip_with_coords(img_h, c_h, 224, schema.HFLIP_PERM)
    assert torch.allclose(c_hh, coords, atol=1e-4)

    img_v, c_v = vflip_with_coords(img, coords, 224, schema.VFLIP_PERM)
    _,    c_vv = vflip_with_coords(img_v, c_v, 224, schema.VFLIP_PERM)
    assert torch.allclose(c_vv, coords, atol=1e-4)


# ---------- heatmap round-trip ----------------------------------------------

def test_coords_to_heatmaps_shape_and_peak():
    coords = torch.tensor([[10.0, 50.0], [200.0, 100.0], [112.0, 112.0]])
    H = 224
    sigma = 2.0
    heatmaps = coords_to_heatmaps(coords, img_size=H, sigma=sigma)
    assert heatmaps.shape == (3, H, H)
    # Peak should be ~1.0 at integer coord locations
    for k in range(3):
        x, y = coords[k]
        assert heatmaps[k, int(y), int(x)] > 0.9


def test_soft_argmax_recovers_input_coords():
    coords = torch.tensor([[10.0, 50.0], [200.0, 100.0], [112.0, 112.0]])
    H = 224
    sigma = 2.0
    heatmaps = coords_to_heatmaps(coords, img_size=H, sigma=sigma).unsqueeze(0)  # [1, 3, H, H]
    decoded = soft_argmax_2d(heatmaps, beta=100.0).squeeze(0)  # [3, 2]
    # Sub-pixel accuracy from soft-argmax with beta=100 should be within 1px
    assert torch.allclose(decoded, coords, atol=1.0)
