# File: test_patches_module.py
# This module tests all patching functions from nightskycam_images.patches.

from __future__ import annotations

from pathlib import Path
from typing import Dict

import imageio.v3 as iio
import numpy as np
import pytest

from nightskycam_images.patches import (
    extract_overlapping_patches,
    load_image_and_extract_patches,
    load_images_from_folder,
    save_patches_from_folder,
)

# --------------------------
# Helpers
# --------------------------


def _make_rgb_uint8(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _make_rgb_uint16(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65535, size=(h, w, 3), dtype=np.uint16)


def _expected_patch_count(
    h: int, w: int, margin: int, patch_size: int, overlap: int
) -> int:
    th, tw = h - 2 * margin, w - 2 * margin
    stride = patch_size - overlap

    def count_axis(length: int) -> int:
        if length - patch_size <= 0:
            return 1
        starts = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if starts[-1] != last:
            starts.append(last)
        return len(starts)

    return count_axis(th) * count_axis(tw)


# --------------------------
# Tests for extract_overlapping_patches
# --------------------------


def test_extract_overlapping_patches_basic_no_overlap() -> None:
    img = _make_rgb_uint8(10, 10)
    margin = 1
    patch_size = 4
    overlap = 0  # stride = 4

    patches = extract_overlapping_patches(img, margin, patch_size, overlap)

    # After trimming margin=1 on all sides: 8x8 region
    # With stride=4 and patch_size=4: starts [0, 4] on both axes => 4 patches
    assert patches.shape == (4, 4, 4, 3)
    assert patches.dtype == np.uint8

    trimmed = img[margin : img.shape[0] - margin, margin : img.shape[1] - margin, :]
    # Top-left patch should match
    np.testing.assert_array_equal(patches[0], trimmed[0:4, 0:4, :])


def test_extract_overlapping_patches_with_overlap_and_end_anchor() -> None:
    # Choose sizes so that stride doesn't perfectly tile the trimmed area
    img = _make_rgb_uint8(15, 13)
    margin = 1  # trimmed: 13 x 11
    patch_size = 5
    overlap = 2  # stride = 3

    patches = extract_overlapping_patches(img, margin, patch_size, overlap)

    # Height: length=13, k=5, stride=3 -> starts [0,3,6,8] (8 is the "end-anchored" start)
    # Width:  length=11, k=5, stride=3 -> starts [0,3,6]
    # Total patches: 4 * 3 = 12
    assert patches.shape == (12, 5, 5, 3)
    assert patches.dtype == np.uint8

    # Verify last start aligns to end on height dimension
    trimmed_h = img.shape[0] - 2 * margin
    last_start_h = trimmed_h - patch_size  # 13 - 5 = 8
    stride = patch_size - overlap
    starts_h = list(range(0, trimmed_h - patch_size + 1, stride))
    if starts_h[-1] != last_start_h:
        starts_h.append(last_start_h)
    assert last_start_h in starts_h


def test_extract_overlapping_patches_invalid_args() -> None:
    img = _make_rgb_uint8(8, 8)

    # overlap >= patch_size should raise
    with pytest.raises(ValueError):
        extract_overlapping_patches(img, margin=0, patch_size=4, overlap=4)

    with pytest.raises(ValueError):
        extract_overlapping_patches(img, margin=0, patch_size=4, overlap=5)

    # margin too large
    with pytest.raises(ValueError):
        extract_overlapping_patches(img, margin=4, patch_size=4, overlap=1)

    # not RGB shape
    with pytest.raises(ValueError):
        extract_overlapping_patches(img[:, :, :2], margin=0, patch_size=4, overlap=1)  # type: ignore[arg-type]

    # unsupported dtype
    with pytest.raises(TypeError):
        extract_overlapping_patches(img.astype(np.int16), margin=0, patch_size=4, overlap=1)  # type: ignore[arg-type]


# --------------------------
# Tests for load_image_and_extract_patches
# --------------------------


def test_load_image_and_extract_patches_jpeg_uint8(tmp_path: Path) -> None:
    img = _make_rgb_uint8(12, 10, seed=1)
    path = tmp_path / "im.jpg"
    iio.imwrite(path, img)

    patches = load_image_and_extract_patches(
        path=path, margin=1, patch_size=4, overlap=2
    )

    assert patches.ndim == 4 and patches.shape[-1] == 3
    assert patches.dtype == np.uint8
    # Count check against expected
    expected = _expected_patch_count(12, 10, margin=1, patch_size=4, overlap=2)
    assert patches.shape[0] == expected


def test_load_image_and_extract_patches_tiff_uint16(tmp_path: Path) -> None:
    img = _make_rgb_uint16(20, 21, seed=3)
    path = tmp_path / "im.tiff"
    iio.imwrite(path, img)

    patches = load_image_and_extract_patches(
        path=path, margin=2, patch_size=7, overlap=3
    )

    assert patches.ndim == 4 and patches.shape[-1] == 3
    assert patches.dtype == np.uint16
    expected = _expected_patch_count(20, 21, margin=2, patch_size=7, overlap=3)
    assert patches.shape[0] == expected


# --------------------------
# Tests for load_images_from_folder
# --------------------------


def test_load_images_from_folder_basic_and_non_recursive(tmp_path: Path) -> None:
    # Create supported images at top level
    img_jpg = _make_rgb_uint8(10, 12, seed=1)
    img_tif = _make_rgb_uint16(7, 9, seed=2)
    p_jpg = tmp_path / "a.jpg"
    p_tif = tmp_path / "b.tiff"
    iio.imwrite(p_jpg, img_jpg)
    iio.imwrite(p_tif, img_tif)

    # Create unsupported and nested items to be ignored
    (tmp_path / "sub").mkdir()
    iio.imwrite(tmp_path / "sub" / "nested.jpg", _make_rgb_uint8(5, 5, seed=3))
    (tmp_path / "note.txt").write_text("not an image")
