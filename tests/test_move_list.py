"""Tests for ns.files.move-list (`_move_selected_from_list`)."""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest
import tomli_w

from nightskycam_images.constants import THUMBNAIL_DIR_NAME, THUMBNAIL_FILE_FORMAT
from nightskycam_images.main import _move_selected_from_list


def _make_image(root: Path, system: str, date: str, time: str) -> str:
    stem = f"{system}_{date}_{time}"
    date_dir = root / system / date
    thumb_dir = date_dir / THUMBNAIL_DIR_NAME
    thumb_dir.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)
    cv2.imwrite(str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), img[:4, :4])
    with open(date_dir / f"{stem}.toml", "wb") as f:
        tomli_w.dump({"process": "raw"}, f)
    return stem


def _write_list(path: Path, lines) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def tree():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "root"
        dest = tmp / "dest"
        a = _make_image(root, "cam1", "2025_06_01", "20_00_00")
        b = _make_image(root, "cam1", "2025_06_01", "21_00_00")
        c = _make_image(root, "cam2", "2025_06_02", "22_00_00")
        yield tmp, root, dest, (a, b, c)


def test_move_keeps_structure(tree):
    tmp, root, dest, (a, b, c) = tree
    lst = _write_list(tmp / "sel.txt", ["# selection", a, b])
    stats = _move_selected_from_list(lst, [root], dest)

    assert stats["moved_images"] == 2
    assert stats["moved_toml"] == 2
    assert stats["moved_thumbs"] == 2

    for stem in (a, b):
        d = dest / "cam1" / "2025_06_01"
        assert (d / f"{stem}.jpg").is_file()
        assert (d / THUMBNAIL_DIR_NAME / f"{stem}.{THUMBNAIL_FILE_FORMAT}").is_file()
        assert (d / f"{stem}.toml").is_file()
        assert not (root / "cam1" / "2025_06_01" / f"{stem}.jpg").exists()
    # An unlisted image is untouched.
    assert (root / "cam2" / "2025_06_02" / f"{c}.jpg").is_file()


def test_dry_run_moves_nothing(tree):
    tmp, root, dest, (a, b, c) = tree
    lst = _write_list(tmp / "sel.txt", [a])
    stats = _move_selected_from_list(lst, [root], dest, dry_run=True)
    assert stats["moved_images"] == 1  # counted, but not moved
    assert (root / "cam1" / "2025_06_01" / f"{a}.jpg").is_file()
    assert not dest.exists() or not any(dest.rglob("*.jpg"))


def test_second_root(tree):
    tmp, root, dest, _ = tree
    root2 = tmp / "root2"
    d = _make_image(root2, "cam3", "2025_06_03", "23_00_00")
    lst = _write_list(tmp / "sel.txt", [d])
    stats = _move_selected_from_list(lst, [root, root2], dest)
    assert stats["moved_images"] == 1
    assert (dest / "cam3" / "2025_06_03" / f"{d}.jpg").is_file()
    assert not (root2 / "cam3" / "2025_06_03" / f"{d}.jpg").exists()


def test_not_found_and_invalid(tree):
    tmp, root, dest, (a, b, c) = tree
    lst = _write_list(
        tmp / "sel.txt", [a, "cam9_2099_01_01_00_00_00", "not a stem at all"]
    )
    stats = _move_selected_from_list(lst, [root], dest)
    assert stats["moved_images"] == 1  # only `a` exists on disk
    assert stats["not_found"] == 1  # cam9 stem parses but is absent
    assert stats["invalid"] == 1  # unparseable line


def test_after_midnight_stem_found_in_night_folder(tree):
    tmp, root, dest, _ = tree
    # An after-midnight capture is filed under the previous day's (observing-
    # night) folder, but its filename carries the next calendar date. move-list
    # must still find it (it derives the folder from the stem's calendar date).
    night = "2025_06_10"
    stem = "cam1_2025_06_11_00_34_30"  # 00:34 -> belongs to the 06_10 night
    date_dir = root / "cam1" / night
    thumb_dir = date_dir / THUMBNAIL_DIR_NAME
    thumb_dir.mkdir(parents=True)
    img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(date_dir / f"{stem}.jpg"), img)
    cv2.imwrite(str(thumb_dir / f"{stem}.{THUMBNAIL_FILE_FORMAT}"), img[:4, :4])
    with open(date_dir / f"{stem}.toml", "wb") as f:
        tomli_w.dump({"process": "raw"}, f)

    lst = _write_list(tmp / "sel.txt", [stem])
    stats = _move_selected_from_list(lst, [root], dest)

    assert stats["moved_images"] == 1
    assert stats["not_found"] == 0
    moved = dest / "cam1" / night  # night-folder structure preserved
    assert (moved / f"{stem}.jpg").is_file()
    assert (moved / THUMBNAIL_DIR_NAME / f"{stem}.{THUMBNAIL_FILE_FORMAT}").is_file()
    assert (moved / f"{stem}.toml").is_file()
    assert not (date_dir / f"{stem}.jpg").exists()


def test_collision_is_fail_fast(tree):
    tmp, root, dest, (a, b, c) = tree
    # Pre-create the target for `a`; the run must abort before moving anything.
    target = dest / "cam1" / "2025_06_01" / f"{a}.jpg"
    target.parent.mkdir(parents=True)
    target.write_text("x")
    lst = _write_list(tmp / "sel.txt", [a, b])

    with pytest.raises(FileExistsError):
        _move_selected_from_list(lst, [root], dest)

    # Atomic: nothing was moved from the source root.
    assert (root / "cam1" / "2025_06_01" / f"{a}.jpg").is_file()
    assert (root / "cam1" / "2025_06_01" / f"{b}.jpg").is_file()
