"""
Full-resolution image conversion for browser display, with a disk cache.

The display pipeline reuses the primitives from ``convert_npy.py``:
``to_npy`` reads npy/tiff/jpeg into a numpy array (16-bit TIFF stays
uint16, channel order is cv2's BGR convention), ``Stretch.array``
applies dtype-preserving auto-stretch, ``_to_8bits`` scales by dtype
max, and ``cv2.imencode`` writes JPEG bytes fully in memory using the
same BGR convention — so colors match the existing on-disk thumbnails.
"""

from collections import defaultdict
import os
from pathlib import Path
import threading
from typing import Dict

import cv2

from ..convert_npy import Stretch, _to_8bits, to_npy

JPEG_QUALITY = 90

# Guards against converting the same image concurrently in one process
# (gunicorn threads); a second process at worst duplicates work, the
# atomic os.replace keeps the cache consistent either way.
_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


class ConversionError(RuntimeError):
    pass


def convert_to_jpeg(hd_path: Path, stretch: bool) -> bytes:
    """Convert an original image file to browser-displayable JPEG bytes."""
    array = to_npy(hd_path)
    if array is None:
        raise ConversionError(f"could not read image: {hd_path}")
    if stretch:
        array = Stretch.array(array)
    array = _to_8bits(array)
    ok, buffer = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ConversionError(f"JPEG encoding failed: {hd_path}")
    return buffer.tobytes()


def cache_path(cache_dir: Path, filename_stem: str, stretch: bool) -> Path:
    return cache_dir / f"{filename_stem}.s{int(stretch)}.jpeg"


def cached_jpeg(
    cache_dir: Path,
    filename_stem: str,
    hd_path: Path,
    stretch: bool,
    max_mb: int,
) -> Path:
    """
    Return the path of the cached JPEG conversion, converting on miss.

    The file is written atomically (tmp + rename); after a miss the
    cache is opportunistically pruned to roughly ``max_mb``.
    """
    target = cache_path(cache_dir, filename_stem, stretch)
    if target.is_file():
        return target
    with _locks_guard:
        lock = _locks[target.name]
    with lock:
        if target.is_file():  # converted while we waited on the lock
            return target
        data = convert_to_jpeg(hd_path, stretch)
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    prune(cache_dir, max_mb)
    return target


def prune(cache_dir: Path, max_mb: int) -> None:
    """Delete oldest cached conversions until the cache fits ``max_mb``."""
    max_bytes = max_mb * 1024 * 1024
    entries = []
    total = 0
    for path in cache_dir.glob("*.jpeg"):
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
        total += stat.st_size
    if total <= max_bytes:
        return
    entries.sort()  # oldest mtime first
    for _, size, path in entries:
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        if total <= max_bytes:
            return
