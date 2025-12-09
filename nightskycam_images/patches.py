from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import imageio.v3 as iio
import numpy as np


def extract_overlapping_patches(
    image: np.ndarray, margin: int, patch_size: int, overlap: int
) -> np.ndarray:
    """
    Extract square overlapping patches from an RGB image after trimming a margin.

    Parameters
    ----------
    image : np.ndarray
        Input image as a H x W x 3 array of dtype uint8 or uint16.
    margin : int
        Number of pixels to trim from each border before patch extraction.
        The effective region is image[margin: H - margin, margin: W - margin].
    patch_size : int
        Side length (in pixels) of square patches to extract.
    overlap : int
        Overlap between neighboring patches (in pixels).
        The stride is computed as patch_size - overlap.

    Returns
    -------
    np.ndarray
        Array of patches with shape (N, patch_size, patch_size, 3), where N is
        the number of patches extracted. The dtype matches the input image.

    Raises
    ------
    TypeError
        If image is not a numpy array or dtype is not uint8/uint16.
    ValueError
        If input validation fails (shape, margin, patch_size/overlap, etc.).
    """
    # ---- Validate input types and shapes ----
    logger = logging.getLogger(__name__)
    logger.debug(
        "extract_overlapping_patches called: image_shape=%s margin=%s patch_size=%s overlap=%s",
        getattr(image, "shape", None),
        margin,
        patch_size,
        overlap,
    )
    if not isinstance(image, np.ndarray):
        logger.error("image must be a numpy.ndarray (got %s)", type(image))
        raise TypeError("image must be a numpy.ndarray")

    if image.ndim != 3 or image.shape[2] != 3:
        logger.error(
            "image must have shape (H, W, 3) for RGB input (got %s)",
            getattr(image, "shape", None),
        )
        raise ValueError("image must have shape (H, W, 3) for RGB input")

    # Accept uint8, or any unsigned 16-bit dtype (normalize to native-endian)
    if image.dtype == np.uint8:
        pass
    elif image.dtype.kind == "u" and image.dtype.itemsize == 2:
        if image.dtype != np.uint16:
            image = image.astype(np.uint16, copy=False)
    else:
        logger.error("image dtype must be uint8 or uint16 (got %s)", image.dtype)
        raise TypeError("image dtype must be uint8 or uint16")

    if not isinstance(margin, int) or margin < 0:
        logger.error("margin must be a non-negative integer (got %s)", margin)
        raise ValueError("margin must be a non-negative integer")

    if not isinstance(patch_size, int) or patch_size <= 0:
        logger.error("patch_size must be a positive integer (got %s)", patch_size)
        raise ValueError("patch_size must be a positive integer")

    if not isinstance(overlap, int) or overlap < 0 or overlap >= patch_size:
        logger.error(
            "overlap must be an integer with 0 <= overlap < patch_size (got %s, patch_size=%s)",
            overlap,
            patch_size,
        )
        raise ValueError("overlap must be an integer with 0 <= overlap < patch_size")

    H, W, _ = image.shape
    if 2 * margin >= H or 2 * margin >= W:
        logger.error(
            "margin is too large: trimmed dimensions would be non-positive (H=%s, W=%s, margin=%s)",
            H,
            W,
            margin,
        )
        raise ValueError(
            "margin is too large: trimmed dimensions would be non-positive"
        )

    # ---- Trim the image by the given margin ----
    trimmed = image[margin : H - margin, margin : W - margin, :]
    th, tw, _ = trimmed.shape

    if th < patch_size or tw < patch_size:
        logger.error(
            "Trimmed image is smaller than patch_size (trimmed=%sx%s, patch_size=%s)",
            th,
            tw,
            patch_size,
        )
        raise ValueError(
            "Trimmed image is smaller than patch_size in at least one dimension"
        )

    # ---- Compute stride from overlap ----
    stride = patch_size - overlap  # stride > 0 guaranteed by validation

    # ---- Helper to compute start indices with end anchoring ----
    def _compute_starts(length: int, k: int, st: int) -> List[int]:
        if length - k <= 0:
            return [0]
        starts = list(range(0, length - k + 1, st))
        last = length - k
        if starts[-1] != last:
            starts.append(last)
        return starts

    ys = _compute_starts(th, patch_size, stride)
    xs = _compute_starts(tw, patch_size, stride)

    # ---- Extract patches ----
    num_patches = len(ys) * len(xs)
    patches = np.empty((num_patches, patch_size, patch_size, 3), dtype=image.dtype)

    idx = 0
    for y in ys:
        y_end = y + patch_size
        for x in xs:
            x_end = x + patch_size
            patches[idx] = trimmed[y:y_end, x:x_end, :]
            idx += 1

    logger.info(
        "Extracted %d patches (patch_size=%s stride=%s) from trimmed image %sx%s, dtype=%s",
        num_patches,
        patch_size,
        stride,
        th,
        tw,
        image.dtype,
    )

    return patches


def load_image_and_extract_patches(
    path: Path,
    margin: int,
    patch_size: int,
    overlap: int,
) -> np.ndarray:
    """
    Read an RGB image (JPEG or TIFF) from a file path and return overlapping patches.

    Assumes the image is RGB (H, W, 3). Dtype must be uint8 or uint16.

    Parameters
    ----------
    path : Path
        Path to the input image file (JPEG or TIFF).
    margin : int
        Number of pixels to trim from each border before patch extraction.
    patch_size : int
        Side length (in pixels) of square patches to extract.
    overlap : int
        Overlap between neighboring patches (in pixels).

    Returns
    -------
    np.ndarray
        Array of patches with shape (N, patch_size, patch_size, 3), dtype uint8 or uint16.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")

    logger = logging.getLogger(__name__)
    logger.info("Loading image from %s", path)
    img = iio.imread(path)

    # Validate RGB shape
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3):
        raise ValueError(
            f"Expected an RGB image with shape (H, W, 3). Got shape {getattr(img, 'shape', None)}"
        )

    # Validate dtype (accept any unsigned 16-bit by converting to native-endian)
    if img.dtype == np.uint8:
        pass
    elif img.dtype.kind == "u" and img.dtype.itemsize == 2:
        if img.dtype != np.uint16:
            img = img.astype(np.uint16, copy=False)
    else:
        logger.error(
            "Unsupported dtype %s. Only uint8 and uint16 are supported.", img.dtype
        )
        raise ValueError(
            f"Unsupported dtype {img.dtype}. Only uint8 and uint16 are supported."
        )

    patches = extract_overlapping_patches(
        image=img,
        margin=margin,
        patch_size=patch_size,
        overlap=overlap,
    )

    logger.info("Loaded %s and extracted %d patches", path.name, patches.shape[0])
    return patches


def load_images_from_folder(folder: Path) -> Dict[str, np.ndarray]:
    """
    Read all RGB JPEG/TIFF images in a folder (non-recursive) and return a dict of arrays.

    Assumes images are RGB (H, W, 3). Dtype is preserved as uint8 or uint16.

    Parameters
    ----------
    folder : Path
        Path to the folder containing images. Only immediate files are considered.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from filename (including extension) to the corresponding RGB image array.
        If the folder contains no supported images, returns an empty dict.
    """
    if not isinstance(folder, Path):
        raise TypeError("folder must be a pathlib.Path")
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    logger = logging.getLogger(__name__)
    logger.info("Loading images from folder %s", folder)

    # Supported extensions (case-insensitive)
    exts = {".jpg", ".jpeg", ".tif", ".tiff"}

    files = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.name.lower(),
    )

    images: Dict[str, np.ndarray] = {}

    for p in files:
        img = iio.imread(p)

        # Validate RGB shape
        if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3):
            logger.error(
                "%s: expected an RGB image with shape (H, W, 3), got %s",
                p.name,
                getattr(img, "shape", None),
            )
            raise ValueError(
                f"{p.name}: expected an RGB image with shape (H, W, 3), got {getattr(img, 'shape', None)}"
            )

        # Validate dtype (accept any unsigned 16-bit by converting to native-endian)
        if img.dtype == np.uint8:
            pass
        elif img.dtype.kind == "u" and img.dtype.itemsize == 2:
            if img.dtype != np.uint16:
                img = img.astype(np.uint16, copy=False)
        else:
            logger.error(
                "%s: unsupported dtype %s. Only uint8/uint16 are supported.",
                p.name,
                img.dtype,
            )
            raise ValueError(
                f"{p.name}: unsupported dtype {img.dtype}. Only uint8/uint16 are supported."
            )

        images[p.name] = img

    logger.info("Loaded %d images from %s", len(images), folder)
    return images


def save_patches_from_folder(
    input_folder: Path,
    output_folder: Path,
    margin: int,
    patch_size: int,
    overlap: int,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    For each RGB image in input_folder, extract patches and save them to output_folder.

    - Non-recursive: only files directly under input_folder are processed.
    - Output format matches input file extension (JPEG stays JPEG, TIFF stays TIFF).
    - Output filenames: <original_filename_stem>_<patch_index><original_suffix>.
    - Patch index starts at 0 for each source image.

    Parameters
    ----------
    input_folder : Path
        Folder containing input images (.jpg, .jpeg, .tif, .tiff).
    output_folder : Path
        Folder where patches will be saved. Created if it does not exist.
    margin : int
        Number of pixels to trim from each border before patch extraction.
    patch_size : int
        Side length (in pixels) of square patches to extract.
    overlap : int
        Overlap between neighboring patches (in pixels).

    Returns
    -------
    Dict[str, int]
        Mapping from input filename (with extension) to the number of patches saved.

    Raises
    ------
    FileNotFoundError
        If input_folder does not exist.
    NotADirectoryError
        If input_folder is not a directory.
    ValueError
        If an image is not RGB or has an unsupported dtype for its format.
    """
    if not isinstance(input_folder, Path):
        raise TypeError("input_folder must be a pathlib.Path")
    if not isinstance(output_folder, Path):
        raise TypeError("output_folder must be a pathlib.Path")

    if not input_folder.exists():
        raise FileNotFoundError(f"Folder not found: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info(
        "Saving patches from %s to %s (margin=%s patch_size=%s overlap=%s)",
        input_folder,
        output_folder,
        margin,
        patch_size,
        overlap,
    )

    # Supported extensions (case-insensitive)
    jpeg_exts = {".jpg", ".jpeg"}
    tiff_exts = {".tif", ".tiff"}
    exts = jpeg_exts | tiff_exts

    files = sorted(
        (p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: p.name.lower(),
    )

    counts: Dict[str, int] = {}

    for p in files:
        img = iio.imread(p)

        # Validate RGB shape
        if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3):
            logger.error(
                "%s: expected an RGB image with shape (H, W, 3), got %s",
                p.name,
                getattr(img, "shape", None),
            )
            raise ValueError(
                f"{p.name}: expected an RGB image with shape (H, W, 3), got {getattr(img, 'shape', None)}"
            )

        suffix = p.suffix  # preserve original suffix including case
        suffix_lower = suffix.lower()

        # Validate dtype compatibility with format
        if suffix_lower in jpeg_exts:
            if img.dtype != np.uint8:
                raise ValueError(f"{p.name}: JPEG must be uint8, got {img.dtype}")
        elif suffix_lower in tiff_exts:
            # Accept uint8 or any unsigned 16-bit (normalize to native-endian)
            if img.dtype == np.uint8:
                pass
            elif img.dtype.kind == "u" and img.dtype.itemsize == 2:
                if img.dtype != np.uint16:
                    img = img.astype(np.uint16, copy=False)
            else:
                logger.error(
                    "%s: TIFF must be uint8 or uint16, got %s", p.name, img.dtype
                )
                raise ValueError(
                    f"{p.name}: TIFF must be uint8 or uint16, got {img.dtype}"
                )
        else:
            # Should not happen due to filter
            continue

        # Extract patches
        patches = extract_overlapping_patches(
            image=img,
            margin=margin,
            patch_size=patch_size,
            overlap=overlap,
        )

        # Save each patch with matching format and naming
        stem = p.stem
        n_saved = 0
        for i, patch in enumerate(patches):
            out_path = output_folder / f"{stem}_{i}{suffix}"
            # imageio selects plugin by extension; dtype drives bit depth.
            if out_path.exists() and not overwrite:
                logger.warning("Skipping existing file (overwrite=False): %s", out_path)
                continue
            iio.imwrite(out_path, patch)
            n_saved += 1

        counts[p.name] = n_saved
        logger.info("Saved %d patches for %s", n_saved, p.name)

    logger.info("Saved patches for %d files", len(counts))
    return counts
