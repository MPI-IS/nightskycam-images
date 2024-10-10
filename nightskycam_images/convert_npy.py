from pathlib import Path
import tempfile

import PIL.Image as PILImage
import cv2
import numpy as np
import numpy.typing as npt


def _bits_reduction(data: npt.NDArray, target: type) -> npt.NDArray:
    original_max = np.iinfo(data.dtype).max
    target_max = np.iinfo(target).max
    ratio = target_max / original_max
    return (data * ratio).astype(target)


def _to_8bits(image: npt.NDArray) -> npt.NDArray:
    return _bits_reduction(image, np.uint8)


def npy_file_to_pil(image_path: Path) -> PILImage.Image:
    """
    Read the file (expected to be an .npy file) and
    return a corresponding instance of PIL Image.
    It first converts the image to 8 bits.
    """
    img_array = np.load(image_path)
    img_array = _to_8bits(img_array)

    # Create a temporary directory for intermediary tiff file.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiff_file_path = Path(tmp_dir) / f"{image_path.stem}.tiff"
        cv2.imwrite(str(tiff_file_path), img_array, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        return PILImage.open(str(tiff_file_path))


def npy_file_to_numpy(image_path: Path) -> np.ndarray:
    """
    Read the file (expected to be an .npy file) and
    return a corresponding numpy array, converted to 8 bits.
    """
    img_array = np.load(image_path)
    img_array = _to_8bits(img_array)
    return img_array
