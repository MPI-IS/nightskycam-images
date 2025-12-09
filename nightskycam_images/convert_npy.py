from pathlib import Path
import random
import string
import tempfile

import PIL.Image as PILImage
from auto_stretch.stretch import Stretch as AutoStretch
import cv2
import numpy as np
import numpy.typing as npt


def _bits_reduction(data: npt.NDArray, target: type) -> npt.NDArray:
    original_max = np.iinfo(data.dtype).max
    target_max = np.iinfo(target).max
    ratio = target_max / original_max
    return (data * ratio).astype(target)


def _to_8bits(image: npt.NDArray) -> npt.NDArray:
    if image.dtype == np.uint8:
        return image
    return _bits_reduction(image, np.uint8)


def _random_string(length=8):
    char_set = string.ascii_uppercase + string.digits
    random_string = "".join(random.choices(char_set, k=length))
    return random_string


def npy_array_to_pil(img_array: npt.NDArray) -> PILImage.Image:
    img_array = _to_8bits(img_array)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiff_file_path = Path(tmp_dir) / f"{_random_string()}.tiff"
        cv2.imwrite(str(tiff_file_path), img_array, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        return PILImage.open(str(tiff_file_path))


def npy_file_to_pil(image_path: Path) -> PILImage.Image:
    """
    Read the file (expected to be an .npy file) and
    return a corresponding instance of PIL Image.
    It first converts the image to 8 bits.
    """
    img_array = np.load(image_path)
    return npy_array_to_pil(img_array)


def npy_file_to_numpy(image_path: Path) -> np.ndarray:
    """
    Read the file (expected to be an .npy file) and
    return a corresponding numpy array, converted to 8 bits.
    """
    img_array = np.load(image_path)
    return img_array


def to_npy(image_path: Path) -> np.ndarray:
    """
    Read an image file to a numpy array.
    """
    if image_path.suffix == ".npy":
        img_array = np.load(image_path)
        return img_array
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image
    pil_image = PILImage.open(image_path)
    image = np.array(pil_image)
    return image


class Stretch:
    _stretch = AutoStretch()

    @classmethod
    def array(cls, image: np.ndarray) -> np.ndarray:
        stretched = cls._stretch.stretch(image)
        return (stretched * np.iinfo(image.dtype).max).astype(image.dtype)

    @classmethod
    def file(cls, source_path: Path, target_path: Path) -> None:
        array = to_npy(source_path)
        array = cls.array(array)
        if source_path.suffix == ".npy":
            np.save(target_path, array)
            return
        if source_path.suffix == ".tiff":
            cv2.imwrite(str(target_path), array, params=(cv2.IMWRITE_TIFF_COMPRESSION, 1))
            return
        else:
            cv2.imwrite(str(target_path), array)
            return
