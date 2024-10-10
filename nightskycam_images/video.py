from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import logging
from multiprocessing import cpu_count
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Generator, Iterable, Optional, Sequence

import cv2
import numpy as np

from .constants import FILE_PERMISSIONS
from .convert_npy import npy_file_to_numpy
from .folder_change import folder_has_changed

# TODO: Confirm.
#   @Vincent:
#     If I remember correctly, the modules mainly use type hints in the
#     format of Tuple[int, int] instead of tuple[int, int].
#     Is the former format preferred?


class TextFormat:
    def __init__(self) -> None:
        self.position: tuple[int, int] = (10, 50)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale: int = 1
        self.color: tuple[int, int, int] = (255, 0, 0)
        self.thickness: int = 2


class VideoFormat:
    def __init__(self) -> None:
        self.filename_stem: str = "day_summary"
        self.size: tuple[int, int] = (480, 360)  # (width, height).
        self.codec = cv2.VideoWriter_fourcc(*"VP90")  # type: ignore
        self.format: str = "webm"
        self.fps: float = 10.0
        self.text_format: TextFormat = TextFormat()

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "VideoFormat":
        instance = cls()
        vkeys = ("size", "codec", "format", "fps", "filename")
        for k in vkeys:
            if k not in config.keys():
                raise KeyError(
                    "failed to create an instance of" f"VideoFormat: missing key {k}"
                )
            setattr(instance, k, config[k])
        tkeys = (
            "font",
            "font_scale",
            "font_color",
            "font_thickness",
            "text_position",
        )
        for k in tkeys:
            if k not in config.keys():
                raise KeyError(
                    "failed to create an instance of" f"VideoFormat: missing key {k}"
                )
        instance.text_format.font = config["font"]
        instance.text_format.font_scale = config["font_scale"]
        instance.text_format.color = config["font_color"]
        instance.text_format.thickness = config["font_thickness"]
        instance.text_format.position = config["text_position"]
        return instance


_video_format = VideoFormat()
_nb_workers = 1 if cpu_count() == 1 else cpu_count() - 1


@contextmanager
def _get_video_capture(
    video_path: Path,
) -> Generator[cv2.VideoCapture, None, None]:
    """
    Get video capture instance,
    which is used for opening/reading video files.

    Parameters
    ----------
    video_path
        Path to video file.

    Yields
    ------
    cv2.VideoCapture
        Video capture instance.
    """
    capture = cv2.VideoCapture(str(video_path))
    yield capture
    capture.release()


@contextmanager
def _get_video_writer(
    video_path: Path, video_format: VideoFormat
) -> Generator[cv2.VideoWriter, None, None]:
    """
    Get video writer instance,
    which is used for writing video files.

    Parameters
    ----------
    video_path
        Path to output video file.

    Yields
    ------
    cv2.VideoWriter
        Video writer instance.
    """
    writer = cv2.VideoWriter(
        str(video_path),
        video_format.codec,
        video_format.fps,
        video_format.size,
    )
    yield writer
    writer.release()


def _write_to_image(image: np.ndarray, text: str, text_format: TextFormat) -> None:
    """
    Write text to the image.

    Parameters
    ----------
    image
        Image.
    text
        Text that will be added to the image.
    text_format
        Format specification for the text.
    """
    cv2.putText(  # type: ignore
        image,
        text,
        text_format.position,
        text_format.font,
        text_format.font_scale,
        text_format.color,
        text_format.thickness,
    )


def _setup_image_array(
    image_path: Path, text: Optional[str], video_format: VideoFormat
) -> Optional[np.ndarray]:
    """
    Set up image array in preparation for creating video:
    - Resize image to video format size.
    - Add (optional) text to image.

    Parameters
    ----------
    image_path
        Path to image file.
    text
        Text that will be added to the image.
    video_format
        Format configurations (e.g. size) for video.

    Returns
    -------
    np.ndarray
        Numpy array of the result image.
    """
    # If it is a numpy pickle file.
    if image_path.suffix == ".npy":
        # Load pickled array from file.
        image_array = npy_file_to_numpy(image_path)
    # If it is NOT a numpy pickle file.
    else:
        image_array = cv2.imread(str(image_path))

    # Resize image to set video format size.
    image_array = cv2.resize(image_array, video_format.size)
    if text:
        # Add text to image.
        _write_to_image(image_array, text, video_format.text_format)

    return image_array


def _count_frames(video_path: Path) -> int:
    """
    Get number of frames for video.

    Parameters
    ----------
    video_path
        Path to video file.

    Returns
    -------
    int
        Number of frames.
    """
    with _get_video_capture(video_path) as capture:
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return n_frames


def _write_video(
    video_path: Path,
    image_path_s: Iterable[Path],
    text_s: Iterable[Optional[str]] = tuple(),
    video_format: VideoFormat = _video_format,
    permissions: int = FILE_PERMISSIONS,
) -> None:
    """
    Write video to file.

    Parameters
    ----------
    video_path
        Path to video file.
    image_path_s
        Paths to image files.
    text_s
        Texts that will be added to images.
    video_format
        Format configurations (e.g. size) for video.
    permissions
        File permissions for resulting video file.
    """
    with _get_video_writer(video_path, video_format) as writer:
        for image, text in zip(image_path_s, text_s):
            image_array = _setup_image_array(image, text, video_format)
            if image_array is not None:
                writer.write(image_array)
    # Set file permissions.
    video_path.chmod(permissions)


def _create_video(
    video_path: Path,
    image_path_s: Sequence[Path],
    text_s: Sequence[Optional[str]] = tuple(),
    video_format: VideoFormat = _video_format,
    permissions: int = FILE_PERMISSIONS,
) -> None:
    """
    Write/overwrite video file, if necessary.

    If the target video file already exists but does NOT contain all images,
    overwrite existing video with updated video (containing all images).

    Parameters
    ----------
    video_path
        Path to video file.
    image_path_s
        Paths to image files.
    text_s
        Texts that will be added to images.
    video_format
        Format configurations (e.g. size) for video.
    permissions
        File permissions for resulting video file.
    """
    if not image_path_s:
        return

    # If there is NO already existing video file.
    if not video_path.is_file():
        # Create new video file.
        logging.info(
            "using %s images to create video %s.", len(image_path_s), video_path
        )
        _write_video(video_path, image_path_s, text_s, video_format, permissions)
        return

    # If there is already an existing video file.

    # Get number of frames in existing video file.
    n_frames = _count_frames(video_path)
    # If existing video already contains all images.
    if n_frames == len(image_path_s):
        return

    # Only used for logging.
    if n_frames < 0:
        n_frames = 0

    # Create new video file to replace existing one.
    logging.info("adding %s new images to %s", len(image_path_s) - n_frames, video_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_output = Path(tmp_dir) / f"out.{video_format.format}"
        _write_video(tmp_output, image_path_s, text_s, video_format)
        # Replace existing video file.
        shutil.move(str(tmp_output), video_path)
    # Set file permissions.
    video_path.chmod(permissions)


def create_video(
    video_path: Path,
    image_path_s: Sequence[Path],
    text_s: Sequence[Optional[str]] = tuple(),
    video_format: VideoFormat = _video_format,
    # TODO: confirm.
    #   @Vincent:
    #     Is it safe to ignore all errors as the default behaviour?
    skip_error: bool = True,
    permissions: int = FILE_PERMISSIONS,
) -> None:
    """
    Write/overwrite video file, if necessary.

    If the target video file already exists but does NOT contain all images,
    overwrite existing video with updated video (containing all images).

    Parameters
    ----------
    video_path
        Path to video file.
    image_path_s
        Paths to image files.
    text_s
        Texts that will be added to images.
    video_format
        Format configurations (e.g. size) for video.
    skip_error
        Whether to ignore errors.
    permissions
        File permissions for resulting video file.
    """
    try:
        _create_video(
            video_path,
            image_path_s,
            text_s=text_s,
            video_format=video_format,
            permissions=permissions,
        )
    except Exception as e:
        logging.error("failed to create/update video %s: %s", video_path, e)
        if skip_error:
            pass
        else:
            raise e


def create_all_videos(
    filename: str,
    walk_folders: Callable[[], Generator[Path, None, None]],
    list_images: Callable[[Path], Sequence[Path]],
    extract_text: Callable[[Path], str],
    video_format: VideoFormat = _video_format,
    skip_error: bool = True,
    nb_workers: int = _nb_workers,
    history: Optional[dict[Path, Optional[float]]] = None,
    permissions: int = 0o755,
) -> None:
    with ProcessPoolExecutor(max_workers=nb_workers) as executor:
        for folder in walk_folders():
            if folder_has_changed(folder, history):
                images = list_images(folder)
                output = folder / "thumbnails" / f"{filename}.{video_format.format}"
                texts = [extract_text(image) for image in images]
                executor.submit(
                    create_video,
                    output,
                    images,
                    texts,
                    video_format,
                    skip_error,
                    permissions=permissions,
                )
