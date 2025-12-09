from concurrent.futures import ProcessPoolExecutor
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Generator, Iterable, Optional

from PIL import Image as PILImage
import cv2
import numpy as np
import numpy.typing as npt

from .constants import (
    FILE_PERMISSIONS,
    THUMBNAIL_DIR_NAME,
    THUMBNAIL_FILE_FORMAT,
    THUMBNAIL_WIDTH,
)
from .convert_npy import Stretch, npy_array_to_pil, npy_file_to_pil, to_npy
from .folder_change import folder_has_changed

logging.getLogger("PIL").setLevel(logging.ERROR)

_nb_workers = 1 if cpu_count() == 1 else cpu_count() - 1


class ThumbnailText:
    def __init__(self) -> None:
        self.position: tuple[int, int] = (0, 20)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale: int = 1
        self.color: tuple[int, int, int] = (0, 0, 255)
        self.thickness: int = 1


def _thumbnail_path(
    image_path: Path,
    thumbnail_format: str = THUMBNAIL_FILE_FORMAT,
) -> Path:
    """
    Get path to thumbnail image file for given HD image path.

    Parameters
    ----------
    image_path
        Path to HD image file.
    thumbnail_format
        Format of thumbnail image file.

    Returns
    -------
    Path
        Path to thumbnail image file.
    """
    return (
        image_path.parent / THUMBNAIL_DIR_NAME / f"{image_path.stem}.{thumbnail_format}"
    )


def create_thumbnail(
    image_path: Path,
    # TODO: Remove parameter and only keep the default behaviour?
    #   Default behaviour:
    #     Automatic creation of thumbnail image path.
    #   Reasoning:
    #     Makes it impossible to input faulty combinations of
    #     image_path/thumbnail_path.
    #   @Vincent:
    #     Would this make sense?
    thumbnail_path: Optional[Path] = None,
    thumbnail_width: int = THUMBNAIL_WIDTH,
    thumbnail_format: str = THUMBNAIL_FILE_FORMAT,
    stretch: bool = True,
    overwrite: bool = False,
    permissions: int = FILE_PERMISSIONS,
) -> Path:
    """
    Write thumbnail image file for given HD image.

    Parameters
    ----------
    image_path
        Path to HD image file.
    thumbnail_path
        Path to output thumbnail image file.
    thumbnail_width
        Width of the thumbnail image (in pixels).
    thumbnail_format
        File format of the thumbnail image.
    stretch
        If True, autostretch will be applied first to the image
    overwrite
        Whether to overwrite an existing thumbnail image file.
    permissions
        File permissions for the thumbnail image.

    Returns
    -------
    Path
        Path to output thumbnail image file.
    """
    if thumbnail_path is None:
        thumbnail_path = _thumbnail_path(image_path)

    if thumbnail_path.is_file() and overwrite is False:
        logging.info("thumbnail for %s already exists, skipping", image_path.stem)
        return thumbnail_path

    logging.info("creating thumbnail for %s", image_path.stem)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

    if not stretch:
        if image_path.suffix == ".npy":
            img = npy_file_to_pil(image_path)
        else:
            img = PILImage.open(image_path)

    else:
        array = to_npy(image_path)
        stretched = Stretch.array(array)
        img = npy_array_to_pil(stretched)

    # Height for the thumbnail image (in pixels).
    # Use same scaling ratio between thumbnail and HD image
    # as for width.
    thumbnail_height = int((thumbnail_width / img.width) * img.height)
    # Convert HD image into thumbnail image (in-place).
    img.thumbnail((thumbnail_width, thumbnail_height))
    # Save thumbnail image.
    img.save(thumbnail_path, thumbnail_format)
    # Set file permissions.
    thumbnail_path.chmod(permissions)

    return thumbnail_path


def create_thumbnails(
    image_path_s: Iterable[Path],
    thumbnail_width: int = THUMBNAIL_WIDTH,
    thumbnail_format: str = THUMBNAIL_FILE_FORMAT,
    stretch: bool = True,
    skip_error: bool = True,
    overwrite: bool = False,
    permissions: int = FILE_PERMISSIONS,
) -> list[Path]:
    """
    Write thumbnail image file for each given HD image.

    Parameters
    ----------
    image_path_s
        Paths to HD image files.
    thumbnail_width
        Width of the thumbnail image (in pixels).
    thumbnail_format
        File format of the thumbnail image.
    stretch
        If True autostretch will be applied to the images
    skip_error
        Whether to ignore errors.
    overwrite
        Whether to overwrite an existing thumbnail image file.
    permissions
        File permissions for the thumbnail image.

    Returns
    -------
    Path
        Paths to output thumbnail image files.
    """
    thumbnail_path_s: list[Path] = []
    for image_path in image_path_s:
        if isinstance(image_path, str):
            image_path = Path(image_path)
        try:
            thumbnail_path_s.append(
                create_thumbnail(
                    image_path,
                    thumbnail_width=thumbnail_width,
                    thumbnail_format=thumbnail_format,
                    stretch=stretch,
                    overwrite=overwrite,
                    permissions=permissions,
                )
            )
        except Exception as e:
            logging.error("failed to create thumbnail for %s: %s", image_path.stem, e)
            if skip_error:
                logging.error("-> skipping the exception")
            else:
                raise e
    return thumbnail_path_s


# TODO: Improve maintainability of docstrings?
#   @Vincent, @Jean-Claude:
#     Would it make sense to document parameters like `history`
#     with `parameter for function 'folder_change.folder_has_changed'`
#     instead of duplicating the documentation?
#     This might make it easier to maintain the code in the case that
#     the used function changes?
def create_missing_thumbnails(
    root_dir: Path,
    thumbnail_width: int = THUMBNAIL_WIDTH,
    thumbnail_format: str = THUMBNAIL_FILE_FORMAT,
    stretch: bool = True,
    skip_error: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    permissions: int = FILE_PERMISSIONS,
) -> dict[str, int]:
    """
    Create thumbnails only for images that are missing them.

    Parameters
    ----------
    root_dir
        Root directory with nightskycam structure (system/date/).
    thumbnail_width
        Width of the thumbnail image (in pixels).
    thumbnail_format
        File format of the thumbnail image.
    stretch
        If True, autostretch will be applied to the images.
    skip_error
        Whether to ignore errors.
    dry_run
        If True, don't actually create thumbnails.
    verbose
        If True, log detailed progress.
    permissions
        File permissions for the thumbnail image.

    Returns
    -------
    dict
        Statistics: images_scanned, thumbnails_missing, thumbnails_created, errors
    """
    from .constants import IMAGE_FILE_FORMATS

    stats = {
        "images_scanned": 0,
        "thumbnails_missing": 0,
        "thumbnails_created": 0,
        "errors": 0,
    }

    if not root_dir.exists() or not root_dir.is_dir():
        logging.error(f"Root directory does not exist: {root_dir}")
        return stats

    # Walk through system directories
    for system_dir in root_dir.iterdir():
        if not system_dir.is_dir():
            continue

        if verbose:
            logging.info(f"Processing system: {system_dir.name}")

        # Walk through date directories
        for date_dir in system_dir.iterdir():
            if not date_dir.is_dir():
                continue

            if verbose:
                logging.info(f"  Processing date: {date_dir.name}")

            # Find all image files
            for item in date_dir.iterdir():
                if not item.is_file():
                    continue

                # Check if it's an image file
                if item.suffix.lstrip(".") not in IMAGE_FILE_FORMATS:
                    continue

                stats["images_scanned"] += 1

                # Check if thumbnail exists
                thumbnail_path = _thumbnail_path(item, thumbnail_format)

                if not thumbnail_path.exists():
                    stats["thumbnails_missing"] += 1

                    if dry_run:
                        logging.info(f"[DRY-RUN] Would create thumbnail for: {item}")
                        stats["thumbnails_created"] += 1
                    else:
                        try:
                            if verbose:
                                logging.info(f"    Creating thumbnail for: {item.name}")
                            create_thumbnail(
                                item,
                                thumbnail_path=thumbnail_path,
                                thumbnail_width=thumbnail_width,
                                thumbnail_format=thumbnail_format,
                                stretch=stretch,
                                overwrite=False,
                                permissions=permissions,
                            )
                            stats["thumbnails_created"] += 1
                        except Exception as e:
                            logging.error(f"Failed to create thumbnail for {item}: {e}")
                            stats["errors"] += 1
                            if not skip_error:
                                raise

    return stats


def create_all_thumbnails(
    walk_folders: Callable[[], Generator[Path, None, None]],
    list_images: Callable[[Path], Iterable[Path]],
    thumbnail_width: int = THUMBNAIL_WIDTH,
    thumbnail_format: str = THUMBNAIL_FILE_FORMAT,
    stretch: bool = True,
    skip_error: bool = True,
    nb_workers: int = _nb_workers,
    overwrite: bool = False,
    history: Optional[dict[Path, Optional[float]]] = None,
    permissions: int = FILE_PERMISSIONS,
) -> None:
    """
    Write thumbnail image file for all images in the date directories.

    Parameters
    ----------
    walk_folders
        Function for iterating over date directories
        (containing image files).
    list_images
        Function for detecting and listing all image files
        in the date directories.
    thumbnail_width
        Width of the thumbnail image (in pixels).
    thumbnail_format
        File format of the thumbnail image.
    stretch
        If True autostretch will be applied to the images
    skip_error
        Whether to ignore errors.
    overwrite
        Whether to overwrite an existing thumbnail image file.
    history
        Last modification times found in previous check:
        - key:
            Path to directory.
        - value:
            Time of last modification of directory.
    permissions
        File permissions for the thumbnail image.
    """
    # With process pool
    # for executing tasks asynchronously.
    with ProcessPoolExecutor(max_workers=nb_workers) as executor:
        # Iterate over date directories
        # (containing image files).
        for folder in walk_folders():
            # If last-modified-time of directory changed.
            if folder_has_changed(folder, history):
                # Image files in date directory.
                image_path_s = list(list_images(folder))
                # Submit task to process pool.
                executor.submit(
                    # Callable.
                    create_thumbnails,
                    # Parameters of callable.
                    image_path_s,
                    thumbnail_width=thumbnail_width,
                    thumbnail_format=thumbnail_format,
                    stretch=stretch,
                    skip_error=skip_error,
                    overwrite=overwrite,
                    permissions=permissions,
                )
