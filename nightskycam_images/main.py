import argparse
from pathlib import Path
import sys

from .video import VideoFormat, create_video
from .walk import walk_thumbnails

# TODO: Move to resp. combine with project-level constant?
#   @Vincent:
#     Is it intended that there is an additional image format "png"?
#     If yes, could this be merged with the project-level constant
#     IMAGE_FILE_FORMATS: Tuple[str, ...] = ("npy", "tiff", "jpg", "jpeg")
#     by adding "png" to the project-level constant?
image_formats = ("jpeg", "jpg", "png", "tiff", "npy")

_video_format = VideoFormat()


def thumbnails():
    parser = argparse.ArgumentParser(description="list the thumbnails folders")
    parser.add_argument("folder_path", type=str, help="Path to the folder")
    args = parser.parse_args()
    p = Path(args.folder_path)
    if not p.is_dir():
        sys.stderr.write(
            f"The path {args.folder_path} does not exist or is not a directory."
        )
        sys.exit(1)

    for tp in walk_thumbnails(p):
        sys.stdout.write(f"{tp}\n")
    sys.exit(0)


def _list_images(current: Path) -> list[Path]:
    images: list[Path] = []
    for format_ in image_formats:
        images.extend(list(current.glob(f"*.{format_}")))
    return images


def main(video_format: VideoFormat = _video_format) -> None:
    current_path = Path(".")
    output = current_path / f"output.{video_format.format}"
    image_files = _list_images(current_path)
    image_files.sort()
    create_video(
        output,
        image_files,
        [str(img_file) for img_file in image_files],
        video_format,
    )
