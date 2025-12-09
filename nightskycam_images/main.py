import argparse
import datetime as dt
import logging
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import imageio.v3 as iio
from loguru import logger
import tomli
import tomli_w
import typer

from nightskycam_scorer.model.infer import SkyScorer
from nightskycam_scorer.utils import to_float_image

from .annotator_webapp import create_app as create_annotator_app
from .constants import IMAGE_FILE_FORMATS, THUMBNAIL_DIR_NAME
from .convert_npy import to_npy
from .patches import load_image_and_extract_patches, save_patches_from_folder
from .stats import generate_stats_report
from .thumbnail import create_missing_thumbnails
from .video import VideoFormat, create_video
from .view_webapp import create_app as create_view_app
from .walk import (
    _create_symlink_safe,
    _is_within_date_range,
    copy_and_retarget_symlinks,
    filter_and_export_images,
    get_images,
    move_clear_images,
    walk_dates,
    walk_systems,
    walk_thumbnails,
)
 



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


def stats():
    """
    CLI entry point for generating statistics reports.
    """
    parser = argparse.ArgumentParser(
        description="Generate statistics report for nightskycam images"
    )
    parser.add_argument(
        "folder_path", type=str, help="Path to the media root directory"
    )
    args = parser.parse_args()

    p = Path(args.folder_path)
    if not p.is_dir():
        sys.stderr.write(
            f"The path {args.folder_path} does not exist or is not a directory.\n"
        )
        sys.exit(1)

    try:
        generate_stats_report(p)
        sys.exit(0)
    except Exception as e:
        sys.stderr.write(f"Error generating statistics: {e}\n")
        sys.exit(1)


def view_webapp():
    """
    CLI entry point for the image viewer web application.
    """
    parser = argparse.ArgumentParser(
        description="Start the Nightskycam image viewer web application"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the root directory (original or filtered structure)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5002, help="Port to bind to (default: 5002)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    root_path = Path(args.root_dir)
    if not root_path.exists() or not root_path.is_dir():
        sys.stderr.write(
            f"Error: {args.root_dir} does not exist or is not a directory.\n"
        )
        sys.exit(1)

    # Import the Flask app from view_webapp module
    app = create_view_app(root_path)

    print(f"Starting Nightskycam Image Viewer on http://{args.host}:{args.port}")
    print(f"Root directory: {root_path}")
    print("Press Ctrl+C to stop")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


def annotator_webapp():
    """
    CLI entry point for the thumbnail annotator web application.
    """
    parser = argparse.ArgumentParser(
        description="Start the Nightskycam thumbnail annotator web application"
    )
    parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file and exit",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5003, help="Port to bind to (default: 5003)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Handle --create-config flag
    if args.create_config:
        output_path = Path.cwd() / "nightskycam_thumbnails_annotator_config.toml"
        if output_path.exists():
            sys.stderr.write(
                f"Error: Configuration file already exists at {output_path}\n"
            )
            sys.exit(1)

        try:
            default_config = {
                "root_dir": "/path/to/media/root",
                "output_dir": "/path/to/output",
                "systems": [],  # Empty list means all systems
                "start_date": "",  # Empty means no start limit (format: YYYY-MM-DD)
                "end_date": "",  # Empty means no end limit (format: YYYY-MM-DD)
            }
            _save_config(default_config, output_path)
            print(f"Created default configuration file: {output_path}")
            print("Edit this file with your desired settings, then run:")
            print(f"  nightskycam-thumbnails-annotator-webapp {output_path}")
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"Error creating configuration file: {e}\n")
            sys.exit(1)

    # Require config_path if not creating config
    if args.config_path is None:
        sys.stderr.write(
            "Error: You must provide a config file path or use --create-config\n\n"
            "Usage:\n"
            "  nightskycam-thumbnails-annotator-webapp --create-config       # Create default config\n"
            "  nightskycam-thumbnails-annotator-webapp <config.toml>         # Run with config file\n"
        )
        sys.exit(1)

    config_path = Path(args.config_path)
    if not config_path.exists():
        sys.stderr.write(f"Error: Configuration file not found: {config_path}\n")
        sys.exit(1)

    # Import the Flask app from annotator_webapp module
    try:
        app = create_annotator_app(config_path)
    except Exception as e:
        sys.stderr.write(f"Error loading configuration: {e}\n")
        sys.exit(1)

    print(f"Starting Nightskycam Image Annotator on http://{args.host}:{args.port}")
    print(f"Configuration file: {config_path}")
    print("Press Ctrl+C to stop")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


def _list_images(current: Path) -> list[Path]:
    images: list[Path] = []
    for format_ in image_formats:
        images.extend(list(current.glob(f"*.{format_}")))
    return images


def save_patches() -> None:
    """CLI entry point to save patches from a file or folder.

    Usage examples:
      nightskycam-save-patches <path> <output_dir> -m 0 -s 256 -l 0 --overwrite
    """
    parser = argparse.ArgumentParser(description="Save patches from image or folder")
    parser.add_argument(
        "path", type=str, help="Path to an image file or a folder containing images"
    )
    parser.add_argument("output_dir", type=str, help="Directory to write patches into")
    parser.add_argument(
        "-m", "--margin", type=int, default=0, help="Pixels to trim from each border"
    )
    parser.add_argument(
        "-s",
        "--patch-size",
        type=int,
        default=256,
        help="Side length of square patches",
    )
    parser.add_argument(
        "-l",
        "--overlap",
        type=int,
        default=0,
        help="Overlap in pixels between neighboring patches",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    path = Path(args.path)
    output_dir = Path(args.output_dir)

    if not path.exists():
        logger.error("Path does not exist: %s", path)
        sys.exit(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    # If path is a file, process single image
    if path.is_file():
        logger.info("Processing single image: %s", path)
        try:
            patches = load_image_and_extract_patches(
                path,
                margin=args.margin,
                patch_size=args.patch_size,
                overlap=args.overlap,
            )
        except Exception as e:
            logger.exception("Failed to load and extract patches from %s: %s", path, e)
            sys.exit(3)

        suffix = path.suffix or ".jpg"
        stem = path.stem
        n_saved = 0
        for i, patch in enumerate(patches):
            out_path = output_dir / f"{stem}_{i}{suffix}"
            if out_path.exists() and not args.overwrite:
                logger.warning("Skipping existing file (overwrite=False): %s", out_path)
                continue
            iio.imwrite(out_path, patch)
            n_saved += 1

        logger.info("Saved %d patches to %s", n_saved, output_dir)
        return

    # If path is directory, delegate to save_patches_from_folder which supports overwrite
    if path.is_dir():
        logger.info("Processing folder: %s", path)
        try:
            counts = save_patches_from_folder(
                input_folder=path,
                output_folder=output_dir,
                margin=args.margin,
                patch_size=args.patch_size,
                overlap=args.overlap,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logger.exception("Failed to save patches from folder %s: %s", path, e)
            sys.exit(4)

        total = sum(counts.values())
        logger.info(
            "Saved %d patches for %d files to %s", total, len(counts), output_dir
        )
        return

    logger.error("Path is neither a file nor a directory: %s", path)
    sys.exit(5)


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


@dataclass
class _Config:
    """Configuration for filtering and exporting images."""

    root: Path
    output_dir: Path
    systems: Optional[List[str]] = None
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    time_window: Optional[Tuple[Optional[dt.time], Optional[dt.time]]] = None
    process_substring: Optional[str] = None
    process_not_substring: Optional[str] = None
    cloud_cover_range: Optional[Tuple[int, int]] = None
    weather_values: Optional[List[str]] = None
    cache_process_filter: bool = False


def _config_to_dict(config: _Config) -> Dict[str, Any]:
    """
    Convert a _Config instance to a dictionary suitable for TOML serialization.

    Converts Path objects to strings, date/time objects to strings, and handles None values.
    """
    result: Dict[str, Any] = {}

    # Convert Path objects to strings
    result["root"] = str(config.root)
    result["output_dir"] = str(config.output_dir)

    # Optional fields
    if config.systems is not None:
        result["systems"] = config.systems

    if config.start_date is not None:
        result["start_date"] = config.start_date.strftime("%Y-%m-%d")

    if config.end_date is not None:
        result["end_date"] = config.end_date.strftime("%Y-%m-%d")

    if config.time_window is not None:
        start_time, end_time = config.time_window
        if start_time is not None:
            result["start_time"] = start_time.strftime("%H:%M")
        if end_time is not None:
            result["end_time"] = end_time.strftime("%H:%M")

    if config.process_substring is not None:
        result["process"] = config.process_substring

    if config.process_not_substring is not None:
        result["process_not"] = config.process_not_substring

    if config.cloud_cover_range is not None:
        cloud_min, cloud_max = config.cloud_cover_range
        result["cloud_min"] = cloud_min
        result["cloud_max"] = cloud_max

    if config.weather_values is not None:
        result["weather"] = config.weather_values

    result["cache_process"] = config.cache_process_filter

    return result


@dataclass
class _ScorerConfig:
    """Configuration for scorer-based filtering."""

    root: Path
    output_dir: Path
    model_path: Path
    systems: Optional[List[str]] = None
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    classify_positive: bool = True
    probability_threshold: float = 0.5


def _scorer_config_to_dict(config: _ScorerConfig) -> Dict[str, Any]:
    """
    Convert a _ScorerConfig instance to a dictionary suitable for TOML serialization.

    Converts Path objects to strings, date objects to strings, and handles None values.
    """
    result: Dict[str, Any] = {}

    # Convert Path objects to strings
    result["root"] = str(config.root)
    result["output_dir"] = str(config.output_dir)
    result["model_path"] = str(config.model_path)

    # Optional fields
    if config.systems is not None:
        result["systems"] = config.systems

    if config.start_date is not None:
        result["start_date"] = config.start_date.strftime("%Y-%m-%d")

    if config.end_date is not None:
        result["end_date"] = config.end_date.strftime("%Y-%m-%d")

    # Classification settings
    result["classify_positive"] = config.classify_positive
    result["probability_threshold"] = config.probability_threshold

    return result


def _create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary with example values."""
    # Create a _Config instance with example values
    config = _Config(
        root=Path("/path/to/media/root"),
        output_dir=Path("/path/to/output"),
        systems=["nightskycam5", "nightskycam6"],
        start_date=dt.date(2025, 1, 1),
        end_date=dt.date(2025, 12, 31),
        time_window=(dt.time(20, 0), dt.time(23, 0)),
        process_substring="stretching and 8bits",
        process_not_substring=None,
        cloud_cover_range=(0, 30),
        weather_values=["clear", "partly_cloudy"],
        cache_process_filter=False,
    )
    # Convert to dictionary for TOML serialization
    return _config_to_dict(config)


def _save_config(config: Dict[str, Any], path: Path) -> None:
    """Save configuration to a TOML file."""
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def _load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from a TOML file."""
    with open(path, "rb") as f:
        return tomli.load(f)


def _parse_config(config: Dict[str, Any]) -> _Config:
    """
    Parse and validate configuration values.

    Returns a _Config instance with parsed values ready to pass to filter_and_export_images.
    """
    # Required fields
    if "root" not in config:
        raise ValueError("Configuration must include 'root' field")
    if "output_dir" not in config:
        raise ValueError("Configuration must include 'output_dir' field")

    root = Path(config["root"])
    output_dir = Path(config["output_dir"])

    # Validate root exists
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root}")

    # Optional: systems (list of strings or None)
    systems = config.get("systems", None)
    if systems is not None and not isinstance(systems, list):
        raise ValueError("'systems' must be a list of strings")

    # Optional: dates
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None

    if "start_date" in config and config["start_date"]:
        try:
            start_date = dt.datetime.strptime(
                config["start_date"], "%Y-%m-%d"
            ).date()
        except ValueError:
            raise ValueError(
                f"Invalid start_date format '{config['start_date']}'. Expected YYYY-MM-DD."
            )

    if "end_date" in config and config["end_date"]:
        try:
            end_date = dt.datetime.strptime(
                config["end_date"], "%Y-%m-%d"
            ).date()
        except ValueError:
            raise ValueError(
                f"Invalid end_date format '{config['end_date']}'. Expected YYYY-MM-DD."
            )

    # Optional: time window
    time_window: Optional[Tuple[Optional[dt.time], Optional[dt.time]]] = None
    start_time_str = config.get("start_time")
    end_time_str = config.get("end_time")

    if start_time_str or end_time_str:
        try:
            start_time_parsed: Optional[dt.time] = None
            end_time_parsed: Optional[dt.time] = None

            if start_time_str:
                start_time_parsed = dt.datetime.strptime(start_time_str, "%H:%M").time()
            if end_time_str:
                end_time_parsed = dt.datetime.strptime(end_time_str, "%H:%M").time()

            time_window = (start_time_parsed, end_time_parsed)
        except ValueError as e:
            raise ValueError(f"Invalid time format. Expected HH:MM. Details: {e}")

    # Optional: process filters
    process_substring = config.get("process")
    process_not_substring = config.get("process_not")

    # Optional: cloud cover range
    cloud_cover_range: Optional[Tuple[int, int]] = None
    cloud_min = config.get("cloud_min")
    cloud_max = config.get("cloud_max")

    if cloud_min is not None or cloud_max is not None:
        min_val = cloud_min if cloud_min is not None else 0
        max_val = cloud_max if cloud_max is not None else 100

        if not (0 <= min_val <= 100):
            raise ValueError(f"cloud_min must be between 0 and 100, got {min_val}")
        if not (0 <= max_val <= 100):
            raise ValueError(f"cloud_max must be between 0 and 100, got {max_val}")
        if min_val > max_val:
            raise ValueError(
                f"cloud_min ({min_val}) cannot be greater than cloud_max ({max_val})"
            )

        cloud_cover_range = (min_val, max_val)

    # Optional: weather values
    weather = config.get("weather")
    weather_values: Optional[List[str]] = None
    if weather is not None:
        if isinstance(weather, list):
            weather_values = weather
        else:
            raise ValueError("'weather' must be a list of strings")

    # Optional: cache_process
    cache_process_filter = config.get("cache_process", False)

    return _Config(
        root=root,
        output_dir=output_dir,
        systems=systems,
        start_date=start_date,
        end_date=end_date,
        time_window=time_window,
        process_substring=process_substring,
        process_not_substring=process_not_substring,
        cloud_cover_range=cloud_cover_range,
        weather_values=weather_values,
        cache_process_filter=cache_process_filter,
    )


def filter_export() -> None:
    """
    CLI tool to filter and export nightskycam images using a TOML configuration file.
    """
    app = typer.Typer(
        help="Filter nightskycam images and create symlinks based on a configuration file."
    )

    @app.command()
    def run(
        config_path: Optional[Path] = typer.Argument(
            None,
            help="Path to the TOML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        create_config: bool = typer.Option(
            False,
            "--create-config",
            help="Create a default configuration file named 'nightskycam_filter_config.toml' in the current directory.",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug logging (shows detailed filter operations).",
        ),
    ) -> None:
        """
        Filter nightskycam images and create symlinks based on configuration file.

        Use --create-config to generate a default configuration file that you can edit.
        """
        # Configure logging level
        logger.remove()  # Remove default handler
        if debug:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")
        # Handle --create-config flag
        if create_config:
            output_path = Path.cwd() / "nightskycam_filter_config.toml"
            if output_path.exists():
                typer.echo(
                    f"Error: Configuration file already exists at {output_path}",
                    err=True,
                )
                raise typer.Exit(code=1)

            try:
                default_config = _create_default_config()
                _save_config(default_config, output_path)
                typer.echo(
                    f"Created default configuration file: {output_path}\n"
                    f"Edit this file with your desired filter criteria, then run:\n"
                    f"  nightskycam-filter-export {output_path}"
                )
                raise typer.Exit(code=0)
            except Exception as e:
                typer.echo(f"Error creating configuration file: {e}", err=True)
                raise typer.Exit(code=1)

        # Require config_path if not creating config
        if config_path is None:
            typer.echo(
                "Error: You must provide a config file path or use --create-config\n\n"
                "Usage:\n"
                "  nightskycam-filter-export --create-config    # Create default config\n"
                "  nightskycam-filter-export <config.toml>      # Run with config file",
                err=True,
            )
            raise typer.Exit(code=1)

        # Load and parse configuration
        try:
            config = _load_config(config_path)
            parsed = _parse_config(config)
        except FileNotFoundError:
            typer.echo(f"Error: Configuration file not found: {config_path}", err=True)
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"Error loading configuration: {e}", err=True)
            raise typer.Exit(code=1)

        # Run the filter and export
        try:
            filter_and_export_images(
                root=parsed.root,
                output_dir=parsed.output_dir,
                systems=parsed.systems,
                start_date=parsed.start_date,
                end_date=parsed.end_date,
                time_window=parsed.time_window,
                cache_process_filter=parsed.cache_process_filter,
                process_substring=parsed.process_substring,
                process_not_substring=parsed.process_not_substring,
                cloud_cover_range=parsed.cloud_cover_range,
                weather_values=parsed.weather_values,
            )
            typer.echo("Filter and export completed successfully!")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    app()


def filter_copy() -> None:
    """
    CLI tool to copy a symlink directory and retarget symlinks to a different root.
    """
    app = typer.Typer(
        help="Copy a symlink directory (created by nightskycam-filter-export) "
        "and retarget symlinks to point to files in a different root directory."
    )

    @app.command()
    def run(
        input_dir: Path = typer.Argument(
            ...,
            help="Path to the input directory containing symlinks (created by nightskycam-filter-export).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        output_dir: Path = typer.Argument(
            ...,
            help="Path to the output directory to create (will contain retargeted symlinks).",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        root_dir: Path = typer.Argument(
            ...,
            help="Path to the root directory containing the target files (nightskycam structure).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug logging (shows detailed operations).",
        ),
    ) -> None:
        """
        Copy a symlink directory and retarget symlinks to a different root.

        This command takes a directory created by nightskycam-filter-export
        (containing symlinks) and creates a copy with symlinks retargeted to
        point to corresponding files in a different root directory.
        """
        # Configure logging level
        logger.remove()  # Remove default handler
        if debug:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        # Validate that output_dir doesn't exist
        if output_dir.exists():
            typer.echo(
                f"Error: Output directory already exists: {output_dir}",
                err=True,
            )
            raise typer.Exit(code=1)

        # Run the copy and retarget operation
        try:
            stats = copy_and_retarget_symlinks(
                input_dir=input_dir,
                output_dir=output_dir,
                root_dir=root_dir,
            )

            # Display statistics
            logger.info("=== Copy and Retarget Summary ===")
            logger.info(f"Symlinks processed: {stats['processed']}")
            logger.info(f"Symlinks created: {stats['created']}")
            logger.info(f"Files skipped (not symlinks): {stats['skipped']}")
            logger.info(f"Warnings (missing targets): {stats['warnings']}")

            typer.echo("Copy and retarget completed successfully!")
        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    app()


def _remove_file_safe(file_path: Path, dry_run: bool = False) -> bool:
    """
    Safely remove a file if it exists and is a symlink.

    Parameters
    ----------
    file_path
        Path to the file to remove.
    dry_run
        If True, don't actually remove files.

    Returns
    -------
    bool
        True if file was removed (or would be in dry-run), False otherwise.
    """
    if not file_path.exists():
        logger.debug(f"File does not exist: {file_path}")
        return False

    if not file_path.is_symlink():
        logger.warning(f"File is not a symlink, skipping: {file_path}")
        return False

    if dry_run:
        logger.info(f"[DRY-RUN] Would remove: {file_path}")
        return True

    try:
        file_path.unlink()
        logger.debug(f"Removed: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove {file_path}: {e}")
        return False


def _remove_from_list(
    target_folder: Path,
    list_file: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Remove images and TOML files listed in text file from target folder.

    Parameters
    ----------
    target_folder
        Root directory containing filtered images (symlinks).
    list_file
        Text file with lines in format: system/date/filename
    dry_run
        If True, don't actually remove files.
    verbose
        If True, log detailed progress.

    Returns
    -------
    dict
        Statistics: lines_processed, images_removed, tomls_removed, errors
    """
    stats = {
        "lines_processed": 0,
        "images_removed": 0,
        "tomls_removed": 0,
        "errors": 0,
    }

    if not list_file.exists():
        logger.error(f"List file not found: {list_file}")
        raise FileNotFoundError(f"List file not found: {list_file}")

    if not target_folder.exists() or not target_folder.is_dir():
        logger.error(f"Target folder not found or not a directory: {target_folder}")
        raise NotADirectoryError(
            f"Target folder not found or not a directory: {target_folder}"
        )

    with open(list_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        stats["lines_processed"] += 1

        try:
            # Parse line: system/date/filename
            parts = line.split("/")
            if len(parts) != 3:
                logger.warning(
                    f"Invalid line format (expected system/date/filename): {line}"
                )
                stats["errors"] += 1
                continue

            system, date, filename = parts
            image_path = target_folder / system / date / filename
            stem = Path(filename).stem
            toml_path = target_folder / system / date / f"{stem}.toml"

            if verbose:
                logger.info(f"Processing: {line}")

            # Remove image file
            if _remove_file_safe(image_path, dry_run):
                stats["images_removed"] += 1
            else:
                stats["errors"] += 1

            # Remove corresponding TOML file
            if _remove_file_safe(toml_path, dry_run):
                stats["tomls_removed"] += 1
            # Note: TOML might not exist, so we don't count as error

        except Exception as e:
            logger.error(f"Error processing line '{line}': {e}")
            stats["errors"] += 1

    return stats


def remove_selected():
    """
    CLI entry point for removing selected images from filtered directory.

    Reads a text file with image paths and removes both the image symlinks
    and their corresponding TOML metadata files from the target directory.
    """
    parser = argparse.ArgumentParser(
        description="Remove selected images from filtered directory"
    )
    parser.add_argument(
        "target_folder",
        type=str,
        help="Path to filtered directory containing symlinks",
    )
    parser.add_argument(
        "list_file",
        type=str,
        help="Path to text file with images to remove (system/date/filename format)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    target_folder = Path(args.target_folder)
    list_file = Path(args.list_file)

    if args.dry_run:
        logger.info("=== DRY-RUN MODE ===")

    logger.info(f"Target folder: {target_folder}")
    logger.info(f"List file: {list_file}")

    try:
        stats = _remove_from_list(
            target_folder=target_folder,
            list_file=list_file,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        logger.info("=== Removal Statistics ===")
        logger.info(f"Lines processed: {stats['lines_processed']}")
        logger.info(f"Images removed: {stats['images_removed']}")
        logger.info(f"TOML files removed: {stats['tomls_removed']}")
        logger.info(f"Errors: {stats['errors']}")

        if args.dry_run:
            logger.info("=== DRY-RUN MODE (no files were actually removed) ===")

        if stats["errors"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


def _get_thumbnail_path_from_image(image_path: Path) -> Path:
    """
    Get thumbnail path for an image.

    Parameters
    ----------
    image_path
        Path to the HD image file.

    Returns
    -------
    Path
        Path to thumbnail (may not exist).
    """
    return image_path.parent / THUMBNAIL_DIR_NAME / f"{image_path.stem}.jpeg"


def _copy_thumbnail_from_list(
    list_file: Path,
    root_dir: Path,
    dest_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Copy thumbnails for images listed in text file to a flat directory.

    All thumbnails are copied directly into dest_dir without preserving
    the directory structure.

    Parameters
    ----------
    list_file
        Text file with relative image paths (format: system/date/filename)
    root_dir
        Root directory to resolve relative paths
    dest_dir
        Destination directory for thumbnails (flat structure)
    dry_run
        If True, don't actually copy files
    verbose
        If True, log detailed progress

    Returns
    -------
    dict
        Statistics: lines_processed, thumbnails_copied, already_exists,
                   missing_thumbnails, errors, blank_lines
    """
    stats = {
        "lines_processed": 0,
        "thumbnails_copied": 0,
        "already_exists": 0,
        "missing_thumbnails": 0,
        "errors": 0,
        "blank_lines": 0,
    }

    # Read all lines
    with open(list_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Skip blank lines and comments
        if not line or line.startswith("#"):
            stats["blank_lines"] += 1
            continue

        stats["lines_processed"] += 1

        try:
            # Parse relative path: system/date/filename
            parts = line.split("/")
            if len(parts) != 3:
                logger.warning(
                    f"Invalid line format (expected system/date/filename): {line}"
                )
                stats["errors"] += 1
                continue

            system, date, filename = parts

            # Resolve full image path
            image_path = root_dir / system / date / filename
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                stats["errors"] += 1
                continue

            # Construct thumbnail path
            thumbnail_path = _get_thumbnail_path_from_image(image_path)
            if not thumbnail_path.exists():
                logger.warning(f"Thumbnail not found: {thumbnail_path}")
                stats["missing_thumbnails"] += 1
                continue

            # Construct destination path (flat structure)
            dest_thumbnail_path = dest_dir / f"{image_path.stem}.jpeg"

            # Skip if already exists
            if dest_thumbnail_path.exists():
                logger.info(f"Already exists, skipping: {dest_thumbnail_path}")
                stats["already_exists"] += 1
                continue

            # Copy file
            if dry_run:
                logger.info(
                    f"[DRY-RUN] Would copy: {thumbnail_path} -> {dest_thumbnail_path}"
                )
                stats["thumbnails_copied"] += 1
            else:
                shutil.copy2(thumbnail_path, dest_thumbnail_path)
                if verbose:
                    logger.debug(f"Copied: {thumbnail_path} -> {dest_thumbnail_path}")
                stats["thumbnails_copied"] += 1

        except Exception as e:
            logger.error(f"Error processing line '{line}': {e}")
            stats["errors"] += 1

    return stats


def _move_file_safe(source: Path, dest: Path, dry_run: bool = False) -> bool:
    """
    Safely move file, creating parent directories as needed.

    Parameters
    ----------
    source
        Source file path.
    dest
        Destination file path.
    dry_run
        If True, don't actually move files.

    Returns
    -------
    bool
        True if file was moved (or would be in dry-run).

    Raises
    ------
    Exception
        If move operation fails (fail-fast behavior).
    """
    if not source.exists():
        raise FileNotFoundError(f"Source file does not exist: {source}")

    if dry_run:
        logger.info(f"[DRY-RUN] Would move: {source} -> {dest}")
        return True

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        logger.debug(f"Moved: {source} -> {dest}")
        return True
    except Exception as e:
        logger.error(f"Failed to move {source} to {dest}: {e}")
        raise


def _delete_path_safe(path: Path, dry_run: bool = False) -> bool:
    """
    Safely delete file or directory.

    Parameters
    ----------
    path
        Path to file or directory to delete.
    dry_run
        If True, don't actually delete.

    Returns
    -------
    bool
        True if path was deleted (or would be in dry-run).

    Raises
    ------
    Exception
        If delete operation fails (fail-fast behavior).
    """
    if not path.exists():
        logger.debug(f"Path does not exist (already deleted?): {path}")
        return False

    if dry_run:
        if path.is_dir():
            logger.info(f"[DRY-RUN] Would delete directory: {path}")
        else:
            logger.info(f"[DRY-RUN] Would delete file: {path}")
        return True

    try:
        if path.is_dir():
            shutil.rmtree(path)
            logger.debug(f"Deleted directory: {path}")
        else:
            path.unlink()
            logger.debug(f"Deleted file: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete {path}: {e}")
        raise


def _is_date_folder_empty(date_path: Path) -> bool:
    """
    Check if date folder has no images or image-related TOML files.

    Parameters
    ----------
    date_path
        Path to date directory.

    Returns
    -------
    bool
        True if folder has no images or image TOMLs.
        weathers.toml and thumbnails folder are ignored.
    """
    for item in date_path.iterdir():
        if item.is_file():
            # Has an image file
            if item.suffix.lstrip(".") in IMAGE_FILE_FORMATS:
                return False
            # Has a TOML file (but ignore weathers.toml - it will be deleted with folder)
            if item.suffix == ".toml" and item.name != "weathers.toml":
                return False
        # Ignore directories (including thumbnails)
    return True


def _cleanup_empty_directories(
    root_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Clean up empty directories after moving files.

    Deletes:
    - Thumbnail folders (even if not empty, when date folder has no images)
    - Empty date folders
    - Empty system folders

    Parameters
    ----------
    root_dir
        Root directory to clean up.
    dry_run
        If True, don't actually delete directories.
    verbose
        If True, log detailed progress.

    Returns
    -------
    dict
        Statistics: thumbnail_folders_deleted, date_folders_deleted, system_folders_deleted
    """
    stats = {
        "thumbnail_folders_deleted": 0,
        "date_folders_deleted": 0,
        "system_folders_deleted": 0,
    }

    if not root_dir.exists() or not root_dir.is_dir():
        logger.warning(f"Root directory does not exist: {root_dir}")
        return stats

    # Collect empty date folders
    empty_date_folders: List[Path] = []

    # Walk through system directories
    for system_dir in root_dir.iterdir():
        if not system_dir.is_dir():
            continue

        # Walk through date directories
        for date_dir in system_dir.iterdir():
            if not date_dir.is_dir():
                continue

            # Check if date folder is empty
            if _is_date_folder_empty(date_dir):
                empty_date_folders.append(date_dir)

                # Delete thumbnail folder if exists (even if not empty)
                thumbnail_dir = date_dir / THUMBNAIL_DIR_NAME
                if thumbnail_dir.exists() and thumbnail_dir.is_dir():
                    if verbose:
                        logger.info(f"Deleting thumbnail folder: {thumbnail_dir}")
                    _delete_path_safe(thumbnail_dir, dry_run)
                    stats["thumbnail_folders_deleted"] += 1

    # Delete empty date folders
    for date_dir in empty_date_folders:
        if verbose:
            logger.info(f"Deleting empty date folder: {date_dir}")
        _delete_path_safe(date_dir, dry_run)
        stats["date_folders_deleted"] += 1

    # Delete empty system folders
    for system_dir in root_dir.iterdir():
        if not system_dir.is_dir():
            continue

        # Check if system folder is empty
        if not any(system_dir.iterdir()):
            if verbose:
                logger.info(f"Deleting empty system folder: {system_dir}")
            _delete_path_safe(system_dir, dry_run)
            stats["system_folders_deleted"] += 1

    return stats


def move_to_backup() -> None:
    """
    CLI entry point for moving filtered images to backup.

    Moves original images (and TOMLs) to backup directory based on symlinks
    in filter-export directory. Deletes thumbnails and cleans up empty folders.
    """
    parser = argparse.ArgumentParser(
        description="Move original images to backup based on filter-export symlinks"
    )
    parser.add_argument(
        "filter_output_dir",
        type=str,
        help="Path to filter-export directory containing symlinks",
    )
    parser.add_argument(
        "backup_dir",
        type=str,
        help="Path to backup directory (will preserve system/date structure)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without actually moving/deleting files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    filter_output_dir = Path(args.filter_output_dir)
    backup_dir = Path(args.backup_dir)

    if args.dry_run:
        logger.info("=== DRY-RUN MODE ===")

    logger.info(f"Filter-export directory: {filter_output_dir}")
    logger.info(f"Backup directory: {backup_dir}")

    # Validate directories
    if not filter_output_dir.exists() or not filter_output_dir.is_dir():
        logger.error(f"Filter-export directory not found: {filter_output_dir}")
        sys.exit(1)

    try:
        stats = {
            "symlinks_processed": 0,
            "images_moved": 0,
            "tomls_moved": 0,
            "thumbnails_deleted": 0,
        }

        # Track original root directory for cleanup
        # We'll determine this from the first symlink
        original_root: Optional[Path] = None
        processed_originals: Set[Path] = set()  # Track to avoid duplicates

        # Walk through filter-export directory
        for system_dir in filter_output_dir.iterdir():
            if not system_dir.is_dir():
                continue

            system_name = system_dir.name

            for date_dir in system_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                date_name = date_dir.name

                for item in date_dir.iterdir():
                    # Only process image symlinks
                    if not item.is_symlink():
                        continue

                    if item.suffix.lstrip(".") not in IMAGE_FILE_FORMATS:
                        continue

                    stats["symlinks_processed"] += 1

                    try:
                        # Resolve symlink to original file
                        original_image = item.resolve()

                        # Skip if already processed
                        if original_image in processed_originals:
                            logger.debug(f"Already processed: {original_image}")
                            continue

                        processed_originals.add(original_image)

                        # Determine original root from first symlink
                        if original_root is None:
                            # Original structure: root/system/date/image.ext
                            # Go up 2 levels from image to get system, then 1 more to get root
                            original_root = original_image.parent.parent.parent
                            logger.debug(f"Detected original root: {original_root}")

                        if args.verbose:
                            logger.info(f"Processing: {original_image}")

                        # Calculate backup destination (preserve tree structure)
                        relative_path = original_image.relative_to(original_root)
                        backup_dest = backup_dir / relative_path

                        # Move image file
                        _move_file_safe(original_image, backup_dest, args.dry_run)
                        stats["images_moved"] += 1

                        # Move TOML metadata file
                        original_toml = original_image.with_suffix(".toml")
                        if original_toml.exists():
                            backup_toml = backup_dest.with_suffix(".toml")
                            _move_file_safe(original_toml, backup_toml, args.dry_run)
                            stats["tomls_moved"] += 1
                        else:
                            logger.warning(f"TOML not found: {original_toml}")

                        # Delete thumbnail
                        thumbnail_path = _get_thumbnail_path_from_image(original_image)
                        if thumbnail_path.exists():
                            _delete_path_safe(thumbnail_path, args.dry_run)
                            stats["thumbnails_deleted"] += 1

                    except Exception as e:
                        logger.error(f"Error processing {item}: {e}")
                        raise  # Fail-fast

        # Cleanup empty directories
        if original_root is not None:
            logger.info("Cleaning up empty directories...")
            cleanup_stats = _cleanup_empty_directories(
                original_root,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            stats.update(cleanup_stats)
        else:
            logger.warning("No symlinks found, nothing to clean up")

        # Print statistics
        logger.info("=== Operation Statistics ===")
        logger.info(f"Symlinks processed: {stats['symlinks_processed']}")
        logger.info(f"Images moved: {stats['images_moved']}")
        logger.info(f"TOMLs moved: {stats['tomls_moved']}")
        logger.info(f"Thumbnails deleted: {stats['thumbnails_deleted']}")
        logger.info(
            f"Thumbnail folders deleted: {stats.get('thumbnail_folders_deleted', 0)}"
        )
        logger.info(f"Date folders deleted: {stats.get('date_folders_deleted', 0)}")
        logger.info(f"System folders deleted: {stats.get('system_folders_deleted', 0)}")

        if args.dry_run:
            logger.info("=== DRY-RUN MODE (no files were actually moved/deleted) ===")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


def move_clear_images_cli():
    """
    CLI entry point for moving clear-weather night images.

    Moves images with clear weather (22:00-04:00) to target directory.
    Filters by weather field containing "clear" and time between 22:00-04:00.
    """
    parser = argparse.ArgumentParser(
        description="Move clear-weather night images to target directory"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Source media root directory (nightskycam structure)",
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Target directory (will preserve system/date structure)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview moves without actually moving files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()  # Remove default handler
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    try:
        # Validate paths
        if not args.root_dir.exists():
            logger.error(f"Root directory does not exist: {args.root_dir}")
            sys.exit(1)

        if not args.root_dir.is_dir():
            logger.error(f"Root path is not a directory: {args.root_dir}")
            sys.exit(1)

        # Run move operation
        logger.info("=" * 60)
        logger.info("nightskycam-move-clear-images")
        logger.info("=" * 60)
        logger.info(f"Source: {args.root_dir}")
        logger.info(f"Target: {args.target_dir}")
        logger.info(f"Filters: weather contains 'clear' AND time 22:00-04:00")
        logger.info("=" * 60)

        stats = move_clear_images(
            root=args.root_dir,
            target_dir=args.target_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        # Display statistics
        logger.info("=" * 60)
        logger.info("Operation Statistics")
        logger.info("=" * 60)
        logger.info(f"Images scanned: {stats['images_scanned']}")
        logger.info(f"Images matched: {stats['images_matched']}")
        logger.info(f"Images moved: {stats['images_moved']}")
        logger.info(f"TOMLs moved: {stats['tomls_moved']}")
        logger.info(f"Images skipped (already exist): {stats['images_skipped']}")
        logger.info(f"Errors: {stats['errors']}")

        if not args.dry_run:
            logger.info("")
            logger.info("Empty directories cleaned up:")
            logger.info(
                f"  Thumbnail folders deleted: {stats.get('thumbnail_folders_deleted', 0)}"
            )
            logger.info(
                f"  Date folders deleted: {stats.get('date_folders_deleted', 0)}"
            )
            logger.info(
                f"  System folders deleted: {stats.get('system_folders_deleted', 0)}"
            )

        if args.dry_run:
            logger.info("")
            logger.info("=== DRY-RUN MODE (no files were actually moved) ===")

        logger.info("=" * 60)
        logger.info("Operation completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def _get_default_scorer_config() -> Dict[str, Any]:
    """
    Return a default configuration dictionary for scorer-based filtering.
    """
    # Create a _ScorerConfig instance with example values
    config = _ScorerConfig(
        root=Path("/path/to/media/root"),
        output_dir=Path("/path/to/output"),
        model_path=Path("/path/to/best_model.pt"),
        systems=["nightskycam5", "nightskycam7"],
        start_date=dt.date(2025, 1, 1),
        end_date=dt.date(2025, 12, 31),
        classify_positive=True,
        probability_threshold=0.5,
    )
    # Convert to dictionary for TOML serialization
    return _scorer_config_to_dict(config)


def _parse_scorer_config(config: Dict[str, Any]) -> _ScorerConfig:
    """
    Parse and validate scorer configuration values.

    Returns a _ScorerConfig instance with parsed values ready to use.
    """
    # Required fields
    if "root" not in config:
        raise ValueError("Configuration must include 'root' field")
    if "output_dir" not in config:
        raise ValueError("Configuration must include 'output_dir' field")
    if "model_path" not in config:
        raise ValueError("Configuration must include 'model_path' field")

    root = Path(config["root"])
    output_dir = Path(config["output_dir"])
    model_path = Path(config["model_path"])

    # Validate paths exist
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root}")
    if not model_path.exists() or not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")

    # Optional: systems (list of strings or None)
    systems = config.get("systems", None)
    if systems is not None and not isinstance(systems, list):
        raise ValueError("'systems' must be a list of strings")

    # Optional: dates
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None

    if "start_date" in config and config["start_date"]:
        try:
            start_date = dt.datetime.strptime(
                config["start_date"], "%Y-%m-%d"
            ).date()
        except ValueError:
            raise ValueError(
                f"Invalid start_date format '{config['start_date']}'. Expected YYYY-MM-DD."
            )

    if "end_date" in config and config["end_date"]:
        try:
            end_date = dt.datetime.strptime(
                config["end_date"], "%Y-%m-%d"
            ).date()
        except ValueError:
            raise ValueError(
                f"Invalid end_date format '{config['end_date']}'. Expected YYYY-MM-DD."
            )

    # Classification settings
    classify_positive = config.get("classify_positive", True)
    if not isinstance(classify_positive, bool):
        raise ValueError("'classify_positive' must be a boolean (true/false)")

    probability_threshold = config.get("probability_threshold", 0.5)
    if not isinstance(probability_threshold, (int, float)):
        raise ValueError("'probability_threshold' must be a number")
    if not (0.0 <= probability_threshold <= 1.0):
        raise ValueError("'probability_threshold' must be between 0.0 and 1.0")

    return _ScorerConfig(
        root=root,
        output_dir=output_dir,
        model_path=model_path,
        systems=systems,
        start_date=start_date,
        end_date=end_date,
        classify_positive=classify_positive,
        probability_threshold=probability_threshold,
    )


def scorer_filter() -> None:
    """
    CLI tool to filter nightskycam images using a trained classifier model.

    This tool uses the nightskycam-scorer classifier to predict image quality
    and creates symlinks to images that match the classification criteria.
    """
    app = typer.Typer(
        help="Filter nightskycam images using a trained classifier model."
    )

    @app.command()
    def run(
        config_path: Optional[Path] = typer.Argument(
            None,
            help="Path to the TOML configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        create_config: bool = typer.Option(
            False,
            "--create-config",
            help="Create a default configuration file named 'nightskycam_scorer_filter_config.toml' in the current directory.",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug logging (shows detailed inference operations).",
        ),
    ) -> None:
        """
        Filter images based on classifier predictions and create symlinks.
        """
        # Configure logging level
        logger.remove()
        if debug:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        # Handle --create-config
        if create_config:
            default_config = _get_default_scorer_config()
            config_file_path = Path("nightskycam_scorer_filter_config.toml")

            if config_file_path.exists():
                logger.error(f"Configuration file already exists: {config_file_path}")
                logger.info("Remove the existing file or specify a different name.")
                sys.exit(1)

            _save_config(default_config, config_file_path)
            logger.info(f"Created default configuration file: {config_file_path}")
            logger.info(
                "Please edit the file with your actual paths and settings, then run again."
            )
            sys.exit(0)

        # Require config_path if not creating config
        if config_path is None:
            logger.error("No configuration file provided.")
            logger.info(
                "Use --create-config to generate a template, or provide a config file path."
            )
            sys.exit(1)

        try:
            # Load and parse configuration
            logger.info(f"Loading configuration from: {config_path}")
            config = _load_config(config_path)
            parsed = _parse_scorer_config(config)

            root = parsed.root
            output_dir = parsed.output_dir
            model_path = parsed.model_path
            systems = parsed.systems
            start_date = parsed.start_date
            end_date = parsed.end_date
            classify_positive = parsed.classify_positive
            probability_threshold = parsed.probability_threshold

            logger.info(f"Root directory: {root}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Model path: {model_path}")
            logger.info(
                f"Classifying: {'positive' if classify_positive else 'negative'}"
            )
            logger.info(f"Probability threshold: {probability_threshold}")

            if systems:
                logger.info(f"Systems filter: {systems}")
            if start_date or end_date:
                logger.info(f"Date range: {start_date or 'any'} to {end_date or 'any'}")

 
            # Load the scorer model
            logger.info("Loading classifier model...")
            try:
                scorer = SkyScorer(str(model_path), validate_size=False)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                sys.exit(1)

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {output_dir}")

            # Statistics
            stats = {
                "total_scanned": 0,
                "total_with_thumbnails": 0,
                "total_matched": 0,
                "symlinks_created": 0,
                "errors": 0,
            }

            logger.info("Starting image classification and filtering...")

            # Iterate through systems
            for system_path in walk_systems(root):
                system_name = system_path.name

                # Filter by system name if specified
                if systems is not None and system_name not in systems:
                    logger.debug(f"Skipping system (not in filter): {system_name}")
                    continue

                logger.info(f"Processing system: {system_name}")

                # Iterate through dates in the system
                for date_, date_path in walk_dates(system_path):
                    # Filter by date range
                    if not _is_within_date_range(date_, start_date, end_date):
                        logger.debug(f"Skipping date (out of range): {date_}")
                        continue

                    date_str = date_.strftime("%Y_%m_%d")
                    logger.info(f"  Processing date: {date_str}")

                    # Get all images for this date
                    try:
                        images = get_images(date_path)
                    except PermissionError as e:
                        logger.error(
                            f"    Permission denied accessing date directory: {e}"
                        )
                        stats["errors"] += 1
                        continue
                    except Exception as e:
                        logger.error(f"    Error reading date directory: {e}")
                        stats["errors"] += 1
                        if debug:
                            raise
                        continue

                    if not images:
                        logger.debug(f"    No images with thumbnails found")
                        continue

                    # Create output subdirectory matching the source structure
                    output_subdir = output_dir / system_name / date_str
                    output_subdir.mkdir(parents=True, exist_ok=True)

                    date_matched = 0

                    for image in images:
                        stats["total_scanned"] += 1

                        # Check if thumbnail exists
                        thumbnail_path = image.thumbnail
                        if thumbnail_path is None or not thumbnail_path.exists():
                            logger.debug(
                                f"    Skipping image (no thumbnail): {image.filename_stem}"
                            )
                            continue

                        stats["total_with_thumbnails"] += 1

                        # Check if HD image exists
                        hd_path = image.hd
                        if hd_path is None or not hd_path.exists():
                            logger.debug(
                                f"    Skipping image (no HD file): {image.filename_stem}"
                            )
                            continue

                        try:
                            # Load thumbnail for inference
                            logger.debug(f"    Classifying: {image.filename_stem}")
                            thumbnail_array = to_npy(thumbnail_path)
                            rgb_float = to_float_image(thumbnail_array)

                            # Run inference
                            result_raw = scorer.predict(rgb_float)

                            # Handle both single result and list of results
                            result = result_raw[0] if isinstance(result_raw, list) else result_raw

                            logger.debug(
                                f"      Prediction: {result.prediction}, "
                                f"Probability: {result.probability:.3f}, "
                                f"Confidence: {result.confidence:.3f}"
                            )

                            # Apply filtering logic
                            match = False
                            if classify_positive:
                                # Export positive predictions
                                match = result.probability >= probability_threshold
                            else:
                                # Export negative predictions
                                match = result.probability < probability_threshold

                            if match:
                                # Create symlinks to HD image and TOML file
                                image_dest = output_subdir / hd_path.name
                                if _create_symlink_safe(hd_path, image_dest):
                                    stats["symlinks_created"] += 1
                                    date_matched += 1
                                    logger.debug(
                                        f"      Created symlink: {image_dest.name}"
                                    )

                                # Also symlink TOML file if it exists
                                toml_path = image.meta_path
                                if toml_path is not None and toml_path.exists():
                                    toml_dest = output_subdir / toml_path.name
                                    if _create_symlink_safe(toml_path, toml_dest):
                                        logger.debug(
                                            f"      Created symlink: {toml_dest.name}"
                                        )

                                stats["total_matched"] += 1

                        except Exception as e:
                            logger.error(
                                f"    Error processing {image.filename_stem}: {e}"
                            )
                            stats["errors"] += 1
                            if debug:
                                raise

                    if date_matched > 0:
                        logger.info(f"    Matched: {date_matched} images")

            # Print summary statistics
            logger.info("")
            logger.info("=== Classification Summary ===")
            logger.info(f"Total images scanned: {stats['total_scanned']}")
            logger.info(f"Images with thumbnails: {stats['total_with_thumbnails']}")
            logger.info(f"Images matched filter: {stats['total_matched']}")
            logger.info(f"Symlinks created: {stats['symlinks_created']}")
            if stats["errors"] > 0:
                logger.warning(f"Errors encountered: {stats['errors']}")

            logger.info(f"Output directory: {output_dir}")
            logger.info("Done!")

            sys.exit(0)

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            if debug:
                raise
            sys.exit(1)

    app()


def copy_thumbnails() -> None:
    """
    CLI tool to copy thumbnails for images listed in a text file.
    """
    app = typer.Typer(
        help="Copy thumbnails for images listed in a text file "
        "to a flat destination directory."
    )

    @app.command()
    def run(
        list_file: Path = typer.Argument(
            ...,
            help="Path to text file containing relative image paths (one per line, format: system/date/filename).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        root_dir: Path = typer.Argument(
            ...,
            help="Root directory to resolve relative paths (media root).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        dest_dir: Path = typer.Argument(
            ...,
            help="Destination directory for copied thumbnails.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Show what would be copied without actually copying files.",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug logging (shows detailed operations).",
        ),
    ) -> None:
        """
        Copy thumbnails for images listed in a text file to a flat directory.

        All thumbnails are copied directly into the destination directory
        (flat structure, no subdirectories).

        The text file should contain relative image paths, one per line:
        system/date/filename

        For example:
        nightskycam5/2025_08_25/nightskycam5_2025_08_25_23_30_03.tiff

        Blank lines and lines starting with '#' are ignored.
        """
        # Configure logging
        logger.remove()
        if debug:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        # Create destination directory if needed
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Log configuration
        logger.info(f"List file: {list_file}")
        logger.info(f"Root directory: {root_dir}")
        logger.info(f"Destination directory: {dest_dir}")
        if dry_run:
            logger.info("=== DRY-RUN MODE ===")

        # Run the copy operation
        try:
            stats = _copy_thumbnail_from_list(
                list_file=list_file,
                root_dir=root_dir,
                dest_dir=dest_dir,
                dry_run=dry_run,
                verbose=debug,
            )

            # Display statistics
            logger.info("=== Copy Thumbnails Summary ===")
            logger.info(f"Lines processed: {stats['lines_processed']}")
            logger.info(f"Thumbnails copied: {stats['thumbnails_copied']}")
            logger.info(f"Already exists (skipped): {stats['already_exists']}")
            logger.info(f"Missing thumbnails: {stats['missing_thumbnails']}")
            logger.info(f"Blank/comment lines: {stats['blank_lines']}")
            logger.info(f"Errors: {stats['errors']}")

            if stats["errors"] > 0:
                typer.echo(
                    f"Completed with {stats['errors']} errors. See log for details.",
                    err=True,
                )
                raise typer.Exit(code=1)
            else:
                typer.echo("Copy thumbnails completed successfully!")

        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

    app()


def create_missing_thumbnails_cli() -> None:
    """
    CLI entry point for creating missing thumbnails.

    Creates thumbnails only for images that don't already have them.
    """
    parser = argparse.ArgumentParser(
        description="Create thumbnails for images that are missing them"
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory with nightskycam structure (system/date/)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=200,
        help="Width of thumbnails in pixels (default: 200)",
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        default=True,
        help="Apply auto-stretch to images (default: True)",
    )
    parser.add_argument(
        "--no-stretch",
        dest="stretch",
        action="store_false",
        help="Disable auto-stretch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be created without actually creating thumbnails",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        # Validate path
        if not args.root_dir.exists():
            logger.error(f"Root directory does not exist: {args.root_dir}")
            sys.exit(1)

        if not args.root_dir.is_dir():
            logger.error(f"Root path is not a directory: {args.root_dir}")
            sys.exit(1)

        # Import the function
        logger.info("=" * 60)
        logger.info("nightskycam-create-missing-thumbnails")
        logger.info("=" * 60)
        logger.info(f"Root directory: {args.root_dir}")
        logger.info(f"Thumbnail width: {args.width} pixels")
        logger.info(f"Auto-stretch: {args.stretch}")
        if args.dry_run:
            logger.info("DRY-RUN MODE: No thumbnails will be created")
        logger.info("=" * 60)

        # Run the function
        stats = create_missing_thumbnails(
            root_dir=args.root_dir,
            thumbnail_width=args.width,
            stretch=args.stretch,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        # Display statistics
        logger.info("=" * 60)
        logger.info("Operation Statistics")
        logger.info("=" * 60)
        logger.info(f"Images scanned: {stats['images_scanned']}")
        logger.info(f"Thumbnails missing: {stats['thumbnails_missing']}")
        logger.info(f"Thumbnails created: {stats['thumbnails_created']}")
        logger.info(f"Errors: {stats['errors']}")

        if args.dry_run:
            logger.info("")
            logger.info("=== DRY-RUN MODE (no thumbnails were actually created) ===")

        logger.info("=" * 60)
        logger.info("Operation completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
