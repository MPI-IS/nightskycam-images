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
import numpy as np
import tomli
import tomli_w
import typer

from nightskycam_scorer.model.infer import SkyScorer
from nightskycam_scorer.utils import to_float_image

from .annotator_webapp import create_app as create_annotator_app
from .constants import IMAGE_FILE_FORMATS, THUMBNAIL_DIR_NAME, VIDEO_FILE_NAME
from .convert_npy import to_npy
from .patches import load_image_and_extract_patches, save_patches_from_folder
from .stats import generate_stats_report
from .thumbnail import create_missing_thumbnails, _thumbnail_path
from .video import VideoFormat, create_video
from .view_webapp import create_app as create_view_app
from .walk import (
    _create_symlink_safe,
    _get_images_from_hd,
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
    """List the thumbnails folders."""
    app = typer.Typer(help="List the thumbnails folders")

    @app.command()
    def run(
        folder_path: Path = typer.Argument(
            ...,
            help="Path to the folder",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ) -> None:
        """List the thumbnails folders in the specified directory."""
        for tp in walk_thumbnails(folder_path):
            typer.echo(f"{tp}")

    app()


def stats():
    """CLI entry point for generating statistics reports."""
    app = typer.Typer(help="Generate statistics report for nightskycam images")

    @app.command()
    def run(
        folder_path: Path = typer.Argument(
            ...,
            help="Path to the media root directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ) -> None:
        """Generate statistics report for nightskycam images."""
        try:
            generate_stats_report(folder_path)
        except Exception as e:
            typer.echo(f"Error generating statistics: {e}", err=True)
            raise typer.Exit(code=1)

    app()


def view_webapp():
    """CLI entry point for the image viewer web application."""
    app = typer.Typer(help="Start the Nightskycam image viewer web application")

    @app.command()
    def run(
        root_dir: Path = typer.Argument(
            ...,
            help="Path to the root directory (original or filtered structure)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        host: str = typer.Option(
            "127.0.0.1",
            "--host",
            help="Host to bind to",
        ),
        port: int = typer.Option(
            5002,
            "--port",
            help="Port to bind to",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug mode",
        ),
    ) -> None:
        """Start the Nightskycam image viewer web application."""
        # Import the Flask app from view_webapp module
        flask_app = create_view_app(root_dir)

        typer.echo(f"Starting Nightskycam Image Viewer on http://{host}:{port}")
        typer.echo(f"Root directory: {root_dir}")
        typer.echo("Press Ctrl+C to stop")

        try:
            flask_app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            typer.echo("\nShutting down...")
            raise typer.Exit(code=0)

    app()


def annotator_webapp():
    """CLI entry point for the thumbnail annotator web application."""
    app = typer.Typer(help="Start the Nightskycam thumbnail annotator web application")

    @app.command()
    def run(
        config_path: Optional[Path] = typer.Argument(
            None,
            help="Path to TOML configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        create_config: bool = typer.Option(
            False,
            "--create-config",
            help="Create a default configuration file and exit",
        ),
        host: str = typer.Option(
            "127.0.0.1",
            "--host",
            help="Host to bind to",
        ),
        port: int = typer.Option(
            5003,
            "--port",
            help="Port to bind to",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug mode",
        ),
    ) -> None:
        """Start the Nightskycam thumbnail annotator web application."""
        # Handle --create-config flag
        if create_config:
            output_path = Path.cwd() / "nightskycam_thumbnails_annotator_config.toml"
            if output_path.exists():
                typer.echo(
                    f"Error: Configuration file already exists at {output_path}",
                    err=True,
                )
                raise typer.Exit(code=1)

            try:
                default_config = {
                    "root_dir": "/path/to/media/root",
                    "output_dir": "/path/to/output",
                    "systems": [],  # Empty list means all systems
                    "start_date": "",  # Empty means no start limit (format: YYYY-MM-DD)
                    "end_date": "",  # Empty means no end limit (format: YYYY-MM-DD)
                }
                _save_config(default_config, output_path)
                typer.echo(f"Created default configuration file: {output_path}")
                typer.echo("Edit this file with your desired settings, then run:")
                typer.echo(f"  nightskycam-thumbnails-annotator-webapp {output_path}")
            except Exception as e:
                typer.echo(f"Error creating configuration file: {e}", err=True)
                raise typer.Exit(code=1)

            raise typer.Exit(code=0)

        # Require config_path if not creating config
        if config_path is None:
            typer.echo(
                "Error: You must provide a config file path or use --create-config\n\n"
                "Usage:\n"
                "  nightskycam-thumbnails-annotator-webapp --create-config       # Create default config\n"
                "  nightskycam-thumbnails-annotator-webapp <config.toml>         # Run with config file",
                err=True,
            )
            raise typer.Exit(code=1)

        # Import the Flask app from annotator_webapp module
        try:
            flask_app = create_annotator_app(config_path)
        except Exception as e:
            typer.echo(f"Error loading configuration: {e}", err=True)
            raise typer.Exit(code=1)

        typer.echo(f"Starting Nightskycam Image Annotator on http://{host}:{port}")
        typer.echo(f"Configuration file: {config_path}")
        typer.echo("Press Ctrl+C to stop")

        try:
            flask_app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            typer.echo("\nShutting down...")
            raise typer.Exit(code=0)

    app()


def _list_images(current: Path) -> list[Path]:
    images: list[Path] = []
    for format_ in image_formats:
        images.extend(list(current.glob(f"*.{format_}")))
    return images


def save_patches() -> None:
    """CLI entry point to save patches from a file or folder."""
    app = typer.Typer(help="Save patches from image or folder")

    @app.command()
    def run(
        path: Path = typer.Argument(
            ...,
            help="Path to an image file or a folder containing images",
            exists=True,
            resolve_path=True,
        ),
        output_dir: Path = typer.Argument(
            ...,
            help="Directory to write patches into",
            resolve_path=True,
        ),
        margin: int = typer.Option(
            0,
            "-m",
            "--margin",
            help="Pixels to trim from each border",
        ),
        patch_size: int = typer.Option(
            256,
            "-s",
            "--patch-size",
            help="Side length of square patches",
        ),
        overlap: int = typer.Option(
            0,
            "-l",
            "--overlap",
            help="Overlap in pixels between neighboring patches",
        ),
        overwrite: bool = typer.Option(
            False,
            "--overwrite",
            help="Overwrite existing output files if present",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            help="Enable verbose debug logging",
        ),
    ) -> None:
        """Save patches from image or folder.

        Usage examples:
          nightskycam-save-patches <path> <output_dir> -m 0 -s 256 -l 0 --overwrite
        """
        # Configure logging
        logger.remove()
        if verbose:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        output_dir.mkdir(parents=True, exist_ok=True)

        # If path is a file, process single image
        if path.is_file():
            logger.info("Processing single image: %s", path)
            try:
                patches = load_image_and_extract_patches(
                    path,
                    margin=margin,
                    patch_size=patch_size,
                    overlap=overlap,
                )
            except Exception as e:
                logger.exception("Failed to load and extract patches from %s: %s", path, e)
                raise typer.Exit(code=3)

            suffix = path.suffix or ".jpg"
            stem = path.stem
            n_saved = 0
            for i, patch in enumerate(patches):
                out_path = output_dir / f"{stem}_{i}{suffix}"
                if out_path.exists() and not overwrite:
                    logger.warning("Skipping existing file (overwrite=False): %s", out_path)
                    continue
                iio.imwrite(out_path, patch)
                n_saved += 1

            logger.info("Saved %d patches to %s", n_saved, output_dir)
            return

        # If path is directory, delegate to save_patches_from_folder
        if path.is_dir():
            logger.info("Processing folder: %s", path)
            try:
                counts = save_patches_from_folder(
                    input_folder=path,
                    output_folder=output_dir,
                    margin=margin,
                    patch_size=patch_size,
                    overlap=overlap,
                    overwrite=overwrite,
                )
            except Exception as e:
                logger.exception("Failed to save patches from folder %s: %s", path, e)
                raise typer.Exit(code=4)

            total = sum(counts.values())
            logger.info(
                "Saved %d patches for %d files to %s", total, len(counts), output_dir
            )
            return

        logger.error("Path is neither a file nor a directory: %s", path)
        raise typer.Exit(code=5)

    app()


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
    nb_images: Optional[int] = None
    folder_step: Optional[int] = None
    first_folder_step: Optional[int] = None


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

    if config.nb_images is not None:
        result["nb_images"] = config.nb_images

    if config.folder_step is not None:
        result["folder_step"] = config.folder_step

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


@dataclass
class _ModelConfig:
    """Configuration for a single model in the classifier."""

    model_path: Path
    probability_threshold: float
    classify_positive: bool


@dataclass
class _ClassifierConfig:
    """Configuration for multi-model classification."""

    root: Path
    filtered_dir: Path
    not_filtered_dir: Path
    models: List[_ModelConfig]
    systems: Optional[List[str]] = None
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None


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
        process_not_substring="bad_substring",
        cloud_cover_range=(0, 30),
        weather_values=["clear", "partly_cloudy"],
        cache_process_filter=False,
        nb_images=1000,
        folder_step=2,
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

    # Optional: nb_images
    nb_images = config.get("nb_images")
    if nb_images is not None:
        if not isinstance(nb_images, int):
            raise ValueError("'nb_images' must be an integer")
        if nb_images < 0:
            raise ValueError("'nb_images' must be non-negative")

    # Optional: folder_step
    folder_step = config.get("folder_step")
    if folder_step is not None:
        if not isinstance(folder_step, int):
            raise ValueError("'folder_step' must be an integer")
        if folder_step < 1:
            raise ValueError("'folder_step' must be a positive integer (>= 1)")

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
        nb_images=nb_images,
        folder_step=folder_step,
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
            except Exception as e:
                typer.echo(f"Error creating configuration file: {e}", err=True)
                raise typer.Exit(code=1)

            raise typer.Exit(code=0)

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
                nb_images=parsed.nb_images,
                folder_step=parsed.folder_step,
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
    """CLI entry point for removing selected images from filtered directory."""
    app = typer.Typer(help="Remove selected images from filtered directory")

    @app.command()
    def run(
        target_folder: Path = typer.Argument(
            ...,
            help="Path to filtered directory containing symlinks",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        list_file: Path = typer.Argument(
            ...,
            help="Path to text file with images to remove (system/date/filename format)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Show what would be removed without actually removing files",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            help="Enable verbose logging",
        ),
    ) -> None:
        """Remove selected images from filtered directory.

        Reads a text file with image paths and removes both the image symlinks
        and their corresponding TOML metadata files from the target directory.
        """
        # Configure logging
        logger.remove()
        if verbose:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        if dry_run:
            logger.info("=== DRY-RUN MODE ===")

        logger.info(f"Target folder: {target_folder}")
        logger.info(f"List file: {list_file}")

        try:
            stats = _remove_from_list(
                target_folder=target_folder,
                list_file=list_file,
                dry_run=dry_run,
                verbose=verbose,
            )

            logger.info("=== Removal Statistics ===")
            logger.info(f"Lines processed: {stats['lines_processed']}")
            logger.info(f"Images removed: {stats['images_removed']}")
            logger.info(f"TOML files removed: {stats['tomls_removed']}")
            logger.info(f"Errors: {stats['errors']}")

            if dry_run:
                logger.info("=== DRY-RUN MODE (no files were actually removed) ===")

            if stats["errors"] > 0:
                raise typer.Exit(code=1)

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise typer.Exit(code=1)

    app()


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
    """CLI entry point for moving filtered images to backup."""
    app = typer.Typer(help="Move original images to backup based on filter-export symlinks")

    @app.command()
    def run(
        filter_output_dir: Path = typer.Argument(
            ...,
            help="Path to filter-export directory containing symlinks",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        backup_dir: Path = typer.Argument(
            ...,
            help="Path to backup directory (will preserve system/date structure)",
            resolve_path=True,
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Preview operations without actually moving/deleting files",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            help="Enable verbose logging",
        ),
    ) -> None:
        """Move original images to backup based on filter-export symlinks.

        Moves original images, TOMLs, and thumbnails to backup directory based on symlinks
        in filter-export directory. Preserves system/date/thumbnails structure and cleans up empty folders.
        """
        # Configure logging
        logger.remove()
        if verbose:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        if dry_run:
            logger.info("=== DRY-RUN MODE ===")

        logger.info(f"Filter-export directory: {filter_output_dir}")
        logger.info(f"Backup directory: {backup_dir}")

        try:
            stats = {
                "symlinks_processed": 0,
                "images_moved": 0,
                "tomls_moved": 0,
                "thumbnails_moved": 0,
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

                            if verbose:
                                logger.info(f"Processing: {original_image}")

                            # Calculate backup destination (preserve tree structure)
                            relative_path = original_image.relative_to(original_root)
                            backup_dest = backup_dir / relative_path

                            # Move image file
                            _move_file_safe(original_image, backup_dest, dry_run)
                            stats["images_moved"] += 1

                            # Move TOML metadata file
                            original_toml = original_image.with_suffix(".toml")
                            if original_toml.exists():
                                backup_toml = backup_dest.with_suffix(".toml")
                                _move_file_safe(original_toml, backup_toml, dry_run)
                                stats["tomls_moved"] += 1
                            else:
                                logger.warning(f"TOML not found: {original_toml}")

                            # Move thumbnail
                            thumbnail_path = _get_thumbnail_path_from_image(original_image)
                            if thumbnail_path.exists():
                                # Calculate backup destination for thumbnail preserving structure
                                relative_thumbnail_path = thumbnail_path.relative_to(original_root)
                                backup_thumbnail = backup_dir / relative_thumbnail_path
                                _move_file_safe(thumbnail_path, backup_thumbnail, dry_run)
                                stats["thumbnails_moved"] += 1

                        except Exception as e:
                            logger.error(f"Error processing {item}: {e}")
                            raise  # Fail-fast

            # Cleanup empty directories
            if original_root is not None:
                logger.info("Cleaning up empty directories...")
                cleanup_stats = _cleanup_empty_directories(
                    original_root,
                    dry_run=dry_run,
                    verbose=verbose,
                )
                stats.update(cleanup_stats)
            else:
                logger.warning("No symlinks found, nothing to clean up")

            # Print statistics
            logger.info("=== Operation Statistics ===")
            logger.info(f"Symlinks processed: {stats['symlinks_processed']}")
            logger.info(f"Images moved: {stats['images_moved']}")
            logger.info(f"TOMLs moved: {stats['tomls_moved']}")
            logger.info(f"Thumbnails moved: {stats['thumbnails_moved']}")
            logger.info(
                f"Thumbnail folders deleted: {stats.get('thumbnail_folders_deleted', 0)}"
            )
            logger.info(f"Date folders deleted: {stats.get('date_folders_deleted', 0)}")
            logger.info(f"System folders deleted: {stats.get('system_folders_deleted', 0)}")

            if dry_run:
                logger.info("=== DRY-RUN MODE (no files were actually moved/deleted) ===")

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise typer.Exit(code=1)

    app()


def delete_from_other_root() -> None:
    """CLI entry point for deleting images from another root based on filter-export symlinks."""
    app = typer.Typer(
        help="Delete images from another root directory based on filter-export symlinks"
    )

    @app.command()
    def run(
        filter_output_dir: Path = typer.Argument(
            ...,
            help="Path to filter-export directory containing symlinks",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        other_root_dir: Path = typer.Argument(
            ...,
            help="Path to other root directory from which to delete matching images",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Preview operations without actually deleting files",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            help="Enable verbose logging",
        ),
        yes: bool = typer.Option(
            False,
            "--yes",
            "-y",
            help="Skip confirmation prompt and proceed with deletion",
        ),
    ) -> None:
        """Delete images from another root directory based on filter-export symlinks.

        This command walks through a directory of symlinks (created by nightskycam-filter-export)
        and deletes corresponding images from a different root directory if they exist.

        IMPORTANT: This ONLY deletes files from 'other_root_dir', never the symlinks themselves.

        For each symlink in filter_output_dir:
        1. Extracts the relative path (system/date/filename)
        2. Looks for the corresponding file in other_root_dir
        3. If found, deletes the image, TOML, and thumbnail from other_root_dir
        4. Cleans up empty directories in other_root_dir

        WARNING: This operation is destructive and cannot be undone!
        """
        # Configure logging
        logger.remove()
        if verbose:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        if dry_run:
            logger.info("=== DRY-RUN MODE ===")

        logger.info(f"Filter-export directory: {filter_output_dir}")
        logger.info(f"Other root directory (target for deletion): {other_root_dir}")

        # First pass: count how many files would be affected
        try:
            preview_stats = _preview_deletion_from_other_root(
                filter_output_dir=filter_output_dir,
                other_root_dir=other_root_dir,
            )

            # Display preview
            logger.warning("=" * 70)
            logger.warning("DELETION PREVIEW")
            logger.warning("=" * 70)
            logger.warning(f"Symlinks to process: {preview_stats['symlinks_found']}")
            logger.warning(
                f"Images that will be DELETED from {other_root_dir}:"
            )
            logger.warning(f"  - HD images: {preview_stats['images_found']}")
            logger.warning(f"  - TOML files: {preview_stats['tomls_found']}")
            logger.warning(f"  - Thumbnails: {preview_stats['thumbnails_found']}")
            logger.warning(
                f"Images not found in other root: {preview_stats['not_found']}"
            )
            logger.warning("=" * 70)
            logger.warning("WARNING: This operation is DESTRUCTIVE and CANNOT BE UNDONE!")
            logger.warning("=" * 70)

            # Ask for confirmation unless --yes flag is provided
            if not yes and not dry_run:
                response = typer.prompt(
                    "\nDo you want to proceed with deletion? Type 'DELETE' to confirm",
                    type=str,
                )
                if response != "DELETE":
                    logger.info("Deletion cancelled by user")
                    raise typer.Exit(code=0)

            # Proceed with deletion
            logger.info("Starting deletion process...")

            stats = _delete_images_from_other_root(
                filter_output_dir=filter_output_dir,
                other_root_dir=other_root_dir,
                dry_run=dry_run,
                verbose=verbose,
            )

            # Print statistics
            logger.info("=" * 70)
            logger.info("DELETION STATISTICS")
            logger.info("=" * 70)
            logger.info(f"Symlinks processed: {stats['symlinks_processed']}")
            logger.info(f"Images deleted: {stats['images_deleted']}")
            logger.info(f"TOMLs deleted: {stats['tomls_deleted']}")
            logger.info(f"Thumbnails deleted: {stats['thumbnails_deleted']}")
            logger.info(f"Images not found: {stats['not_found']}")
            logger.info(
                f"Thumbnail folders deleted: {stats.get('thumbnail_folders_deleted', 0)}"
            )
            logger.info(f"Date folders deleted: {stats.get('date_folders_deleted', 0)}")
            logger.info(
                f"System folders deleted: {stats.get('system_folders_deleted', 0)}"
            )
            logger.info("=" * 70)

            if dry_run:
                logger.info("=== DRY-RUN MODE (no files were actually deleted) ===")
            else:
                logger.info("Deletion completed successfully!")

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise typer.Exit(code=1)

    app()


def _preview_deletion_from_other_root(
    filter_output_dir: Path,
    other_root_dir: Path,
) -> Dict[str, int]:
    """
    Preview what would be deleted from other_root_dir.

    Returns statistics about files that would be affected.
    """
    stats = {
        "symlinks_found": 0,
        "images_found": 0,
        "tomls_found": 0,
        "thumbnails_found": 0,
        "not_found": 0,
    }

    processed_files: Set[str] = set()

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

                stats["symlinks_found"] += 1

                # Extract relative path structure from symlink location
                relative_path = f"{system_name}/{date_name}/{item.name}"

                # Skip if already processed (avoid duplicates)
                if relative_path in processed_files:
                    continue
                processed_files.add(relative_path)

                # Construct path in other root
                other_image_path = other_root_dir / system_name / date_name / item.name

                # Check if image exists
                if other_image_path.exists():
                    stats["images_found"] += 1

                    # Check for TOML
                    other_toml_path = other_image_path.with_suffix(".toml")
                    if other_toml_path.exists():
                        stats["tomls_found"] += 1

                    # Check for thumbnail
                    other_thumbnail_path = _get_thumbnail_path_from_image(other_image_path)
                    if other_thumbnail_path.exists():
                        stats["thumbnails_found"] += 1
                else:
                    stats["not_found"] += 1

    return stats


def _delete_images_from_other_root(
    filter_output_dir: Path,
    other_root_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Delete images from other_root_dir based on symlinks in filter_output_dir.

    Parameters
    ----------
    filter_output_dir
        Directory containing symlinks (from nightskycam-filter-export).
    other_root_dir
        Root directory from which to delete matching images.
    dry_run
        If True, don't actually delete files.
    verbose
        If True, log detailed progress.

    Returns
    -------
    dict
        Statistics about deletion operations.
    """
    stats = {
        "symlinks_processed": 0,
        "images_deleted": 0,
        "tomls_deleted": 0,
        "thumbnails_deleted": 0,
        "not_found": 0,
    }

    processed_files: Set[str] = set()

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
                    # Extract relative path structure from symlink location
                    relative_path = f"{system_name}/{date_name}/{item.name}"

                    # Skip if already processed (avoid duplicates)
                    if relative_path in processed_files:
                        logger.debug(f"Already processed: {relative_path}")
                        continue
                    processed_files.add(relative_path)

                    # Construct path in other root
                    other_image_path = (
                        other_root_dir / system_name / date_name / item.name
                    )

                    if verbose:
                        logger.info(f"Processing: {relative_path}")

                    # Check if image exists in other root
                    if not other_image_path.exists():
                        stats["not_found"] += 1
                        logger.debug(f"Not found in other root: {relative_path}")
                        continue

                    # Delete image file
                    _delete_path_safe(other_image_path, dry_run)
                    stats["images_deleted"] += 1

                    # Delete TOML metadata file
                    other_toml_path = other_image_path.with_suffix(".toml")
                    if other_toml_path.exists():
                        _delete_path_safe(other_toml_path, dry_run)
                        stats["tomls_deleted"] += 1
                    else:
                        logger.debug(f"TOML not found: {other_toml_path}")

                    # Delete thumbnail
                    other_thumbnail_path = _get_thumbnail_path_from_image(
                        other_image_path
                    )
                    if other_thumbnail_path.exists():
                        _delete_path_safe(other_thumbnail_path, dry_run)
                        stats["thumbnails_deleted"] += 1

                except Exception as e:
                    logger.error(f"Error processing {item}: {e}")
                    raise  # Fail-fast

    # Cleanup empty directories in other_root_dir
    logger.info("Cleaning up empty directories in other root...")
    cleanup_stats = _cleanup_empty_directories(
        other_root_dir,
        dry_run=dry_run,
        verbose=verbose,
    )
    stats.update(cleanup_stats)

    return stats


def move_clear_images_cli():
    """CLI entry point for moving clear-weather night images."""
    app = typer.Typer(help="Move clear-weather night images to target directory")

    @app.command()
    def run(
        root_dir: Path = typer.Argument(
            ...,
            help="Source media root directory (nightskycam structure)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        target_dir: Path = typer.Argument(
            ...,
            help="Target directory (will preserve system/date structure)",
            resolve_path=True,
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Preview moves without actually moving files",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            help="Enable verbose debug logging",
        ),
    ) -> None:
        """Move clear-weather night images to target directory.

        Moves images with clear weather (22:00-04:00) to target directory.
        Filters by weather field containing "clear" and time between 22:00-04:00.
        """
        # Configure logging
        logger.remove()
        if verbose:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        try:
            # Run move operation
            logger.info("=" * 60)
            logger.info("nightskycam-move-clear-images")
            logger.info("=" * 60)
            logger.info(f"Source: {root_dir}")
            logger.info(f"Target: {target_dir}")
            logger.info(f"Filters: weather contains 'clear' AND time 22:00-04:00")
            logger.info("=" * 60)

            stats = move_clear_images(
                root=root_dir,
                target_dir=target_dir,
                dry_run=dry_run,
                verbose=verbose,
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

            if not dry_run:
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

            if dry_run:
                logger.info("")
                logger.info("=== DRY-RUN MODE (no files were actually moved) ===")

            logger.info("=" * 60)
            logger.info("Operation completed successfully!")

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            if verbose:
                traceback.print_exc()
            raise typer.Exit(code=1)

    app()


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


# ============================================================================
# Multi-Model Classifier Command
# ============================================================================


class _ModelInferenceEngine:
    """Wrapper for a loaded model with its configuration."""

    def __init__(
        self, model_path: Path, threshold: float, classify_positive: bool
    ) -> None:
        """
        Initialize the inference engine with a model and its parameters.

        Parameters
        ----------
        model_path
            Path to the trained model file (.pt)
        threshold
            Probability threshold for classification (0.0-1.0)
        classify_positive
            If True, classify images with probability >= threshold as positive
            If False, classify images with probability < threshold as positive
        """
        self.scorer = SkyScorer(str(model_path), validate_size=False)
        self.threshold = threshold
        self.classify_positive = classify_positive
        self.model_name = model_path.stem

    def should_filter(self, thumbnail_array: np.ndarray) -> bool:
        """
        Run inference and return whether image should be filtered.

        Parameters
        ----------
        thumbnail_array
            Numpy array of thumbnail image

        Returns
        -------
        bool
            True if image should be filtered (matches criteria), False otherwise
        """
        rgb_float = to_float_image(thumbnail_array)
        result_raw = self.scorer.predict(rgb_float)
        result = result_raw[0] if isinstance(result_raw, list) else result_raw

        if self.classify_positive:
            return result.probability >= self.threshold
        else:
            return result.probability < self.threshold


def _copy_image_files(
    image: Any,
    output_dir: Path,
    system_name: str,
    date_str: str,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    """
    Copy HD image, TOML, and thumbnail preserving directory structure.

    Parameters
    ----------
    image
        Image object with .hd, .meta_path, and .thumbnail properties
    output_dir
        Base output directory (filtered_dir or not_filtered_dir)
    system_name
        Name of the system (e.g., "nightskycam5")
    date_str
        Date string in YYYY_MM_DD format
    dry_run
        If True, don't actually copy files

    Returns
    -------
    tuple
        (hd_copied, toml_copied, thumbnail_copied) counts (0 or 1 each)
    """
    # Create output structure: output_dir/system/date/
    output_date_dir = output_dir / system_name / date_str
    output_thumbnail_dir = output_date_dir / THUMBNAIL_DIR_NAME

    if not dry_run:
        output_date_dir.mkdir(parents=True, exist_ok=True)
        output_thumbnail_dir.mkdir(parents=True, exist_ok=True)

    counts = [0, 0, 0]

    # Copy HD image
    if image.hd and image.hd.exists():
        dest = output_date_dir / image.hd.name
        if not dest.exists():
            if not dry_run:
                shutil.copy2(image.hd, dest)
            counts[0] = 1

    # Copy TOML metadata
    if image.meta_path and image.meta_path.exists():
        dest = output_date_dir / image.meta_path.name
        if not dest.exists():
            if not dry_run:
                shutil.copy2(image.meta_path, dest)
            counts[1] = 1

    # Copy thumbnail
    if image.thumbnail and image.thumbnail.exists():
        dest = output_thumbnail_dir / image.thumbnail.name
        if not dest.exists():
            if not dry_run:
                shutil.copy2(image.thumbnail, dest)
            counts[2] = 1

    return tuple(counts)


def _copy_videos_for_dates(
    date_tracking: Dict[Tuple[str, str], bool],
    output_dir: Path,
    root: Path,
    dry_run: bool = False,
) -> int:
    """
    Copy day_summary.webm videos for dates that have classified images.

    Parameters
    ----------
    date_tracking
        Dict mapping (system, date_str) to True for dates with images
    output_dir
        Target directory (filtered_dir or not_filtered_dir)
    root
        Source root directory
    dry_run
        If True, don't actually copy files

    Returns
    -------
    int
        Number of videos copied
    """
    videos_copied = 0

    for (system_name, date_str), _ in date_tracking.items():
        # Source video path: root/system/date/thumbnails/day_summary.webm
        source_video = root / system_name / date_str / THUMBNAIL_DIR_NAME / VIDEO_FILE_NAME

        if not source_video.exists():
            logger.debug(f"No video found for {system_name}/{date_str}")
            continue

        # Destination: output_dir/system/date/thumbnails/day_summary.webm
        dest_video = (
            output_dir / system_name / date_str / THUMBNAIL_DIR_NAME / VIDEO_FILE_NAME
        )

        # Create parent directory
        if not dry_run:
            dest_video.parent.mkdir(parents=True, exist_ok=True)

        # Copy video (skip if already exists - from earlier in same run)
        if not dest_video.exists():
            if not dry_run:
                shutil.copy2(source_video, dest_video)
                logger.debug(f"Copied video: {source_video} -> {dest_video}")
            videos_copied += 1

    return videos_copied


def _validate_model_paths(models: List[_ModelConfig]) -> None:
    """
    Validate that all model files exist before processing.

    Parameters
    ----------
    models
        List of model configurations

    Raises
    ------
    ValueError
        If any model file does not exist
    """
    missing = []
    for i, model in enumerate(models):
        if not model.model_path.exists() or not model.model_path.is_file():
            missing.append(f"Model {i}: {model.model_path}")

    if missing:
        error_msg = "The following model files do not exist:\n" + "\n".join(missing)
        raise ValueError(error_msg)


def _get_default_classifier_config() -> Dict[str, Any]:
    """
    Create a default classifier configuration template.

    Returns
    -------
    dict
        Default configuration dictionary
    """
    return {
        "root": "/path/to/media/root",
        "filtered_dir": "/path/to/filtered/output",
        "not_filtered_dir": "/path/to/not_filtered/output",
        "systems": ["nightskycam5", "nightskycam7"],
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "models": [
            {
                "model_path": "/path/to/model1.pt",
                "probability_threshold": 0.7,
                "classify_positive": True,
            },
            {
                "model_path": "/path/to/model2.pt",
                "probability_threshold": 0.5,
                "classify_positive": False,
            },
        ],
    }


def _parse_classifier_config(config: Dict[str, Any]) -> _ClassifierConfig:
    """
    Parse and validate classifier configuration values.

    Parameters
    ----------
    config
        Configuration dictionary loaded from TOML

    Returns
    -------
    _ClassifierConfig
        Parsed configuration object

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    # Required fields
    required = ["root", "filtered_dir", "not_filtered_dir", "models"]
    for field in required:
        if field not in config:
            raise ValueError(f"Configuration must include '{field}' field")

    root = Path(config["root"])
    filtered_dir = Path(config["filtered_dir"])
    not_filtered_dir = Path(config["not_filtered_dir"])

    # Validate root exists
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root}")

    # Parse models array
    models_data = config["models"]
    if not isinstance(models_data, list) or len(models_data) == 0:
        raise ValueError("'models' must be a non-empty array")

    models = []
    for i, model_data in enumerate(models_data):
        if not isinstance(model_data, dict):
            raise ValueError(f"Model {i} must be a table/dictionary")

        if "model_path" not in model_data:
            raise ValueError(f"Model {i} missing 'model_path'")

        model_path = Path(model_data["model_path"])
        threshold = model_data.get("probability_threshold", 0.5)
        classify_positive = model_data.get("classify_positive", True)

        # Validate threshold
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Model {i} threshold must be a number")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Model {i} threshold must be between 0.0 and 1.0")

        # Validate classify_positive
        if not isinstance(classify_positive, bool):
            raise ValueError(f"Model {i} classify_positive must be boolean")

        models.append(_ModelConfig(model_path, threshold, classify_positive))

    # Optional fields (systems, dates)
    systems = config.get("systems", None)
    if systems is not None and not isinstance(systems, list):
        raise ValueError("'systems' must be a list of strings")

    start_date = None
    end_date = None

    if "start_date" in config and config["start_date"]:
        start_date = dt.datetime.strptime(config["start_date"], "%Y-%m-%d").date()

    if "end_date" in config and config["end_date"]:
        end_date = dt.datetime.strptime(config["end_date"], "%Y-%m-%d").date()

    return _ClassifierConfig(
        root=root,
        filtered_dir=filtered_dir,
        not_filtered_dir=not_filtered_dir,
        models=models,
        systems=systems,
        start_date=start_date,
        end_date=end_date,
    )


def scorer_classifier() -> None:
    """
    CLI tool to classify nightskycam images using multiple trained classifier models.

    This tool uses multiple nightskycam-scorer classifiers to predict image quality
    and copies images (with metadata and thumbnails) to filtered/not-filtered directories.
    If ANY model returns true, the image is copied to filtered_dir, otherwise to not_filtered_dir.
    """
    app = typer.Typer(
        help="Classify nightskycam images using multiple trained classifier models."
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
            help="Create a default configuration file named 'nightskycam_classifier_config.toml' in the current directory.",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Preview operations without copying files.",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug logging (shows detailed inference operations).",
        ),
    ) -> None:
        """
        Classify images using multiple models and copy to filtered/not-filtered directories.
        """
        # Configure logging level
        logger.remove()
        if debug:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        # Handle --create-config
        if create_config:
            default_config = _get_default_classifier_config()
            config_file_path = Path("nightskycam_classifier_config.toml")

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
            parsed = _parse_classifier_config(config)

            root = parsed.root
            filtered_dir = parsed.filtered_dir
            not_filtered_dir = parsed.not_filtered_dir
            models = parsed.models
            systems = parsed.systems
            start_date = parsed.start_date
            end_date = parsed.end_date

            logger.info(f"Root directory: {root}")
            logger.info(f"Filtered output directory: {filtered_dir}")
            logger.info(f"Not-filtered output directory: {not_filtered_dir}")
            logger.info(f"Number of models: {len(models)}")

            if systems:
                logger.info(f"Systems filter: {systems}")
            if start_date or end_date:
                logger.info(f"Date range: {start_date or 'any'} to {end_date or 'any'}")

            # Validate all model paths before loading
            logger.info("Validating model paths...")
            _validate_model_paths(models)
            logger.info("All model paths validated successfully")

            # Load all models
            logger.info(f"Loading {len(models)} model(s)...")
            model_engines = []
            for i, model_config in enumerate(models):
                logger.info(
                    f"  Loading model {i+1}/{len(models)}: {model_config.model_path.name}"
                )
                try:
                    engine = _ModelInferenceEngine(
                        model_config.model_path,
                        model_config.probability_threshold,
                        model_config.classify_positive,
                    )
                    model_engines.append(engine)
                    logger.info(
                        f"    Threshold: {model_config.probability_threshold}, "
                        f"Classify positive: {model_config.classify_positive}"
                    )
                except Exception as e:
                    logger.error(f"Failed to load model {i+1}: {e}")
                    sys.exit(1)

            logger.info("All models loaded successfully")

            # Create output directories
            if not dry_run:
                filtered_dir.mkdir(parents=True, exist_ok=True)
                not_filtered_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Output directories ready")

            if dry_run:
                logger.info("[DRY-RUN MODE] No files will be copied")

            # Statistics
            stats = {
                "total_images_scanned": 0,
                "images_with_thumbnails": 0,
                "images_without_thumbnails": 0,
                "images_filtered": 0,
                "images_not_filtered": 0,
                "hd_files_copied": 0,
                "toml_files_copied": 0,
                "thumbnails_copied": 0,
                "videos_copied_to_filtered": 0,
                "videos_copied_to_not_filtered": 0,
                "inference_errors": 0,
                "copy_errors": 0,
            }

            # Date-level tracking for videos
            dates_with_filtered: Dict[Tuple[str, str], bool] = {}
            dates_with_not_filtered: Dict[Tuple[str, str], bool] = {}

            logger.info("Starting image classification...")

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
                        stats["copy_errors"] += 1
                        continue

                    if not images:
                        logger.debug(f"    No images found in {date_path}")
                        continue

                    logger.info(f"    Found {len(images)} images")

                    date_filtered = 0
                    date_not_filtered = 0

                    # Process each image
                    for image in images:
                        stats["total_images_scanned"] += 1

                        # Check thumbnail exists
                        if not image.thumbnail or not image.thumbnail.exists():
                            logger.debug(
                                f"    Skipping {image.filename_stem} (no thumbnail)"
                            )
                            stats["images_without_thumbnails"] += 1
                            continue

                        stats["images_with_thumbnails"] += 1

                        # Run inference with ALL models
                        try:
                            thumbnail_array = to_npy(image.thumbnail)

                            # Check if ANY model returns True (OR logic)
                            is_filtered = any(
                                engine.should_filter(thumbnail_array)
                                for engine in model_engines
                            )

                            if debug:
                                # Log per-model results
                                logger.debug(f"    Image: {image.filename_stem}")
                                for engine in model_engines:
                                    result = engine.should_filter(thumbnail_array)
                                    logger.debug(
                                        f"      {engine.model_name}: {result}"
                                    )
                                logger.debug(f"      Final decision: {'FILTERED' if is_filtered else 'NOT FILTERED'}")

                        except Exception as e:
                            logger.error(
                                f"    Inference error for {image.filename_stem}: {e}"
                            )
                            stats["inference_errors"] += 1
                            if debug:
                                raise
                            continue

                        # Copy files based on classification
                        try:
                            if is_filtered:
                                hd, toml, thumb = _copy_image_files(
                                    image, filtered_dir, system_name, date_str, dry_run
                                )
                                stats["images_filtered"] += 1
                                stats["hd_files_copied"] += hd
                                stats["toml_files_copied"] += toml
                                stats["thumbnails_copied"] += thumb
                                dates_with_filtered[(system_name, date_str)] = True
                                date_filtered += 1
                            else:
                                hd, toml, thumb = _copy_image_files(
                                    image,
                                    not_filtered_dir,
                                    system_name,
                                    date_str,
                                    dry_run,
                                )
                                stats["images_not_filtered"] += 1
                                stats["hd_files_copied"] += hd
                                stats["toml_files_copied"] += toml
                                stats["thumbnails_copied"] += thumb
                                dates_with_not_filtered[(system_name, date_str)] = True
                                date_not_filtered += 1

                        except Exception as e:
                            logger.error(
                                f"    Copy error for {image.filename_stem}: {e}"
                            )
                            stats["copy_errors"] += 1
                            if debug:
                                raise

                    logger.info(
                        f"    Date summary - Filtered: {date_filtered}, Not filtered: {date_not_filtered}"
                    )

            # Copy videos for tracked dates
            logger.info("Copying videos for dates with classified images...")
            stats["videos_copied_to_filtered"] = _copy_videos_for_dates(
                dates_with_filtered, filtered_dir, root, dry_run
            )
            stats["videos_copied_to_not_filtered"] = _copy_videos_for_dates(
                dates_with_not_filtered, not_filtered_dir, root, dry_run
            )

            # Print summary statistics
            logger.info("")
            logger.info("=== Classification Summary ===")
            logger.info(f"Total images scanned: {stats['total_images_scanned']}")
            logger.info(
                f"Images with thumbnails: {stats['images_with_thumbnails']}"
            )
            logger.info(
                f"Images without thumbnails: {stats['images_without_thumbnails']}"
            )
            logger.info(f"Images filtered: {stats['images_filtered']}")
            logger.info(f"Images not filtered: {stats['images_not_filtered']}")
            logger.info(f"HD files copied: {stats['hd_files_copied']}")
            logger.info(f"TOML files copied: {stats['toml_files_copied']}")
            logger.info(f"Thumbnails copied: {stats['thumbnails_copied']}")
            logger.info(
                f"Videos copied to filtered: {stats['videos_copied_to_filtered']}"
            )
            logger.info(
                f"Videos copied to not_filtered: {stats['videos_copied_to_not_filtered']}"
            )
            if stats["inference_errors"] > 0:
                logger.warning(f"Inference errors: {stats['inference_errors']}")
            if stats["copy_errors"] > 0:
                logger.warning(f"Copy errors: {stats['copy_errors']}")

            logger.info(f"Filtered directory: {filtered_dir}")
            logger.info(f"Not-filtered directory: {not_filtered_dir}")
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
    """CLI entry point for creating missing thumbnails."""
    app = typer.Typer(help="Create thumbnails for images that are missing them")

    @app.command()
    def run(
        root_dir: Path = typer.Argument(
            ...,
            help="Root directory with nightskycam structure (system/date/)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        width: int = typer.Option(
            200,
            "--width",
            help="Width of thumbnails in pixels",
        ),
        stretch: bool = typer.Option(
            True,
            "--stretch/--no-stretch",
            help="Apply auto-stretch to images",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Preview what would be created without actually creating thumbnails",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            help="Enable verbose logging",
        ),
    ) -> None:
        """Create thumbnails for images that are missing them.

        Creates thumbnails only for images that don't already have them.
        """
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(levelname)s: %(message)s",
        )

        try:
            logger.info("=" * 60)
            logger.info("nightskycam-create-missing-thumbnails")
            logger.info("=" * 60)
            logger.info(f"Root directory: {root_dir}")
            logger.info(f"Thumbnail width: {width} pixels")
            logger.info(f"Auto-stretch: {stretch}")
            if dry_run:
                logger.info("DRY-RUN MODE: No thumbnails will be created")
            logger.info("=" * 60)

            # Run the function
            stats = create_missing_thumbnails(
                root_dir=root_dir,
                thumbnail_width=width,
                stretch=stretch,
                dry_run=dry_run,
                verbose=verbose,
            )

            # Display statistics
            logger.info("=" * 60)
            logger.info("Operation Statistics")
            logger.info("=" * 60)
            logger.info(f"Images scanned: {stats['images_scanned']}")
            logger.info(f"Thumbnails missing: {stats['thumbnails_missing']}")
            logger.info(f"Thumbnails created: {stats['thumbnails_created']}")
            logger.info(f"Errors: {stats['errors']}")

            if dry_run:
                logger.info("")
                logger.info("=== DRY-RUN MODE (no thumbnails were actually created) ===")

            logger.info("=" * 60)
            logger.info("Operation completed successfully!")

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            if verbose:
                traceback.print_exc()
            raise typer.Exit(code=1)

    app()


def check_thumbnails():
    """CLI entry point for checking thumbnail completeness."""
    app = typer.Typer(help="Check if all full images have related thumbnails")

    @app.command()
    def run(
        root_dir: Path = typer.Argument(
            ...,
            help="Root directory with nightskycam structure (system/date/)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ) -> None:
        """Check if all full images have related thumbnails."""
        # Initialize data structure
        missing_by_folder: Dict[Path, List[Path]] = {}
        total_images = 0
        total_folders = 0

        # Walk directory structure
        for system_path in walk_systems(root_dir):
            for date, date_path in walk_dates(system_path):
                total_folders += 1

                # Show progress
                rel_path = date_path.relative_to(root_dir)
                typer.echo(f"Checking: {rel_path}")

                # Get all HD images in this date folder
                images = _get_images_from_hd(date_path)
                total_images += len(images)

                # Check each image for missing thumbnail
                missing_in_folder = []
                for image in images:
                    # Get HD image path
                    hd_path = image.hd
                    if hd_path is None:
                        continue

                    # Check if thumbnail exists
                    thumbnail_path = _thumbnail_path(hd_path)
                    if not thumbnail_path.exists():
                        missing_in_folder.append(hd_path)

                # Store if any missing
                if missing_in_folder:
                    missing_by_folder[date_path] = missing_in_folder

        # Section 1: List folders with missing thumbnails
        if missing_by_folder:
            typer.echo("Date folders with missing thumbnails:")
            for date_path in sorted(missing_by_folder.keys()):
                # Show relative path from root
                rel_path = date_path.relative_to(root_dir)
                typer.echo(f"  {rel_path}")

            typer.echo("")  # Blank line

            # Section 2: Detailed list by folder
            typer.echo("Missing thumbnails by folder:")
            typer.echo("")
            for date_path in sorted(missing_by_folder.keys()):
                rel_path = date_path.relative_to(root_dir)
                typer.echo(f"{rel_path}:")
                for hd_path in missing_by_folder[date_path]:
                    typer.echo(f"  {hd_path}")
                typer.echo("")

            # Section 3: Summary statistics
            total_missing = sum(len(images) for images in missing_by_folder.values())
            typer.echo("Summary:")
            typer.echo(f"  Total date folders checked: {total_folders}")
            typer.echo(f"  Date folders with missing thumbnails: {len(missing_by_folder)}")
            typer.echo(f"  Total images checked: {total_images}")
            typer.echo(f"  Total missing thumbnails: {total_missing}")
        else:
            typer.echo("All images have thumbnails!")

        # Always exit with code 0
        raise typer.Exit(code=0)

    app()


def symlink_annotator_webapp():
    """CLI entry point for the symlink annotator web application."""
    app = typer.Typer(help="Start the Nightskycam symlink annotator web application")

    @app.command()
    def run(
        filter_dir: Path = typer.Argument(
            ...,
            help="Path to filter-export directory (symlinks)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        output_dir: Path = typer.Argument(
            ...,
            help="Path to output directory (will create positive/negative subdirs)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
        host: str = typer.Option(
            "127.0.0.1",
            "--host",
            help="Host to bind to",
        ),
        port: int = typer.Option(
            5004,
            "--port",
            help="Port to bind to",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug mode",
        ),
    ) -> None:
        """Start the Nightskycam symlink annotator web application."""
        from .symlink_annotator_webapp import create_app as create_symlink_annotator_app

        # Validate filter_dir has symlink structure
        has_symlinks = False
        for system_dir in filter_dir.iterdir():
            if system_dir.is_dir():
                for date_dir in system_dir.iterdir():
                    if date_dir.is_dir():
                        for item in date_dir.iterdir():
                            if item.is_symlink():
                                has_symlinks = True
                                break
                        if has_symlinks:
                            break
                if has_symlinks:
                    break

        if not has_symlinks:
            typer.echo(
                f"Warning: No symlinks found in {filter_dir}. "
                "This tool is designed for filter-export directories.",
                err=True,
            )

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create Flask app
        flask_app = create_symlink_annotator_app(filter_dir, output_dir)

        typer.echo(f"Starting Nightskycam Symlink Annotator on http://{host}:{port}")
        typer.echo(f"Filter directory: {filter_dir}")
        typer.echo(f"Output directory: {output_dir}")
        typer.echo("  - Positive: {}/positive".format(output_dir))
        typer.echo("  - Negative: {}/negative".format(output_dir))
        typer.echo("Press Ctrl+C to stop")

        try:
            flask_app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            typer.echo("\nShutting down...")
            raise typer.Exit(code=0)

    app()


# ============================================================================
# Classify Images Command
# ============================================================================


@dataclass
class _ClassifyImagesConfig:
    """Configuration for classify-images command."""

    root: Path
    models: Dict[str, Path]  # classifier_name -> model_path
    systems: Optional[List[str]] = None
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None


def _classify_images_config_to_dict(config: _ClassifyImagesConfig) -> Dict[str, Any]:
    """Convert a _ClassifyImagesConfig to a dictionary for TOML serialization."""
    result: Dict[str, Any] = {}
    result["root"] = str(config.root)
    if config.systems is not None:
        result["systems"] = config.systems
    if config.start_date is not None:
        result["start_date"] = config.start_date.strftime("%Y-%m-%d")
    if config.end_date is not None:
        result["end_date"] = config.end_date.strftime("%Y-%m-%d")
    result["models"] = {name: str(path) for name, path in config.models.items()}
    return result


def _get_default_classify_images_config() -> Dict[str, Any]:
    """Return a default configuration dictionary for classify-images."""
    config = _ClassifyImagesConfig(
        root=Path("/path/to/media/root"),
        models={
            "quality": Path("/path/to/quality_model.pt"),
            "clouds": Path("/path/to/clouds_model.pt"),
        },
        systems=["nightskycam5"],
        start_date=dt.date(2025, 1, 1),
        end_date=dt.date(2025, 12, 31),
    )
    return _classify_images_config_to_dict(config)


def _parse_classify_images_config(config: Dict[str, Any]) -> _ClassifyImagesConfig:
    """Parse and validate classify-images configuration values."""
    # Required: root
    if "root" not in config:
        raise ValueError("Configuration must include 'root' field")
    root = Path(config["root"])
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root}")

    # Required: models
    if "models" not in config:
        raise ValueError("Configuration must include 'models' section")
    models_raw = config["models"]
    if not isinstance(models_raw, dict) or len(models_raw) == 0:
        raise ValueError(
            "'models' must be a non-empty table mapping names to .pt file paths"
        )

    models: Dict[str, Path] = {}
    for name, path_str in models_raw.items():
        if not isinstance(path_str, str):
            raise ValueError(f"Model '{name}' path must be a string")
        p = Path(path_str)
        if not p.exists() or not p.is_file():
            raise ValueError(f"Model file for '{name}' does not exist: {p}")
        models[name] = p

    # Optional: systems
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
            end_date = dt.datetime.strptime(config["end_date"], "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(
                f"Invalid end_date format '{config['end_date']}'. Expected YYYY-MM-DD."
            )

    return _ClassifyImagesConfig(
        root=root,
        models=models,
        systems=systems,
        start_date=start_date,
        end_date=end_date,
    )


def classify_images() -> None:
    """
    CLI tool to run multiple classifier models on nightskycam images
    and write prediction scores into TOML metadata files.
    """
    app = typer.Typer(
        help="Classify nightskycam images and write scores to TOML metadata."
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
            help="Create a default configuration file named 'nightskycam_classify_images_config.toml' in the current directory.",
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug logging.",
        ),
    ) -> None:
        """Run classifiers on images and write scores to metadata TOML files."""
        # Configure logging
        logger.remove()
        if debug:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")

        # Handle --create-config
        if create_config:
            default_config = _get_default_classify_images_config()
            config_file_path = Path("nightskycam_classify_images_config.toml")
            if config_file_path.exists():
                logger.error(
                    f"Configuration file already exists: {config_file_path}"
                )
                logger.info(
                    "Remove the existing file or specify a different name."
                )
                sys.exit(1)
            _save_config(default_config, config_file_path)
            logger.info(f"Created default configuration file: {config_file_path}")
            logger.info(
                "Please edit the file with your actual paths and settings, then run again."
            )
            sys.exit(0)

        # Require config_path
        if config_path is None:
            logger.error("No configuration file provided.")
            logger.info(
                "Use --create-config to generate a template, or provide a config file path."
            )
            sys.exit(1)

        try:
            # Load and parse config
            logger.info(f"Loading configuration from: {config_path}")
            config = _load_config(config_path)
            parsed = _parse_classify_images_config(config)

            root = parsed.root
            model_map = parsed.models
            systems = parsed.systems
            start_date = parsed.start_date
            end_date = parsed.end_date

            logger.info(f"Root directory: {root}")
            logger.info(f"Models: {', '.join(model_map.keys())}")
            if systems:
                logger.info(f"Systems filter: {systems}")
            if start_date or end_date:
                logger.info(
                    f"Date range: {start_date or 'any'} to {end_date or 'any'}"
                )

            # Load all models
            logger.info(f"Loading {len(model_map)} model(s)...")
            scorers: Dict[str, SkyScorer] = {}
            for name, model_path in model_map.items():
                logger.info(f"  Loading model '{name}': {model_path}")
                try:
                    scorers[name] = SkyScorer(str(model_path), validate_size=False)
                except Exception as e:
                    logger.error(f"Failed to load model '{name}': {e}")
                    sys.exit(1)
            logger.info("All models loaded successfully")

            # Statistics
            stats = {
                "total_scanned": 0,
                "total_with_thumbnails": 0,
                "total_classified": 0,
                "total_written": 0,
                "errors": 0,
            }

            logger.info("Starting image classification...")

            # Iterate through systems
            for system_path in walk_systems(root):
                system_name = system_path.name

                if systems is not None and system_name not in systems:
                    logger.debug(
                        f"Skipping system (not in filter): {system_name}"
                    )
                    continue

                logger.info(f"Processing system: {system_name}")

                for date_, date_path in walk_dates(system_path):
                    if not _is_within_date_range(date_, start_date, end_date):
                        logger.debug(f"Skipping date (out of range): {date_}")
                        continue

                    date_str = date_.strftime("%Y_%m_%d")
                    logger.info(f"  Processing date: {date_str}")

                    try:
                        images = get_images(date_path)
                    except PermissionError as e:
                        logger.error(f"    Permission denied: {e}")
                        stats["errors"] += 1
                        continue
                    except Exception as e:
                        logger.error(f"    Error reading date directory: {e}")
                        stats["errors"] += 1
                        if debug:
                            raise
                        continue

                    if not images:
                        logger.debug("    No images with thumbnails found")
                        continue

                    date_classified = 0

                    for image in images:
                        stats["total_scanned"] += 1

                        # Skip images without thumbnails
                        thumbnail_path = image.thumbnail
                        if thumbnail_path is None or not thumbnail_path.exists():
                            logger.debug(
                                f"    Skipping (no thumbnail): {image.filename_stem}"
                            )
                            continue

                        stats["total_with_thumbnails"] += 1

                        # Skip images without TOML metadata files
                        meta_path = image.meta_path
                        if meta_path is None:
                            logger.debug(
                                f"    Skipping (no TOML): {image.filename_stem}"
                            )
                            continue

                        try:
                            # Load thumbnail
                            thumbnail_array = to_npy(thumbnail_path)
                            rgb_float = to_float_image(thumbnail_array)

                            # Run all classifiers
                            classifier_scores: Dict[str, float] = {}
                            for name, scorer in scorers.items():
                                result_raw = scorer.predict(rgb_float)
                                result = (
                                    result_raw[0]
                                    if isinstance(result_raw, list)
                                    else result_raw
                                )
                                classifier_scores[name] = float(result.probability)
                                logger.debug(
                                    f"      {name}: probability={result.probability:.3f}"
                                )

                            stats["total_classified"] += 1

                            # Read existing TOML, update classifiers section, write back
                            with open(meta_path, "rb") as f:
                                meta_data = tomli.load(f)

                            meta_data["classifiers"] = classifier_scores

                            with open(meta_path, "wb") as f:
                                tomli_w.dump(meta_data, f)

                            stats["total_written"] += 1
                            date_classified += 1

                        except Exception as e:
                            logger.error(
                                f"    Error processing {image.filename_stem}: {e}"
                            )
                            stats["errors"] += 1
                            if debug:
                                raise

                    if date_classified > 0:
                        logger.info(
                            f"    Classified: {date_classified} images"
                        )

            # Summary
            logger.info("")
            logger.info("=== Classification Summary ===")
            logger.info(f"Total images scanned: {stats['total_scanned']}")
            logger.info(
                f"Images with thumbnails: {stats['total_with_thumbnails']}"
            )
            logger.info(f"Images classified: {stats['total_classified']}")
            logger.info(f"TOML files updated: {stats['total_written']}")
            if stats["errors"] > 0:
                logger.warning(f"Errors encountered: {stats['errors']}")
            logger.info("Done!")
            sys.exit(0)

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            if debug:
                raise
            sys.exit(1)

    app()
