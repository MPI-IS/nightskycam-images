"""
Statistics reporting for nightskycam images.

This module provides functionality to collect and display statistics
about nightskycam image collections, including counts per system,
weather distribution, cloud cover binning, and metadata coverage.
"""

from collections import Counter, defaultdict
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .constants import IMAGE_FILE_FORMATS, THUMBNAIL_DIR_NAME
from .image import Image
from .walk import get_images, parse_image_path, walk_dates, walk_systems


def _get_images_flexible(date_dir_path: Path) -> List[Image]:
    """
    Get images from date directory, handling both standard and filtered structures.

    Standard structure: HD images in date folder, thumbnails in thumbnails/ subfolder
    Filtered structure: Symlinks to HD images and TOML files directly in date folder

    Parameters
    ----------
    date_dir_path
        Path to date directory.

    Returns
    -------
    List[Image]
        List of Image instances found.
    """
    # First try the standard structure (with thumbnails subdirectory)
    images = get_images(date_dir_path)
    if images:
        return images

    # If no thumbnails found, look for HD images directly in the date folder
    # This handles filtered/exported data structure
    hd_files: List[Path] = []
    for ext in IMAGE_FILE_FORMATS:
        hd_files.extend(date_dir_path.glob(f"*.{ext}"))

    if not hd_files:
        return []

    # Create Image instances for HD files found directly
    images = []
    for hd_file in hd_files:
        try:
            system_name, datetime = parse_image_path(hd_file)

            # Create Image instance
            img = Image()
            img.filename_stem = hd_file.stem
            img.date_and_time = datetime
            img.system = system_name
            img.dir_path = date_dir_path

            images.append(img)
        except Exception as e:
            logger.debug(f"Could not parse image file {hd_file.name}: {e}")
            continue

    return images


def collect_statistics(root: Path) -> Dict[str, Any]:
    """
    Traverse all systems/dates and collect comprehensive statistics.

    Parameters
    ----------
    root
        Path to media root directory containing nightskycam systems.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - total_images: Total number of images found
        - systems: Dict mapping system name to system statistics
        - weather_distribution: Counter of weather values
        - cloud_cover_bins: List of 15 integers (counts per bin)
        - process_types: Counter of process field values
        - missing_metadata: Dict with counts of missing fields
    """
    stats: Dict[str, Any] = {
        "total_images": 0,
        "systems": {},
        "weather_distribution": Counter(),
        "cloud_cover_bins": [0] * 15,  # 15 bins: 0-6, 7-13, ..., 98-100
        "process_types": Counter(),
        "missing_metadata": {
            "no_process": 0,
            "no_cloud_cover": 0,
            "no_weather": 0,
            "no_toml": 0,
        },
    }

    logger.info(f"Scanning root directory: {root}")

    for system_path in walk_systems(root):
        system_name = system_path.name
        logger.info(f"Processing system: {system_name}")

        system_stats: Dict[str, Any] = {
            "image_count": 0,
            "date_range": None,
            "dates": set(),
        }

        for date_, date_path in walk_dates(system_path):
            images = _get_images_flexible(date_path)
            system_stats["image_count"] += len(images)
            system_stats["dates"].add(date_)

            for image in images:
                stats["total_images"] += 1

                # Check if TOML exists
                if image.meta_path is None or not image.meta_path.exists():
                    stats["missing_metadata"]["no_toml"] += 1
                    # Count as missing for all fields
                    stats["missing_metadata"]["no_process"] += 1
                    stats["missing_metadata"]["no_cloud_cover"] += 1
                    stats["missing_metadata"]["no_weather"] += 1
                    continue

                meta = image.meta  # Dict, empty if TOML parse failed

                # Process field
                if "process" in meta and meta["process"]:
                    stats["process_types"][meta["process"]] += 1
                else:
                    stats["missing_metadata"]["no_process"] += 1

                # Cloud cover
                if "cloud_cover" in meta:
                    try:
                        cc = int(meta["cloud_cover"])
                        # Calculate bin index: 0-6->0, 7-13->1, etc.
                        bin_index = min(cc // 7, 14)
                        stats["cloud_cover_bins"][bin_index] += 1
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid cloud_cover value in {image.meta_path}: {meta['cloud_cover']}"
                        )
                        stats["missing_metadata"]["no_cloud_cover"] += 1
                else:
                    stats["missing_metadata"]["no_cloud_cover"] += 1

                # Weather
                if "weather" in meta and meta["weather"]:
                    stats["weather_distribution"][meta["weather"]] += 1
                else:
                    stats["missing_metadata"]["no_weather"] += 1

        # Calculate date range
        if system_stats["dates"]:
            sorted_dates = sorted(system_stats["dates"])
            system_stats["date_range"] = (sorted_dates[0], sorted_dates[-1])

        stats["systems"][system_name] = system_stats
        logger.info(f"  Found {system_stats['image_count']} images")

    return stats


def display_statistics(stats: Dict[str, Any]) -> None:
    """
    Display formatted statistics report using rich.

    Parameters
    ----------
    stats
        Statistics dictionary returned by collect_statistics().
    """
    console = Console()

    # Overall summary panel
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Total Images:[/bold cyan] {stats['total_images']:,}\n"
            f"[bold cyan]Systems Found:[/bold cyan] {len(stats['systems'])}",
            title="[bold]NightSkyCam Statistics Report[/bold]",
            border_style="cyan",
        )
    )
    console.print()

    # Per-system table
    if stats["systems"]:
        system_table = Table(
            title="Images per System", show_header=True, header_style="bold cyan"
        )
        system_table.add_column("System", style="cyan", no_wrap=True)
        system_table.add_column("Images", justify="right", style="green")
        system_table.add_column("Date Range", style="yellow")

        for system_name, system_stats in sorted(stats["systems"].items()):
            date_range_str = "No images"
            if system_stats["date_range"]:
                start, end = system_stats["date_range"]
                date_range_str = f"{start} to {end}"
            system_table.add_row(
                system_name, f"{system_stats['image_count']:,}", date_range_str
            )
        console.print(system_table)
        console.print()

    # Weather distribution table
    if stats["weather_distribution"]:
        weather_table = Table(
            title="Weather Distribution", show_header=True, header_style="bold cyan"
        )
        weather_table.add_column("Weather", style="cyan")
        weather_table.add_column("Count", justify="right", style="green")
        weather_table.add_column("Percentage", justify="right", style="yellow")

        total_with_weather = sum(stats["weather_distribution"].values())
        for weather, count in stats["weather_distribution"].most_common():
            pct = (count / total_with_weather * 100) if total_with_weather else 0
            weather_table.add_row(weather, f"{count:,}", f"{pct:.1f}%")
        console.print(weather_table)
        console.print()

    # Cloud cover bins
    total_with_cloud_cover = sum(stats["cloud_cover_bins"])
    if total_with_cloud_cover > 0:
        cloud_table = Table(
            title="Cloud Cover Distribution (15 bins)",
            show_header=True,
            header_style="bold cyan",
        )
        cloud_table.add_column("Range (%)", style="cyan")
        cloud_table.add_column("Count", justify="right", style="green")
        cloud_table.add_column("Percentage", justify="right", style="yellow")

        bin_labels = [
            "0-6",
            "7-13",
            "14-20",
            "21-27",
            "28-34",
            "35-41",
            "42-48",
            "49-55",
            "56-62",
            "63-69",
            "70-76",
            "77-83",
            "84-90",
            "91-97",
            "98-100",
        ]
        for label, count in zip(bin_labels, stats["cloud_cover_bins"]):
            if count > 0:
                pct = count / total_with_cloud_cover * 100
                cloud_table.add_row(label, f"{count:,}", f"{pct:.1f}%")
        console.print(cloud_table)
        console.print()

    # Process types
    if stats["process_types"]:
        process_table = Table(
            title="Process Types", show_header=True, header_style="bold cyan"
        )
        process_table.add_column("Process", style="cyan")
        process_table.add_column("Count", justify="right", style="green")

        for process, count in stats["process_types"].most_common():
            process_table.add_row(process, f"{count:,}")
        console.print(process_table)
        console.print()

    # Missing metadata
    missing = stats["missing_metadata"]
    console.print(
        Panel.fit(
            f"[yellow]Missing TOML files:[/yellow] {missing['no_toml']:,}\n"
            f"[yellow]Missing process field:[/yellow] {missing['no_process']:,}\n"
            f"[yellow]Missing cloud_cover field:[/yellow] {missing['no_cloud_cover']:,}\n"
            f"[yellow]Missing weather field:[/yellow] {missing['no_weather']:,}",
            title="[bold]Missing Metadata[/bold]",
            border_style="yellow",
        )
    )
    console.print()


def generate_stats_report(root: Path) -> None:
    """
    Generate and display statistics report for nightskycam images.

    Parameters
    ----------
    root
        Path to media root directory containing nightskycam systems.
    """
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    stats = collect_statistics(root)
    display_statistics(stats)
