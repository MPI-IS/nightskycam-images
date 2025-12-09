"""Flask web application for annotating images with binary classification (+/-)."""

from datetime import datetime
from pathlib import Path
import random
import secrets
import shutil
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request, send_file
from loguru import logger
import tomli

from .constants import THUMBNAIL_DIR_NAME
from .walk import get_images, walk_dates, walk_systems


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    with open(config_path, "rb") as f:
        return tomli.load(f)


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string in YYYY-MM-DD format."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.warning(f"Invalid date format: {date_str}, expected YYYY-MM-DD")
        return None


def _get_valid_system_date_tuples(
    root_dir: Path,
    systems_filter: List[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> List[Tuple[str, str, Path]]:
    """
    Build list of valid (system, date_str, date_path) tuples matching criteria.

    Returns:
        List of tuples: (system_name, date_string, date_path)
    """
    valid_tuples = []

    # Get all systems or filter by config
    all_systems = walk_systems(root_dir)

    for system_path in all_systems:
        system_name = system_path.name

        # Filter by system name if specified
        if systems_filter and system_name not in systems_filter:
            continue

        # Iterate through dates in this system
        for date_obj, date_path in walk_dates(system_path):
            # Filter by date range
            if start_date and date_obj < start_date:
                continue
            if end_date and date_obj > end_date:
                continue

            # Check if this date has any images with thumbnails
            images = get_images(date_path)
            has_thumbnails = any(
                img.thumbnail and img.thumbnail.exists() for img in images
            )

            if has_thumbnails:
                date_str = date_obj.strftime("%Y_%m_%d")
                valid_tuples.append((system_name, date_str, date_path))

    return valid_tuples


def _get_random_thumbnails(
    valid_tuples: List[Tuple[str, str, Path]],
    count: int = 20,
) -> List[Dict[str, str]]:
    """
    Randomly select thumbnails from valid system/date combinations.

    For each thumbnail needed, randomly selects a (system, date) combination,
    then randomly picks one thumbnail from that date directory.

    Args:
        valid_tuples: List of (system_name, date_str, date_path) tuples
        count: Number of thumbnails to select

    Returns:
        List of dicts with thumbnail info
    """
    if not valid_tuples:
        return []

    selected_thumbnails = []

    # For each thumbnail we need, pick a random system/date and then a random image
    for _ in range(count):
        # Randomly select a (system, date) tuple
        system_name, date_str, date_path = random.choice(valid_tuples)

        # Get images for this date
        images = get_images(date_path)

        # Filter to only images with existing thumbnails
        images_with_thumbnails = [
            img for img in images if img.thumbnail and img.thumbnail.exists()
        ]

        if not images_with_thumbnails:
            # This shouldn't happen since we validated during startup, but handle it
            continue

        # Randomly pick ONE image from this date
        img = random.choice(images_with_thumbnails)

        # img.thumbnail is guaranteed to be not None because of the filter above
        assert img.thumbnail is not None
        thumbnail_filename = img.thumbnail.name

        selected_thumbnails.append(
            {
                "system": system_name,
                "date": date_str,
                "filename": thumbnail_filename,
                "thumbnail_url": f"/api/thumbnail/{system_name}/{date_str}/{thumbnail_filename}",
            }
        )

    return selected_thumbnails


def _count_images_in_dir(directory: Path) -> int:
    """Count number of JPEG images in a directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.jpeg")))


def _copy_thumbnail_to_folder(
    source_path: Path,
    dest_folder: Path,
    filename: str,
) -> bool:
    """
    Copy thumbnail to destination folder.

    Args:
        source_path: Path to source thumbnail
        dest_folder: Destination folder (positive or negative)
        filename: Filename for the copied thumbnail

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create destination folder if needed
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Destination path
        dest_path = dest_folder / filename

        # Copy the file
        shutil.copy2(source_path, dest_path)
        logger.debug(f"Copied thumbnail: {source_path} -> {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy thumbnail {source_path}: {e}")
        return False


def _remove_from_opposite_folder(
    filename: str,
    current_folder: Path,
    opposite_folder: Path,
) -> None:
    """Remove file from opposite classification folder if it exists."""
    opposite_path = opposite_folder / filename
    if opposite_path.exists():
        try:
            opposite_path.unlink()
            logger.debug(f"Removed from opposite folder: {opposite_path}")
        except Exception as e:
            logger.warning(
                f"Failed to remove from opposite folder {opposite_path}: {e}"
            )


def create_app(config_path: Path) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_path: Path to TOML configuration file

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)

    # Load configuration
    config = _load_config(config_path)
    root_dir = Path(config["root_dir"])
    output_dir = Path(config["output_dir"])
    systems_filter = config.get("systems", [])
    start_date = _parse_date(config.get("start_date", ""))
    end_date = _parse_date(config.get("end_date", ""))

    # Store in app config
    app.config["ROOT_DIR"] = root_dir
    app.config["OUTPUT_DIR"] = output_dir
    app.config["POSITIVE_DIR"] = output_dir / "positive"
    app.config["NEGATIVE_DIR"] = output_dir / "negative"

    # Build list of valid (system, date, path) tuples at startup
    logger.info("Building list of valid system/date combinations...")
    valid_tuples = _get_valid_system_date_tuples(
        root_dir, systems_filter, start_date, end_date
    )
    app.config["VALID_TUPLES"] = valid_tuples
    logger.info(f"Found {len(valid_tuples)} valid system/date combinations")

    @app.route("/")
    def index():
        """Main annotation interface."""
        return render_template("annotator/index.html")

    @app.route("/api/random-images")
    def get_random_images():
        """Get random selection of thumbnails."""
        count = request.args.get("count", default=10, type=int)
        valid_tuples = app.config["VALID_TUPLES"]

        thumbnails = _get_random_thumbnails(valid_tuples, count)
        return jsonify(thumbnails)

    @app.route("/api/counts")
    def get_counts():
        """Get current counts of positive and negative classifications."""
        positive_count = _count_images_in_dir(app.config["POSITIVE_DIR"])
        negative_count = _count_images_in_dir(app.config["NEGATIVE_DIR"])

        return jsonify(
            {
                "positive": positive_count,
                "negative": negative_count,
            }
        )

    @app.route("/api/classify", methods=["POST"])
    def classify_image():
        """
        Classify an image as positive or negative.

        Expected JSON body:
        {
            "system": "nightskycam5",
            "date": "2025_01_15",
            "filename": "image.jpeg",
            "classification": "positive" or "negative"
        }
        """
        data = request.get_json()
        system = data["system"]
        date = data["date"]
        filename = data["filename"]
        classification = data["classification"]

        if classification not in ["positive", "negative"]:
            return jsonify({"success": False, "error": "Invalid classification"}), 400

        # Determine source thumbnail path
        root_dir = app.config["ROOT_DIR"]
        thumbnail_path = root_dir / system / date / THUMBNAIL_DIR_NAME / filename

        if not thumbnail_path.exists():
            return jsonify({"success": False, "error": "Thumbnail not found"}), 404

        # Determine destination folder
        if classification == "positive":
            dest_folder = app.config["POSITIVE_DIR"]
            opposite_folder = app.config["NEGATIVE_DIR"]
        else:
            dest_folder = app.config["NEGATIVE_DIR"]
            opposite_folder = app.config["POSITIVE_DIR"]

        # Remove from opposite folder if it exists
        _remove_from_opposite_folder(filename, dest_folder, opposite_folder)

        # Copy to destination folder
        success = _copy_thumbnail_to_folder(thumbnail_path, dest_folder, filename)

        if success:
            # Get updated counts
            positive_count = _count_images_in_dir(app.config["POSITIVE_DIR"])
            negative_count = _count_images_in_dir(app.config["NEGATIVE_DIR"])

            return jsonify(
                {
                    "success": True,
                    "counts": {
                        "positive": positive_count,
                        "negative": negative_count,
                    },
                }
            )
        else:
            return jsonify({"success": False, "error": "Failed to copy thumbnail"}), 500

    @app.route("/api/thumbnail/<system>/<date>/<filename>")
    def serve_thumbnail(system: str, date: str, filename: str):
        """Serve a thumbnail image."""
        root_dir = app.config["ROOT_DIR"]
        thumbnail_path = root_dir / system / date / THUMBNAIL_DIR_NAME / filename

        if not thumbnail_path.exists():
            return jsonify({"error": "Thumbnail not found"}), 404

        response = send_file(thumbnail_path, mimetype="image/jpeg")
        response.cache_control.max_age = 3600  # Cache for 1 hour
        return response

    @app.route("/api/check-classification/<filename>")
    def check_classification(filename: str):
        """
        Check if a thumbnail is already classified.

        Returns:
            {"classification": "positive" | "negative" | null}
        """
        positive_path = app.config["POSITIVE_DIR"] / filename
        negative_path = app.config["NEGATIVE_DIR"] / filename

        if positive_path.exists():
            return jsonify({"classification": "positive"})
        elif negative_path.exists():
            return jsonify({"classification": "negative"})
        else:
            return jsonify({"classification": None})

    return app
