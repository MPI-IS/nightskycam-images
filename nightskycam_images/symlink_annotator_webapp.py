"""Flask web application for annotating images from filter-export directories.

This module provides a web interface to:
- Browse images by system and date from symlink directories (filter-export output)
- Display all thumbnails for a selected date
- Classify thumbnails as positive/negative
- Copy thumbnails to output directories based on classification
"""

from pathlib import Path
from typing import Dict, List, Optional
import secrets
import shutil

from flask import Flask, jsonify, render_template, request, send_file
from loguru import logger

from .constants import IMAGE_FILE_FORMATS, THUMBNAIL_DIR_NAME
from .walk import parse_image_path, walk_dates, walk_systems


def detect_symlinks(date_path: Path) -> bool:
    """
    Check if directory contains symlink image files.

    Parameters
    ----------
    date_path
        Path to date directory.

    Returns
    -------
    bool
        True if symlinks found, False otherwise.
    """
    if not date_path.exists() or not date_path.is_dir():
        return False

    for ext in IMAGE_FILE_FORMATS:
        for image_file in date_path.glob(f"*.{ext}"):
            if image_file.is_symlink():
                return True
            break  # Just check first file

    return False


def get_thumbnail_path_from_symlink(symlink_path: Path) -> Optional[Path]:
    """
    Get thumbnail path by resolving symlink to original location.

    Parameters
    ----------
    symlink_path
        Path to HD image symlink.

    Returns
    -------
    Optional[Path]
        Path to thumbnail, or None if not found.
    """
    if not symlink_path.is_symlink():
        return None

    try:
        # Resolve symlink to original file
        original_path = symlink_path.resolve()

        # Build thumbnail path from original location
        thumbnail_path = (
            original_path.parent / THUMBNAIL_DIR_NAME / f"{original_path.stem}.jpeg"
        )

        return thumbnail_path if thumbnail_path.exists() else None

    except Exception as e:
        logger.error(f"Error resolving symlink {symlink_path}: {e}")
        return None


def count_images_in_dir(directory: Path) -> int:
    """
    Count number of JPEG images in a directory.

    Parameters
    ----------
    directory
        Path to directory.

    Returns
    -------
    int
        Number of JPEG files.
    """
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.jpeg")))


def copy_thumbnail_to_folder(
    thumbnail_path: Path,
    dest_folder: Path,
    filename: str,
) -> bool:
    """
    Copy thumbnail to destination folder.

    Parameters
    ----------
    thumbnail_path
        Path to source thumbnail.
    dest_folder
        Destination folder (positive or negative).
    filename
        Filename for the copied thumbnail.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    try:
        # Create destination folder if needed
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Destination path
        dest_path = dest_folder / filename

        # Copy the file
        shutil.copy2(thumbnail_path, dest_path)
        logger.debug(f"Copied thumbnail: {thumbnail_path} -> {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy thumbnail {thumbnail_path}: {e}")
        return False


def remove_from_opposite_folder(
    filename: str,
    opposite_folder: Path,
) -> None:
    """
    Remove file from opposite classification folder if it exists.

    Parameters
    ----------
    filename
        Filename to remove.
    opposite_folder
        Path to opposite classification folder.
    """
    opposite_path = opposite_folder / filename
    if opposite_path.exists():
        try:
            opposite_path.unlink()
            logger.debug(f"Removed from opposite folder: {opposite_path}")
        except Exception as e:
            logger.warning(
                f"Failed to remove from opposite folder {opposite_path}: {e}"
            )


def create_app(filter_dir: Path, output_dir: Path) -> Flask:
    """
    Create and configure the Flask application.

    Parameters
    ----------
    filter_dir
        Path to filter-export directory (with symlinks).
    output_dir
        Path to output directory for classified thumbnails.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)

    # Store configuration
    app.config["FILTER_DIR"] = filter_dir
    app.config["OUTPUT_DIR"] = output_dir
    app.config["POSITIVE_DIR"] = output_dir / "positive"
    app.config["NEGATIVE_DIR"] = output_dir / "negative"

    # Create output directories
    app.config["POSITIVE_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["NEGATIVE_DIR"].mkdir(parents=True, exist_ok=True)

    logger.info(f"Filter directory: {filter_dir}")
    logger.info(f"Output directory: {output_dir}")

    @app.route("/")
    def index():
        """Main annotation interface."""
        return render_template("symlink_annotator/index.html")

    @app.route("/api/systems")
    def get_systems():
        """Get list of available systems."""
        try:
            filter_dir = app.config["FILTER_DIR"]
            systems = [path.name for path in walk_systems(filter_dir)]
            systems.sort()
            return jsonify(systems)
        except Exception as e:
            logger.error(f"Error getting systems: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/dates/<system>")
    def get_dates(system: str):
        """
        Get list of dates for a system.

        Only returns dates that contain symlink image files.
        """
        try:
            filter_dir = app.config["FILTER_DIR"]
            system_path = filter_dir / system

            if not system_path.exists():
                return jsonify({"error": "System not found"}), 404

            dates = []
            for date_obj, date_path in walk_dates(system_path):
                # Only include dates with symlinks
                if detect_symlinks(date_path):
                    date_str = date_obj.strftime("%Y_%m_%d")
                    display_str = date_obj.strftime("%Y-%m-%d")
                    dates.append({"value": date_str, "display": display_str})

            # Sort chronologically (oldest first)
            dates.sort(key=lambda x: x["value"])

            return jsonify(dates)

        except Exception as e:
            logger.error(f"Error getting dates for system {system}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/images/<system>/<date>")
    def get_images(system: str, date: str):
        """Get list of images for system/date."""
        try:
            filter_dir = app.config["FILTER_DIR"]
            date_path = filter_dir / system / date

            if not date_path.exists():
                return jsonify({"error": "Date directory not found"}), 404

            images = []

            # Find all symlink image files
            for ext in IMAGE_FILE_FORMATS:
                for image_file in date_path.glob(f"*.{ext}"):
                    if image_file.is_symlink():
                        try:
                            # Parse timestamp from filename
                            _, datetime_obj = parse_image_path(image_file)
                            timestamp = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

                            images.append(
                                {
                                    "filename": image_file.name,
                                    "timestamp": timestamp,
                                    "thumbnail_url": f"/api/thumbnail/{system}/{date}/{image_file.name}",
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error parsing image {image_file.name}: {e}"
                            )
                            continue

            # Sort by timestamp
            images.sort(key=lambda x: x["timestamp"])

            return jsonify(images)

        except Exception as e:
            logger.error(f"Error getting images for {system}/{date}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/thumbnail/<system>/<date>/<filename>")
    def serve_thumbnail(system: str, date: str, filename: str):
        """Serve thumbnail image (resolve from symlink)."""
        try:
            filter_dir = app.config["FILTER_DIR"]
            symlink_path = filter_dir / system / date / filename

            if not symlink_path.exists():
                return jsonify({"error": "Image not found"}), 404

            # Get thumbnail path by resolving symlink
            thumbnail_path = get_thumbnail_path_from_symlink(symlink_path)

            if thumbnail_path is None or not thumbnail_path.exists():
                logger.warning(f"Thumbnail not found for {filename}")
                return jsonify({"error": "Thumbnail not found"}), 404

            response = send_file(thumbnail_path, mimetype="image/jpeg")
            response.cache_control.max_age = 3600  # Cache for 1 hour
            return response

        except Exception as e:
            logger.error(f"Error serving thumbnail {filename}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/classify", methods=["POST"])
    def classify_image():
        """
        Classify an image as positive or negative.

        Expected JSON body:
        {
            "system": "nightskycam5",
            "date": "2025_01_15",
            "filename": "image.tiff",
            "classification": "positive" or "negative"
        }
        """
        try:
            data = request.get_json()
            system = data["system"]
            date = data["date"]
            filename = data["filename"]
            classification = data["classification"]

            if classification not in ["positive", "negative"]:
                return jsonify({"success": False, "error": "Invalid classification"}), 400

            # Get symlink path
            filter_dir = app.config["FILTER_DIR"]
            symlink_path = filter_dir / system / date / filename

            if not symlink_path.exists():
                return jsonify({"success": False, "error": "Image not found"}), 404

            # Get thumbnail path by resolving symlink
            thumbnail_path = get_thumbnail_path_from_symlink(symlink_path)

            if thumbnail_path is None or not thumbnail_path.exists():
                return jsonify({"success": False, "error": "Thumbnail not found"}), 404

            # Determine destination folder
            if classification == "positive":
                dest_folder = app.config["POSITIVE_DIR"]
                opposite_folder = app.config["NEGATIVE_DIR"]
            else:
                dest_folder = app.config["NEGATIVE_DIR"]
                opposite_folder = app.config["POSITIVE_DIR"]

            # Use thumbnail filename (always .jpeg)
            thumbnail_filename = thumbnail_path.name

            # Remove from opposite folder if it exists
            remove_from_opposite_folder(thumbnail_filename, opposite_folder)

            # Copy to destination folder
            success = copy_thumbnail_to_folder(
                thumbnail_path, dest_folder, thumbnail_filename
            )

            if success:
                # Get updated counts
                positive_count = count_images_in_dir(app.config["POSITIVE_DIR"])
                negative_count = count_images_in_dir(app.config["NEGATIVE_DIR"])

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

        except KeyError as e:
            return jsonify({"success": False, "error": f"Missing field: {e}"}), 400
        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/counts")
    def get_counts():
        """Get current counts of positive and negative classifications."""
        try:
            positive_count = count_images_in_dir(app.config["POSITIVE_DIR"])
            negative_count = count_images_in_dir(app.config["NEGATIVE_DIR"])

            return jsonify(
                {
                    "positive": positive_count,
                    "negative": negative_count,
                }
            )
        except Exception as e:
            logger.error(f"Error getting counts: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/check-classification/<filename>")
    def check_classification(filename: str):
        """
        Check if a thumbnail is already classified.

        Returns:
            {"classification": "positive" | "negative" | null}
        """
        try:
            # Convert image filename to thumbnail filename
            thumbnail_filename = Path(filename).stem + ".jpeg"

            positive_path = app.config["POSITIVE_DIR"] / thumbnail_filename
            negative_path = app.config["NEGATIVE_DIR"] / thumbnail_filename

            if positive_path.exists():
                return jsonify({"classification": "positive"})
            elif negative_path.exists():
                return jsonify({"classification": "negative"})
            else:
                return jsonify({"classification": None})

        except Exception as e:
            logger.error(f"Error checking classification for {filename}: {e}")
            return jsonify({"error": str(e)}), 500

    return app
