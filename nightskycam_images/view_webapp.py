"""
Flask web application for viewing nightskycam images.

This module provides a minimalistic web interface to browse and view
nightskycam images from both original and filtered directory structures.
"""

from io import BytesIO
from pathlib import Path
import secrets
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import Flask, Response, jsonify, render_template, request, send_file, session
from loguru import logger

from .constants import IMAGE_FILE_FORMATS, THUMBNAIL_DIR_NAME
from .image import Image
from .walk import get_images, parse_image_path, walk_dates, walk_systems

# Server-side storage for image lists (per session ID)
# This avoids cookie size limits
image_lists: Dict[str, List[Dict[str, str]]] = {}


def detect_structure(root: Path, system: str, date: str) -> str:
    """
    Detect if directory structure is original or filtered.

    Parameters
    ----------
    root
        Root directory path.
    system
        System name.
    date
        Date string in YYYY_MM_DD format.

    Returns
    -------
    str
        Either "original" or "filtered".
    """
    date_path = root / system / date

    if not date_path.exists():
        return "original"

    # Check if thumbnails directory exists
    thumbnail_dir = date_path / THUMBNAIL_DIR_NAME
    if thumbnail_dir.exists() and thumbnail_dir.is_dir():
        return "original"

    # Check if any image files are symlinks
    for ext in IMAGE_FILE_FORMATS:
        for image_file in date_path.glob(f"*.{ext}"):
            if image_file.is_symlink():
                return "filtered"
            break  # Just check first file

    # Default to original
    return "original"


def get_thumbnail_path(image_path: Path, structure: str) -> Optional[Path]:
    """
    Get thumbnail path for an image, handling both structures.

    Parameters
    ----------
    image_path
        Path to the HD image (may be symlink or real file).
    structure
        Either "original" or "filtered".

    Returns
    -------
    Optional[Path]
        Path to thumbnail, or None if not found.
    """
    if structure == "original":
        # Simple case: thumbnails are in same directory tree
        thumbnail_path = (
            image_path.parent / THUMBNAIL_DIR_NAME / f"{image_path.stem}.jpeg"
        )
        return thumbnail_path if thumbnail_path.exists() else None

    else:  # filtered
        # If image is a symlink, resolve to original location
        if image_path.is_symlink():
            original_path = image_path.resolve()
            thumbnail_path = (
                original_path.parent / THUMBNAIL_DIR_NAME / f"{original_path.stem}.jpeg"
            )
            return thumbnail_path if thumbnail_path.exists() else None
        else:
            # Shouldn't happen in filtered structure, but handle gracefully
            return None


def create_app(root_dir: Path) -> Flask:
    """
    Factory function to create Flask application.

    Parameters
    ----------
    root_dir
        Root directory containing nightskycam images.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    app = Flask(__name__)
    app.config["ROOT_DIR"] = root_dir
    app.secret_key = secrets.token_hex(16)

    @app.route("/")
    def index() -> str:
        """Main viewer page."""
        return render_template("viewer/index.html")

    @app.route("/api/systems")
    def get_systems_route() -> Union[Response, Tuple[Response, int]]:
        """Return list of available systems."""
        try:
            systems = [path.name for path in walk_systems(app.config["ROOT_DIR"])]
            systems.sort()
            return jsonify(systems)
        except Exception as e:
            logger.error(f"Error listing systems: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/dates/<system>")
    def get_dates_route(system: str) -> Union[Response, Tuple[Response, int]]:
        """Return list of available dates for a system (only dates with images)."""
        try:
            system_path = app.config["ROOT_DIR"] / system
            if not system_path.exists():
                return jsonify({"error": "System not found"}), 404

            dates: List[Dict[str, str]] = []
            for date, date_path in walk_dates(system_path):
                date_str = date.strftime("%Y_%m_%d")
                structure = detect_structure(app.config["ROOT_DIR"], system, date_str)

                # Check if this date has any images
                has_images = False
                if structure == "original":
                    # Check using get_images()
                    if get_images(date_path):
                        has_images = True
                else:
                    # Check for HD files directly
                    for ext in IMAGE_FILE_FORMATS:
                        if list(date_path.glob(f"*.{ext}")):
                            has_images = True
                            break

                # Only add dates that have images
                if has_images:
                    dates.append(
                        {
                            "value": date_str,
                            "display": date.strftime("%Y-%m-%d"),
                        }
                    )

            # Sort chronologically (oldest first)
            dates.sort(key=lambda x: x["value"])
            return jsonify(dates)
        except Exception as e:
            logger.error(f"Error listing dates for {system}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/images/<system>/<date>")
    def get_images_route(system: str, date: str) -> Union[Response, Tuple[Response, int]]:
        """Return list of images for a system/date combination."""
        try:
            date_path = app.config["ROOT_DIR"] / system / date
            if not date_path.exists():
                return jsonify({"error": "Date directory not found"}), 404

            structure = detect_structure(app.config["ROOT_DIR"], system, date)
            logger.debug(f"Detected structure: {structure} for {system}/{date}")

            images_data: List[Dict[str, str]] = []

            # For original structure, use get_images()
            if structure == "original":
                for img in get_images(date_path):
                    if img.hd and img.date_and_time:
                        images_data.append(
                            {
                                "filename": img.hd.name,
                                "timestamp": img.date_and_time.strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "thumbnail_url": f"/api/thumbnail/{system}/{date}/{img.hd.name}",
                                "image_url": f"/api/image/{system}/{date}/{img.hd.name}",
                            }
                        )
            else:
                # For filtered structure, find HD files directly
                hd_files: List[Path] = []
                for ext in IMAGE_FILE_FORMATS:
                    hd_files.extend(date_path.glob(f"*.{ext}"))

                for hd_file in hd_files:
                    try:
                        system_name, datetime = parse_image_path(hd_file)
                        images_data.append(
                            {
                                "filename": hd_file.name,
                                "timestamp": datetime.strftime("%Y-%m-%d %H:%M:%S"),
                                "thumbnail_url": f"/api/thumbnail/{system}/{date}/{hd_file.name}",
                                "image_url": f"/api/image/{system}/{date}/{hd_file.name}",
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Could not parse image {hd_file.name}: {e}")
                        continue

            # Sort by timestamp
            images_data.sort(key=lambda x: x["timestamp"])
            return jsonify(images_data)

        except Exception as e:
            logger.error(f"Error listing images for {system}/{date}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/thumbnail/<system>/<date>/<filename>")
    def serve_thumbnail(system: str, date: str, filename: str) -> Union[Response, Tuple[str, int], Tuple[Response, int]]:
        """Serve thumbnail image file."""
        try:
            date_path = app.config["ROOT_DIR"] / system / date
            structure = detect_structure(app.config["ROOT_DIR"], system, date)

            # Find the HD image first
            image_path: Optional[Path] = None
            stem = Path(filename).stem
            for ext in IMAGE_FILE_FORMATS:
                candidate = date_path / f"{stem}.{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if not image_path:
                logger.warning(f"Image not found: {filename}")
                return "Image not found", 404

            thumbnail_path = get_thumbnail_path(image_path, structure)
            if not thumbnail_path or not thumbnail_path.exists():
                logger.warning(f"Thumbnail not found for: {filename}")
                return "Thumbnail not found", 404

            response: Response = send_file(thumbnail_path, mimetype="image/jpeg")
            response.cache_control.max_age = 3600  # Cache for 1 hour
            return response

        except Exception as e:
            logger.error(f"Error serving thumbnail {filename}: {e}")
            return str(e), 500

    @app.route("/api/image/<system>/<date>/<filename>")
    def serve_image(system: str, date: str, filename: str) -> Union[Response, Tuple[str, int]]:
        """Serve full-size image."""
        try:
            date_path = app.config["ROOT_DIR"] / system / date
            structure = detect_structure(app.config["ROOT_DIR"], system, date)

            # Find the HD image
            image_path: Optional[Path] = None
            stem = Path(filename).stem
            for ext in IMAGE_FILE_FORMATS:
                candidate = date_path / f"{stem}.{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if not image_path:
                return "Image not found", 404

            # For filtered structure with symlinks, resolve to original
            if structure == "filtered" and image_path.is_symlink():
                image_path = image_path.resolve()

            # Determine mimetype based on extension
            ext = image_path.suffix.lower()
            mimetype_map: Dict[str, str] = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".tiff": "image/tiff",
                ".npy": "application/octet-stream",
            }

            # For NPY files, convert to JPEG on the fly
            if ext == ".npy":
                try:
                    from .convert_npy import npy_file_to_pil

                    pil_image = npy_file_to_pil(image_path)
                    img_io = BytesIO()
                    pil_image.save(img_io, "JPEG", quality=95)
                    img_io.seek(0)
                    return send_file(img_io, mimetype="image/jpeg")
                except Exception as e:
                    logger.error(f"Error converting NPY file {filename}: {e}")
                    return f"Error converting image: {e}", 500

            return send_file(
                image_path, mimetype=mimetype_map.get(ext, "application/octet-stream")
            )

        except Exception as e:
            logger.error(f"Error serving image {filename}: {e}")
            return str(e), 500

    def get_session_id() -> str:
        """Get or create session ID."""
        if "session_id" not in session:
            session["session_id"] = secrets.token_hex(16)
        return session["session_id"]

    def get_image_list() -> List[Dict[str, str]]:
        """Get image list for current session."""
        session_id = get_session_id()
        if session_id not in image_lists:
            image_lists[session_id] = []
        return image_lists[session_id]

    @app.route("/list")
    def list_page() -> str:
        """List management page."""
        return render_template("viewer/list.html")

    @app.route("/api/list", methods=["GET"])
    def get_list() -> Response:
        """Return current image list."""
        return jsonify(get_image_list())

    @app.route("/api/list/add", methods=["POST"])
    def add_to_list() -> Union[Response, Tuple[Response, int]]:
        """Add single image to list."""
        try:
            data: Optional[Dict[str, Any]] = request.get_json()
            if (
                not data
                or "system" not in data
                or "date" not in data
                or "filename" not in data
            ):
                return jsonify({"error": "Missing required fields"}), 400

            img_list = get_image_list()

            # Check if already in list
            image_entry: Dict[str, str] = {
                "system": data["system"],
                "date": data["date"],
                "filename": data["filename"],
            }

            if image_entry not in img_list:
                img_list.append(image_entry)
                return jsonify({"success": True, "count": len(img_list)})
            else:
                return jsonify({"success": False, "message": "Image already in list"})

        except Exception as e:
            logger.error(f"Error adding to list: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/list/add-all", methods=["POST"])
    def add_all_to_list() -> Union[Response, Tuple[Response, int]]:
        """Add all images from current system/date to list."""
        try:
            data: Optional[Dict[str, Any]] = request.get_json()
            if not data or "system" not in data or "date" not in data:
                return jsonify({"error": "Missing required fields"}), 400

            system = data["system"]
            date = data["date"]

            # Get all images for this system/date
            date_path = app.config["ROOT_DIR"] / system / date
            if not date_path.exists():
                return jsonify({"error": "Date directory not found"}), 404

            structure = detect_structure(app.config["ROOT_DIR"], system, date)
            filenames: List[str] = []

            if structure == "original":
                for img in get_images(date_path):
                    if img.hd:
                        filenames.append(img.hd.name)
            else:
                hd_files: List[Path] = []
                for ext in IMAGE_FILE_FORMATS:
                    hd_files.extend(date_path.glob(f"*.{ext}"))
                filenames = [f.name for f in hd_files]

            # Add all to list
            img_list = get_image_list()

            added_count = 0
            for filename in filenames:
                image_entry = {"system": system, "date": date, "filename": filename}
                if image_entry not in img_list:
                    img_list.append(image_entry)
                    added_count += 1

            return jsonify(
                {"success": True, "added": added_count, "count": len(img_list)}
            )

        except Exception as e:
            logger.error(f"Error adding all to list: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/list/remove/<int:index>", methods=["DELETE"])
    def remove_from_list(index: int) -> Union[Response, Tuple[Response, int]]:
        """Remove image at index from list."""
        try:
            img_list = get_image_list()

            if 0 <= index < len(img_list):
                img_list.pop(index)
                return jsonify({"success": True, "count": len(img_list)})
            else:
                return jsonify({"error": "Invalid index"}), 400

        except Exception as e:
            logger.error(f"Error removing from list: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/list/clear", methods=["DELETE"])
    def clear_list() -> Union[Response, Tuple[Response, int]]:
        """Clear entire image list."""
        try:
            img_list = get_image_list()
            img_list.clear()
            return jsonify({"success": True})

        except Exception as e:
            logger.error(f"Error clearing list: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/list/export")
    def export_list() -> Union[Response, Tuple[str, int]]:
        """Export list as text file."""
        try:
            img_list = get_image_list()

            # Format: system/date/filename (one per line)
            lines: List[str] = []
            for item in img_list:
                lines.append(f"{item['system']}/{item['date']}/{item['filename']}")

            content = "\n".join(lines)

            return Response(
                content,
                mimetype="text/plain",
                headers={"Content-Disposition": "attachment; filename=image_list.txt"},
            )

        except Exception as e:
            logger.error(f"Error exporting list: {e}")
            return str(e), 500

    return app
