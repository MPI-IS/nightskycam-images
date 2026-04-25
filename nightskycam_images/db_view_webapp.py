"""
Flask web application for viewing nightskycam images via the SQLite database.

Unlike the file-based view webapp, this version queries the database for
filtering by stretched/format/cloud cover/classifier scores, and resolves
image paths using the root stored per image.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from flask import Flask, Response, jsonify, render_template, request, send_file
from loguru import logger
from PIL import Image as PILImage
import tifffile

from .constants import IMAGE_FILE_FORMATS, THUMBNAIL_DIR_NAME
from .db import (
    get_classifier_names,
    get_classifier_scores,
    get_dates,
    get_systems,
    open_db,
    query_images,
)


def create_app(db_path: Path) -> Flask:
    """
    Factory function to create the DB-backed viewer Flask application.

    Parameters
    ----------
    db_path
        Path to the SQLite database file.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path

    @app.route("/")
    def index() -> str:
        return render_template("db_viewer/index.html")

    @app.route("/api/systems")
    def systems_route() -> Union[Response, Tuple[Response, int]]:
        try:
            return jsonify(get_systems(app.config["DB_PATH"]))
        except Exception as e:
            logger.error(f"Error listing systems: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/dates/<system>")
    def dates_route(system: str) -> Union[Response, Tuple[Response, int]]:
        try:
            dates = get_dates(app.config["DB_PATH"], system)
            result = [
                {"value": d, "display": d.replace("_", "-")}
                for d in dates
            ]
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error listing dates for {system}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/classifiers")
    def classifiers_route() -> Union[Response, Tuple[Response, int]]:
        try:
            return jsonify(get_classifier_names(app.config["DB_PATH"]))
        except Exception as e:
            logger.error(f"Error listing classifiers: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/images/<system>/<date>")
    def images_route(
        system: str, date: str
    ) -> Union[Response, Tuple[Response, int]]:
        try:
            # Parse filter query parameters.
            kwargs: Dict[str, Any] = {
                "systems": [system],
                "start_date": date,
                "end_date": date,
                "has_thumbnail": True,
            }

            stretched_param = request.args.get("stretched")
            if stretched_param == "yes":
                kwargs["stretched"] = True
            elif stretched_param == "no":
                kwargs["stretched"] = False

            fmt = request.args.get("format")
            if fmt and fmt != "all":
                kwargs["image_format"] = fmt

            cc_max = request.args.get("cloud_cover_max")
            if cc_max is not None and cc_max != "":
                try:
                    kwargs["cloud_cover_max"] = int(cc_max)
                except ValueError:
                    pass

            # Classifier thresholds: ?clf_cloudy=0.5&clf_rainy=0.3
            classifier_max: Dict[str, float] = {}
            for key, val in request.args.items():
                if key.startswith("clf_") and val != "":
                    name = key[4:]
                    try:
                        classifier_max[name] = float(val)
                    except ValueError:
                        pass
            if classifier_max:
                kwargs["classifier_max"] = classifier_max

            rows = query_images(app.config["DB_PATH"], **kwargs)

            images_data: List[Dict[str, Any]] = []
            for row in rows:
                stem = row["filename_stem"]
                fmt = row["image_format"]
                scores = get_classifier_scores(
                    app.config["DB_PATH"], stem
                )
                entry: Dict[str, Any] = {
                    "filename_stem": stem,
                    "time": row["time"],
                    "timestamp": row["datetime"],
                    "image_format": fmt,
                    "weather": row["weather"],
                    "cloud_cover": row["cloud_cover"],
                    "stretched": bool(row["stretched"]),
                    "classifiers": scores,
                    "thumbnail_url": f"/api/thumbnail/{stem}",
                    "jpeg_url": f"/api/image/jpeg/{stem}",
                }
                if fmt and fmt not in ("jpg", "jpeg"):
                    entry["raw_url"] = f"/api/image/raw/{stem}"
                images_data.append(entry)
            return jsonify(images_data)
        except Exception as e:
            logger.error(f"Error listing images for {system}/{date}: {e}")
            return jsonify({"error": str(e)}), 500

    def _resolve_image_row(
        filename_stem: str,
    ) -> Optional[Dict[str, Any]]:
        """Look up an image row by filename_stem."""
        conn = open_db(app.config["DB_PATH"])
        row = conn.execute(
            "SELECT * FROM images WHERE filename_stem = ?",
            (filename_stem,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    @app.route("/api/thumbnail/<filename_stem>")
    def serve_thumbnail(
        filename_stem: str,
    ) -> Union[Response, Tuple[str, int]]:
        try:
            row = _resolve_image_row(filename_stem)
            if not row:
                return "Image not found in database", 404

            root = Path(row["root"])
            date_dir = root / row["system"] / row["date"]
            thumb_path = (
                date_dir / THUMBNAIL_DIR_NAME / f"{filename_stem}.jpeg"
            )
            if not thumb_path.exists():
                return "Thumbnail not found", 404

            response: Response = send_file(thumb_path, mimetype="image/jpeg")
            response.cache_control.max_age = 3600
            return response
        except Exception as e:
            logger.error(f"Error serving thumbnail {filename_stem}: {e}")
            return str(e), 500

    def _find_image_path(
        filename_stem: str,
    ) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
        """Resolve the on-disk image path for a filename_stem."""
        row = _resolve_image_row(filename_stem)
        if not row:
            return None, None
        root = Path(row["root"])
        date_dir = root / row["system"] / row["date"]
        for fmt in IMAGE_FILE_FORMATS:
            candidate = date_dir / f"{filename_stem}.{fmt}"
            if candidate.exists():
                return candidate, row
        return None, row

    @app.route("/api/image/jpeg/<filename_stem>")
    def serve_image_jpeg(
        filename_stem: str,
    ) -> Union[Response, Tuple[str, int]]:
        """Serve the image as JPEG, converting from tiff/npy if needed."""
        try:
            image_path, row = _find_image_path(filename_stem)
            if not row:
                return "Image not found in database", 404
            if not image_path:
                return "Image file not found on disk", 404

            ext = image_path.suffix.lower()

            if ext in (".jpg", ".jpeg"):
                return send_file(image_path, mimetype="image/jpeg")

            if ext == ".tiff":
                data = tifffile.imread(str(image_path))
                pil_image = PILImage.fromarray(data)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                img_io = BytesIO()
                pil_image.save(img_io, "JPEG", quality=95)
                img_io.seek(0)
                return send_file(img_io, mimetype="image/jpeg")

            if ext == ".npy":
                from .convert_npy import npy_file_to_pil

                pil_image = npy_file_to_pil(image_path)
                img_io = BytesIO()
                pil_image.save(img_io, "JPEG", quality=95)
                img_io.seek(0)
                return send_file(img_io, mimetype="image/jpeg")

            return "Unsupported image format", 400

        except Exception as e:
            logger.error(f"Error serving JPEG for {filename_stem}: {e}")
            return str(e), 500

    @app.route("/api/image/raw/<filename_stem>")
    def serve_image_raw(
        filename_stem: str,
    ) -> Union[Response, Tuple[str, int]]:
        """Serve the original image file for download."""
        try:
            image_path, row = _find_image_path(filename_stem)
            if not row:
                return "Image not found in database", 404
            if not image_path:
                return "Image file not found on disk", 404

            ext = image_path.suffix.lower()
            mimetype_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".tiff": "image/tiff",
                ".npy": "application/octet-stream",
            }
            return send_file(
                image_path,
                mimetype=mimetype_map.get(ext, "application/octet-stream"),
                as_attachment=True,
                download_name=image_path.name,
            )
        except Exception as e:
            logger.error(f"Error serving raw image {filename_stem}: {e}")
            return str(e), 500

    return app
