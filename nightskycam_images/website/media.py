"""Image bytes: thumbnails, converted JPEGs, original downloads."""

from typing import Any

from flask import (
    Blueprint,
    abort,
    current_app,
    redirect,
    request,
    send_file,
    url_for,
)

from ..db_api import ImageDB, ImageRecord
from .convert import ConversionError, cached_jpeg

media_bp = Blueprint("media", __name__)

DAY_SECONDS = 86400

_DOWNLOAD_MIMETYPES = {
    "tiff": "image/tiff",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "npy": "application/octet-stream",
}


def _record_or_404(filename_stem: str) -> ImageRecord:
    record = ImageDB(current_app.config["DB_PATH"]).image(filename_stem)
    if record is None:
        abort(404)
    return record


@media_bp.route("/thumb/<filename_stem>.jpeg")
def thumb(filename_stem: str) -> Any:
    record = _record_or_404(filename_stem)
    path = record.thumbnail_path
    if path is None:
        return redirect(url_for("static", filename="placeholder.svg"))
    response = send_file(path, mimetype="image/jpeg")
    response.cache_control.public = True
    response.cache_control.max_age = DAY_SECONDS
    return response


@media_bp.route("/jpeg/<filename_stem>.jpeg")
def jpeg(filename_stem: str) -> Any:
    record = _record_or_404(filename_stem)
    hd_path = record.hd_path
    if hd_path is None:
        abort(404, "original image file not found on disk")
    stretch = request.args.get("stretch", "0") in ("1", "true", "yes")

    if hd_path.suffix in (".jpg", ".jpeg") and not stretch:
        # Already browser-displayable; stream from disk unmodified.
        response = send_file(hd_path, mimetype="image/jpeg")
    else:
        try:
            converted = cached_jpeg(
                current_app.config["CACHE_DIR"],
                filename_stem,
                hd_path,
                stretch,
                current_app.config["CACHE_MAX_MB"],
            )
        except ConversionError as error:
            abort(500, str(error))
        response = send_file(converted, mimetype="image/jpeg")
    response.cache_control.public = True
    response.cache_control.max_age = DAY_SECONDS
    return response


@media_bp.route("/download/<filename_stem>")
def download(filename_stem: str) -> Any:
    record = _record_or_404(filename_stem)
    hd_path = record.hd_path
    if hd_path is None:
        abort(404, "original image file not found on disk")
    mimetype = _DOWNLOAD_MIMETYPES.get(
        hd_path.suffix.lstrip("."), "application/octet-stream"
    )
    return send_file(
        hd_path,
        mimetype=mimetype,
        as_attachment=True,
        download_name=hd_path.name,
    )
