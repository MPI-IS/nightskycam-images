"""
nightskycam-images: management of nightskycam image folders and their
SQLite metadata database.

The names re-exported here form the package's PUBLIC API — the surface
external consumers (e.g. the nightskycam-website package) may rely on.
Everything else is internal and may change without notice.

Exports are lazy (PEP 562): importing the package does not import heavy
optional dependencies (``convert_npy`` pulls OpenCV, which also needs
system libraries absent from slim containers that only query the
database).
"""

import importlib
from typing import Any, List

_EXPORTS = {
    "IMAGE_FILE_FORMATS": ".constants",
    "ImageDB": ".db_api",
    "ImageRecord": ".db_api",
    "Stretch": ".convert_npy",
    "get_weather_icon": ".weather",
    "open_db_readonly": ".db",
    "to_8bits": ".convert_npy",
    "to_npy": ".convert_npy",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_EXPORTS))
