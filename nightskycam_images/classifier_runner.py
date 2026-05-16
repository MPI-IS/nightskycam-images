"""
Per-image classifier helper shared by ``ns.ml.classify`` and
``ns.db.update --classifier-config``.

The core function (:func:`classify_image_inplace`) decides which configured
classifiers must be run for a single image — only those whose scores are
missing from the image's TOML ``[classifiers]`` section, or all of them when
``overwrite=True`` — runs them on the image's thumbnail, merges the resulting
scores into the TOML, and writes the file back. Pre-existing classifier keys
not named in the current configuration are always preserved.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from loguru import logger
import tomli_w

from nightskycam_scorer.model.infer import SkyScorer
from nightskycam_scorer.utils import to_float_image

from .convert_npy import to_npy


def load_classifiers(model_map: Dict[str, Path]) -> Dict[str, SkyScorer]:
    """
    Load each ``.pt`` model into a :class:`SkyScorer`.

    Parameters
    ----------
    model_map
        Mapping of classifier name to model file path.

    Returns
    -------
    Dict[str, SkyScorer]
        Loaded scorers keyed by classifier name.

    Raises
    ------
    Exception
        Re-raises whatever :class:`SkyScorer` raises if a model fails to load.
    """
    scorers: Dict[str, SkyScorer] = {}
    for name, model_path in model_map.items():
        logger.info(f"Loading classifier '{name}': {model_path}")
        scorers[name] = SkyScorer(str(model_path), validate_size=False)
    return scorers


def _scorers_to_run(
    meta: Dict[str, Any],
    scorers: Dict[str, SkyScorer],
    overwrite: bool,
) -> Dict[str, SkyScorer]:
    """Return the subset of scorers whose scores need to be (re)computed."""
    if overwrite:
        return scorers
    existing = meta.get("classifiers", {}) or {}
    return {name: s for name, s in scorers.items() if name not in existing}


def classify_image_inplace(
    thumbnail_path: Path,
    meta_path: Path,
    meta: Dict[str, Any],
    scorers: Dict[str, SkyScorer],
    overwrite: bool,
) -> Tuple[Dict[str, Any], bool]:
    """
    Ensure ``meta['classifiers']`` contains scores for every configured model.

    Only the scorers whose names are absent from ``meta['classifiers']`` are
    run (or all of them when ``overwrite=True``). Any classifier keys already
    in the TOML that are not named in ``scorers`` are preserved.

    Caller is responsible for skipping when ``thumbnail_path`` or ``meta_path``
    don't exist on disk.

    Parameters
    ----------
    thumbnail_path
        Path to the image's thumbnail (must exist).
    meta_path
        Path to the image's TOML metadata file (must exist).
    meta
        The TOML metadata already loaded as a dict. Mutated in place.
    scorers
        Configured classifiers keyed by name.
    overwrite
        When ``False`` (default), only fill missing classifier keys.
        When ``True``, rerun and overwrite every configured key.

    Returns
    -------
    Tuple[Dict[str, Any], bool]
        ``(meta, modified)`` — ``meta`` is the same dict mutated in place;
        ``modified`` is ``True`` iff the TOML file was rewritten.
    """
    to_run = _scorers_to_run(meta, scorers, overwrite)
    if not to_run:
        return meta, False

    thumbnail_array = to_npy(thumbnail_path)
    rgb_float = to_float_image(thumbnail_array)

    new_scores: Dict[str, float] = {}
    for name, scorer in to_run.items():
        result_raw = scorer.predict(rgb_float)
        result = result_raw[0] if isinstance(result_raw, list) else result_raw
        new_scores[name] = float(result.probability)
        logger.debug(f"      {name}: probability={result.probability:.3f}")

    existing = meta.get("classifiers", {}) or {}
    existing.update(new_scores)
    meta["classifiers"] = existing

    with open(meta_path, "wb") as f:
        tomli_w.dump(meta, f)

    return meta, True


EnricherStats = Dict[str, int]
Enricher = Callable[[Path, Path, Dict[str, Any]], Dict[str, Any]]


def make_populate_enricher(
    scorers: Dict[str, SkyScorer],
    overwrite: bool,
) -> Tuple[Enricher, EnricherStats]:
    """
    Build the per-image callback used by :func:`nightskycam_images.db.populate`.

    The returned callable has signature
    ``(thumbnail_path, meta_path, meta) -> meta`` and updates a mutable stats
    dict (also returned) so the caller can report classification activity.

    Returns
    -------
    Tuple[Enricher, EnricherStats]
        ``(enricher, stats)`` where ``stats`` has keys
        ``classifier_runs`` (images for which any scorer was executed),
        ``classifier_writes`` (TOML files rewritten),
        ``classifier_errors``.
    """
    stats: EnricherStats = {
        "classifier_runs": 0,
        "classifier_writes": 0,
        "classifier_errors": 0,
    }

    def enricher(
        thumbnail_path: Path, meta_path: Path, meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            updated_meta, modified = classify_image_inplace(
                thumbnail_path, meta_path, meta, scorers, overwrite
            )
            if modified:
                stats["classifier_runs"] += 1
                stats["classifier_writes"] += 1
            return updated_meta
        except Exception as e:
            logger.error(
                f"Classifier failed on {thumbnail_path.name}: {e}"
            )
            stats["classifier_errors"] += 1
            return meta

    return enricher, stats
