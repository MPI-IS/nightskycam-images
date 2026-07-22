"""
CLI-level checks for ``ns.db.update`` (nightskycam_images.main:db_update).

The location mapping is REQUIRED so newly scanned images are never silently
left with an unknown location — the update must fail loudly when it is omitted
rather than upserting rows with NULL locations.
"""

import sys

import pytest

from nightskycam_images.main import db_update


def test_db_update_requires_locations(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    # No --locations: Typer/Click rejects the missing required option (exit 2)
    # before the command body runs, so no scan/upsert happens.
    monkeypatch.setattr(sys, "argv", ["ns.db.update", str(root)])
    with pytest.raises(SystemExit) as exc:
        db_update()
    assert exc.value.code == 2


def test_db_update_rejects_missing_locations_file(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    # A --locations path that does not exist is rejected by `exists=True`.
    monkeypatch.setattr(
        sys,
        "argv",
        ["ns.db.update", str(root), "--locations", str(tmp_path / "nope.toml")],
    )
    with pytest.raises(SystemExit) as exc:
        db_update()
    assert exc.value.code == 2
