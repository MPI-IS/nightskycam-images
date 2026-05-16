---
name: nightskycam-database
description: Build, refresh, or inspect the SQLite metadata cache for a nightskycam-images dataset. Use before nightskycam-query if the cache is missing or stale, or when the user wants DB stats or the web viewer.
---

# nightskycam-database

The `nightskycam-images` package keeps a SQLite mirror of per-image TOML metadata so queries don't have to walk the filesystem. This skill covers populating, refreshing, and inspecting that cache.

## When to use

- Before any **nightskycam-query** call when the DB file does not exist or is suspected stale.
- After any **nightskycam-export** operation that modifies metadata (`ns.ml.classify`, `ns.ml.scorer`).
- When the user asks for "stats", "dataset summary", or wants the DB-backed web UI.

For querying the DB contents, defer to **nightskycam-query**. For producing files, defer to **nightskycam-export**.

## What gets stored

Two tables:

- **`images`** — one row per image stem with `root`, `system`, `date` (`YYYY_MM_DD`), `time` (`HH_MM_SS`), `datetime` (ISO), `nightstart_date`, `image_format`, `process`, `weather`, `cloud_cover`, `stretched`, `has_thumbnail`, `has_toml`.
- **`classifier_scores`** — `(image_id, classifier_name, probability)` for every classifier output found in the per-image TOML's `[classifiers]` section.

Plus a `scan_metadata` table holding `last_scan_timestamp` for incremental refresh.

## Default DB path

`<root>/.nightskycam_images.db`. Override with `--db-path` if the user wants the cache elsewhere (e.g. read-only roots).

## Primary tools

### `ns.db.update <root> [<second_root>] [--db-path PATH] [--full]` — populate or refresh

```bash
poetry run ns.db.update /data
poetry run ns.db.update /data /backup --db-path /tmp/nsc.db
poetry run ns.db.update /data --full     # ignore mtime cache, rescan everything
```

- **Incremental by default.** Date directories whose mtime is older than the last scan are skipped. Cheap to run repeatedly.
- **`--full`** forces a complete rescan. Use after a bulk metadata edit that doesn't bump the parent date dir's mtime, or after restoring from backup.
- **Two-root mode.** Pass an optional `second_root` to index two media trees into the same DB. Each row carries its source `root`, so paths resolve correctly.
- **Idempotent.** Safe to run on a partially-populated DB — uses `INSERT OR REPLACE` on `filename_stem`.

### `ns.db.stats <root>` — printable summary

```bash
poetry run ns.db.stats /data
```

Rich-formatted tables: total images, per-system counts and date ranges, weather distribution, missing-metadata counts (no_toml / no_process / no_weather / no_cloud_cover). Operates on the default DB path under `<root>`.

### `ns.db.web.view <db_path>` — Flask viewer for humans

```bash
poetry run ns.db.web.view /data/.nightskycam_images.db
```

Starts a Flask server (default `127.0.0.1:5002`-ish) that renders the DB. Tell the user the URL and exit; do not block waiting for them. Only invoke when the user explicitly asks for a UI.

## Programmatic access

If a step needs to populate inside Python (e.g. a notebook):

```python
from pathlib import Path
from nightskycam_images.db import populate, get_default_db_path

stats = populate(Path("/data"), full=False)
print(stats)        # {"images_scanned": ..., "images_upserted": ..., "errors": ..., ...}
print(get_default_db_path(Path("/data")))   # /data/.nightskycam_images.db
```

`populate` accepts a single root or a list of roots, an optional `db_path`, and `full=True` for a complete rescan.

## When to refresh

| Trigger                                                    | Run                                  |
| ---------------------------------------------------------- | ------------------------------------ |
| First-time setup on an unindexed root                       | `ns.db.update <root>`                |
| New images arrived in the dataset                           | `ns.db.update <root>` (incremental)  |
| `ns.ml.classify` / `ns.ml.scorer` updated TOMLs             | `ns.db.update <root>`                |
| TOML edits done by hand that don't bump the date-dir mtime  | `ns.db.update <root> --full`         |
| User reports stale stats or missing recent images           | Incremental first; `--full` if still off |

## Gotchas

- **Cache, not source of truth.** The filesystem is authoritative. If the user reports a discrepancy, re-run `ns.db.update <root> --full` before debugging.
- **Mtime-based incremental skip is fragile to in-place TOML edits.** Editing a `.toml` file does not change its parent directory's mtime on every filesystem. When in doubt, `--full`.
- **Default DB path is hidden** (leading dot). On some viewers it won't show — confirm existence with `ls -la <root> | grep nightskycam_images.db`.
- **Two-root caveat.** A given `(system, date)` pair should live in only one root. If the same date dir exists under both, the DB row's `root` reflects whichever was scanned last.
- **`ns.db.web.view` takes the DB path, not the root.** Distinct from `ns.files.web.view` which takes the root.

## Worked examples

### "Set up the cache for /data."

```bash
poetry run ns.db.update /data
poetry run ns.db.stats /data
```

### "I just classified everything — refresh the cache."

```bash
poetry run ns.db.update /data       # incremental picks up modified date dirs
```

### "I edited some TOMLs by hand and the stats look wrong."

```bash
poetry run ns.db.update /data --full
poetry run ns.db.stats /data
```

### "Open the web viewer."

```bash
poetry run ns.db.web.view /data/.nightskycam_images.db
# tell the user the URL printed by the server, then return control
```
