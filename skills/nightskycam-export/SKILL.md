---
name: nightskycam-export
description: Materialize files from a nightskycam-images dataset — symlink subsets, classifier-filtered copies, thumbnails, backups. Use whenever the user wants a new on-disk output rather than just a query.
---

# nightskycam-export

Drives the `ns.*` CLI to produce on-disk artifacts: symlinks, copies, thumbnails, backups. All commands run via `poetry run`.

## When to use

Trigger when the user wants files written, moved, deleted, or symlinked — anything that changes the filesystem. Common phrasings: "export", "create a folder of", "make symlinks to", "back up", "copy to", "delete", "generate thumbnails", "split into good/bad", "build a video summary".

For pure read queries, defer to **nightskycam-query**.

## How most commands are configured

Most non-trivial commands take a TOML config file (positional arg). Each supports `--create-config` to scaffold a starter file in the current directory; the user typically edits it and re-runs without the flag. Use `--debug` (or `--verbose` on file-mover commands) for detailed logging, and `--dry-run` on destructive ops.

Workflow when scaffolding:

```bash
poetry run ns.filter.export --create-config   # writes nightskycam_filter_config.toml
# edit the file
poetry run ns.filter.export nightskycam_filter_config.toml
```

You can also write the TOML directly with the schemas below — no need to scaffold first if you already know what the user wants.

## Filtering & symlink subsets

### `ns.filter.export <config.toml>` — create a symlinked subset

Walks a root, applies metadata filters, creates symlinks (image + .toml) under `output_dir/<system>/<date>/`. Thumbnails are also symlinked.

Config schema (dates use `YYYY-MM-DD`, times use `HH:MM`):

```toml
root = "/path/to/media/root"
output_dir = "/path/to/output"
systems = ["nightskycam5", "nightskycam6"]      # empty/missing = all
start_date = "2025-01-01"
end_date = "2025-12-31"
start_time = "20:00"                             # optional time window
end_time = "23:00"
process = "stretching and 8bits"                 # substring with and/or operators
process_not = "bad_substring"                    # substring to exclude
cloud_min = 0
cloud_max = 30
weather = ["clear", "partly_cloudy"]             # OR-combined
cache_process = false                            # true = check process only on first image per folder
nb_images = 1000                                 # cap total
folder_step = 2                                  # take every Nth date folder
```

### `ns.filter.copy <input_dir> <output_dir> <new_root>` — retarget existing symlinks

Takes a previously-exported symlink tree and rewrites the targets to point inside `<new_root>`. Use when the user wants the same filtered selection but pointing at a different storage location.

### `ns.filter.scorer <config.toml>` — filter via a single classifier

Runs a single SkyScorer model on thumbnails, symlinks images that pass.

```toml
root = "/path/to/media/root"
output_dir = "/path/to/output"
model_path = "/path/to/best_model.pt"
systems = ["nightskycam5", "nightskycam7"]
start_date = "2025-01-01"
end_date = "2025-12-31"
classify_positive = true                         # true = export positives, false = negatives
probability_threshold = 0.5
```

## Classifier scoring

### `ns.ml.classify <config.toml>` — write scores into per-image TOMLs

Runs multiple models and updates each image's `.toml` with a `[classifiers]` table. After this, **nightskycam-query** can filter by `classifier_max={...}`. Re-run `ns.db.update` afterward so the SQLite cache picks up the new scores.

```toml
root = "/path/to/media/root"
systems = ["nightskycam5"]
start_date = "2025-01-01"
end_date = "2025-12-31"

[models]
quality = "/path/to/quality_model.pt"
clouds = "/path/to/clouds_model.pt"
```

### `ns.ml.scorer <config.toml>` — multi-model copy into filtered/not-filtered

Runs an array of models and copies images (with metadata) into `filtered_dir` or `not_filtered_dir` based on each model's threshold and direction.

```toml
root = "/path/to/media/root"
filtered_dir = "/path/to/filtered/output"
not_filtered_dir = "/path/to/not_filtered/output"
systems = ["nightskycam5", "nightskycam7"]
start_date = "2025-01-01"
end_date = "2025-12-31"

[[models]]
model_path = "/path/to/model1.pt"
probability_threshold = 0.7
classify_positive = true

[[models]]
model_path = "/path/to/model2.pt"
probability_threshold = 0.5
classify_positive = false
```

## Thumbnails

- `poetry run ns.thumb.create <root_dir> [--width 200] [--no-stretch] [--dry-run] [--verbose]` — generate JPEG thumbnails for any image missing one. Idempotent.
- `poetry run ns.thumb.list <root>` — print all `thumbnails/` directories.
- `poetry run ns.thumb.check <root>` — print images with no thumbnail.
- `poetry run ns.thumb.copy <list_file> <root> <dest_dir>` — copy thumbnails for images named in `<list_file>` (lines of `system/date/filename`) into a flat destination.

## File operations on existing exports

- `ns.files.backup <root> <backup_dir>` — move images + metadata to backup.
- `ns.files.move-clear <source_root> <target_dir>` — move clear-weather night images (22:00–04:00) into a target tree.
- `ns.files.delete-from-other <filter_output_dir> <other_root>` — delete from a *secondary* root the files referenced by symlinks in `<filter_output_dir>`. Always supports `--dry-run` and prompts for `DELETE` confirmation; pass `--yes` only when the user has explicitly authorized it.
- `ns.files.remove-selected <filter_dir> <list_file>` — remove specific symlinks (and their TOMLs) named in `<list_file>` from a filter-export directory.

## End-to-end pipeline

`ns.backup.route <config.toml>` runs the full backup pipeline: classify → route into good/bad backup subdirs → copy videos and metadata. Use only when the user explicitly asks for the full automated route, not for one-off filtering.

## Web UIs (interactive — only when asked)

- `ns.files.web.view <root>` — file-based viewer.
- `ns.files.web.annotate <config.toml>` — labelling UI on images directly.
- `ns.files.web.annotate-symlinks <filter_dir> <output_dir>` — label a filter-export by binning into positive/negative subdirs.

These start a Flask server. Tell the user the URL, do not block waiting for them.

## Gotchas

- **Date/time format split-brain.** Filter-export TOMLs use `YYYY-MM-DD` and `HH:MM`. The query skill's `ImageDB` uses `YYYY_MM_DD` and `HH_MM_SS`. Convert at the boundary.
- **Always offer `--dry-run` for destructive ops** (`ns.files.backup`, `ns.files.move-clear`, `ns.files.delete-from-other`, `ns.files.remove-selected`) before running for real. `delete-from-other` is irreversible and can affect a different machine — never pass `--yes` without explicit user authorization.
- **`ns.filter.export` writes symlinks.** Targets must remain reachable; if the user moves the source root afterward, run `ns.filter.copy` to retarget rather than re-exporting.
- **After classification, refresh the DB.** `ns.ml.classify` and `ns.ml.scorer` modify per-image TOMLs. The SQLite cache is stale until `ns.db.update <root>` runs (full refresh not needed — the incremental scan picks up modified date folders by mtime).
- **No standalone `--create-config` for `ns.ml.classify` / `ns.ml.scorer` is needed** — both accept `--create-config`. Use the inline schemas above when scaffolding fresh.
- **Broken entry points — never invoke.** `ns.ml.train` (→ missing `nightskycam_images.classifier.main`) and `ns.annotate` (→ missing `nightskycam_images.annotator.cli`).
- **HD image format priority** when resolving stems on disk: `npy → tiff → jpg → jpeg`. Thumbnails are always `.jpeg`.
- **Filename pattern.** Operations that derive a system+datetime from a filename expect `<system>_YYYY_MM_DD_HH_MM_SS.<ext>` (or the `DD_MM_YYYY_HH_MM_SS` variant). Files that don't match are silently skipped.

## Worked examples

### "Symlink all clear nights from cam1 in June 2025 with low cloud cover."

```bash
cat > /tmp/clear_june.toml <<'EOF'
root = "/data"
output_dir = "/data/exports/cam1_clear_june"
systems = ["cam1"]
start_date = "2025-06-01"
end_date = "2025-06-30"
start_time = "20:00"
end_time = "04:00"
weather = ["clear"]
cloud_min = 0
cloud_max = 20
EOF
poetry run ns.filter.export /tmp/clear_june.toml
```

### "Generate any missing thumbnails under /data."

```bash
poetry run ns.thumb.create /data --width 200
```

### "Score every image with the quality and clouds models, then refresh the DB."

```bash
poetry run ns.ml.classify /tmp/classify.toml      # writes [classifiers] into TOMLs
poetry run ns.db.update /data                     # incremental refresh
```
