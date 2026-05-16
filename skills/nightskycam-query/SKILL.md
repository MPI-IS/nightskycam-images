---
name: nightskycam-query
description: Find images, thumbnails, or metadata in a nightskycam-images dataset by system, date, weather, cloud cover, or classifier score. Use whenever the user wants to list, count, or locate captures.
---

# nightskycam-query

Read-only access to a `nightskycam-images` dataset. Translate natural-language requests into queries against the SQLite metadata cache via the `ImageDB` Python API.

## When to use

Trigger on any user request that involves *finding* or *describing* nightskycam captures without producing new files. Examples:

- "How many clear-sky images do we have from cam3 in March?"
- "Give me thumbnail paths for all stretched images under 20% cloud cover."
- "What systems and date ranges are in this dataset?"
- "Which images has the `clouds` classifier scored below 0.3?"

If the user wants to *create* a filtered subset on disk (symlinks, copies, videos, thumbnails), defer to the **nightskycam-export** skill instead.

## Prerequisites

`ImageDB` reads from `<root>/.nightskycam_images.db`. If that file does not exist, run **nightskycam-database** first (`ns.db.update <root>`).

## Primary tool — Python API

Always prefer the Python API over the CLI. Run inside `poetry run python` or as a one-shot `poetry run python -c "..."`:

```python
from nightskycam_images.db_api import ImageDB

db = ImageDB("/path/to/root/.nightskycam_images.db")

# Inventory
db.systems()                 # ["cam1", "cam2", ...]
db.dates("cam1")             # ["2025_06_01", "2025_06_02", ...]
db.classifier_names()        # ["clouds", "quality", ...]
db.stats()                   # totals, per-system counts, weather distribution

# Filtered queries — every kwarg is optional and AND-combined
records = db.images(
    system="cam1",                 # or systems=["cam1", "cam2"]
    date="2025_06_15",             # or start_date / end_date
    start_time="22_00_00",         # crosses midnight if start_time > end_time
    end_time="04_00_00",
    weather=["clear", "partly"],   # OR-combined substrings
    cloud_cover_min=0,
    cloud_cover_max=30,
    stretched=True,                # see "Gotchas"
    image_format="jpg",
    has_thumbnail=True,
    classifier_max={"clouds": 0.3, "rainy": 0.5},
)

# Counting without materializing the rows — same kwargs as images()
n = db.count(system="cam1", weather=["clear"])

# Just the paths
hd_paths    = db.hd_paths(system="cam1", weather=["clear"])
thumb_paths = db.thumbnail_paths(system="cam1", weather=["clear"])

# Single image lookup by stem
rec = db.image("cam1_2025_06_15_22_30_00")
```

Each `ImageRecord` exposes `.hd_path`, `.thumbnail_path`, `.toml_path` (each `Optional[Path]`, `None` if the file is missing on disk), `.classifier_scores` (dict), plus the metadata columns (`system`, `date`, `time`, `datetime`, `weather`, `cloud_cover`, `process`, `stretched`, `image_format`, `has_thumbnail`, `has_toml`, `nightstart_date`).

## Fallback — CLI

For a quick human-readable summary without writing Python:

```bash
poetry run ns.db.stats <root>
```

The DB-backed Flask viewer (`ns.db.web.view <db_path>`) is for the user to browse interactively; only invoke it when they ask for a UI.

## Gotchas

- **Date / time format conversion.** `ImageDB` and the SQLite columns use `YYYY_MM_DD` and `HH_MM_SS`. If the user gives `2025-06-15` or `22:30`, convert to `2025_06_15` and `22_30_00` before passing to `ImageDB`. (The export skill's TOML configs use the dashed forms — do not mix them up.)
- **Time window crossing midnight.** Pass `start_time > end_time` (e.g. `start_time="22_00_00", end_time="04_00_00"`) and `ImageDB` handles the wraparound. Do not split into two queries.
- **`stretched` is derived, not stored verbatim.** It comes from the `process` TOML field containing the substring `"stretching"`. Querying `stretched=True` is the supported path; do not pattern-match on `process` yourself unless the user asked for a specific substring.
- **`classifier_max` is a *ceiling*, not equality.** `classifier_max={"clouds": 0.3}` returns images whose `clouds` score is `<= 0.3`. There is no `classifier_min` — if the user asks for *high* scores, invert the framing or run a custom SQL query against `classifier_scores`.
- **Files can vanish from disk.** A row in the DB only proves the metadata existed at last scan. `ImageRecord.hd_path` returns `None` if the file is gone. Filter with `if rec.hd_path is not None` before reading bytes.
- **HD image format priority.** When a stem has multiple formats on disk, resolution order is `npy → tiff → jpg → jpeg`. The `image_format` column reflects what was found first; pass `image_format="tiff"` (etc.) to filter explicitly.
- **Two roots are supported.** A populated DB may have rows from two different `root` columns. `ImageRecord` resolves paths per-row using its own `root`, so this is transparent — but be aware when summarizing "where the data lives".
- **`ns.ml.train` and `ns.annotate` are broken entry points.** Never invoke them; the underlying modules (`nightskycam_images.classifier.main`, `nightskycam_images.annotator.cli`) do not exist on disk.

## Worked examples

### "How many clear-sky stretched images from cam1 in June 2025 have a `clouds` classifier score below 0.2?"

```python
from nightskycam_images.db_api import ImageDB
db = ImageDB("/data/.nightskycam_images.db")
n = db.count(
    system="cam1",
    start_date="2025_06_01",
    end_date="2025_06_30",
    weather=["clear"],
    stretched=True,
    classifier_max={"clouds": 0.2},
)
print(n)
```

### "Give me thumbnail paths for nights between 22:00 and 04:00 in March."

```python
paths = db.thumbnail_paths(
    start_date="2025_03_01",
    end_date="2025_03_31",
    start_time="22_00_00",
    end_time="04_00_00",
    has_thumbnail=True,
)
for p in paths:
    print(p)
```

### "Summarize the dataset."

```python
import json
print(json.dumps(db.stats(), indent=2, default=str))
```
