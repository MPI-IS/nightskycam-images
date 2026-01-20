# Nightskycam Images

This is the repository for `nightskycam_images`,
a python package for managing images captured by the camera-RaspberryPi systems of the nightskycam project.
These images are managed in a filesystem.

Its functions include:
* managing images (and related data) in a filesystem
* generating thumbnail images
* generating summary videos.

## Requirements

* Operating system: Linux or macOS
* Python 3.10+

## Getting Started as a User (using `pip`)

Dependency management with `pip` is easier to set up than with `poetry`, but the optional dependency-groups are not installable with `pip`.

* Create and activate a new Python virtual environment:
  ```bash
  python3 -m venv --copies venv
  source venv/bin/activate
  ```
* Update `pip` and build package:
  ```bash
  pip install -U pip  # optional but always advised
  pip install .       # -e option for editable mode
  ```

## Getting Started as a Developer (using `poetry`)

Dependency management with `poetry` is required for the installation of the optional dependency-groups.

* Install [poetry](https://python-poetry.org/docs/).
* Install dependencies for package
  (also automatically creates project's virtual environment):
  ```bash
  poetry install
  ```
* Install `dev` dependency group:
  ```bash
  poetry install --with dev
  ```
* Activate project's virtual environment:
  ```bash
  poetry shell
  ```
* Optional: Set up pre-commit git hook (automatic `isort` and `black` formatting):
  ```bash
  pre-commit install
  ```
  The hook will now run automatically on `git commit`. It is not recommended, but the hook can be bypassed with the option `--no-verify`.

  The hook can also be manually run with:
  ```bash
  # Force checking all files (instead of only changed files).
  pre-commit run --all-files
  ```

## Tests (only possible for setup with `poetry`, not with `pip`)

To install `test` dependency group:
```bash
poetry install --with test
```

To run the tests:
```bash
python -m pytest
```

To extract coverage data:
* Get code coverage by measuring how much of the code is executed when running the tests:
  ```bash
  coverage run -m pytest
  ```
* View coverage results:
  ```bash
  # Option 1: simple report in terminal.
  coverage report
  # Option 2: nicer HTML report.
  coverage html  # Open resulting 'htmlcov/index.html' in browser.
  ```

## Command-Line Executables

This package provides several command-line tools for managing and processing nightskycam images. All commands are available after installation via `pip` or `poetry`.

### `nightskycam-thumbnails`

Lists all thumbnail folder paths in a media root directory.

**Usage:**
```bash
nightskycam-thumbnails /path/to/media/root
```

**Output:** Prints paths to all `thumbnails/` subdirectories found in the nightskycam filesystem hierarchy.

**Use case:** Quickly discover all thumbnail directories for batch processing or verification.

---

### `nightskycam-filter-export`

Filters and exports images based on multiple criteria, creating symlinks to matching images in an output directory.

**Usage:**
```bash
# Create a default configuration file
nightskycam-filter-export --create-config

# Edit the generated nightskycam_filter_config.toml file, then run:
nightskycam-filter-export nightskycam_filter_config.toml

# Enable debug logging for detailed filter operations
nightskycam-filter-export nightskycam_filter_config.toml --debug
```

**Configuration options** (in TOML file):
- `root` (required): Path to media root directory
- `output_dir` (required): Path where symlinks will be created
- `systems`: List of system names to include (e.g., `["nightskycam5", "nightskycam6"]`)
- `start_date`, `end_date`: Date range filter (format: `YYYY-MM-DD`)
- `start_time`, `end_time`: Time window filter (format: `HH:MM`)
- `process`: Process field substring to match (supports `and`/`or` operators)
- `process_not`: Process field substring to exclude
- `cloud_min`, `cloud_max`: Cloud coverage percentage range (0-100)
- `weather`: List of weather values to match (e.g., `["clear", "partly_cloudy"]`)
- `cache_process`: Boolean - if true, checks only first image per folder for process field (performance optimization)

**Output:** Creates symlinks in `output_dir` preserving the `system/date/` directory structure. Both image files and their `.toml` metadata files are symlinked.

**Use case:** Extract subsets of images matching specific criteria (e.g., clear nights between 20:00-23:00 with low cloud cover) for analysis, archiving, or further processing.

---

### `nightskycam-filter-copy`

Copies a symlink directory (created by `nightskycam-filter-export`) and retargets the symlinks to point to files in a different root directory.

**Usage:**
```bash
nightskycam-filter-copy <input_symlink_dir> <output_dir> <new_root_dir>

# Enable debug logging
nightskycam-filter-copy <input_symlink_dir> <output_dir> <new_root_dir> --debug
```

**Arguments:**
- `input_symlink_dir`: Directory containing symlinks (from `nightskycam-filter-export`)
- `output_dir`: Destination directory for new symlinks (must not exist)
- `new_root_dir`: Root directory containing the target files to link to

**Output:** Creates a new directory structure with symlinks retargeted to `new_root_dir`. Useful when moving filtered sets between different storage locations or machines.

**Use case:** You've filtered images on one machine and want to create the same filtered structure pointing to images on another machine or storage location.

---

### `nightskycam-scorer-filter`

Filters images using a trained PyTorch classifier model from the `nightskycam-scorer` package. Creates symlinks to images that match the classification criteria.

**Usage:**
```bash
# Create a default configuration file
nightskycam-scorer-filter --create-config

# Edit the generated nightskycam_scorer_filter_config.toml file, then run:
nightskycam-scorer-filter nightskycam_scorer_filter_config.toml

# Enable debug logging to see per-image predictions
nightskycam-scorer-filter nightskycam_scorer_filter_config.toml --debug
```

**Configuration options** (in TOML file):
- `root` (required): Path to media root directory
- `output_dir` (required): Path where symlinks will be created
- `model_path` (required): Path to trained model file (`.pt`)
- `systems`: Optional list of systems to process
- `start_date`, `end_date`: Optional date range filter
- `classify_positive`: Boolean - if true, export positive predictions; if false, export negative predictions
- `probability_threshold`: Float (0.0-1.0) - classification threshold

**How it works:**
- Uses thumbnails for inference (fast classification)
- Creates symlinks to HD images (not thumbnails) that match criteria
- Preserves directory hierarchy: `output_dir/system/date/`
- Progress logged at INFO level, detailed predictions at DEBUG level

**Output:** Symlinks to images classified according to the specified criteria, along with their `.toml` metadata files.

**Use case:** Automatically filter images based on learned quality criteria (e.g., "clear night sky suitable for observation" vs. "cloudy/unsuitable").

---

### `nightskycam-stats`

Generates comprehensive statistics reports for nightskycam images.

**Usage:**
```bash
nightskycam-stats /path/to/media/root
```

**Output:** Prints statistical summary including image counts, date coverage, system information, and metadata distributions.

**Use case:** Get an overview of your image collection, identify gaps in coverage, or generate reports for documentation.

---

### `nightskycam-view-webapp`

Starts a Flask web application for browsing and viewing nightskycam images.

**Usage:**
```bash
# Start with default settings (127.0.0.1:5002)
nightskycam-view-webapp /path/to/media/root

# Customize host and port
nightskycam-view-webapp /path/to/media/root --host 0.0.0.0 --port 8080

# Enable debug mode (auto-reloads on code changes)
nightskycam-view-webapp /path/to/media/root --debug
```

**Arguments:**
- `root_dir`: Path to media root (can be original structure or filtered output from `nightskycam-filter-export`)
- `--host`: Host to bind to (default: `127.0.0.1`)
- `--port`: Port to bind to (default: `5002`)
- `--debug`: Enable Flask debug mode

**Access:** Open `http://127.0.0.1:5002` (or configured host/port) in a web browser.

**Use case:** Browse, preview, and inspect images through a web interface instead of navigating the filesystem.

---

### `nightskycam-thumbnails-annotator-webapp`

Starts a Flask web application for annotating thumbnail images (labeling for machine learning).

**Usage:**
```bash
# Create a default configuration file
nightskycam-thumbnails-annotator-webapp --create-config

# Edit the generated nightskycam_thumbnails_annotator_config.toml file, then run:
nightskycam-thumbnails-annotator-webapp nightskycam_thumbnails_annotator_config.toml

# Customize host and port
nightskycam-thumbnails-annotator-webapp config.toml --host 0.0.0.0 --port 8080
```

**Configuration options** (in TOML file):
- `root_dir`: Path to media root directory
- `output_dir`: Path where annotations will be saved
- `systems`: List of systems to include (empty = all)
- `start_date`, `end_date`: Date range filter (empty = no limit)

**Access:** Open `http://127.0.0.1:5003` (or configured host/port) in a web browser.

**Use case:** Create labeled datasets for training image classifiers. The webapp presents images and allows users to assign labels/categories.

---

### `nightskycam-annotator`

Command-line interface for image annotation (Note: implementation may be in development - check source for current status).

**Use case:** Annotate images for training machine learning models via CLI instead of web interface.

---

### `nightskycam-classifier`

Command-line interface for running image classification tasks (Note: implementation may be in development - check source for current status).

**Use case:** Apply classifiers to images or manage classification workflows via CLI.

---

### `nightskycam-remove-selected`

Removes selected images (symlinks) and their metadata files from a filtered directory based on a list file.

**Usage:**
```bash
# Remove images listed in removal_list.txt from filtered directory
nightskycam-remove-selected /path/to/filtered_dir removal_list.txt

# Dry-run mode (preview without deleting)
nightskycam-remove-selected /path/to/filtered_dir removal_list.txt --dry-run

# Verbose logging
nightskycam-remove-selected /path/to/filtered_dir removal_list.txt --verbose
```

**List file format:** Each line should contain a relative path in format `system/date/filename`:
```
nightskycam5/2025_01_15/nightskycam5_2025_01_15_20_30_00.tiff
nightskycam6/2025_01_16/nightskycam6_2025_01_16_21_45_30.jpg
```

Lines starting with `#` are ignored (comments). Blank lines are skipped.

**What it does:**
- Removes image symlinks from the filtered directory
- Also removes corresponding `.toml` metadata files
- Only removes symlinks (safety feature - won't delete regular files)

**Output:** Statistics showing lines processed, images removed, TOMLs removed, and errors.

**Use case:** After reviewing filtered images, remove unwanted ones from the filtered set (e.g., manually curating results from automated filtering).

---

### `nightskycam-move-to-backup`

Moves original images to a backup directory based on symlinks in a filter-export directory. Moves thumbnails and cleans up empty folders.

**Usage:**
```bash
# Move originals to backup based on filter-export symlinks
nightskycam-move-to-backup /path/to/filter_output /path/to/backup

# Dry-run mode (preview without moving)
nightskycam-move-to-backup /path/to/filter_output /path/to/backup --dry-run

# Verbose logging
nightskycam-move-to-backup /path/to/filter_output /path/to/backup --verbose
```

**Arguments:**
- `filter_output_dir`: Directory containing symlinks (from `nightskycam-filter-export`)
- `backup_dir`: Destination backup directory (preserves system/date structure)

**What it does:**
1. Follows each symlink to find the original image file
2. Moves the original image to backup directory
3. Moves the corresponding `.toml` metadata file
4. Moves the corresponding thumbnail to backup (preserving system/date/thumbnails structure)
5. Cleans up empty directories (thumbnail folders, date folders, system folders)

**Output:** Statistics showing symlinks processed, images/TOMLs/thumbnails moved, and folders cleaned up.

**Use case:** Archive/backup specific images while removing them from the active collection. For example, moving cloudy images to backup storage to free up space in the main collection.

---

### `nightskycam-delete-from-other`

Deletes images from another root directory based on symlinks in a filter-export directory. This is useful when you have the same images in multiple locations and want to delete them from one location based on a filtered selection.

**Usage:**
```bash
# Delete matching images from another root (with confirmation prompt)
nightskycam-delete-from-other /path/to/filter_output /path/to/other_root

# Dry-run mode (preview without deleting)
nightskycam-delete-from-other /path/to/filter_output /path/to/other_root --dry-run

# Skip confirmation prompt (use with caution!)
nightskycam-delete-from-other /path/to/filter_output /path/to/other_root --yes

# Verbose logging
nightskycam-delete-from-other /path/to/filter_output /path/to/other_root --verbose
```

**Arguments:**
- `filter_output_dir`: Directory containing symlinks (from `nightskycam-filter-export`)
- `other_root_dir`: Root directory from which to delete matching images

**What it does:**
1. Walks through the filter-export directory (symlinks)
2. For each symlink, extracts the relative path (system/date/filename)
3. Looks for the corresponding file in `other_root_dir`
4. If found, deletes the image, `.toml` metadata file, and thumbnail from `other_root_dir`
5. Cleans up empty directories (thumbnail folders, date folders, system folders) in `other_root_dir`
6. **IMPORTANT**: Only deletes files from `other_root_dir`, NEVER the symlinks themselves

**Safety features:**
- Shows a preview of files to be deleted before proceeding
- Requires user to type "DELETE" to confirm (unless `--yes` flag is used)
- Supports `--dry-run` mode to preview without deleting
- Fail-fast behavior: stops immediately if any error occurs
- Warns prominently that operation is destructive and cannot be undone

**Output:** Statistics showing symlinks processed, images/TOMLs/thumbnails deleted from other root, images not found, and folders cleaned up.

**Use case:** You have images stored in two locations (e.g., local SSD and network storage). You filter images on the local SSD using `nightskycam-filter-export`, then want to delete those same filtered images from the network storage to free up space. This command looks up the corresponding files in the network storage and deletes them, without touching your local filtered symlinks.

**Example workflow:**
```bash
# Step 1: Filter cloudy images on local SSD
nightskycam-filter-export config_cloudy.toml
# (creates symlinks in /filtered/cloudy/ pointing to /local/ssd/images/)

# Step 2: Preview deletion from network storage
nightskycam-delete-from-other /filtered/cloudy/ /network/storage/images/ --dry-run

# Step 3: Delete from network storage (asks for confirmation)
nightskycam-delete-from-other /filtered/cloudy/ /network/storage/images/
# Type "DELETE" when prompted

# Result: Cloudy images are deleted from /network/storage/images/
#         Your /filtered/cloudy/ symlinks and /local/ssd/images/ remain untouched
```

---

### `nightskycam-move-clear-images`

Moves clear-weather night images (22:00-04:00) to a target directory based on metadata filters.

**Usage:**
```bash
# Move clear night images from source to target
nightskycam-move-clear-images /path/to/source_root /path/to/target_dir

# Dry-run mode (preview without moving)
nightskycam-move-clear-images /path/to/source_root /path/to/target_dir --dry-run

# Verbose logging
nightskycam-move-clear-images /path/to/source_root /path/to/target_dir --verbose
```

**Filter criteria:**
- Weather metadata contains "clear"
- Image time between 22:00 and 04:00

**What it does:**
1. Scans all images in source root
2. Filters by weather="clear" and time window 22:00-04:00
3. Moves matching images to target directory (preserves system/date structure)
4. Moves corresponding `.toml` metadata files
5. Skips images that already exist in target
6. Cleans up empty directories after moving

**Output:** Statistics showing images scanned, matched, moved, skipped (already exist), and errors. Also reports cleanup statistics (folders deleted).

**Use case:** Extract the highest-quality observation images (clear weather during core night hours) for archiving, sharing, or creating curated collections.

---

### `nightskycam-copy-thumbnails`

Copies thumbnails for images listed in a text file to a flat destination directory.

**Usage:**
```bash
# Copy thumbnails to destination directory
nightskycam-copy-thumbnails image_list.txt /path/to/media/root /path/to/dest_dir

# Dry-run mode (preview without copying)
nightskycam-copy-thumbnails image_list.txt /path/to/media/root /path/to/dest_dir --dry-run

# Debug logging
nightskycam-copy-thumbnails image_list.txt /path/to/media/root /path/to/dest_dir --debug
```

**List file format:** Each line should contain a relative path in format `system/date/filename`:
```
nightskycam5/2025_08_25/nightskycam5_2025_08_25_23_30_03.tiff
nightskycam6/2025_08_26/nightskycam6_2025_08_26_20_15_00.jpg
```

Blank lines and lines starting with `#` are ignored.

**What it does:**
- Reads image paths from list file
- Locates corresponding thumbnails in the nightskycam hierarchy
- Copies thumbnails to destination directory with flat structure (no subdirectories)
- Skips thumbnails that already exist in destination
- Warns about missing thumbnails

**Output:** Statistics showing lines processed, thumbnails copied, already existing (skipped), missing thumbnails, and errors.

**Use case:** Create a flat collection of thumbnails for quick preview, sharing, or importing into annotation tools that expect flat directory structures.
