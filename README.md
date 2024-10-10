[![Python package](https://github.com/MPI-IS/nightskycam-images/actions/workflows/tests.yml/badge.svg)](https://github.com/MPI-IS/nightskycam-images/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/nightskycam-images.svg)](https://pypi.org/project/nightskycam-images/)

> 🚧 **Under Construction**  
> This project is currently under development. Please check back later for updates.


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
* Python 3.9+

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
