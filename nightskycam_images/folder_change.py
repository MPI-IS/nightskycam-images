import os
from pathlib import Path
from typing import Optional


def folder_has_changed(
    folder: Path, history: Optional[dict[Path, Optional[float]]]
) -> bool:
    """
    Check whether the last-modified-time of the directory changed
    since the previous check.

    If there is no history of a previous check:
    return True and initialise history.

    Parameters
    ----------
    folder
        Path to directory.
    history
        Last modification times found in previous check:
        - key:
            Path to directory.
        - value:
            Time of last modification of directory.
    """
    history = history if history else {}

    # Float: number of seconds since epoch.
    last_modified_time = os.path.getmtime(folder)

    # Set to dummy value when key does NOT exist.
    previous_modified_time = history.get(folder, None)

    # If the folder has NO previous history
    # OR was modified since previous history,
    # update the history.
    if not previous_modified_time or previous_modified_time < last_modified_time:
        history[folder] = last_modified_time
        return True

    return False
