"""
Predicate functions for filtering nightskycam images.

This module provides predicate functions for use with the filter_and_export_images function.
These predicates can be passed directly or used with functools.partial for parameterization.
"""

from functools import partial
from pathlib import Path
import re
from typing import Callable, Container, List, Optional, Union

from loguru import logger
import toml


def _parse_process_expression(expression: str, process_value: str) -> bool:
    """
    Parse and evaluate a boolean expression for substring matching.

    Supports:
    - Single substrings: "stretching"
    - AND logic: "stretching and 8bits" or "'stretching' and '8bits'"
    - OR logic: "stretching or 8bits" or "'stretching' or '8bits'"
    - Mixed: "stretching and (8bits or 16bits)"

    Parameters
    ----------
    expression
        Boolean expression with substring matching.
    process_value
        The process value string to search in.

    Returns
    -------
    bool
        Result of evaluating the expression.

    Examples
    --------
    >>> _parse_process_expression("stretching", "auto-stretching applied")
    True
    >>> _parse_process_expression("stretching and 8bits", "auto-stretching 8bits")
    True
    >>> _parse_process_expression("stretching or 8bits", "8bits conversion")
    True
    >>> _parse_process_expression("stretching and 8bits", "stretching 16bits")
    False
    """
    # Simple case: no logical operators
    if " and " not in expression.lower() and " or " not in expression.lower():
        # Remove quotes if present
        cleaned = expression.strip().strip("'\"")
        return cleaned in process_value

    # Replace quoted strings with tokens to preserve them
    quoted_strings = []

    def replace_quoted(match):
        quoted_strings.append(match.group(1))
        return f"__TOKEN_{len(quoted_strings) - 1}__"

    # Replace single and double quoted strings
    expr = re.sub(r"'([^']*)'", replace_quoted, expression)
    expr = re.sub(r'"([^"]*)"', replace_quoted, expr)

    # Split by 'and' and 'or' while preserving parentheses
    # Convert to Python boolean expression
    # Replace 'and' with ' and ' and 'or' with ' or '
    expr = re.sub(r"\band\b", "and", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bor\b", "or", expr, flags=re.IGNORECASE)

    # Build evaluation by checking each substring
    def check_substring(token: str) -> bool:
        token = token.strip()
        # Check if it's a token reference
        if token.startswith("__TOKEN_"):
            idx = int(token.replace("__TOKEN_", "").replace("__", ""))
            return quoted_strings[idx] in process_value
        # Otherwise it's an unquoted substring
        return token in process_value

    # Parse the expression manually to handle AND/OR precedence
    # Split by OR first (lower precedence)
    or_parts = re.split(r"\s+or\s+", expr, flags=re.IGNORECASE)

    for or_part in or_parts:
        # Split by AND (higher precedence)
        and_parts = re.split(r"\s+and\s+", or_part.strip(), flags=re.IGNORECASE)

        # Check if all AND conditions are met
        and_result = all(check_substring(part.strip()) for part in and_parts)

        # If any OR part is true, return True
        if and_result:
            return True

    return False


def has_process_substring(
    image_path: Path, toml_path: Optional[Path], substring: str, default: bool = False
) -> bool:
    """
    Check if a substring expression matches the 'process' field of the TOML file.

    Supports boolean expressions with 'and'/'or' operators.

    Parameters
    ----------
    image_path
        Path to the image file.
    toml_path
        Path to the TOML metadata file.
    substring
        Substring or boolean expression to evaluate against the 'process' field.
        Examples:
        - "stretching" - simple substring
        - "stretching and 8bits" - both must be present
        - "'stretching' or '8bits'" - either must be present
        - "auto and (8bits or 16bits)" - complex expression
    default
        Default value to return if TOML cannot be read.

    Returns
    -------
    bool
        True if the expression matches the 'process' field.

    Examples
    --------
    >>> from functools import partial
    >>> predicate = partial(has_process_substring, substring="stretching")
    >>> predicate = partial(has_process_substring, substring="stretching and 8bits")
    """
    if toml_path is None or not toml_path.exists():
        logger.debug(f"TOML file not found for {image_path.name}, skipping")
        return False

    try:
        data = toml.load(toml_path)
        process_value = data.get("process", "")

        if not isinstance(process_value, str):
            logger.warning(
                f"'process' field is not a string in {toml_path.name}, skipping"
            )
            return default

        return _parse_process_expression(substring, process_value)

    except toml.decoder.TomlDecodeError as e:
        logger.error(f"Failed to parse TOML file {toml_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error reading {toml_path.name}: {e}")
        return False


def not_has_process_substring(
    image_path: Path, toml_path: Optional[Path], substring: str, default: bool = False
) -> bool:
    """
    Check if a substring expression does NOT match the 'process' field of the TOML file.

    Supports boolean expressions with 'and'/'or' operators.

    Parameters
    ----------
    image_path
        Path to the image file.
    toml_path
        Path to the TOML metadata file.
    substring
        Substring or boolean expression to evaluate against the 'process' field.
        Returns True if the expression does NOT match.
        Examples:
        - "stretching" - field must not contain "stretching"
        - "stretching and 8bits" - field must not contain both
        - "'stretching' or '8bits'" - field must not contain either
    default
        Default value to return if TOML cannot be read.

    Returns
    -------
    bool
        True if the expression does NOT match the 'process' field.

    Examples
    --------
    >>> from functools import partial
    >>> # Filter images that have NOT been stretched
    >>> predicate = partial(not_has_process_substring, substring="stretching")
    >>> # Filter images without stretching AND without 8bits
    >>> predicate = partial(not_has_process_substring, substring="stretching or 8bits")
    """
    if toml_path is None or not toml_path.exists():
        logger.debug(f"TOML file not found for {image_path.name}, skipping")
        return False

    try:
        data = toml.load(toml_path)
        process_value = data.get("process", "")

        if not isinstance(process_value, str):
            logger.warning(
                f"'process' field is not a string in {toml_path.name}, skipping"
            )
            return default

        return not _parse_process_expression(substring, process_value)

    except toml.decoder.TomlDecodeError as e:
        logger.error(f"Failed to parse TOML file {toml_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error reading {toml_path.name}: {e}")
        return False


def has_cloud_cover_in_range(
    image_path: Path,
    toml_path: Optional[Path],
    min_cover: int = 0,
    max_cover: int = 100,
    default: bool = False,
) -> bool:
    """
    Check if 'cloud_cover' is within a specified range.

    Parameters
    ----------
    image_path
        Path to the image file.
    toml_path
        Path to the TOML metadata file.
    min_cover
        Minimum cloud cover value (inclusive). Default: 0.
    max_cover
        Maximum cloud cover value (inclusive). Default: 100.

    Returns
    -------
    bool
        True if cloud_cover is within the specified range.
    """
    if toml_path is None or not toml_path.exists():
        logger.debug(f"TOML file not found for {image_path.name}, skipping")
        return False

    try:
        data = toml.load(toml_path)

        if "cloud_cover" not in data:
            logger.debug(f"'cloud_cover' key not found in {toml_path.name}")
            return False

        cloud_cover = data["cloud_cover"]

        if not isinstance(cloud_cover, int):
            logger.warning(
                f"'cloud_cover' field is not an integer in {toml_path.name}, skipping"
            )
            return default

        return min_cover <= cloud_cover <= max_cover

    except toml.decoder.TomlDecodeError as e:
        logger.error(f"Failed to parse TOML file {toml_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error reading {toml_path.name}: {e}")
        return False


def has_weather_value(
    image_path: Path,
    toml_path: Optional[Path],
    weather: Union[str, Container[str]],
    default: bool = False,
) -> bool:
    """
    Check if 'weather' field matches a specified value or is in a set of values.

    Parameters
    ----------
    image_path
        Path to the image file.
    toml_path
        Path to the TOML metadata file.
    weather
        Weather value(s) to match. Can be:
        - A single string for exact match (case-sensitive)
        - A container of strings (list, tuple, set) for matching any value

    Returns
    -------
    bool
        True if the weather field matches the specified value (or any value in container).
    """
    if toml_path is None or not toml_path.exists():
        logger.debug(f"TOML file not found for {image_path.name}, skipping")
        return False

    try:
        data = toml.load(toml_path)

        if "weather" not in data:
            logger.debug(f"'weather' key not found in {toml_path.name}")
            return default

        weather_value = data["weather"]

        if not isinstance(weather_value, str):
            logger.warning(
                f"'weather' field is not a string in {toml_path.name}, skipping"
            )
            return False

        # Check if weather is a string or a container
        if isinstance(weather, str):
            return weather_value == weather
        else:
            # Assume it's a container (list, tuple, set, etc.)
            return weather_value in weather

    except toml.decoder.TomlDecodeError as e:
        logger.error(f"Failed to parse TOML file {toml_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error reading {toml_path.name}: {e}")
        return False


def create_combined_predicate(
    process_substring: Optional[str] = None,
    process_not_substring: Optional[str] = None,
    cloud_cover_range: Optional[tuple[int, int]] = None,
    weather_values: Optional[Union[str, Container[str]]] = None,
) -> Optional[Callable[[Path, Optional[Path]], bool]]:
    """
    Factory function to create a combined predicate from multiple filter criteria.

    This function combines the available predicate functions (has_process_substring,
    not_has_process_substring, has_cloud_cover_in_range, has_weather_value) into a
    single predicate with AND logic. Only non-None criteria are included.

    Parameters
    ----------
    process_substring
        Substring that must be present in the 'process' field. None means no inclusion filter.
    process_not_substring
        Substring that must NOT be present in the 'process' field. None means no exclusion filter.
    cloud_cover_range
        Tuple of (min_cover, max_cover) for cloud cover filtering.
        None means no filtering on cloud cover.
    weather_values
        Weather value(s) to match. Can be a single string or container of strings.
        None means no filtering on weather.

    Returns
    -------
    Optional[Callable[[Path, Optional[Path]], bool]]
        Combined predicate function with signature (image_path, toml_path) -> bool,
        or None if no filters were specified.

    Examples
    --------
    >>> # Images that have been stretched AND have low cloud cover
    >>> predicate = create_combined_predicate(
    ...     process_substring="stretching",
    ...     cloud_cover_range=(0, 30)
    ... )
    >>> # Images that have NOT been stretched
    >>> predicate = create_combined_predicate(
    ...     process_not_substring="stretching"
    ... )
    """
    predicates = []

    # Add process inclusion filter if specified
    if process_substring is not None:
        predicates.append(partial(has_process_substring, substring=process_substring))

    # Add process exclusion filter if specified
    if process_not_substring is not None:
        predicates.append(
            partial(not_has_process_substring, substring=process_not_substring)
        )

    # Add cloud cover range filter if specified
    if cloud_cover_range is not None:
        min_cover, max_cover = cloud_cover_range
        predicates.append(
            partial(has_cloud_cover_in_range, min_cover=min_cover, max_cover=max_cover)
        )

    # Add weather value filter if specified
    if weather_values is not None:
        predicates.append(partial(has_weather_value, weather=weather_values))

    # If no predicates were specified, return None
    if not predicates:
        return None

    # If only one predicate, return it directly
    if len(predicates) == 1:
        return predicates[0]

    # Combine multiple predicates with AND logic
    def combined_predicate(image_path: Path, toml_path: Optional[Path]) -> bool:
        """Apply all predicates with AND logic."""
        return all(pred(image_path, toml_path) for pred in predicates)

    return combined_predicate
