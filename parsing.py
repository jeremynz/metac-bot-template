"""Deterministic parse-first helpers (project-backlog#613).

`main.py`'s four `_run_forecast_on_*` methods already require a fixed
final-answer format from the forecasting prompt (`Probability: ZZ%`,
`Percentile 10: XX`, ...). These functions parse that fixed format
directly, without an LLM call, and return the matching
`forecasting_tools` type -- or `None` when the text fails validation,
which signals the caller to fall back to the existing `structure_output`
LLM-parser path unchanged.

Pure functions: no network, no LLM, no side effects.
"""

from __future__ import annotations

import re
from datetime import datetime

from forecasting_tools import BinaryPrediction, DatePercentile, Percentile, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

REQUIRED_PERCENTILES = (10, 20, 40, 60, 80, 90)

_BINARY_RE = re.compile(r"Probability:\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_PERCENTILE_LINE_RE = re.compile(r"Percentile\s*(\d+)\s*:\s*(.+)", re.IGNORECASE)
_OPTION_LINE_RE = re.compile(r"^(?P<name>.+?)\s*:\s*(?P<value>-?\$?[\d,]*\.?\d+)\s*%?\s*$")
_NUMBER_RE = re.compile(
    r"""^\s*
    (?P<neg1>-)?
    \s*\$?\s*
    (?P<neg2>-)?
    (?P<num>[\d,]*\.?\d+)
    \s*
    (?P<suffix>[kKmMbB])?
    """,
    re.VERBOSE,
)
_SUFFIX_MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9}
_DATE_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}:\d{2}))?Z?$"
)


def parse_binary(text: str) -> BinaryPrediction | None:
    """Use the LAST "Probability: X%" match. X must be in [0, 100]."""
    matches = _BINARY_RE.findall(text)
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    if not (0 <= value <= 100):
        return None
    try:
        return BinaryPrediction(prediction_in_decimal=value / 100)
    except Exception:
        return None


def parse_multiple_choice(text: str, options: list[str]) -> PredictedOptionList | None:
    """Every option must be present (exact or case-insensitive name match).
    Probabilities must be >=0 and are normalised to sum to 1. A missing
    option means None."""
    parsed: dict[str, float] = {}
    for raw_line in text.splitlines():
        match = _OPTION_LINE_RE.match(raw_line.strip())
        if not match:
            continue
        value = _parse_number(match.group("value"))
        if value is None:
            continue
        parsed[match.group("name").strip()] = value

    probabilities: list[float] = []
    for option in options:
        if option in parsed:
            value = parsed[option]
        else:
            value = None
            for name, candidate in parsed.items():
                if name.lower() == option.lower():
                    value = candidate
                    break
        if value is None or value < 0:
            return None
        probabilities.append(value)

    total = sum(probabilities)
    if total <= 0:
        return None
    normalized = [value / total for value in probabilities]
    try:
        return PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=option, probability=probability)
                for option, probability in zip(options, normalized)
            ]
        )
    except Exception:
        return None


def parse_numeric_percentiles(text: str) -> list[Percentile] | None:
    """All six percentiles (10/20/40/60/80/90) must be present and
    strictly increasing. Handles commas, k/M suffixes, a leading $ and
    trailing units, and negatives."""
    found: dict[int, float] = {}
    for raw_line in text.splitlines():
        match = _PERCENTILE_LINE_RE.search(raw_line)
        if not match:
            continue
        percentile = int(match.group(1))
        if percentile not in REQUIRED_PERCENTILES:
            continue
        value = _parse_number(match.group(2))
        if value is None:
            continue
        found[percentile] = value

    if any(percentile not in found for percentile in REQUIRED_PERCENTILES):
        return None
    values = [found[percentile] for percentile in REQUIRED_PERCENTILES]
    if not all(earlier < later for earlier, later in zip(values, values[1:])):
        return None
    try:
        return [
            Percentile(percentile=percentile / 100, value=value)
            for percentile, value in zip(REQUIRED_PERCENTILES, values)
        ]
    except Exception:
        return None


def parse_date_percentiles(text: str) -> list[DatePercentile] | None:
    """All six percentiles must be present, ISO dates, strictly
    increasing."""
    found: dict[int, datetime] = {}
    for raw_line in text.splitlines():
        match = _PERCENTILE_LINE_RE.search(raw_line)
        if not match:
            continue
        percentile = int(match.group(1))
        if percentile not in REQUIRED_PERCENTILES:
            continue
        value = _parse_date(match.group(2))
        if value is None:
            continue
        found[percentile] = value

    if any(percentile not in found for percentile in REQUIRED_PERCENTILES):
        return None
    values = [found[percentile] for percentile in REQUIRED_PERCENTILES]
    if not all(earlier < later for earlier, later in zip(values, values[1:])):
        return None
    try:
        return [
            DatePercentile(percentile=percentile / 100, value=value)
            for percentile, value in zip(REQUIRED_PERCENTILES, values)
        ]
    except Exception:
        return None


def _parse_number(raw: str) -> float | None:
    """Parse a numeric-percentile value: commas, k/M/B suffixes, a
    leading $ (before or after a negative sign), and trailing units are
    all tolerated; anything without at least one digit fails."""
    match = _NUMBER_RE.match(raw.strip())
    if not match or not match.group("num"):
        return None
    try:
        value = float(match.group("num").replace(",", ""))
    except ValueError:
        return None
    suffix = match.group("suffix")
    if suffix:
        value *= _SUFFIX_MULTIPLIER[suffix.lower()]
    if match.group("neg1") or match.group("neg2"):
        value = -value
    return value


def _parse_date(raw: str) -> datetime | None:
    """Parse a strict ISO date/datetime (YYYY-MM-DD, optionally
    T/space-separated HH:MM:SS, optional trailing Z). Always returned
    naive (Z/timezone is dropped) so percentiles compare consistently."""
    cleaned = raw.strip().strip("\"'")
    match = _DATE_LINE_RE.match(cleaned)
    if not match:
        return None
    date_part, time_part = match.groups()
    try:
        return datetime.fromisoformat(f"{date_part}T{time_part or '00:00:00'}")
    except ValueError:
        return None
