"""Argument checks shared by the pattern builders, solvers, and exporters."""

from __future__ import annotations

import math


def integer(name: str, value, minimum: int | None = None) -> int:
    """Return ``value`` as an int; reject booleans, fractions, and non-numbers."""
    try:
        number = int(value)
        valid = not isinstance(value, bool) and number == value
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return number


def finite(
    name: str,
    value,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Return ``value`` as a finite float; reject booleans, text, NaN, and infinity."""
    kind = "positive " if positive else "non-negative " if nonnegative else ""
    try:
        number = math.nan if isinstance(value, (bool, str, bytes)) else float(value)
    except (TypeError, ValueError, OverflowError):
        number = math.nan
    if (
        not math.isfinite(number)
        or (positive and number <= 0.0)
        or (nonnegative and number < 0.0)
    ):
        raise ValueError(f"{name} must be a finite {kind}number; got {value!r}.")
    return number
