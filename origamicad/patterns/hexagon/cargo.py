"""Cargo dimensions for a folded hexagon packaging pattern."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

from ..._validation import finite


class CargoSize(NamedTuple):
    """Transverse and longitudinal cargo lengths."""

    transverse: float
    longitudinal: float


def _side_and_angle(l: float, theta: float, unit: str) -> tuple[float, float]:
    """Validate the side length and return the dihedral angle in radians."""
    side_length = finite("l", l, positive=True)
    angle = finite("theta", theta)
    if unit == "deg":
        angle = math.radians(angle)
    elif unit != "rad":
        raise ValueError("unit must be 'rad' or 'deg'.")
    if not 0.0 <= angle <= math.pi:
        raise ValueError("theta must be between 0 and 180 degrees, inclusive.")
    return side_length, angle


def calculate_cargo_height(
    l: float,
    theta: float,
    *,
    unit: Literal["rad", "deg"] = "deg",
    valley_height: float = 0.0,
) -> float:
    """Calculate the folded cargo height from panel side and dihedral angle.

    The height is measured from the valley-panel level and follows the
    hexagon relation ``sqrt(3) / 2 * l * sin(theta)``. ``valley_height``
    translates the result to a non-zero valley-panel elevation.

    Parameters
    ----------
    l:
        Panel side length. It must be positive.
    theta:
        Dihedral angle, expressed in degrees by default.
    unit:
        Angle unit for ``theta``: ``"deg"`` or ``"rad"``.
    valley_height:
        Elevation of the valley panels. It defaults to zero.
    """
    side_length, angle = _side_and_angle(l, theta, unit)
    return finite("valley_height", valley_height) + (
        math.sqrt(3.0) / 2.0 * side_length * math.sin(angle)
    )


def calculate_cargo_size(
    l: float,
    theta: float,
    gamma: float,
    delta: float,
    *,
    unit: Literal["rad", "deg"] = "deg",
) -> CargoSize:
    """Calculate the available transverse and longitudinal cargo lengths.

    Parameters
    ----------
    l:
        Panel side length. It must be positive.
    theta:
        Dihedral angle, expressed in degrees by default.
    gamma:
        Transverse geometric parameter.
    delta:
        Longitudinal geometric parameter.
    unit:
        Angle unit for ``theta``: ``"deg"`` or ``"rad"``.

    Returns
    -------
    CargoSize
        A tuple-like result whose fields are ``transverse`` and
        ``longitudinal``. Both lengths use the same unit as ``l``.
    """
    side_length, angle = _side_and_angle(l, theta, unit)
    transverse_parameter = finite("gamma", gamma)
    longitudinal_parameter = finite("delta", delta)

    cosine = math.cos(angle)
    d = 7.0 - 12.0 * cosine + 9.0 * cosine**2
    sqrt_d = math.sqrt(d)

    transverse = (
        math.sqrt(3.0)
        * side_length
        / 4.0
        * (
            (transverse_parameter - 1.0) * sqrt_d
            - 2.0
            * (1.0 - cosine)
            * (1.0 + 3.0 * cosine)
            / sqrt_d
        )
    )
    longitudinal = (
        side_length
        / (4.0 * sqrt_d)
        * (
            9.0 * cosine**2
            - 5.0
            + 2.0 * (longitudinal_parameter - 1.0) * d
        )
    )

    return CargoSize(transverse=transverse, longitudinal=longitudinal)


__all__ = ["CargoSize", "calculate_cargo_height", "calculate_cargo_size"]
