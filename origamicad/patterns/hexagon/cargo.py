"""Cargo dimensions for a folded hexagon packaging pattern."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple


class CargoSize(NamedTuple):
    """Transverse and longitudinal cargo lengths."""

    transverse: float
    longitudinal: float


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
    side_length = _finite_float(l, "l")
    angle = _finite_float(theta, "theta")
    transverse_parameter = _finite_float(gamma, "gamma")
    longitudinal_parameter = _finite_float(delta, "delta")

    if side_length <= 0.0:
        raise ValueError("l must be positive.")

    if unit == "deg":
        angle = math.radians(angle)
    elif unit != "rad":
        raise ValueError("unit must be 'rad' or 'deg'.")

    if not 0.0 <= angle <= math.pi:
        raise ValueError("theta must be between 0 and 180 degrees, inclusive.")

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


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc

    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


__all__ = ["CargoSize", "calculate_cargo_size"]
