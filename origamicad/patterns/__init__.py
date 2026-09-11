"""Reusable crease-pattern generators and pattern-specific operations."""

from .hexagon import (
    CargoSize,
    analyze_kinematics,
    build_packaging,
    calculate_cargo_height,
    calculate_cargo_size,
    iter_local_creases,
    solve_kinematics,
    stack_layers,
    stack_mixed_layers,
)

__all__ = [
    "CargoSize",
    "analyze_kinematics",
    "build_packaging",
    "calculate_cargo_height",
    "calculate_cargo_size",
    "iter_local_creases",
    "solve_kinematics",
    "stack_layers",
    "stack_mixed_layers",
]
