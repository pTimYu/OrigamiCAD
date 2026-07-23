"""Hexagon crease-pattern generation and pattern-specific kinematics."""

from .kinematics import solve_kinematics, solve_simple_hexagon_kinematics
from .layout import (
    build_packaging,
    draw_hex_two_loops,
    hex_unit_chain,
    hexagon_packaging,
)
from .metadata import HexUnit
from .stacking import (
    LayerStackResult,
    analytical_layer_height,
    layer_panel_levels,
    stack_layers,
    stack_simple_hexagon_layers,
)

__all__ = [
    "HexUnit",
    "LayerStackResult",
    "analytical_layer_height",
    "build_packaging",
    "draw_hex_two_loops",
    "hex_unit_chain",
    "hexagon_packaging",
    "layer_panel_levels",
    "solve_kinematics",
    "solve_simple_hexagon_kinematics",
    "stack_layers",
    "stack_simple_hexagon_layers",
]
