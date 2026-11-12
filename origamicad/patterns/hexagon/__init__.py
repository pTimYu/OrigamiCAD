"""Hexagon crease-pattern generation and pattern-specific kinematics."""

from importlib import import_module
from typing import TYPE_CHECKING

from ...core.clearance import minimum_panel_clearance
from .cargo import CargoSize, calculate_cargo_height, calculate_cargo_size
from .kinematics import analyze_kinematics, solve_kinematics
from .layout import build_packaging, draw_hex_loops, hex_unit_chain
from .metadata import HexUnit, iter_local_creases
from .stacking import (
    LayerInterfaceResult,
    LayerStackResult,
    MixedLayerStackResult,
    layer_panel_levels,
    stack_layers,
    stack_mixed_layers,
)

if TYPE_CHECKING:
    from .insertion_simulation import (
        DEFAULT_INNER_DIHEDRAL_DEG,
        DEFAULT_NUM_LOOPS,
        InsertionPanelState,
        InsertionSimulationResult,
        LoopDihedralStats,
        classify_insertion_panel_states,
        draw_insertion_simulation,
        draw_insertion_stack_3d,
        print_insertion_report,
        simulate_insertion,
    )

# Loaded on first use, because the insertion module needs the plotting stack.
_INSERTION_EXPORTS = [
    "DEFAULT_INNER_DIHEDRAL_DEG",
    "DEFAULT_NUM_LOOPS",
    "InsertionPanelState",
    "InsertionSimulationResult",
    "LoopDihedralStats",
    "classify_insertion_panel_states",
    "draw_insertion_simulation",
    "draw_insertion_stack_3d",
    "print_insertion_report",
    "simulate_insertion",
]


def __getattr__(name: str):
    if name in _INSERTION_EXPORTS:
        return getattr(import_module(f"{__name__}.insertion_simulation"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted([
    "CargoSize",
    "HexUnit",
    "LayerInterfaceResult",
    "LayerStackResult",
    "MixedLayerStackResult",
    "analyze_kinematics",
    "build_packaging",
    "calculate_cargo_height",
    "calculate_cargo_size",
    "draw_hex_loops",
    "hex_unit_chain",
    "iter_local_creases",
    "layer_panel_levels",
    "minimum_panel_clearance",
    "solve_kinematics",
    "stack_layers",
    "stack_mixed_layers",
    *_INSERTION_EXPORTS,
])
