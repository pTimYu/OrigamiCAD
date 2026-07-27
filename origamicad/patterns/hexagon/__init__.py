"""Hexagon crease-pattern generation and pattern-specific kinematics."""

from importlib import import_module
from typing import TYPE_CHECKING

from .kinematics import solve_kinematics, solve_simple_hexagon_kinematics
from .layout import (
    build_packaging,
    draw_hex_loops,
    hex_unit_chain,
    hexagon_packaging,
)
from .metadata import HexUnit
from .stacking import (
    LayerStackResult,
    layer_panel_levels,
    stack_layers,
    stack_simple_hexagon_layers,
)

if TYPE_CHECKING:
    from .insertion_simulation import (
        CombinationAttempt,
        DEFAULT_ALL_ACUTE_PRECONTACT_INNER_DIHEDRAL_DEG,
        DEFAULT_INNER_DIHEDRAL_DEG,
        DEFAULT_MAX_NO_CLIPPING_INNER_DIHEDRAL_DEG,
        DEFAULT_NUM_LOOPS,
        HEXAGON_LOCK_ACUTE_DIHEDRAL_DEG,
        HEXAGON_LOCK_OBTUSE_DIHEDRAL_DEG,
        InsertionPanelState,
        InsertionSimulationResult,
        LoopDihedralStats,
        classify_insertion_panel_states,
        draw_insertion_simulation,
        draw_insertion_stack_3d,
        minimum_panel_clearance,
        print_insertion_report,
        simulate_insertion,
    )

_INSERTION_EXPORTS = {
    "CombinationAttempt",
    "DEFAULT_ALL_ACUTE_PRECONTACT_INNER_DIHEDRAL_DEG",
    "DEFAULT_INNER_DIHEDRAL_DEG",
    "DEFAULT_MAX_NO_CLIPPING_INNER_DIHEDRAL_DEG",
    "DEFAULT_NUM_LOOPS",
    "HEXAGON_LOCK_ACUTE_DIHEDRAL_DEG",
    "HEXAGON_LOCK_OBTUSE_DIHEDRAL_DEG",
    "InsertionPanelState",
    "InsertionSimulationResult",
    "LoopDihedralStats",
    "classify_insertion_panel_states",
    "draw_insertion_simulation",
    "draw_insertion_stack_3d",
    "minimum_panel_clearance",
    "print_insertion_report",
    "simulate_insertion",
}


def __getattr__(name: str):
    """Load the executable insertion module only when one of its APIs is used."""
    if name in _INSERTION_EXPORTS:
        module = import_module(f"{__name__}.insertion_simulation")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CombinationAttempt",
    "DEFAULT_ALL_ACUTE_PRECONTACT_INNER_DIHEDRAL_DEG",
    "DEFAULT_INNER_DIHEDRAL_DEG",
    "DEFAULT_MAX_NO_CLIPPING_INNER_DIHEDRAL_DEG",
    "DEFAULT_NUM_LOOPS",
    "HEXAGON_LOCK_ACUTE_DIHEDRAL_DEG",
    "HEXAGON_LOCK_OBTUSE_DIHEDRAL_DEG",
    "HexUnit",
    "InsertionPanelState",
    "InsertionSimulationResult",
    "LayerStackResult",
    "LoopDihedralStats",
    "build_packaging",
    "classify_insertion_panel_states",
    "draw_hex_loops",
    "draw_insertion_simulation",
    "draw_insertion_stack_3d",
    "hex_unit_chain",
    "hexagon_packaging",
    "layer_panel_levels",
    "minimum_panel_clearance",
    "print_insertion_report",
    "simulate_insertion",
    "solve_kinematics",
    "solve_simple_hexagon_kinematics",
    "stack_layers",
    "stack_simple_hexagon_layers",
]
