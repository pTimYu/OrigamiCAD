"""Hexagon crease-pattern generation and pattern-specific kinematics."""

from ...core.clearance import minimum_panel_clearance
from .cargo import CargoSize, calculate_cargo_height, calculate_cargo_size
from .compact_fold import (
    CargoRectangle,
    CompactFold,
    cargo_rectangle,
    clipping_pairs,
    most_compact_fold,
)
from .insertion_simulation import (
    DEFAULT_INNER_DIHEDRAL_DEG,
    DEFAULT_NUM_LOOPS,
    InsertionPanelState,
    InsertionSimulationResult,
    SnapEvent,
    cavity_panel_ids,
    draw_insertion_simulation,
    draw_insertion_stack_3d,
    print_insertion_report,
    simulate_insertion,
)
from .kinematics import analyze_kinematics, solve_kinematics
from .layout import build_packaging, draw_hex_loops, hex_unit_chain
from .metadata import HexUnit, iter_local_creases
from .snap_lines import fold_with_snapped_panels, snap_lines
from .stacking import (
    LayerInterfaceResult,
    LayerStackResult,
    MixedLayerStackResult,
    layer_panel_levels,
    stack_layers,
    stack_mixed_layers,
)

__all__ = [
    "CargoRectangle",
    "CargoSize",
    "CompactFold",
    "DEFAULT_INNER_DIHEDRAL_DEG",
    "DEFAULT_NUM_LOOPS",
    "HexUnit",
    "InsertionPanelState",
    "InsertionSimulationResult",
    "LayerInterfaceResult",
    "LayerStackResult",
    "MixedLayerStackResult",
    "SnapEvent",
    "analyze_kinematics",
    "build_packaging",
    "calculate_cargo_height",
    "calculate_cargo_size",
    "cargo_rectangle",
    "cavity_panel_ids",
    "clipping_pairs",
    "draw_hex_loops",
    "draw_insertion_simulation",
    "draw_insertion_stack_3d",
    "fold_with_snapped_panels",
    "hex_unit_chain",
    "iter_local_creases",
    "layer_panel_levels",
    "minimum_panel_clearance",
    "most_compact_fold",
    "print_insertion_report",
    "simulate_insertion",
    "snap_lines",
    "solve_kinematics",
    "stack_layers",
    "stack_mixed_layers",
]
