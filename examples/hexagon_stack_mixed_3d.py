"""Build a mixed ten-layer hexagon-packaging stack.

The bottom four layers use a completely filled packaging pattern. The top
six layers use the matching pattern with its cavity open on the left. Run
from the project root with:

    python -m examples.hexagon_stack_mixed_3d
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import TwoDDrawer
from origamicad.io.cad_export import save_cad
from origamicad.patterns.hexagon import (
    build_packaging,
    stack_mixed_layers,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SIDE_LENGTH = 15.0
TARGET_DIHEDRAL_DEG = 135.0
BOTTOM_FILLED_LAYERS = 4
TOP_CAVITY_LAYERS = 6
GEOMETRY_TOLERANCE = 1e-5


def build_layer_pattern(*, fill_cavity: bool) -> TwoDDrawer:
    """Build one of the two aligned layer patterns used by the stack."""
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    build_packaging(
        pattern,
        l=15,
        alpha=2,
        beta=2,
        gamma=2,
        delta=3,
        # Keep this True for both patterns. fill_cavity=True overrides the
        # opening in the bottom pattern, while the common flag makes the
        # horizon rotation use the same center for both patterns.
        enable_left_open=True,
        enable_hole_punch_outer=2.5,
        enable_hole_punch_cavity=2.5,
        rotate_cavity_to_horizon=True,
        fill_cavity=fill_cavity,
    )
    return pattern


def main() -> None:
    filled_pattern = build_layer_pattern(fill_cavity=True)
    cavity_pattern = build_layer_pattern(fill_cavity=False)

    stack = stack_mixed_layers(
        [filled_pattern, cavity_pattern],
        layer_counts=[BOTTOM_FILLED_LAYERS, TOP_CAVITY_LAYERS],
        tolerance=GEOMETRY_TOLERANCE,
        require_upper_support=True,
        reference_layer=0,
        include_hole_punches=True,
        solve_options={
            "final_dihedral": TARGET_DIHEDRAL_DEG,
            "tol": 1e-10,
        },
    )
    assembly = stack["model"]
    mixed_interface = stack["interfaces"][BOTTOM_FILLED_LAYERS - 1]

    OUTPUT_DIR.mkdir(exist_ok=True)
    step_path = OUTPUT_DIR / "hexagon_stack_mixed_3d.step"
    json_path = OUTPUT_DIR / "hexagon_stack_mixed_3d.json"
    save_cad(assembly, step_path, separate_layer_parts=True)
    save_cad(assembly, json_path)

    print(f"Filled bottom layers:           {BOTTOM_FILLED_LAYERS}")
    print(f"Open-cavity top layers:        {TOP_CAVITY_LAYERS}")
    print(f"Solved layer height:           {stack['layer_height']:.6f} mm")
    print(
        "Mixed adhesive triangles:      "
        f"{mixed_interface['num_matched_surfaces']}"
    )
    print(
        "Mixed adhesive area:           "
        f"{mixed_interface['adhesive_area']:.6f} mm^2"
    )
    print(
        "Maximum mixed-interface error: "
        f"{mixed_interface['max_interface_error']:.3e} mm"
    )
    print(f"Saved STEP assembly:           {step_path}")
    print(f"Saved JSON assembly:           {json_path}")

    assembly.draw(
        show_surfaces=True,
        figsize=(11, 10),
    )


if __name__ == "__main__":
    main()
