"""Solve and export a four-layer stack of the two-loop hexagon pattern.

Run from the project root:

    python -m examples.simple_hexagon_four_layer_stack

Odd-numbered layers reverse the mountain/valley assignment.  Their geometry is
the vertical reflection of the solved base layer, so every upper valley
triangle rests on a mountain triangle from the layer below.
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.io.cad_export import save_cad
from origamicad.patterns.hexagon import (
    analytical_layer_height,
    draw_hex_two_loops,
    solve_kinematics,
    stack_layers,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SIDE_LENGTH = 15.0
TARGET_DIHEDRAL_DEG = 135.0
NUM_LAYERS = 4


def main() -> None:
    base_pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    draw_hex_two_loops(
        base_pattern,
        start_point=(0.0, 0.0),
        l=SIDE_LENGTH,
        reverse=False,
    )

    base_model = Cadder.from_drawer(base_pattern)
    solve_kinematics(
        base_model,
        final_dihedral=TARGET_DIHEDRAL_DEG,
        start_dihedral=175.0,
        steps=4,
        unit="deg",
        fixed_triangle_surface_id="tri_0_1",
        valley_z=0.0,
        strict_unique_edges=False,
        mountain_height=2.0,
        valley_height=0.0,
        max_nfev_per_step=8000,
        tol=1e-10,
        verbose=True,
    )

    # d is the perpendicular distance between the two parallel crease axes
    # across a quadrilateral panel.
    d = SIDE_LENGTH * np.sqrt(3.0) / 2.0
    expected_layer_height = analytical_layer_height(
        panel_distance=d,
        dihedral_angle=TARGET_DIHEDRAL_DEG,
        unit="deg",
    )
    stack = stack_layers(
        base_model,
        num_layers=NUM_LAYERS,
        expected_layer_height=expected_layer_height,
        tolerance=1e-6 * max(1.0, SIDE_LENGTH),
    )
    assembly = stack["model"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "simple_hexagon_four_layer_stack.step"
    save_cad(assembly, output_path)

    print(f"Analytical layer height: {expected_layer_height:.6f} mm")
    print(f"Solved layer height:     {stack['layer_height']:.6f} mm")
    print(f"Maximum interface error: {stack['max_interface_error']:.3e} mm")
    print(f"Saved four-layer stack:  {output_path}")

    assembly.draw(
        show_surfaces=True,
        figsize=(10, 10),
    )


if __name__ == "__main__":
    main()
