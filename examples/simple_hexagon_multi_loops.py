"""Generate 2D and folded 3D plots of a three-loop hexagon pattern.

Run from the project root:

    python -m examples.simple_hexagon_multi_loops

You can also run this file directly from an IDE.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import draw_hex_loops, solve_kinematics


N_LOOPS = 3
SIDE_LENGTH = 15.0

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    units = draw_hex_loops(
        pattern,
        n=N_LOOPS,
        start_point=(0.0, 0.0),
        l=SIDE_LENGTH,
    )

    expected_units = 1 + 3 * N_LOOPS * (N_LOOPS - 1)
    print(f"Generated {len(units)} unit cells (expected {expected_units}).")
    pattern.print_summary()
    pattern.save_dxf(
            filename=f"{OUTPUT_DIR}/{N_LOOPS}loops_dxf.dxf",
            profile="solidworks"
    )
    pattern.draw(figsize=(12, 12))

    # model = Cadder.from_drawer(pattern)
    # solve_kinematics(
    #     model,
    #     final_dihedral=150.0,
    #     start_dihedral=175.0,
    #     steps=4,
    #     unit="deg",
    #     fixed_triangle_surface_id="tri_0_1",
    #     valley_z=0.0,
    #     strict_unique_edges=False,
    #     mountain_height=2.0,
    #     valley_height=0.0,
    #     max_nfev_per_step=8000,
    #     tol=1e-10,
    #     verbose=True,
    # )

    # model.draw(
    #     show_surfaces=True,
    #     figsize=(12, 12),
    #     view=(25, -60),
    # )


if __name__ == "__main__":
    main()
