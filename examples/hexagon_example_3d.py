"""Interactive simple-hexagon example.

Run from the project root:

    python -m examples.interactive_simple_hexagon

You can also run this file directly from an IDE.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon_packaging import hexagon_packaging


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    hexagon_packaging(
    pattern,
    start_point=(0, 0),
    l=15,
    alpha=2,
    beta=2,
    gamma=3,
    delta=4
    )

    model = Cadder.from_drawer(pattern)

    model.solve_simple_hexagon_kinematics(
        final_dihedral=135.0,
        start_dihedral=175.0,
        steps=2,
        unit="deg",
        # Anchor a valley panel because all valley panels are constrained to z=0.
        fixed_triangle_surface_id="tri_0_1",
        valley_z=0.0,
        strict_unique_edges=False,
        mountain_height=2.0,
        valley_height=0.0,
        max_nfev_per_step=8000,
        tol=1e-10,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    model.draw(
        # save_fig=True,
        # save_path=f"{OUTPUT_DIR}/3D_hexagon.png"
    )


if __name__ == "__main__":
    main()
