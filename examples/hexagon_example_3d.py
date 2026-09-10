"""Generate and solve a 3D hexagon-packaging pattern.

Run from the project root:

    python -m examples.hexagon_example_3d

You can also run this file directly from an IDE.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import build_packaging, solve_kinematics


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    build_packaging(
        pattern,
        l=15,
        alpha=2,
        beta=2,
        gamma=2,
        delta=3,
        enable_left_open=True,
        rotate_cavity_to_horizon=True,
        fill_cavity=True
    )

    model = Cadder.from_drawer(pattern)

    solve_kinematics(
        model,
        final_dihedral=135.0,
        steps=2,
        tol=1e-10,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    model.draw(
        # save_fig=True,
        # save_path=f"{OUTPUT_DIR}/3D_hexagon.png"
    )

    model.save_cad(filename=f"{OUTPUT_DIR}/3D_hexagon.step")

if __name__ == "__main__":
    main()
