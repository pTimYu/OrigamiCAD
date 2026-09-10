"""Generate and solve the two-loop hexagon pattern in 3D.

Run from the project root:

    python -m examples.simple_hexagon_example_3d

You can also run this file directly from an IDE.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import draw_hex_loops, solve_kinematics
from origamicad.io.cad_export import save_cad

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    draw_hex_loops(
        pattern,
        n=2,
        start_point=(0, 0),
        l=15,
    )

    model = Cadder.from_drawer(pattern)

    solve_kinematics(
        model,
        final_dihedral=136.03,
        steps=4,
        tol=1e-10,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    save_cad(model, f"{OUTPUT_DIR}/3D_simple_hexagon_136_03deg.step")

    model.draw(
        figsize=(10, 10),
        # save_fig=True,
        # save_path=f"{OUTPUT_DIR}/3D_simple_hexagon.png"
    )


if __name__ == "__main__":
    main()
