"""Solve insertion from a ``Cadder`` model and export its XY projection.

Run from the project root:

    python -m examples.simple_hexagon_insertion_projection_dxf
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import draw_hex_loops, simulate_insertion


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DXF_PATH = OUTPUT_DIR / "simple_hexagon_insertion_projection.dxf"


def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    draw_hex_loops(pattern, n=2, start_point=(0.0, 0.0), l=15.0)
    model = Cadder.from_drawer(pattern)

    result = simulate_insertion(
        model=model,
        inner_dihedral_deg=136.03,
        num_layers=4,
        # num_loops and side_length are inferred from model.
        verbose=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = result["model"].save_xy_dxf(
        DXF_PATH,
        include_creases=True,
        crease_style="dashed",
        include_construction=False,
        include_rigid=True,
        include_side=True,
        point_tol=1e-6,
        profile="solidworks",
    )
    print(f"Saved insertion projection DXF to {saved_path}")


if __name__ == "__main__":
    main()
