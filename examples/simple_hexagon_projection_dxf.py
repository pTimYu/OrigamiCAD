from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import draw_hex_two_loops, solve_kinematics


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DXF_PATH = OUTPUT_DIR / "simple_hexagon_projection.dxf"


def main() -> None:
    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    draw_hex_two_loops(
        pattern,
        start_point=(0, 0),
        l=15,
    )

    model = Cadder.from_drawer(pattern)

    solve_kinematics(
        model,
        final_dihedral=150.0,
        start_dihedral=175.0,
        steps=4,
        unit="deg",
        # Anchor a valley panel because all valley panels are constrained to z=0.
        fixed_triangle_surface_id="tri_0_1",
        valley_z=0.0,
        strict_unique_edges=False,
        mountain_height=2.0,
        valley_height=0.0,
        max_nfev_per_step=8000,
        tol=1e-10,
        verbose=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = model.save_xy_dxf(
        DXF_PATH,
        include_creases=True,
        crease_style="dashed",
        include_construction=False,
        include_rigid=True,
        include_side=True,
        point_tol=1e-6,
        profile="solidworks",
    )
    print(f"Saved projected DXF to {saved_path}")


if __name__ == "__main__":
    main()
