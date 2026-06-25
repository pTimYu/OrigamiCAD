"""Example: solve, project, and export a simple-hexagon origami structure.

Run from the project root:

    python -m examples.simple_hexagon

You can also run this file directly from an IDE.

Generated JSON/DXF/STEP files are written to ``examples/output/`` and ignored
by git through `.gitignore`.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from origamicad import Cadder
from origamicad.patterns.simple_hexagon import pattern


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    model = Cadder.from_drawer(pattern)

    model.print_simple_hexagon_metadata_summary()

    info = model.add_simple_hexagon_kinematic_constraints(
        target_dihedral=175.0,
        unit="deg",
        fixed_triangle_surface_id="tri_0_1",
        valley_z=0.0,
        strict_unique_edges=False,
    )
    print(info)

    X0 = model.simple_hexagon_initial_guess(
        mountain_height=2.0,
        valley_height=0.0,
    )

    report = model.solve_simple_hexagon_continuation(
        final_dihedral=135.0,
        start_dihedral=175.0,
        steps=4,
        unit="deg",
        X0=X0,
        max_nfev_per_step=8000,
        tol=1e-10,
    )

    model.print_solve_report(report)
    model.print_dihedral_signed_status(max_items=20, unit="deg")

    if report.max_abs_residual > 1e-5:
        print("WARNING: constraints are not sufficiently satisfied.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / "simple_hexagon_3d.json"
    step_path = OUTPUT_DIR / "simple_hexagon.step"
    projected_json_path = OUTPUT_DIR / "simple_hexagon_projected_2d.json"
    projected_dxf_path = OUTPUT_DIR / "simple_hexagon_projected.dxf"

    model.save_json(json_path)
    model.save_step(step_path, thickness=0.0)

    projected = model.to_2d_drawer()
    projected.save_json(projected_json_path)
    projected.save_dxf(
        projected_dxf_path,
        include_creases=True,
        crease_style="dashed",
    )

    print(f"Saved {json_path}")
    print(f"Saved {step_path}")
    print(f"Saved {projected_json_path}")
    print(f"Saved {projected_dxf_path}")

    # If you only want cut boundaries without crease lines:
    # projected.save_dxf(
    #     OUTPUT_DIR / "simple_hexagon_cut_only.dxf",
    #     include_creases=False,
    # )

    # Uncomment for an interactive 3D preview.
    # model.draw(
    #     show_points=False,
    #     show_point_ids=False,
    #     show_line_ids=False,
    #     show_surface_ids=False,
    #     figsize=(10, 10),
    # )


if __name__ == "__main__":
    main()
