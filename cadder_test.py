from simple_hexagon import pattern
from cadder import Cadder

model = Cadder.from_drawer(pattern)

model.print_simple_hexagon_metadata_summary()

info = model.add_simple_hexagon_kinematic_constraints(
    target_dihedral=175.0,   # start close to flat
    unit="deg",
    # Anchor a valley panel because all valley panels are constrained to z=0.
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
    final_dihedral=150.0,
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

# Optional exports of the solved configuration. Thickness uses model units (mm).
# model.save_json("simple_hexagon_3d.json")
# model.save_stl("simple_hexagon.stl", thickness=1.0)
model.save_step("simple_hexagon.step")
# The extension-dispatching equivalent is:
# model.save_cad("simple_hexagon.step", thickness=1.0)

model.draw(
    show_points=False,
    show_point_ids=False,
    show_line_ids=False,
    show_surface_ids=False,
    figsize=(10, 10),
)
