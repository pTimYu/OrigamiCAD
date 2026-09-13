# OrigamiCAD

OrigamiCAD is a small Python toolkit for drawing 2D origami crease patterns,
solving folded 3D kinematic configurations, projecting them back to 2D, and
exporting JSON, DXF, STL, or STEP files.

Typical imports:

```python
from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import build_packaging, solve_kinematics

pattern = TwoDDrawer()
build_packaging(pattern)

model = Cadder.from_drawer(pattern)
result = solve_kinematics(model, final_dihedral=135.0, tol=1e-10)
```

`solve_kinematics` solves directly at `final_dihedral`, with no intermediate
angle steps. Remove `steps` and `start_dihedral` from older calls, and rename
`max_nfev_per_step` to `max_nfev` if specifying an evaluation limit. The
default budget is 5000 evaluations for the entire direct solve.

Sparse solves adapt LSMR's inner accuracy by default. The outer stopping
tolerance remains `tol=1e-10`. LSMR starts with `atol=btol=1e-6`; if a stage
has not converged within eight evaluations, it resumes from its latest
coordinates with 100 times tighter inner tolerances. An early `ftol`/`xtol`
exit without satisfying `gtol` also triggers tightening. At the inner
accuracy floor (`max(10 * eps, min(1e-6, tol))`), the solver
uses the remaining evaluation budget without further restarts. All stages
share `max_nfev`, and
`report.nfev` includes their combined evaluations. Set
`adaptive_tolerance=False` in `Cadder.solve` or `solve_kinematics` to use
SciPy's original fixed inner tolerance. Dense solves are unaffected.
The kinematics report describes the direct solve at the requested angle.

The automatic initial guess already retains the flat pattern's XY
coordinates and assigns triangle heights. Unless `mountain_height` or `X0`
is supplied, mountain height is `valley_height + sqrt(3)/2 * d * sin(final_dihedral)`,
using the measured hexagon side length `d` and the requested angle unit.
A partially folded state also requires XY contraction, which the nonlinear
solver determines.

Hexagon patterns store each physical hinge once in `pattern.hex_creases`
(and `model.hex_creases` after conversion). Each unit's `local_creases` list
contains references with its local index, side, and edge direction. To read
the resolved crease geometry, including the original local orientation, use:

```python
from origamicad.patterns.hexagon import iter_local_creases

for unit in pattern.hex_units:
    for crease in iter_local_creases(pattern, unit):
        print(crease["edge"], crease["triangle"], crease["quad"], crease["kind"])
```

The iterator returns snapshots; edit physical definitions through
`hex_creases`. JSON export includes the registry and local references. The
loaders also accept older files with inline crease records and convert them
automatically. Projection and model conversion preserve the registry and unit
memberships. Custom code that read `unit["local_creases"]` directly for physical
fields should use `iter_local_creases` instead.

Consistent shared references are valid even with `strict_unique_edges=True`.
Conflicting fold assignments or adjacent panels raise before constraints are
added; strict mode also rejects missing geometry. `constraint_info` reports
`num_shared_crease_references` and `shared_crease_references`, counting
references beyond the first owner. The older
duplicate-constraint fields remain available and are zero/empty because no
duplicate equations are created. Local ownership is retained for cavity,
hole-boundary, and insertion operations.

Mountain/valley classifications come from the 2D builder's `triangle_kinds`
and physical crease metadata. Build/import validates these records and prepares
an ordered constraint plan. Kinematics and initialization read the stored
labels directly; they do not rediscover them by searching the drawing's lines.
Missing or conflicting triangle labels require correcting/rebuilding the 2D
metadata. Older inline crease records with triangle labels remain supported.

Panel rigidity uses combinations of vertices within each panel, deduplicated
by unordered endpoint IDs. The plan preserves the first constraint ID and
insertion order. A temporary index of existing bar constraints handles repeated
setup and conflicting reference lengths. Plans contain connectivity rather
than coordinates or angle targets: folding reuses them, while edits to public
geometry or metadata trigger validation and rebuilding before reuse. Plan
snapshots are private, derived data; JSON retains the existing metadata format.

`solve_kinematics` returns after folding, without calculating Jacobian rank or
mobility. Its `result["report"].rank` and `.mobility` fields are `-1`, meaning
not computed. Calculate these diagnostics separately when needed:

```python
from origamicad.patterns.hexagon import analyze_kinematics

analysis = analyze_kinematics(model, tol=1e-8)
print(analysis["rank"], analysis["mobility"])
```

This analyzes the current coordinates without solving again or changing the
model. It returns `rank`, `mobility`, `num_variables`, and `num_residuals`,
excluding hole-contour-only points from the variable count. The diagnostic
still uses a dense rank calculation, so it can take substantial time and
memory for large models. It does not update the earlier solve report.

Pass `reverse=True` to `build_packaging` to exchange all mountain and valley
crease labels and their kinematic metadata.

Draw concentric hexagonal loops with an optional central cavity:

```python
from origamicad.patterns.hexagon import draw_hex_loops

pattern = TwoDDrawer()
draw_hex_loops(pattern, n=3, cavity_loops=1)
pattern.draw()
```

`cavity_loops` defaults to `0` and must satisfy `0 <= cavity_loops < n`.
For a three-loop structure, `0` keeps the full pattern, `1` opens the innermost
loop, and `2` leaves only the outer loop. Shared panels and strips projecting
into the cavity are removed, and exposed edges become cut boundaries. The
outer dimensions stay the same. Returned units retain their original numbering
and reference only surviving geometry.

Set `enable_hole_punch_outer` to a hole diameter to punch the outer side
parallelograms only, including when a cavity is present:

```python
draw_hex_loops(TwoDDrawer(), n=3, cavity_loops=1, enable_hole_punch_outer=2.5)
```

The default `0.0` disables holes. The diameter uses the pattern's length units
and must be no larger than `sqrt(3) * l / 2` so it fits within the panel.
Interior panels and panels bordering only the cavity are left unpunched.
When using `Cadder.from_drawer(pattern)` and `solve_kinematics`, hole contours
move with their panels without adding folding degrees of freedom. The 3D
preview and zero-thickness STEP export preserve the openings, including in
stacked layers.

Calculate the transverse and longitudinal cargo dimensions for a folded
hexagon package (angles are in degrees by default):

```python
from origamicad.patterns.hexagon import calculate_cargo_size

cargo = calculate_cargo_size(l=15.0, theta=135.0, gamma=3, delta=4)
print(cargo.transverse, cargo.longitudinal)
```

The corresponding folded height is available separately:

```python
from origamicad.patterns.hexagon import calculate_cargo_height

height = calculate_cargo_height(l=15.0, theta=135.0)
```

Analytic constraint Jacobians are available for the full model or one
constraint. Build constraints first (for example by calling `solve_kinematics`),
then use:

```python
from origamicad import JacobianBuilder

builder = JacobianBuilder(model)
J = builder.build()                         # SciPy CSR matrix
constraint_id = next(iter(model.constraints))
J_constraint = builder.for_constraint(constraint_id)
J_dense = builder.build(sparse=False)        # NumPy array

# Equivalent model methods; X optionally evaluates another configuration
# without changing model.points.
J = model.jacobian(X=model.get_coordinate_vector())
J_constraint = model.constraint_jacobian(constraint_id, sparse=False)
```

Rows follow `residual_vector()` order. Columns follow the coordinate vector
`[x0, y0, z0, x1, y1, z1, ...]`, including all model points for an individual
constraint. The builder covers bar lengths, fixed coordinates, parallel lines
and surfaces, all three dihedral residuals, coplanarity, horizontal surfaces,
and surface heights. Fixed-point helpers create three fixed-coordinate rows.

`Cadder.solve()` uses the sparse analytic Jacobian by default. Set
`use_jac_sparsity=False` for dense derivatives or `use_analytic_jacobian=False`
to use SciPy finite differences. `model.numerical_jacobian()` retains independent
central differences for derivative checks. Rank and mobility diagnostics use
the analytic Jacobian. Derivatives at degenerate geometry or the discontinuity
of a wrapped angle residual raise `ValueError`; surface-normal derivatives
assume the selected non-collinear vertex triplet stays the same locally.

Run the derivative and solver checks with:

```bash
python -m unittest discover -s tests -v
```

The reusable core is kept separate from pattern-specific code:

```text
origamicad/
  core/                 # generic 2D/3D models and constraint solver
    jacobian.py         # analytic per-constraint and assembled Jacobians
  patterns/
    hexagon/
      layout.py         # hexagon geometry and metadata generation
      kinematics.py     # hexagon-specific constraint setup and solving
      metadata.py       # shared hexagon metadata types
  io/                   # JSON, DXF, STL, and STEP export
```

Run the 2D and 3D examples from the project root with:

```bash
python -m examples.hexagon_example_2d
python -m examples.hexagon_example_3d
python -m examples.simple_hexagon_insertion_projection_dxf
```
