# OrigamiCAD

OrigamiCAD is a small Python toolkit for drawing 2D origami crease patterns,
solving folded 3D kinematic configurations, projecting them back to 2D, and
exporting JSON, DXF, STL, or STEP files.

**Contents**

- [Installation](#installation)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Crease patterns](#crease-patterns)
- [Folding kinematics](#folding-kinematics)
- [Cargo dimensions](#cargo-dimensions)
- [Panel clearance](#panel-clearance)
- [Snap-through insertion](#snap-through-insertion)
- [Drawing](#drawing)
- [Crease metadata](#crease-metadata)
- [Jacobians](#jacobians)
- [Solver internals](#solver-internals)
- [Development](#development)
- [License](#license)

## Installation

OrigamiCAD needs Python 3.10 or newer and git. pip installs its dependencies
(NumPy, SciPy, Matplotlib, and ezdxf) automatically.

### Install from GitHub

To use OrigamiCAD without changing its code, install it straight from GitHub:

```bash
python -m pip install "git+https://github.com/pTimYu/OrigamiCAD.git"
```

This installs the latest commit on `main`. To install a particular branch,
tag, or commit instead, append it after `@`:

```bash
python -m pip install "git+https://github.com/pTimYu/OrigamiCAD.git@<branch, tag, or commit>"
```

To update an existing installation, run the same command with
`--force-reinstall --no-deps` added; this reinstalls OrigamiCAD and leaves its
dependencies alone.

### Install for development

To work on the code, clone the repository and install it in editable mode, so
that code changes take effect without reinstalling:

```bash
git clone https://github.com/pTimYu/OrigamiCAD.git
cd OrigamiCAD
python -m pip install -e .
```

### Check the installation

```bash
python -c "import origamicad; print(origamicad.__version__)"
```

## Quick start

Build a hexagon packaging pattern and fold it to 135 degrees:

```python
from origamicad import Cadder, TwoDDrawer
from origamicad.patterns.hexagon import build_packaging, solve_kinematics

pattern = TwoDDrawer()
build_packaging(pattern)

model = Cadder.from_drawer(pattern)
result = solve_kinematics(model, final_dihedral=135.0, steps=3, tol=1e-10)
```

## Examples

The `examples/` folder holds runnable scripts. They import the installed
package, so install OrigamiCAD first (see [Installation](#installation)). Run
them from the project root, for example:

```bash
python -m examples.hexagon_example_2d
python -m examples.hexagon_example_3d
python -m examples.simple_hexagon_insertion_projection_dxf
```

The scripts write their files to `examples/output/`.

## Crease patterns

Pass `reverse=True` to `build_packaging` to exchange all mountain and valley
crease labels and their kinematic metadata.

### Concentric hexagonal loops

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

### Hole punches

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

## Folding kinematics

### Hybrid solver

`solve_kinematics` uses a hybrid algorithm by default: take `steps=3` tangent
steps from the flat 180-degree coordinates toward `final_dihedral`, then refine
the predicted coordinates with the nonlinear solver. There is no nonlinear
starting-pose solve. Set `steps` to any nonnegative integer; `steps=0` solves
directly at the requested angle from a height guess. Each tangent step uses
the sparse analytic Jacobian to solve the first-order constraint equations,
without nonlinear correction between steps. If tangent prediction fails, the
solver falls back to the direct endpoint guess.

`max_nfev` defaults to 5000 evaluations for final refinement.
`result["report"].nfev` counts that nonlinear stage; tangent steps are not
counted in it. `result["hybrid"]` contains `requested_steps`,
`completed_steps`, `refinement_nfev`, `linear_iterations`, and
`fallback_reason` (`None` when no fallback was needed). `linear_iterations`
counts tangent LSMR iterations and is zero with the default direct solver.

### Linear solvers and variable scaling

Both stages use `linear_solver="direct"` by default. Each tangent step and
each step of the trust-region Gauss-Newton refinement factorizes the sparse
normal equations once. Pass `linear_solver="lsmr"` to `solve_kinematics` or
`Cadder.solve` for the iterative LSMR solver instead (`atol=btol=1e-10`,
`maxiter=5000`), which refines inside SciPy's trust-region solver; it is
several times slower on large patterns and is also what a direct step falls
back to if its factorization fails. Both stop on the same `tol=1e-10` tests by
default, and final refinement owns the full `max_nfev` budget. The kinematics report describes
final refinement at the requested angle. `Cadder.solve` remains the generic
nonlinear solver.

`Cadder.solve` and `solve_kinematics` use `x_scale="jac"` by default to scale
coordinate updates using Jacobian column norms. Pass `x_scale=1.0` to disable
variable scaling, or supply positive per-coordinate scales. Hybrid tangent
prediction uses the same scaling option. Residual equations, their physical
units, and the stopping tolerances are unchanged by this option.

### Initial guess

For `steps=0`, the automatic initial guess retains the flat pattern's XY
coordinates and assigns triangle heights. Unless `mountain_height` or `X0`
is supplied, mountain height is
`valley_height + sqrt(3)/2 * d * sin(final_dihedral)`, using the measured
hexagon side length `d`. A partially folded state also requires XY contraction,
which the nonlinear solver determines. Positive-step solves instead begin at
the model's flat 180-degree coordinates; an explicit `X0` may supply those
starting coordinates.

### Solving again at another angle

A solved model can be solved again at another angle:

```python
solve_kinematics(model, final_dihedral=150.0)
solve_kinematics(model, final_dihedral=120.0)   # continues from 150 degrees
```

The second call retargets the existing constraints instead of adding them
again, and its tangent steps start from the angle the model is currently
folded to. A different `valley_z` moves the valley panels to the new height.

### Valley height

Valley triangles use absolute vertex-height constraints, which also enforce
their horizontality, at the height given by `valley_z` (default `0.0`). The
solver omits the implied horizontal equations and the reference frame's Z
equations for those vertices, so the valley height is set by `valley_z` alone
and need not match the pattern's current height. Existing user constraints are
retained, including a height constraint the caller added under a
`valley_z_<triangle>` ID, which keeps its own value.

### Rank and mobility

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

## Cargo dimensions

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

## Panel clearance

`minimum_panel_clearance(model)` returns the smallest gap between panels that
do not share an edge, the pair it occurs at, and whether any panels pass
through each other:

```python
from origamicad.patterns.hexagon import minimum_panel_clearance

gap, closest_pair, clipping = minimum_panel_clearance(model)
```

## Snap-through insertion

Find the snapped state of a folded pattern that is squeezed from all sides:

```python
from origamicad.patterns.hexagon import print_insertion_report, simulate_insertion

result = simulate_insertion(inner_dihedral_deg=135.0, num_loops=2)
print_insertion_report(result)
```

### Compact method (default)

Two facts about rigid folds of this pattern make this fast.

- **Snap lines.** Triangles stay horizontal and can only translate, and a
  parallelogram moves its two triangles apart by its width times the cosine of
  its fold angle. Snapping a parallelogram through 90 degrees gives it the
  supplementary angle, so the same distance with the opposite sign. Around a
  unit chain the six contributions must cancel, which they do when opposite
  parallelograms of the unit are in the same state. Opposite parallelograms
  are parallel and each is shared with the next unit, so the state runs on:
  all parallelograms across one row of unit chains snap together.
  `snap_lines(model)` lists these sets. The one exception is a unit whose six
  parallelograms alternate between the two states; every line through such a
  unit changes state there.
- **Closed-form fold.** For any compatible choice of snapped parallelograms
  `fold_with_snapped_panels(model, dihedral_deg, snapped)` returns the exact
  folded coordinates without solving anything.

Squeezing drives the pattern towards the fold that takes the least room, and
every snapped parallelogram pulls two triangles together by the same amount.
The default `method="compact"` therefore looks for the rigid fold with the
most snapped parallelograms that closes everywhere, keeps the cargo where it
is, and lets no panels pass through each other (`most_compact_fold`). The
first two conditions are linear, so this is a small mixed-integer program;
intersections are checked on the exact fold, and each one found excludes the
panel states that cause it before the program is solved again. Two loops take
0.1 s and a 2-2-3-4 package about 1 s.

For a `build_packaging` pattern the cargo is the rectangle that rests on the
vertices of the cavity in the unsnapped fold (`cargo_rectangle`; its size
equals `calculate_cargo_size`). The vertices it rests on keep resting on the
same face of the cargo and no other point may enter the rectangle. By default
they may slide along that face (`cargo_contact="slide"`), which lets a wall
shorten along the cargo so that it snaps along its whole length;
`cargo_contact="fixed"` keeps them exactly in place, and then nothing can
snap between two of them. `whole_lines=True` allows only complete snap lines
and no alternating units, which shears the walls instead. A side on which the
cavity is open is left free, so an open package needs no second half. For a
pattern without a cavity, or when `held` is given, the held parallelograms
keep the fold angle instead (by default those of the innermost unit chain).
The result is an exact rigid fold, so `result["assembly"]` stacks it.

### Pressure method

`method="pressure"` follows the squeeze itself and returns where a given
pressure leaves the structure. It is far slower, and walls snap one after
another, so a low `pressure_max` leaves some sides unsnapped.

In the pressure method the pattern is folded uniformly to `inner_dihedral_deg`. The creases of the
held cavity panels then stay at that angle while a uniform in-plane pressure
on the outer boundary is raised from zero. Which panels snap through 90
degrees, at which pressure, and where they come to rest is computed, not
prescribed. The model is a bar-and-hinge model:

- triangles are rigid and horizontal; valley triangles stay at one height and
  each mountain triangle is free to rise on its own;
- parallelograms can stretch slightly (`panel_stiffness_ratio`, default 1000,
  is their stiffness relative to a crease);
- every crease that is not held is a rotational spring of stiffness
  `hinge_stiffness` per unit length, relaxed at `rest_dihedral_deg` (flat);
- panels closer than `contact_distance` (default 3 % of the side length)
  repel each other, so they cannot pass through one another;
- the load is the potential `pressure * enclosed area` of the outer boundary,
  which pushes every boundary edge inward with the same force per length.

`held` selects what is held: unit counts hold that unit chain's
parallelograms, and panel IDs hold every crease of that panel. By default it
is the panels bordering the cavity, or the innermost unit chain when the
pattern has no cavity. `constraints` takes a function that receives the folded
working model and may add any other `Cadder` constraints, which are then
enforced exactly as well.

The pressure rises until `pressure_max`. Without it, the ramp ends when no
panel snaps while the pressure triples (`hold_factor`), or when a panel is
strained beyond `strain_limit`. `result["snap_events"]` lists every snap with
its pressure, the panels involved, and the energy released; `result["path"]`
holds the pressure, energy, enclosed area, and lowest stiffness eigenvalue of
every equilibrium. `result["model"]` is the state right after the last snap.
If that state is an exact rigid folding (`rigid_compatible`), the model holds
the exact geometry; otherwise panels are held against each other in a strained
state, and `max_panel_strain` reports the largest strain.
`stable_without_load` tells whether the state stays snapped when the pressure
is removed. Stacking needs all mountain triangles at one height, so
`result["assembly"]` is `None` for a strained state.

Results depend on the stiffness ratio and on the contact distance, and more
than one stable snapped state can exist for the same pressure. The simulation
returns the state reached by raising the pressure quasi-statically from the
uniformly folded pattern.

### Open packages in the pressure method

In the pressure method a pattern from `build_packaging` with one side open
has no closed boundary to press on. Pass it as `model=`: it is completed with the same lattice continued
over its 180-degree rotation, the closed ring is simulated, and the supplied
half is returned (`result["closed_model"]` is the ring). A mirrored copy is
not used because the lattice is chiral.

### The underlying mechanics solver

The solver behind it is generic and can be used directly:
`origamicad.core.mechanics.MechanicalModel` combines hard constraints, elastic
elements, and load potentials; `origamicad.core.equilibrium.ramp_load` raises
a load through stable equilibria and across snaps;
`origamicad.core.contact.ContactBarrier` is the panel contact energy.

## Drawing

`TwoDDrawer.draw()` and `Cadder.draw()` open the figure with `plt.show()` and
return `(fig, ax)`. Pass `show=False` to keep editing the figure, or to draw
without a display; the figure is then yours to close with `plt.close(fig)`.
Matplotlib is imported only when something is drawn, so `import origamicad`
does not load it.

## Crease metadata

### Crease registry

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

### Shared references

Consistent shared references are valid even with `strict_unique_edges=True`.
Conflicting fold assignments or adjacent panels raise before constraints are
added; strict mode also rejects missing geometry. `constraint_info` reports
`num_shared_crease_references` and `shared_crease_references`, counting
references beyond the first owner. Local ownership is retained for cavity,
hole-boundary, and insertion operations.

### Mountain and valley labels

Mountain/valley classifications come from the 2D builder's `triangle_kinds`
and physical crease metadata. Build/import validates these records and prepares
an ordered constraint plan. Kinematics and initialization read the stored
labels directly; they do not rediscover them by searching the drawing's lines.
Missing or conflicting triangle labels require correcting/rebuilding the 2D
metadata. Older inline crease records with triangle labels remain supported.

## Jacobians

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
constraint. The constraint kinds are bar lengths, fixed coordinates, signed
dihedral increments, horizontal surfaces, and surface heights. Fixed-point
helpers create three fixed-coordinate rows.

`Cadder.solve()` always uses the sparse analytic Jacobian, and so do the rank
and mobility diagnostics. `Cadder.solve()` skips those diagnostics unless
`compute_rank=True` is passed, because the dense rank is slow for large
models. Derivatives at degenerate geometry or the discontinuity of a wrapped
angle residual raise `ValueError`.

## Solver internals

### Constraint plans

Panel rigidity uses combinations of vertices within each panel, deduplicated
by unordered endpoint IDs. The plan preserves the first constraint ID and
insertion order. A temporary index of existing bar constraints handles repeated
setup and conflicting reference lengths. Plans contain connectivity rather
than coordinates or angle targets: folding reuses them, while edits to public
geometry or metadata trigger validation and rebuilding before reuse. Plan
snapshots are private, derived data; JSON retains the existing metadata format.

### Compiled evaluation

Each solve compiles point indices and sparse assembly structure once. Residuals
and Jacobians evaluate every constraint kind as one array operation and share
geometry at identical trial coordinates, without writing those coordinates
into the model. The plan is discarded after solving; public evaluations
outside a solve observe direct edits to points and constraints.

## Development

### Package layout

The reusable core is kept separate from pattern-specific code:

```text
origamicad/
  _validation.py        # argument checks shared by builders, solvers, exporters
  core/                 # generic 2D/3D models and constraint solver
    two_d_drawer.py     # TwoDDrawer: 2D points, lines, surfaces, and preview
    cadder.py           # Cadder: 3D model, constraints, and nonlinear solve
    visualization.py    # 3D preview drawn by Cadder.draw()
    topology.py         # panel rigidity plans, boundary loops, loop areas
    panel_holes.py      # keeps panel holes attached without extra freedom
    jacobian.py         # analytic per-constraint and assembled Jacobians
    _evaluation.py      # solve-scoped constraint batches and sparse assembly
    _direct.py          # trust-region solver with sparse direct linear solves
    _tangent.py         # first-order dihedral prediction for hybrid solves
    clearance.py        # gaps and intersections between folded panels
    mechanics.py        # elastic energy, load potentials, hard constraints
    equilibrium.py      # stable equilibria, snaps, and load ramps
    contact.py          # barrier energy that keeps panels apart
  patterns/
    hexagon/
      layout.py         # hexagon geometry and metadata generation
      metadata.py       # shared hexagon metadata types
      kinematics.py     # hexagon-specific constraint setup and solving
      cargo.py          # cargo dimensions of a folded package
      stacking.py       # layer stacks from solved models or 2D patterns
      snap_lines.py     # snap lines and the closed-form rigid fold
      compact_fold.py   # most compact rigid fold around a cargo
      closure.py        # half-turn closure of a pattern open on one side
      insertion_simulation.py  # snap-through insertion under side pressure
  io/                   # JSON, DXF, STL, and STEP export
    cad_export.py       # JSON, STL, and STEP export of 3D models
    dxf_export.py       # DXF export of 2D patterns
```

### Version

The version is written only in `origamicad/__init__.py` (`__version__`);
`pyproject.toml` reads it from there.

## License

OrigamiCAD is released under the MIT License; see `LICENSE`.
