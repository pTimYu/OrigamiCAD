"""Snap-through insertion simulation for the hexagon pattern.

The structure starts folded uniformly at the cavity angle. Constraints hold
the cavity while an evenly distributed in-plane pressure squeezes the outer
boundary. Panels are slightly compliant, creases are rotational springs, and
panels cannot pass through each other, so which panels snap through 90
degrees, and where they come to rest, is the result of the mechanics.

The model follows the bar-and-hinge formulation of Liu and Paulino and the
rigid-panel, hinge-spring description of this pattern family by Jamalimehr
et al.; holding selected panels while the rest snap follows Almessabi et al.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..._validation import finite, integer
from ...core._evaluation import CompiledConstraints
from ...core.cadder import Cadder, Constraint
from ...core.clearance import minimum_panel_clearance
from ...core.contact import ContactBarrier
from ...core.equilibrium import LoadPath, lowest_modes, ramp_load, solve_equilibrium
from ...core.mechanics import AreaPotential, MechanicalModel
from ...core.panel_holes import hole_attachments, panel_only_geometry, place_holes
from ...core.topology import boundary_loops, loop_area
from ...core.two_d_drawer import TwoDDrawer
from .closure import close_open_pattern
from .kinematics import dihedral_constraint_id, solve_kinematics
from .layout import draw_hex_loops
from .metadata import iter_local_creases
from .stacking import stack_layers


DEFAULT_SIDE_LENGTH = 15.0
DEFAULT_INNER_DIHEDRAL_DEG = 150.0
DEFAULT_NUM_LOOPS = 2
DEFAULT_NUM_LAYERS = 4
_SIGNED = "dihedral_signed_increment"

PanelState = Literal["contact", "snap", "partial"]


class InsertionPanelState(TypedDict):
    """Simulated state of one parallelogram.

    ``contact`` panels kept their obtuse creases, ``snap`` panels passed
    through 90 degrees on every crease, and ``partial`` panels did so on only
    some. ``held`` panels belong to the constrained cavity.
    """

    surface_id: str
    state: PanelState
    held: bool
    dihedrals_deg: tuple[float, ...]
    adjacent_triangle_ids: tuple[str, ...]


class SnapEvent(TypedDict):
    """One change of the set of acute creases during the pressure ramp."""

    pressure: float
    snapped_panel_ids: tuple[str, ...]
    released_panel_ids: tuple[str, ...]
    energy_released: float
    area_before: float
    area_after: float


class PathPoint(TypedDict):
    """One equilibrium of the pressure ramp."""

    pressure: float
    energy: float
    elastic_energy: float
    area: float
    lowest_eigenvalue: float
    num_acute_creases: int


class InsertionSimulationResult(TypedDict):
    """Geometry and diagnostics returned by :func:`simulate_insertion`."""

    model: Cadder
    assembly: Optional[Cadder]
    pattern: TwoDDrawer
    status: Literal["panel_contact", "compact_nonclipping", "no_snap"]
    reason: str
    contact_dihedral_deg: float
    snap_dihedral_deg: float
    num_layers: int
    side_length: float
    layer_height: Optional[float]
    max_interface_error: Optional[float]
    final_pressure: float
    highest_pressure: float
    stop_reason: str
    snap_events: list[SnapEvent]
    path: list[PathPoint]
    rigid_compatible: bool
    max_panel_strain: float
    stable_without_load: Optional[bool]
    minimum_panel_gap: float
    closest_panel_pair: tuple[str, str] | None
    contact_detected: bool
    panel_states: dict[str, InsertionPanelState]
    contact_panel_ids: frozenset[str]
    snap_panel_ids: frozenset[str]
    held_panel_ids: frozenset[str]
    crease_assignment_by_edge: dict[tuple[str, str], str]
    closed_model: Optional[Cadder]


def _canonical_edge(point_a: str, point_b: str) -> tuple[str, str]:
    return tuple(sorted((point_a, point_b)))


def _angle(name: str, value: float, lower: float, upper: float) -> float:
    angle = float(value)
    if not np.isfinite(angle) or not (lower < angle < upper):
        raise ValueError(f"{name} must be between {lower:g} and {upper:g} degrees.")
    return angle


def _infer_side_length(pattern: TwoDDrawer) -> float:
    """Measure the hexagon side length from the crease pattern."""
    lengths = [
        float(np.hypot(
            pattern.points[crease["edge"][1]].x - pattern.points[crease["edge"][0]].x,
            pattern.points[crease["edge"][1]].y - pattern.points[crease["edge"][0]].y,
        ))
        for crease in pattern.hex_creases.values()
    ]
    if not lengths:
        raise ValueError("The insertion model has no crease metadata.")
    return float(np.mean(lengths))


def _crease_panels(model: Cadder) -> dict[str, tuple[str, str]]:
    """Map each crease constraint ID to its (parallelogram, triangle)."""
    panels: dict[str, tuple[str, str]] = {}
    for unit in model.hex_units:
        for crease in iter_local_creases(model, unit):
            constraint_id = dihedral_constraint_id(
                int(unit["count"]), crease["local_index"], crease["side"],
            )
            if constraint_id in model.constraints:
                panels[constraint_id] = (crease["quad"], crease["triangle"])
    missing = sorted(
        cid for cid, constraint in model.constraints.items()
        if constraint.kind == _SIGNED and cid not in panels
    )
    if missing:
        raise RuntimeError(f"Could not associate every crease with a panel: {missing}.")
    return panels


def cavity_panel_ids(pattern) -> set[str]:
    """Return the panels that border a cavity of the pattern.

    A cavity is an inner boundary loop that is not the hexagonal void of a
    single unit chain. Patterns without one, such as complete loop patterns,
    return an empty set.
    """
    xy = {pid: (point.x, point.y) for pid, point in pattern.points.items()}
    void_points = [set(unit["mid"]) for unit in pattern.hex_units if len(unit["mid"]) == 6]
    cavity_edges = set()
    for loop in boundary_loops(pattern, xy):
        if loop_area(loop, xy) > 0 or set(loop) in void_points:
            continue
        cavity_edges.update(
            _canonical_edge(start, end) for start, end in zip(loop, loop[1:] + loop[:1])
        )
    panels = set()
    for surface_id, surface in pattern.surfaces.items():
        vertices = list(surface.vertices)
        if any(
            _canonical_edge(start, end) in cavity_edges
            for start, end in zip(vertices, vertices[1:] + vertices[:1])
        ):
            panels.add(surface_id)
    return panels


def _held_panels(pattern: TwoDDrawer, held) -> set[str]:
    """Resolve the ``held`` argument to a set of panel IDs."""
    if held is None:
        panels = cavity_panel_ids(pattern)
        if panels:
            return panels
        inner = min(pattern.hex_units, key=lambda unit: int(unit["count"]))
        return set(inner["parallelograms"])
    units = {int(unit["count"]): unit for unit in pattern.hex_units}
    panels = set()
    for item in held:
        if isinstance(item, str):
            if item not in pattern.surfaces:
                raise ValueError(f"Held panel '{item}' does not exist.")
            panels.add(item)
        elif int(item) in units:
            panels.update(units[int(item)]["parallelograms"])
        else:
            raise ValueError(f"Held unit {item!r} does not exist.")
    return panels


def _build_mechanics(
    model: Cadder,
    pattern: TwoDDrawer,
    held_panels: set[str],
    extra_hard: set[str],
    side_length: float,
    hinge_stiffness: float,
    rest_dihedral_deg: float,
    panel_stiffness_ratio: float,
    contact_distance: float,
):
    """Split the kinematic constraints into hard and elastic sets."""
    crease_panels = _crease_panels(model)
    triangle_pairs = set()
    for surface_id in model.surfaces:
        vertices = model._surface_vertices(surface_id)
        if len(vertices) == 3:
            triangle_pairs.update(
                _canonical_edge(a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1])
            )

    bar_stiffness = panel_stiffness_ratio * hinge_stiffness / side_length
    rest_fold = np.deg2rad(180.0 - rest_dihedral_deg)
    hard, elastic, stiffness, creases, held_creases = [], [], [], [], set()
    for constraint_id, constraint in model.constraints.items():
        data = constraint.data
        if constraint_id in extra_hard:
            hard.append(constraint)
        elif constraint.kind == "bar_length":
            if _canonical_edge(data["p1"], data["p2"]) in triangle_pairs:
                hard.append(constraint)
            else:
                elastic.append(constraint)
                stiffness.append(bar_stiffness * side_length / data["length"])
        elif constraint.kind == _SIGNED and constraint_id in crease_panels:
            creases.append(Constraint(
                constraint_id, _SIGNED, {**data, "target_increment": 0.0},
            ))
            quad, triangle = crease_panels[constraint_id]
            if quad in held_panels or triangle in held_panels:
                hard.append(constraint)
                held_creases.add(constraint_id)
            else:
                length = float(np.linalg.norm(
                    model.point_array(data["edge_end"]) - model.point_array(data["edge_start"])
                ))
                elastic.append(Constraint(constraint_id, _SIGNED, {
                    **data, "target_increment": float(data["sign"] * rest_fold),
                }))
                stiffness.append(hinge_stiffness * length)
        else:
            hard.append(constraint)

    xy = {pid: (point.x, point.y) for pid, point in pattern.points.items()}
    outer = max(boundary_loops(pattern, xy), key=lambda loop: loop_area(loop, xy))
    index = {pid: i for i, pid in enumerate(model.point_ids())}
    start = model.get_coordinate_vector()
    area = AreaPotential([index[pid] for pid in outer], model.num_variables(), start)
    potentials = []
    if contact_distance > 0:
        potentials.append(ContactBarrier(
            model, contact_distance, bar_stiffness * contact_distance**2,
        ))
    mechanics = MechanicalModel(model, hard, elastic, stiffness, [area], potentials)

    angle_plan = CompiledConstraints(model, creases)
    signs = np.array([crease.data["sign"] for crease in creases], dtype=float)

    def dihedrals(X) -> np.ndarray:
        """Unsigned dihedral angle of every crease, in degrees."""
        return np.rad2deg(np.pi - signs * angle_plan.residual(X))

    return mechanics, area, dihedrals, [crease.id for crease in creases], crease_panels, held_creases


def _panel_states(crease_ids, crease_panels, angles, held_panels, pattern):
    quad_angles: dict[str, list[float]] = {}
    quad_triangles: dict[str, list[str]] = {}
    for crease_id, angle in zip(crease_ids, angles):
        quad, triangle = crease_panels[crease_id]
        quad_angles.setdefault(quad, []).append(float(angle))
        quad_triangles.setdefault(quad, []).append(triangle)
    states: dict[str, InsertionPanelState] = {}
    for quad, values in sorted(quad_angles.items()):
        acute = [value < 90.0 for value in values]
        states[quad] = {
            "surface_id": quad,
            "state": "snap" if all(acute) else "partial" if any(acute) else "contact",
            "held": quad in held_panels,
            "dihedrals_deg": tuple(values),
            "adjacent_triangle_ids": tuple(sorted(quad_triangles[quad])),
        }
    return states


def _supplied_half(ring, model, pattern, closure, num_layers, geometry_tolerance):
    """Cut the caller's half out of a simulated closed ring."""
    solved = ring["model"]
    attachments = hole_attachments(model)
    for point_id in closure.pattern.points:
        if point_id not in model.points:
            continue
        x, y, z = solved.point_array(point_id)
        point = model.points[point_id]
        point.x, point.y = map(float, closure.rotation @ np.array([x, y]) + closure.translation)
        point.z = float(z)
    place_holes(model, attachments)

    panel_states = {
        sid: state for sid, state in ring["panel_states"].items() if sid in model.surfaces
    }
    minimum_gap, closest_pair, _ = minimum_panel_clearance(model)
    try:
        stack = stack_layers(
            model, num_layers=num_layers,
            tolerance=geometry_tolerance * max(1.0, ring["side_length"]),
        )
        assembly = stack["model"]
        layer_height, interface_error = stack["layer_height"], stack["max_interface_error"]
    except ValueError:
        assembly = layer_height = interface_error = None
    return {
        **ring,
        "model": model,
        "assembly": assembly,
        "pattern": pattern,
        "layer_height": layer_height,
        "max_interface_error": interface_error,
        "minimum_panel_gap": minimum_gap,
        "closest_panel_pair": closest_pair,
        "panel_states": panel_states,
        "contact_panel_ids": frozenset(
            sid for sid, panel in panel_states.items() if panel["state"] == "contact"
        ),
        "snap_panel_ids": frozenset(
            sid for sid, panel in panel_states.items() if panel["state"] == "snap"
        ),
        "held_panel_ids": frozenset(
            sid for sid in ring["held_panel_ids"] if sid in model.surfaces
        ),
        "crease_assignment_by_edge": {
            edge: state for edge, state in ring["crease_assignment_by_edge"].items()
            if edge[0] in model.points and edge[1] in model.points
        },
        "closed_model": solved,
    }


def simulate_insertion(
    inner_dihedral_deg: float = DEFAULT_INNER_DIHEDRAL_DEG,
    num_layers: int = DEFAULT_NUM_LAYERS,
    side_length: float | None = None,
    *,
    num_loops: int | None = None,
    model: Cadder | None = None,
    held: Iterable[int | str] | None = None,
    constraints: Callable[[Cadder], None] | None = None,
    hinge_stiffness: float = 1.0,
    rest_dihedral_deg: float = 180.0,
    panel_stiffness_ratio: float = 1e3,
    contact_distance: float | None = None,
    pressure_max: float | None = None,
    pressure_step: float | None = None,
    hold_factor: float = 3.0,
    strain_limit: float = 0.1,
    check_unloaded: bool = True,
    close_open_side: bool | Literal["auto"] = "auto",
    solve_tolerance: float = 1e-9,
    geometry_tolerance: float = 1e-6,
    contact_tolerance: float | None = None,
    verbose: bool = False,
) -> InsertionSimulationResult:
    """Squeeze a folded pattern from all sides until its panels snap through.

    The pattern is first folded uniformly to ``inner_dihedral_deg``. The
    creases of the ``held`` panels are then fixed at that angle for the whole
    simulation, and a uniform in-plane pressure on the outer boundary is
    raised from zero. Triangles stay rigid and horizontal, each mountain
    triangle at its own height; parallelograms can stretch slightly; the other
    creases are springs; and panels cannot pass through each other. The
    result is the stable state the structure settles in.

    ``held`` lists unit counts and/or panel IDs. A unit count holds that unit
    chain's parallelograms, and a panel ID holds every crease of that panel.
    By default the panels bordering the cavity are held, or the innermost
    unit chain for a pattern without a cavity. ``constraints`` is called with the folded working model and may
    add any further ``Cadder`` constraints; those are enforced exactly too.

    ``hinge_stiffness`` is the crease stiffness per unit length and
    ``rest_dihedral_deg`` the angle at which creases are relaxed.
    ``panel_stiffness_ratio`` is the stiffness of a panel bar of one side
    length times the side length squared, relative to the stiffness of a
    crease of one side length. ``contact_distance`` is the gap at which panels
    start to repel each other (default 3 % of the side length; 0 disables
    contact). Pressures are forces per unit boundary length.

    The pressure rises until ``pressure_max``. Without it the ramp stops once
    the pressure reaches ``hold_factor`` times that of the last snap with no
    further change. It always stops once a panel is strained beyond
    ``strain_limit``, where the nearly rigid panels of this model stop being
    meaningful. The returned state is the equilibrium right after the last
    snap; ``result["snap_events"]`` lists every snap on the way.

    If ``model`` is supplied it must be a flat ``Cadder`` built from a hexagon
    pattern; it is used as a template and left unmodified. Otherwise a loop
    pattern with ``num_loops`` and ``side_length`` is built.

    A supplied pattern whose cavity is open on one side has no closed outer
    boundary to press on. With ``close_open_side`` (``"auto"`` by default) it
    is completed by the same lattice continued over its 180-degree rotation,
    the closed ring is simulated, and the supplied half is returned with the
    ring in ``result["closed_model"]``. ``held`` and ``constraints`` then
    refer to the ring, in which the supplied panels keep their IDs. ``True``
    raises if the pattern cannot be closed this way.

    When the snapped state is an exact rigid folding, ``result["model"]``
    holds that exact geometry and ``rigid_compatible`` is true. Otherwise the
    panels are held in a strained state by contact; the model then holds the
    elastic equilibrium and ``max_panel_strain`` reports its largest strain.
    """
    contact_deg = _angle("inner_dihedral_deg", inner_dihedral_deg, 90.0, 180.0)
    rest_dihedral_deg = _angle("rest_dihedral_deg", rest_dihedral_deg, 0.0, 180.0 + 1e-9)
    if model is not None and not isinstance(model, Cadder):
        raise TypeError("model must be a Cadder instance or None.")
    num_layers = integer("num_layers", num_layers, minimum=1)
    hinge_stiffness = finite("hinge_stiffness", hinge_stiffness, positive=True)
    panel_stiffness_ratio = finite("panel_stiffness_ratio", panel_stiffness_ratio, positive=True)
    hold_factor = finite("hold_factor", hold_factor, positive=True)
    strain_limit = finite("strain_limit", strain_limit, positive=True)

    if model is None:
        side_length = DEFAULT_SIDE_LENGTH if side_length is None else side_length
        num_loops = integer(
            "num_loops", DEFAULT_NUM_LOOPS if num_loops is None else num_loops, minimum=1,
        )
        pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
        draw_hex_loops(pattern, n=num_loops, start_point=(0.0, 0.0), l=side_length)
        model = Cadder.from_drawer(pattern)
    else:
        if not model.hex_units:
            raise ValueError(
                "The insertion model must contain hex-unit metadata. Build it "
                "with a hexagon layout function before calling simulate_insertion()."
            )
        heights = [point.z for point in model.points.values()]
        if max(heights) - min(heights) > geometry_tolerance:
            raise ValueError("The insertion model must be the flat, unfolded pattern.")
        pattern = model.to_2d_drawer(point_tol=1e-6)
        inferred = _infer_side_length(pattern)
        if side_length is not None and not np.isclose(side_length, inferred):
            raise ValueError(
                "side_length does not match the supplied model: "
                f"expected {inferred:g}, got {side_length:g}."
            )
        side_length = inferred
        # Work on a copy so the caller's model keeps its own state.
        model = copy.deepcopy(model)
        model.constraints = {}
        model._constraint_count = 0

        closure = None
        if close_open_side and not cavity_panel_ids(pattern):
            try:
                closure = close_open_pattern(pattern)
            except ValueError:
                if close_open_side is True:
                    raise
            if closure is None and close_open_side is True:
                raise ValueError(
                    "No half-turn continuation of this pattern encloses a cavity."
                )
        if closure is not None:
            ring = simulate_insertion(
                contact_deg, num_layers, model=Cadder.from_drawer(closure.pattern),
                held=held, constraints=constraints, hinge_stiffness=hinge_stiffness,
                rest_dihedral_deg=rest_dihedral_deg,
                panel_stiffness_ratio=panel_stiffness_ratio,
                contact_distance=contact_distance, pressure_max=pressure_max,
                pressure_step=pressure_step, hold_factor=hold_factor,
                strain_limit=strain_limit, check_unloaded=check_unloaded,
                close_open_side=False, solve_tolerance=solve_tolerance,
                geometry_tolerance=geometry_tolerance,
                contact_tolerance=contact_tolerance, verbose=verbose,
            )
            return _supplied_half(ring, model, pattern, closure, num_layers, geometry_tolerance)

    side_length = finite("side_length", side_length, positive=True)
    length_tolerance = finite("geometry_tolerance", geometry_tolerance, positive=True) * max(
        1.0, side_length,
    )
    contact_distance = finite(
        "contact_distance",
        0.03 * side_length if contact_distance is None else contact_distance,
        nonnegative=True,
    )
    contact_tolerance = finite(
        "contact_tolerance",
        max(contact_distance, length_tolerance) if contact_tolerance is None else contact_tolerance,
        positive=True,
    )
    held_panels = _held_panels(pattern, held)

    with panel_only_geometry(model):
        solve_kinematics(model, final_dihedral=contact_deg, verbose=False)
        folded = model.get_coordinate_vector()
        kinematic_ids = set(model.constraints)
        if constraints is not None:
            constraints(model)
        extra_hard = set(model.constraints) - kinematic_ids
        mechanics, area, dihedrals, crease_ids, crease_panels, held_creases = _build_mechanics(
            model, pattern, held_panels, extra_hard, side_length, hinge_stiffness,
            rest_dihedral_deg, panel_stiffness_ratio, contact_distance,
        )
        bars = mechanics.elastic.groups.get("bar_length")

        def strain(X) -> float:
            if bars is None:
                return 0.0
            residual = mechanics.elastic.residual(X)[bars["rows"]]
            return float(np.max(np.abs(residual) / bars["target"]))

        last_change = {"load": None}

        def stop(step):
            if step.changed:
                last_change["load"] = step.load
            if strain(step.x) > strain_limit:
                return "A panel was strained beyond the strain limit."
            if pressure_max is None and last_change["load"] is not None and (
                step.load >= hold_factor * last_change["load"]
            ):
                return "No further snap occurred while the pressure was raised."
            return None

        reference_pressure = hinge_stiffness / side_length
        step_size = finite(
            "pressure_step",
            0.25 * reference_pressure if pressure_step is None else pressure_step,
            positive=True,
        )
        load_max = finite(
            "pressure_max",
            1e3 * reference_pressure if pressure_max is None else pressure_max,
            positive=True,
        )
        path: LoadPath = ramp_load(
            mechanics, folded,
            load_max=load_max, initial_step=min(step_size, load_max),
            min_step=0.02 * step_size, max_step=None if pressure_max is None else load_max / 10,
            signature=lambda X: tuple(dihedrals(X) < 90.0),
            stop=stop, tol=solve_tolerance, verbose=verbose,
        )
        if not path.steps:
            raise RuntimeError(f"The insertion simulation could not start: {path.stop_reason}")

        events: list[SnapEvent] = []
        final_index = 0
        for index in range(1, len(path.steps)):
            before, after = path.steps[index - 1], path.steps[index]
            if not after.changed:
                continue
            final_index = index
            old = _panel_states(crease_ids, crease_panels, dihedrals(before.x), held_panels, pattern)
            new = _panel_states(crease_ids, crease_panels, dihedrals(after.x), held_panels, pattern)
            events.append({
                "pressure": after.load,
                "snapped_panel_ids": tuple(sorted(
                    q for q in new if new[q]["state"] == "snap" and old[q]["state"] != "snap"
                )),
                "released_panel_ids": tuple(sorted(
                    q for q in new if old[q]["state"] == "snap" and new[q]["state"] != "snap"
                )),
                "energy_released": float(
                    mechanics.energy(before.x, after.load) - after.energy
                ),
                "area_before": area.value(before.x),
                "area_after": area.value(after.x),
            })
        final = path.steps[final_index] if events else path.final
        angles = dihedrals(final.x)
        panel_states = _panel_states(crease_ids, crease_panels, angles, held_panels, pattern)
        max_strain = strain(final.x)

        stable_without_load = None
        if check_unloaded and events:
            unloaded = solve_equilibrium(
                mechanics, final.x, 0.0, multipliers=final.multipliers, tol=solve_tolerance,
            )
            stable_without_load = bool(
                unloaded.converged
                and np.array_equal(dihedrals(unloaded.x) < 90.0, angles < 90.0)
                and lowest_modes(mechanics, unloaded.x, 0.0, unloaded.multipliers)[0][0] > 0
            )

        # The state is an exact rigid folding when every crease already sits
        # near the held angle or its supplement and the rigid solver reaches
        # those angles from here without clipping.
        model.set_coordinate_vector(final.x)
        rigid_compatible = False
        nearest = np.where(angles < 90.0, 180.0 - contact_deg, contact_deg)
        if events and float(np.max(np.abs(angles - nearest))) <= 10.0:
            acute = dict(zip(crease_ids, angles < 90.0))
            targets = {}
            for constraint_id in crease_ids:
                data = model.constraints[constraint_id].data
                targets[constraint_id] = data["target_increment"]
                target = 180.0 - contact_deg if acute[constraint_id] else contact_deg
                data["target_increment"] = float(data["sign"] * np.deg2rad(180.0 - target))
            try:
                report = model.solve(X0=final.x, update_model=False, tol=1e-12, max_nfev=2000)
                if report.max_abs_residual <= 1e-7:
                    model.set_coordinate_vector(report.x)
                    if minimum_panel_clearance(model)[2]:
                        model.set_coordinate_vector(final.x)
                    else:
                        rigid_compatible = True
                        angles = dihedrals(report.x)
                        panel_states = _panel_states(
                            crease_ids, crease_panels, angles, held_panels, pattern,
                        )
            except ValueError:
                pass
            if not rigid_compatible:
                for constraint_id, target in targets.items():
                    model.constraints[constraint_id].data["target_increment"] = target

    minimum_gap, closest_pair, clipping = minimum_panel_clearance(model)
    barrier_gap = (
        mechanics.potentials[0].minimum_distance(final.x) if mechanics.potentials else np.inf
    )
    contact_detected = bool(events) and min(minimum_gap, barrier_gap) <= contact_tolerance
    try:
        stack = stack_layers(model, num_layers=num_layers, tolerance=length_tolerance)
        assembly = stack["model"]
        layer_height, interface_error = stack["layer_height"], stack["max_interface_error"]
    except ValueError:
        # Layers only stack when all mountain triangles share one height.
        assembly = layer_height = interface_error = None

    if not events:
        status, reason = "no_snap", f"No panel snapped through. {path.stop_reason}"
    elif contact_detected:
        status = "panel_contact"
        reason = "Panels snapped through and came to rest against each other."
    else:
        status = "compact_nonclipping"
        reason = "Panels snapped through and settled without touching."
    if clipping:
        reason += " Panels intersect in the returned state."

    snap_deg = 180.0 - contact_deg
    return {
        "model": model,
        "assembly": assembly,
        "pattern": pattern,
        "status": status,
        "reason": reason,
        "contact_dihedral_deg": contact_deg,
        "snap_dihedral_deg": snap_deg,
        "num_layers": num_layers,
        "side_length": side_length,
        "layer_height": layer_height,
        "max_interface_error": interface_error,
        "final_pressure": final.load,
        "highest_pressure": path.final.load,
        "stop_reason": path.stop_reason,
        "snap_events": events,
        "path": [
            {
                "pressure": step.load,
                "energy": step.energy,
                "elastic_energy": step.elastic_energy,
                "area": step.load_potential,
                "lowest_eigenvalue": step.lowest_eigenvalue,
                "num_acute_creases": int(sum(step.signature)),
            }
            for step in path.steps
        ],
        "rigid_compatible": rigid_compatible,
        "max_panel_strain": max_strain,
        "stable_without_load": stable_without_load,
        "minimum_panel_gap": minimum_gap,
        "closest_panel_pair": closest_pair,
        "contact_detected": contact_detected,
        "panel_states": panel_states,
        "contact_panel_ids": frozenset(
            sid for sid, panel in panel_states.items() if panel["state"] == "contact"
        ),
        "snap_panel_ids": frozenset(
            sid for sid, panel in panel_states.items() if panel["state"] == "snap"
        ),
        "held_panel_ids": frozenset(held_panels),
        "crease_assignment_by_edge": {
            _canonical_edge(
                model.constraints[cid].data["edge_start"],
                model.constraints[cid].data["edge_end"],
            ): "snap" if angle < 90.0 else "contact"
            for cid, angle in zip(crease_ids, angles)
        },
        "closed_model": None,
    }


def print_insertion_report(result: InsertionSimulationResult) -> None:
    """Print the simulated state, the snap events, and the panel angles."""
    unit = result["model"].unit
    title = "Snap-through insertion simulation"
    print(title)
    print("-" * len(title))

    def optional(value, form):
        return "-" if value is None else form.format(value)

    counts = {state: 0 for state in ("contact", "snap", "partial")}
    for panel in result["panel_states"].values():
        counts[panel["state"]] += 1
    for label, value in (
        ("Status", result["status"]),
        ("Reason", result["reason"]),
        ("Held / nominal snap dihedral", f"{result['contact_dihedral_deg']:.3f} / "
                                         f"{result['snap_dihedral_deg']:.3f} deg"),
        ("Panels contact / snap / partial", f"{counts['contact']} / {counts['snap']} / "
                                            f"{counts['partial']}"),
        ("Held panels", len(result["held_panel_ids"])),
        ("Pressure at returned state", f"{result['final_pressure']:.6g}"),
        ("Highest pressure applied", f"{result['highest_pressure']:.6g}"),
        ("Ramp ended because", result["stop_reason"]),
        ("Exact rigid folding", "yes" if result["rigid_compatible"] else "no"),
        ("Largest panel strain", f"{result['max_panel_strain']:.3%}"),
        ("Stays snapped without load", optional(result["stable_without_load"], "{}")),
        ("Panel contact reached", "yes" if result["contact_detected"] else "no"),
        ("Minimum non-adjacent gap", f"{result['minimum_panel_gap']:.6e} {unit}"),
        ("Closest panel pair", result["closest_panel_pair"]),
        ("Stacked layers", result["num_layers"]),
        ("Layer height", optional(result["layer_height"], "{:.6f} " + unit)),
    ):
        print(f"{label + ':':34s}{value}")

    print("")
    print(f"{'pressure':>12s} {'snapped':>8s} {'released':>9s} {'energy released':>16s} {'area':>22s}")
    for event in result["snap_events"]:
        print(
            f"{event['pressure']:12.6g} {len(event['snapped_panel_ids']):8d} "
            f"{len(event['released_panel_ids']):9d} {event['energy_released']:16.6g} "
            f"{event['area_before']:10.1f} -> {event['area_after']:8.1f}"
        )

    print("")
    print(f"{'state':>10s} {'creases':>8s} {'mean':>10s} {'min':>10s} {'max':>10s}   [deg]")
    for state in ("contact", "snap"):
        angles = [
            angle for panel in result["panel_states"].values()
            for angle in panel["dihedrals_deg"]
            if (angle < 90.0) == (state == "snap")
        ]
        if angles:
            print(
                f"{state:>10s} {len(angles):8d} {np.mean(angles):10.3f} "
                f"{np.min(angles):10.3f} {np.max(angles):10.3f}"
            )


def _finish_figure(figure, save_path, show: bool, dpi: int) -> Path | None:
    resolved_path = None
    if save_path is not None:
        resolved_path = Path(save_path).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return resolved_path


def draw_insertion_simulation(
    result: InsertionSimulationResult,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 240,
) -> Path | None:
    """Draw an orthographic top view of the selected one-layer state."""
    layer = result["model"]
    inner_surfaces = result["held_panel_ids"]
    panel_edge = "#566568"
    # State -> (panel face color, crease color, crease line width).
    styles = {
        "contact": ("#f2a65a", "#c66b16", 1.15),
        "snap": ("#27c8ad", "#009d8f", 2.25),
        "partial": ("#b79ad9", "#7a52b3", 1.6),
    }
    triangle_color = "#dbe7e9"

    figure, axis = plt.subplots(figsize=(9.2, 9.2), constrained_layout=True)
    surface_records = []
    for surface_id in layer.surfaces:
        coordinates = np.array(
            [layer.point_array(pid) for pid in layer._surface_vertices(surface_id)],
            dtype=float,
        )
        if surface_id in inner_surfaces:
            face_color, alpha = "#c7d9dc", 0.82
        elif surface_id in result["panel_states"]:
            face_color = styles[result["panel_states"][surface_id]["state"]][0]
            alpha = 0.90
        else:
            face_color, alpha = triangle_color, 0.78
        surface_records.append(
            (float(np.mean(coordinates[:, 2])), coordinates[:, :2], face_color, alpha)
        )

    # Paint from the lowest panel upward so higher panels cover lower ones.
    surface_records.sort(key=lambda record: record[0])
    minimum_z = surface_records[0][0]
    for mean_z, polygon_xy, face_color, alpha in surface_records:
        axis.add_collection(PolyCollection(
            [polygon_xy],
            facecolors=[face_color],
            edgecolors=[panel_edge],
            linewidths=0.72,
            alpha=alpha,
            zorder=10.0 + mean_z - minimum_z,
        ))

    for line_id in layer.lines:
        start_id, end_id, _ = layer._line_info(line_id)
        assignment = result["crease_assignment_by_edge"].get(
            _canonical_edge(start_id, end_id)
        )
        if assignment is None:
            continue
        start, end = layer.point_array(start_id), layer.point_array(end_id)
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=styles[assignment][1],
            linewidth=styles[assignment][2],
            alpha=0.98,
            solid_capstyle="round",
            zorder=200.0 + float(np.mean([start[2], end[2]])),
        )

    points_xy = np.array(
        [layer.point_array(point_id)[:2] for point_id in layer.points], dtype=float,
    )
    xy_min, xy_max = np.min(points_xy, axis=0), np.max(points_xy, axis=0)
    center = 0.5 * (xy_min + xy_max)
    half_span = (0.5 + 0.075) * float(np.max(xy_max - xy_min))
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.legend(
        handles=[
            Patch(
                facecolor=styles["contact"][0], edgecolor=styles["contact"][1],
                label="Parallelogram that kept its fold",
            ),
            Patch(
                facecolor=styles["snap"][0], edgecolor=styles["snap"][1],
                label="Parallelogram that snapped through",
            ),
            Patch(facecolor="#c7d9dc", edgecolor=panel_edge, label="Held cavity panel"),
            Patch(facecolor=triangle_color, edgecolor=panel_edge, label="Triangular panel"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        fontsize=9,
    )
    return _finish_figure(figure, save_path, show, dpi)


def draw_insertion_stack_3d(
    result: InsertionSimulationResult,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 240,
) -> Path | None:
    """Draw the actual, non-exploded stacked assembly in isometric view."""
    assembly = result["assembly"]
    if assembly is None:
        raise ValueError(
            "This state cannot be stacked: its mountain triangles do not share one height."
        )
    num_layers = result["num_layers"]
    layer_colors = ("#75858b", "#9fbcc2", "#67aeb7", "#25c5aa")
    if num_layers > len(layer_colors):
        layer_colors = tuple(
            plt.get_cmap("viridis")(position)
            for position in np.linspace(0.25, 0.82, num_layers)
        )

    figure = plt.figure(figsize=(11.0, 9.0), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    for layer_index in range(num_layers):
        prefix = f"layer_{layer_index}::"
        axis.add_collection3d(Poly3DCollection(
            [
                [assembly.point_array(pid) for pid in assembly._surface_vertices(sid)]
                for sid in assembly.surfaces
                if sid.startswith(prefix)
            ],
            facecolors=layer_colors[layer_index],
            edgecolors="#3c4b50",
            linewidths=0.34,
            alpha=0.48 if layer_index < num_layers - 1 else 0.62,
        ))

    assembly._set_axes_equal_3d(axis)
    for set_label, name in ((axis.set_xlabel, "x"), (axis.set_ylabel, "y"), (axis.set_zlabel, "z")):
        set_label(f"{name} [{assembly.unit}]", labelpad=8.0)
    axis.view_init(elev=25.0, azim=-55.0)
    axis.grid(True, alpha=0.18)
    axis.legend(
        handles=[
            Patch(
                facecolor=layer_colors[layer_index],
                edgecolor="#3c4b50",
                alpha=0.72,
                label=f"Layer {layer_index + 1}"
                + (" (top)" if layer_index == num_layers - 1 else ""),
            )
            for layer_index in range(num_layers)
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=9,
    )
    return _finish_figure(figure, save_path, show, dpi)
