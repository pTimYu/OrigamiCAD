"""Sequential contact/snap insertion simulation for the hexagon pattern.

Physical parallelograms are labelled with insertion-only metadata.  The
innermost ring is ``contact``.  Each later ring is processed once, from inner
to outer: connection panels are ``snap``; panels sharing a connection panel's
outward triangle are ``contact``; all remaining panels are ``snap``.  Contact
panels use the requested inner dihedral and snap panels use its supplement.

The returned state is checked for non-adjacent panel clipping.
"""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import Bounds, LinearConstraint, milp

from ..._validation import finite, integer
from ...core.cadder import Cadder
from ...core.clearance import minimum_panel_clearance
from ...core.two_d_drawer import TwoDDrawer
from .kinematics import _HexagonKinematics, dihedral_constraint_id
from .layout import draw_hex_loops
from .metadata import iter_local_creases
from .stacking import stack_layers


DEFAULT_SIDE_LENGTH = 15.0
DEFAULT_INNER_DIHEDRAL_DEG = 150.0
DEFAULT_NUM_LOOPS = 2
DEFAULT_NUM_LAYERS = 4
_FIXED_TRIANGLE = "tri_0_1"

PanelState = Literal["contact", "snap"]
PanelStateSource = Literal[
    "inner",
    "connection",
    "same_outer_triangle",
    "remaining",
    "closure_adjustment",
]


class InsertionPanelState(TypedDict):
    """Insertion-only state assigned to one physical parallelogram."""

    surface_id: str
    ring_index: int
    state: PanelState
    requested_state: PanelState
    source: PanelStateSource
    adjacent_triangle_ids: tuple[str, ...]


class LoopDihedralStats(TypedDict):
    """Measured contact/snap dihedrals for one planar loop."""

    loop_index: int
    contact_mean_deg: float | None
    snap_mean_deg: float | None
    num_contact_hinges: int
    num_snap_hinges: int


class InsertionSimulationResult(TypedDict):
    """Geometry and diagnostics returned by :func:`simulate_insertion`."""

    model: Cadder
    assembly: Cadder
    pattern: TwoDDrawer
    status: Literal["panel_contact", "compact_nonclipping"]
    reason: str
    contact_dihedral_deg: float
    snap_dihedral_deg: float
    num_loops: int
    num_layers: int
    side_length: float
    layer_height: float
    max_interface_error: float
    max_solve_residual: float
    feasibility_tolerance: float
    minimum_panel_gap: float
    closest_panel_pair: tuple[str, str] | None
    contact_detected: bool
    panel_states: dict[str, InsertionPanelState]
    contact_panel_ids: frozenset[str]
    snap_panel_ids: frozenset[str]
    crease_ring_by_edge: dict[tuple[str, str], int]
    crease_assignment_by_edge: dict[tuple[str, str], PanelState]
    loop_dihedrals: list[LoopDihedralStats]


def _canonical_edge(point_a: str, point_b: str) -> tuple[str, str]:
    return tuple(sorted((point_a, point_b)))


def _angle(name: str, value: float, lower: float, upper: float) -> float:
    angle = float(value)
    if not np.isfinite(angle) or not (lower < angle < upper):
        raise ValueError(f"{name} must be between {lower:g} and {upper:g} degrees.")
    return angle


def classify_insertion_panel_states(
    pattern: TwoDDrawer,
) -> dict[str, InsertionPanelState]:
    """Propagate contact/snap parallelogram states from inner to outer.

    This metadata belongs only to the insertion workflow.  Unit-chain rings
    are found by breadth-first traversal from unit 0.  A physical surface is
    owned by the innermost ring containing it, so processing a later ring can
    never rewrite a state already assigned to an inner ring.

    Ring 0 starts with every parallelogram in ``contact``.  In each later ring:

    * a parallelogram adjacent to a triangle owned by an inner ring is a
      ``snap`` connection panel;
    * other panels sharing the connection panel's outward triangle are
      ``contact`` panels;
    * all remaining panels in that ring are ``snap`` panels.
    """
    if not pattern.hex_units:
        raise ValueError(
            "No hex-unit metadata found for insertion panel propagation."
        )
    units_by_count = {int(unit["count"]): unit for unit in pattern.hex_units}
    if len(units_by_count) != len(pattern.hex_units):
        raise ValueError("Hex-unit counts must be unique.")
    if 0 not in units_by_count:
        raise ValueError("Insertion propagation requires innermost unit 0.")

    surface_units: dict[str, set[int]] = defaultdict(set)
    for unit_count, unit in units_by_count.items():
        for surface_id in unit["surfaces"]:
            surface_units[surface_id].add(unit_count)
    unit_neighbors: dict[int, set[int]] = defaultdict(set)
    for unit_counts in surface_units.values():
        for unit_count in unit_counts:
            unit_neighbors[unit_count].update(unit_counts - {unit_count})

    unit_ring = {0: 0}
    queue: deque[int] = deque([0])
    while queue:
        unit_count = queue.popleft()
        for neighbor in sorted(unit_neighbors[unit_count]):
            if neighbor not in unit_ring:
                unit_ring[neighbor] = unit_ring[unit_count] + 1
                queue.append(neighbor)
    missing_units = set(units_by_count) - set(unit_ring)
    if missing_units:
        raise ValueError(
            "Hex-unit insertion topology is disconnected; unreachable units: "
            f"{sorted(missing_units)}."
        )

    surface_ring = {
        surface_id: min(unit_ring[unit_count] for unit_count in unit_counts)
        for surface_id, unit_counts in surface_units.items()
    }
    parallelogram_ids = {
        surface_id
        for unit in pattern.hex_units
        for surface_id in unit["parallelograms"]
    }
    quad_triangles: dict[str, set[str]] = defaultdict(set)
    for unit in pattern.hex_units:
        for crease in iter_local_creases(pattern, unit):
            quad_triangles[crease["quad"]].add(crease["triangle"])

    states: dict[str, InsertionPanelState] = {}

    def assign(surface_ids, ring_index: int, state: PanelState, source: PanelStateSource):
        for surface_id in sorted(surface_ids):
            states[surface_id] = {
                "surface_id": surface_id,
                "ring_index": ring_index,
                "state": state,
                "requested_state": state,
                "source": source,
                "adjacent_triangle_ids": tuple(sorted(quad_triangles[surface_id])),
            }

    for ring_index in range(max(unit_ring.values()) + 1):
        ring_panels = {
            surface_id
            for surface_id in parallelogram_ids
            if surface_ring[surface_id] == ring_index
        }
        if ring_index == 0:
            assign(ring_panels, 0, "contact", "inner")
            continue
        connection_panels = {
            surface_id
            for surface_id in ring_panels
            if any(
                surface_ring[triangle_id] < ring_index
                for triangle_id in quad_triangles[surface_id]
            )
        }
        if ring_panels and not connection_panels:
            raise RuntimeError(
                f"Insertion ring {ring_index} has no connection panels."
            )
        outward_triangles = {
            triangle_id
            for surface_id in connection_panels
            for triangle_id in quad_triangles[surface_id]
            if surface_ring[triangle_id] == ring_index
        }
        same_triangle_panels = {
            surface_id
            for surface_id in ring_panels - connection_panels
            if quad_triangles[surface_id] & outward_triangles
        }
        assign(connection_panels, ring_index, "snap", "connection")
        assign(same_triangle_panels, ring_index, "contact", "same_outer_triangle")
        assign(
            ring_panels - connection_panels - same_triangle_panels,
            ring_index, "snap", "remaining",
        )
    return states


def _quad_crease_edges(pattern: TwoDDrawer) -> dict[str, dict[str, list[str]]]:
    """Map every parallelogram to the crease edge it shares with each triangle."""
    edges: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for unit in pattern.hex_units:
        for crease in iter_local_creases(pattern, unit):
            edges[crease["quad"]][crease["triangle"]] = crease["edge"]
    return edges


def _edge_midpoint(pattern: TwoDDrawer, edge) -> np.ndarray:
    return np.mean(
        [(pattern.points[pid].x, pattern.points[pid].y) for pid in edge], axis=0,
    )


def _panel_width_vectors(
    pattern: TwoDDrawer,
    quad_edges: dict[str, dict[str, list[str]]],
) -> dict[int, dict[str, np.ndarray]]:
    """Return, per unit, each panel's flat vector between its two triangle edges."""
    return {
        int(unit["count"]): {
            surface_id: (
                _edge_midpoint(
                    pattern, quad_edges[surface_id][unit["triangles"][(index + 1) % 6]],
                )
                - _edge_midpoint(pattern, quad_edges[surface_id][unit["triangles"][index]])
            )
            for index, surface_id in enumerate(unit["parallelograms"])
        }
        for unit in pattern.hex_units
    }


def _state_sign(panel: InsertionPanelState) -> float:
    return 1.0 if panel["state"] == "contact" else -1.0


def _reconcile_panel_states(
    pattern: TwoDDrawer,
    panel_states: dict[str, InsertionPanelState],
    panel_vectors: dict[int, dict[str, np.ndarray]],
    tolerance: float,
) -> None:
    """Resolve ambiguous outer-corner peers without rewriting inner rings.

    The first surrounding ring follows the literal connection/contact/remain
    rule.  On every later ring, snap panels already fixed on the completed
    inner boundary act as the physical connections.  A binary closure solve
    therefore selects the most-contact outer state compatible with those
    frozen rings.  Any panel whose propagated request must change is recorded
    explicitly as a ``closure_adjustment`` in insertion-only metadata.
    """
    unit_ring = {
        int(unit["count"]): max(
            panel_states[surface_id]["ring_index"]
            for surface_id in unit["parallelograms"]
        )
        for unit in pattern.hex_units
    }
    maximum_ring = max(panel["ring_index"] for panel in panel_states.values())

    for ring_index in range(1, maximum_ring + 1):
        variable_ids = [] if ring_index == 1 else sorted(
            surface_id
            for surface_id, panel in panel_states.items()
            if panel["ring_index"] == ring_index
        )
        variable_index = {sid: index for index, sid in enumerate(variable_ids)}
        rows: list[np.ndarray] = []
        targets: list[float] = []

        for unit in pattern.hex_units:
            count = int(unit["count"])
            if unit_ring[count] > ring_index:
                continue
            for axis in range(2):
                row = np.zeros(len(variable_ids), dtype=float)
                fixed_sum = unknown_flat_sum = 0.0
                for surface_id in unit["parallelograms"]:
                    component = float(panel_vectors[count][surface_id][axis])
                    variable = variable_index.get(surface_id)
                    if variable is None:
                        fixed_sum += _state_sign(panel_states[surface_id]) * component
                    else:
                        row[variable] += 2.0 * component
                        unknown_flat_sum += component
                rows.append(row)
                targets.append(-fixed_sum + unknown_flat_sum)

        if not variable_ids:
            if any(abs(target) > tolerance for target in targets):
                raise RuntimeError(
                    f"Insertion ring {ring_index} cannot close while "
                    "preserving completed inner-ring panel states."
                )
            continue

        target_vector = np.asarray(targets, dtype=float)
        result = milp(
            c=-np.ones(len(variable_ids), dtype=float),
            integrality=np.ones(len(variable_ids), dtype=int),
            bounds=Bounds(np.zeros(len(variable_ids)), np.ones(len(variable_ids))),
            constraints=LinearConstraint(
                np.asarray(rows, dtype=float), target_vector, target_vector,
            ),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(
                "Could not reconcile insertion ring "
                f"{ring_index} with rigid unit-cell closure: {result.message}"
            )

        for surface_id, value in zip(variable_ids, result.x):
            panel = panel_states[surface_id]
            panel["state"] = "contact" if value >= 0.5 else "snap"
            if panel["state"] != panel["requested_state"]:
                panel["source"] = "closure_adjustment"


def _closure_errors(
    pattern: TwoDDrawer,
    panel_states: dict[str, InsertionPanelState],
    panel_vectors: dict[int, dict[str, np.ndarray]],
    tolerance: float,
) -> list[tuple[int, float]]:
    """Return unit cells whose prescribed contact/snap projections cannot close.

    Contact and snap dihedrals are supplementary, so their projected panel
    widths have equal magnitudes and opposite signs.  Each rigid unit cell
    must therefore have a zero signed sum of its six flat panel-width vectors.
    This inexpensive geometric condition catches an incompatible state
    sequence before the nonlinear solver stalls while compromising rigidity.
    """
    errors: list[tuple[int, float]] = []
    for unit in pattern.hex_units:
        count = int(unit["count"])
        closure = np.zeros(2, dtype=float)
        for surface_id in unit["parallelograms"]:
            closure += _state_sign(panel_states[surface_id]) * panel_vectors[count][surface_id]
        closure_error = float(np.linalg.norm(closure))
        if closure_error > tolerance:
            errors.append((count, closure_error))
    return errors


def _constraint_panel_ids(model: Cadder, pattern: TwoDDrawer) -> dict[str, str]:
    """Map each physical dihedral constraint to its parallelogram."""
    panel_by_constraint: dict[str, str] = {}
    for unit in pattern.hex_units:
        for crease in iter_local_creases(pattern, unit):
            constraint_id = dihedral_constraint_id(
                int(unit["count"]), crease["local_index"], crease["side"],
            )
            if constraint_id in model.constraints:
                panel_by_constraint[constraint_id] = crease["quad"]

    missing = sorted(
        constraint_id
        for constraint_id, constraint in model.constraints.items()
        if constraint.kind == "dihedral_signed_increment"
        and constraint_id not in panel_by_constraint
    )
    if missing:
        raise RuntimeError(
            f"Could not associate every insertion crease with a panel: {missing}."
        )
    return panel_by_constraint


def _hinge_rings(
    model: Cadder,
    panel_states: dict[str, InsertionPanelState],
    panel_by_constraint: dict[str, str],
) -> dict[tuple[str, str], int]:
    """Assign each physical hinge to its insertion panel's planar ring."""
    ring_by_edge: dict[tuple[str, str], int] = {}
    for constraint_id, surface_id in panel_by_constraint.items():
        data = model.constraints[constraint_id].data
        edge = _canonical_edge(data["edge_start"], data["edge_end"])
        ring_index = panel_states[surface_id]["ring_index"]
        if ring_by_edge.setdefault(edge, ring_index) != ring_index:
            raise RuntimeError(
                f"Physical hinge {edge} was assigned to insertion rings "
                f"{ring_by_edge[edge]} and {ring_index}."
            )
    if 0 not in ring_by_edge.values():
        raise RuntimeError("Insertion topology has no inner-loop hinges.")
    return ring_by_edge


def _set_panel_targets(
    model: Cadder,
    panel_states: dict[str, InsertionPanelState],
    panel_by_constraint: dict[str, str],
    contact_dihedral_deg: float,
) -> None:
    """Set contact/snap hinge targets from propagated panel metadata."""
    for constraint_id, surface_id in panel_by_constraint.items():
        data = model.constraints[constraint_id].data
        target = (
            contact_dihedral_deg
            if panel_states[surface_id]["state"] == "contact"
            else 180.0 - contact_dihedral_deg
        )
        data["target_increment"] = float(data["sign"] * np.deg2rad(180.0 - target))


def _branch_guess(
    model: Cadder,
    pattern: TwoDDrawer,
    quad_edges: dict[str, dict[str, list[str]]],
    panel_states: dict[str, InsertionPanelState],
    contact_dihedral_deg: float,
    tolerance: float,
) -> np.ndarray:
    """Construct an inside-out coordinate seed on the requested snap branch."""
    fold_angle_rad = np.deg2rad(180.0 - contact_dihedral_deg)
    contact_scale = float(np.cos(fold_angle_rad))
    graph: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    crease_axis_distances: list[float] = []

    for surface_id, triangle_edges in quad_edges.items():
        if len(triangle_edges) != 2:
            raise RuntimeError(
                f"Insertion panel '{surface_id}' must connect two triangles."
            )
        (first_triangle, first_edge), (second_triangle, second_edge) = triangle_edges.items()
        flat_vector = _edge_midpoint(pattern, second_edge) - _edge_midpoint(pattern, first_edge)
        crease_axis_distances.append(float(np.linalg.norm(flat_vector)))
        projected_scale = _state_sign(panel_states[surface_id]) * contact_scale
        translation_delta = (projected_scale - 1.0) * flat_vector
        graph[first_triangle].append((second_triangle, translation_delta))
        graph[second_triangle].append((first_triangle, -translation_delta))

    if _FIXED_TRIANGLE not in graph:
        raise RuntimeError(
            f"Fixed triangle '{_FIXED_TRIANGLE}' is not connected "
            "to the insertion panel graph."
        )
    translations = {_FIXED_TRIANGLE: np.zeros(2, dtype=float)}
    queue: deque[str] = deque([_FIXED_TRIANGLE])
    while queue:
        triangle_id = queue.popleft()
        for neighbor_id, delta in graph[triangle_id]:
            proposed = translations[triangle_id] + delta
            previous = translations.get(neighbor_id)
            if previous is None:
                translations[neighbor_id] = proposed
                queue.append(neighbor_id)
            elif np.linalg.norm(previous - proposed) > tolerance:
                raise RuntimeError(
                    "Insertion branch construction found a non-closing "
                    f"triangle path at '{neighbor_id}'."
                )
    if set(translations) != set(graph):
        raise RuntimeError(
            "Insertion triangle graph is disconnected: "
            f"{sorted(set(graph) - set(translations))}."
        )

    mountain_height = float(np.mean(crease_axis_distances)) * float(np.sin(fold_angle_rad))
    _, triangle_kinds = _HexagonKinematics(model)._triangle_metadata()
    proposed_by_point: dict[str, list[np.ndarray]] = defaultdict(list)
    for triangle_id, translation in translations.items():
        z_value = 0.0 if triangle_kinds[triangle_id] == "valley" else mountain_height
        for point_id in pattern.surfaces[triangle_id].vertices:
            flat_point = pattern.points[point_id]
            proposed_by_point[point_id].append(np.array(
                [flat_point.x + translation[0], flat_point.y + translation[1], z_value],
                dtype=float,
            ))

    coordinates = model.get_coordinate_vector().copy()
    for point_index, point_id in enumerate(model.point_ids()):
        proposals = proposed_by_point.get(point_id)
        if not proposals:
            continue
        if any(
            np.linalg.norm(proposal - proposals[0]) > tolerance
            for proposal in proposals[1:]
        ):
            raise RuntimeError(
                f"Insertion branch gives inconsistent coordinates for "
                f"point '{point_id}'."
            )
        coordinates[3 * point_index: 3 * point_index + 3] = np.mean(proposals, axis=0)
    return coordinates


def _loop_dihedral_stats(
    model: Cadder,
    panel_states: dict[str, InsertionPanelState],
    panel_by_constraint: dict[str, str],
) -> list[LoopDihedralStats]:
    values: dict[int, dict[PanelState, list[float]]] = defaultdict(
        lambda: {"contact": [], "snap": []}
    )
    for constraint_id, surface_id in panel_by_constraint.items():
        data = model.constraints[constraint_id].data
        panel = panel_states[surface_id]
        values[panel["ring_index"]][panel["state"]].append(abs(
            model.signed_dihedral_angle(
                data["edge_start"], data["edge_end"],
                data["point_left"], data["point_right"], unit="deg",
            )
        ))
    return [
        {
            "loop_index": loop_index,
            "contact_mean_deg": float(np.mean(angles["contact"])) if angles["contact"] else None,
            "snap_mean_deg": float(np.mean(angles["snap"])) if angles["snap"] else None,
            "num_contact_hinges": len(angles["contact"]),
            "num_snap_hinges": len(angles["snap"]),
        }
        for loop_index, angles in sorted(values.items())
    ]


def _infer_num_loops(pattern: TwoDDrawer) -> int:
    """Infer the number of concentric loops from hex-unit metadata."""
    num_units = len(pattern.hex_units)
    if num_units < 1:
        raise ValueError("The insertion model must contain at least one hex unit.")

    # draw_hex_loops creates 1 + 6 * (1 + ... + (n - 1)) units.
    discriminant = 1.0 + 4.0 * (num_units - 1) / 3.0
    num_loops = int(round((1.0 + np.sqrt(discriminant)) / 2.0))
    if 1 + 3 * num_loops * (num_loops - 1) != num_units:
        raise ValueError(
            "Could not infer a regular concentric-loop count from the model's "
            f"{num_units} hex units."
        )
    return num_loops


def _infer_side_length(pattern: TwoDDrawer) -> float:
    """Infer the base hexagon side length from the innermost unit."""
    first, second = (pattern.points[pid] for pid in pattern.hex_units[0]["mid"][:2])
    return float(np.hypot(second.x - first.x, second.y - first.y))


def simulate_insertion(
    inner_dihedral_deg: float = DEFAULT_INNER_DIHEDRAL_DEG,
    num_layers: int = DEFAULT_NUM_LAYERS,
    side_length: float | None = None,
    *,
    num_loops: int | None = None,
    model: Cadder | None = None,
    start_dihedral_deg: float = 175.0,
    branch_acute_start_deg: float = 88.0,
    initial_steps: int = 12,
    panel_sequence_steps: int = 12,
    max_nfev_per_step: int = 6000,
    solve_tolerance: float = 1e-10,
    feasibility_tolerance: float = 1e-7,
    geometry_tolerance: float = 1e-6,
    contact_tolerance: float | None = None,
    verbose: bool = False,
) -> InsertionSimulationResult:
    """Fold concentric planar loops toward the requested insertion angle.

    The requested inner obtuse angle is the ``contact`` panel target.  A
    ``snap`` panel receives its supplement ``180 degrees - contact``.  The
    states follow the inner-to-outer propagation rule of
    :func:`classify_insertion_panel_states` for all ``num_loops`` planar
    loops.

    If ``model`` is supplied, it must be a ``Cadder`` built from a compatible
    hexagon-loop pattern. Its points, lines, surfaces, and hex-unit metadata
    are used as the insertion geometry template, and ``num_loops`` and
    ``side_length`` may be omitted because they are inferred from it. The
    supplied model is not modified; the returned ``result["model"]`` is a
    solved deep copy.

    One and two loops are folded uniformly from ``start_dihedral_deg`` to
    ``180 - branch_acute_start_deg`` in ``initial_steps`` solves and then to
    the contact/snap targets in ``panel_sequence_steps`` solves. Three or
    more loops start from a constructed seed on the requested branch and are
    solved once. A state that does not satisfy the constraints within
    ``feasibility_tolerance``, or whose panels clip, raises RuntimeError.
    """
    contact_deg = _angle("inner_dihedral_deg", inner_dihedral_deg, 90.0, 180.0)
    start_dihedral_deg = _angle("start_dihedral_deg", start_dihedral_deg, 0.0, 180.0)
    if start_dihedral_deg <= contact_deg:
        raise ValueError(
            "start_dihedral_deg must be greater than inner_dihedral_deg."
        )
    snap_deg = 180.0 - contact_deg
    branch_acute_start_deg = _angle(
        "branch_acute_start_deg", branch_acute_start_deg, 0.0, 90.0,
    )
    if branch_acute_start_deg <= snap_deg:
        raise ValueError(
            "branch_acute_start_deg must be greater than the final acute "
            "dihedral."
        )
    if model is not None and not isinstance(model, Cadder):
        raise TypeError("model must be a Cadder instance or None.")

    if model is None:
        if side_length is None:
            side_length = DEFAULT_SIDE_LENGTH
        if num_loops is None:
            num_loops = DEFAULT_NUM_LOOPS
        pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
        draw_hex_loops(pattern, n=num_loops, start_point=(0.0, 0.0), l=side_length)
        model = Cadder.from_drawer(pattern)
    else:
        if not model.hex_units:
            raise ValueError(
                "The insertion model must contain hex-unit metadata. Build it "
                "with draw_hex_loops() before calling simulate_insertion()."
            )
        pattern = model.to_2d_drawer(point_tol=1e-6)
        inferred_num_loops = _infer_num_loops(pattern)
        inferred_side_length = _infer_side_length(pattern)
        if num_loops is None:
            num_loops = inferred_num_loops
        elif int(num_loops) != inferred_num_loops:
            raise ValueError(
                "num_loops does not match the supplied model: "
                f"expected {inferred_num_loops}, got {num_loops}."
            )
        if side_length is None:
            side_length = inferred_side_length
        elif not np.isclose(side_length, inferred_side_length):
            raise ValueError(
                "side_length does not match the supplied model: "
                f"expected {inferred_side_length:g}, got {side_length:g}."
            )
        # The insertion solver owns the constraints it creates. Work on a
        # copy so a model that is already solved or constrained remains intact.
        model = copy.deepcopy(model)
        model.constraints = {}
        model._constraint_count = 0

    side_length = finite("side_length", side_length, positive=True)
    num_layers = integer("num_layers", num_layers, minimum=1)
    num_loops = integer("num_loops", num_loops, minimum=1)
    initial_steps = integer("initial_steps", initial_steps, minimum=2)
    panel_sequence_steps = integer("panel_sequence_steps", panel_sequence_steps, minimum=2)
    max_nfev_per_step = integer("max_nfev_per_step", max_nfev_per_step, minimum=1)
    solve_tolerance = finite("solve_tolerance", solve_tolerance, positive=True)
    feasibility_tolerance = finite("feasibility_tolerance", feasibility_tolerance, positive=True)
    length_tolerance = finite(
        "geometry_tolerance", geometry_tolerance, positive=True,
    ) * max(1.0, side_length)
    contact_tolerance = finite(
        "contact_tolerance",
        length_tolerance if contact_tolerance is None else contact_tolerance,
        positive=True,
    )

    panel_states = classify_insertion_panel_states(pattern)
    quad_edges = _quad_crease_edges(pattern)
    panel_vectors = _panel_width_vectors(pattern, quad_edges)
    _reconcile_panel_states(pattern, panel_states, panel_vectors, length_tolerance)
    closure_errors = _closure_errors(pattern, panel_states, panel_vectors, length_tolerance)
    if closure_errors:
        details = ", ".join(
            f"unit {unit_count}: {error:.6g} {pattern.unit}"
            for unit_count, error in closure_errors[:12]
        )
        raise ValueError(
            "The reconciled insertion panel sequence is internally "
            f"inconsistent for num_loops={num_loops}: the signed "
            "projected panel widths do not close for "
            f"{len(closure_errors)} unit cells ({details}). Contact and "
            "snap targets are supplementary, so a nonzero closure cannot "
            "be removed without distorting rigid panels."
        )

    kinematics = _HexagonKinematics(model)
    kinematics._add_kinematic_constraints(
        target_dihedral=start_dihedral_deg,
        fixed_triangle_surface_id=_FIXED_TRIANGLE,
    )
    panel_by_constraint = _constraint_panel_ids(model, pattern)
    crease_ring_by_edge = _hinge_rings(model, panel_states, panel_by_constraint)
    # Insertion-only metadata: deliberately kept out of the core model types.
    pattern.insertion_panel_states = panel_states
    model.insertion_panel_states = panel_states

    def solve(start_coordinates):
        return model.solve(
            X0=start_coordinates, update_model=False,
            max_nfev=max_nfev_per_step, tol=solve_tolerance,
        )

    if num_loops > 2:
        coordinates = _branch_guess(
            model, pattern, quad_edges, panel_states, contact_deg, length_tolerance,
        )
        contact_targets = [contact_deg]
    else:
        coordinates = kinematics._initial_guess(
            mountain_height=max(1.0, 0.15 * side_length),
        )
        branch_deg = 180.0 - branch_acute_start_deg
        for step_index, target in enumerate(
            np.linspace(start_dihedral_deg, branch_deg, initial_steps), start=1,
        ):
            kinematics._update_dihedral_target(float(target))
            report = solve(coordinates)
            if report.max_abs_residual > feasibility_tolerance:
                raise RuntimeError(
                    "Could not reach the uniform starting configuration: "
                    f"target={target:.9g} degrees, "
                    f"residual={report.max_abs_residual:.3e}."
                )
            coordinates = report.x
            model.set_coordinate_vector(coordinates)
            if verbose:
                print(
                    f"[initial {step_index:02d}/{initial_steps}] "
                    f"dihedral={target:.6f} deg, "
                    f"residual={report.max_abs_residual:.3e}"
                )
        contact_targets = np.linspace(branch_deg, contact_deg, panel_sequence_steps)

    for target in contact_targets:
        _set_panel_targets(model, panel_states, panel_by_constraint, float(target))
        report = solve(coordinates)
        coordinates = report.x

    clipping = False
    if report.max_abs_residual <= feasibility_tolerance:
        model.set_coordinate_vector(coordinates)
        minimum_gap, closest_pair, clipping = minimum_panel_clearance(model)
    if verbose:
        print(
            f"[panel sequence] residual={report.max_abs_residual:.3e}, "
            f"clipping={clipping}"
        )
    if report.max_abs_residual > feasibility_tolerance or clipping:
        raise RuntimeError(
            "The propagated contact/snap panel sequence did not produce a "
            "valid non-clipping state: "
            f"residual={report.max_abs_residual:.3e}, clipping={clipping}."
        )

    contact = minimum_gap <= contact_tolerance
    stack = stack_layers(model, num_layers=num_layers, tolerance=length_tolerance)
    return {
        "model": model,
        "assembly": stack["model"],
        "pattern": pattern,
        "status": "panel_contact" if contact else "compact_nonclipping",
        "reason": (
            "The contact/snap sequence reaches non-adjacent panel contact "
            "without proper panel intersection."
            if contact else
            "The inner-to-outer contact/snap sequence reached the requested "
            "angle without panel clipping."
        ),
        "contact_dihedral_deg": contact_deg,
        "snap_dihedral_deg": snap_deg,
        "num_loops": num_loops,
        "num_layers": num_layers,
        "side_length": side_length,
        "layer_height": stack["layer_height"],
        "max_interface_error": stack["max_interface_error"],
        "max_solve_residual": report.max_abs_residual,
        "feasibility_tolerance": feasibility_tolerance,
        "minimum_panel_gap": minimum_gap,
        "closest_panel_pair": closest_pair,
        "contact_detected": contact,
        "panel_states": panel_states,
        "contact_panel_ids": frozenset(
            sid for sid, panel in panel_states.items() if panel["state"] == "contact"
        ),
        "snap_panel_ids": frozenset(
            sid for sid, panel in panel_states.items() if panel["state"] == "snap"
        ),
        "crease_ring_by_edge": crease_ring_by_edge,
        "crease_assignment_by_edge": {
            _canonical_edge(
                model.constraints[cid].data["edge_start"],
                model.constraints[cid].data["edge_end"],
            ): panel_states[surface_id]["state"]
            for cid, surface_id in panel_by_constraint.items()
        },
        "loop_dihedrals": _loop_dihedral_stats(model, panel_states, panel_by_constraint),
    }


def print_insertion_report(result: InsertionSimulationResult) -> None:
    """Print the solved state, the loop dihedrals, and the panel states."""
    unit = result["model"].unit
    title = f"Spatial {result['num_loops']}-loop contact/snap insertion simulation"
    print(title)
    print("-" * len(title))
    for label, value in (
        ("Contact / snap dihedral", f"{result['contact_dihedral_deg']:.6f} / "
                                    f"{result['snap_dihedral_deg']:.6f} deg"),
        ("Solve residual", f"{result['max_solve_residual']:.3e} "
                           f"(tolerance {result['feasibility_tolerance']:.3e})"),
        ("Status", result["status"]),
        ("Reason", result["reason"]),
        ("Panel contact reached", "yes" if result["contact_detected"] else "no"),
        ("Minimum non-adjacent gap", f"{result['minimum_panel_gap']:.6e} {unit}"),
        ("Closest panel pair", result["closest_panel_pair"]),
        ("Planar loops", result["num_loops"]),
        ("Stacked layers", result["num_layers"]),
        ("Layer height", f"{result['layer_height']:.6f} {unit}"),
        ("Maximum interface error", f"{result['max_interface_error']:.3e} {unit}"),
    ):
        print(f"{label + ':':31s}{value}")

    def mean(value):
        return "-" if value is None else f"{value:.6f}"

    print("")
    print(
        f"{'loop':6s} {'contact mean':>14s} {'hinges':>8s} "
        f"{'snap mean':>14s} {'hinges':>8s}"
    )
    for row in result["loop_dihedrals"]:
        print(
            f"{row['loop_index']:<6d} "
            f"{mean(row['contact_mean_deg']):>14s} {row['num_contact_hinges']:8d} "
            f"{mean(row['snap_mean_deg']):>14s} {row['num_snap_hinges']:8d}"
        )

    print("")
    print("Insertion parallelogram states")
    print(f"{'ring':>6s} {'source':>22s} {'contact':>10s} {'snap':>10s}")
    counts: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: {"contact": 0, "snap": 0}
    )
    for panel in result["panel_states"].values():
        counts[panel["ring_index"], panel["source"]][panel["state"]] += 1
    sources = PanelStateSource.__args__
    for (ring_index, source), count in sorted(
        counts.items(), key=lambda item: (item[0][0], sources.index(item[0][1])),
    ):
        print(
            f"{ring_index:6d} {source:>22s} "
            f"{count['contact']:10d} {count['snap']:10d}"
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
    inner_surfaces = set(result["pattern"].hex_units[0]["surfaces"])
    panel_edge = "#566568"
    # State -> (panel face color, crease color, crease line width).
    styles = {
        "contact": ("#f2a65a", "#c66b16", 1.15),
        "snap": ("#27c8ad", "#009d8f", 2.25),
    }
    triangle_color = "#dbe7e9"

    figure, axis = plt.subplots(figsize=(9.2, 9.2), constrained_layout=True)
    surface_records = []
    for surface_id in layer.surfaces:
        coordinates = np.array(
            [layer.point_array(pid) for pid in layer._surface_vertices(surface_id)],
            dtype=float,
        )
        if surface_id in result["panel_states"]:
            face_color = styles[result["panel_states"][surface_id]["state"]][0]
            alpha = 0.90
        elif surface_id in inner_surfaces:
            face_color, alpha = "#c7d9dc", 0.82
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
                label=f"Contact parallelogram ({result['contact_dihedral_deg']:.1f}°)",
            ),
            Patch(
                facecolor=styles["snap"][0], edgecolor=styles["snap"][1],
                label=f"Snap parallelogram ({result['snap_dihedral_deg']:.1f}°)",
            ),
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
