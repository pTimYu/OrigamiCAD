"""Sequential contact/snap insertion simulation for the hexagon pattern.

The default configuration labels physical parallelograms using insertion-only
metadata.  The innermost ring is ``contact``.  Each later ring is processed
once, from inner to outer: connection panels are ``snap``; panels sharing a
connection panel's outward triangle are ``contact``; all remaining panels are
``snap``.  Contact panels use the requested inner dihedral and snap panels use
its supplement.

The older regular six-sector mask search remains available through
``assignment_mode="regular_masks"``.  For its all-acute mask and the
implemented N=6 geometry with ``a = b`` and ``phi = 60 degrees``, the first
panel-contact angle follows Eq. (2) of Jamalimehr et al.:

``A_lock = acos(cot(phi) * tan(pi / N)) = acos(1 / 3)``.

Every returned state is checked for non-adjacent panel clipping.

There is no runnable entry point in this module.  The example program lives in
``main/simple_hexagon_insertion.py``.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Literal, Sequence, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ...core.cadder import Cadder
from ...core.two_d_drawer import TwoDDrawer
from .kinematics import _HexagonKinematics
from .layout import draw_hex_two_loops
from .stacking import stack_layers


DEFAULT_CONTACT_ANGLE_BUFFER_DEG = 1e-3
HEXAGON_LOCK_ACUTE_DIHEDRAL_DEG = float(
    np.rad2deg(np.arccos(1.0 / 3.0))
)
HEXAGON_LOCK_OBTUSE_DIHEDRAL_DEG = (
    180.0 - HEXAGON_LOCK_ACUTE_DIHEDRAL_DEG
)
DEFAULT_ALL_ACUTE_PRECONTACT_INNER_DIHEDRAL_DEG = (
    HEXAGON_LOCK_OBTUSE_DIHEDRAL_DEG
    - DEFAULT_CONTACT_ANGLE_BUFFER_DEG
)
# Backward-compatible alias. This is only the mask-111111 limit; mixed masks
# can remain non-clipping at larger inner angles.
DEFAULT_MAX_NO_CLIPPING_INNER_DIHEDRAL_DEG = (
    DEFAULT_ALL_ACUTE_PRECONTACT_INNER_DIHEDRAL_DEG
)
DEFAULT_SIDE_LENGTH = 15.0
DEFAULT_INNER_DIHEDRAL_DEG = 150.0
DEFAULT_NUM_LAYERS = 4


PanelState = Literal["contact", "snap"]
PanelStateSource = Literal[
    "inner",
    "connection",
    "same_outer_triangle",
    "remaining",
]


class InsertionPanelState(TypedDict):
    """Insertion-only state assigned to one physical parallelogram."""

    surface_id: str
    ring_index: int
    state: PanelState
    source: PanelStateSource
    adjacent_triangle_ids: tuple[str, ...]


class LoopDihedralStats(TypedDict):
    """Measured acute/obtuse dihedrals for one spatial loop."""

    loop_index: int
    role: Literal["inner", "outer"]
    acute_mean_deg: float | None
    obtuse_mean_deg: float | None
    num_acute_hinges: int
    num_obtuse_hinges: int


class CombinationAttempt(TypedDict):
    """Result from one symmetry-structured outer A/O combination."""

    mask: int
    bit_pattern: str
    num_acute_groups: int
    num_acute_hinges: int
    max_solve_residual: float
    kinematically_valid: bool
    minimum_panel_gap: float | None
    top_view_pca_aspect: float | None
    top_view_bbox_aspect: float | None
    clipping: bool
    contact: bool


class InsertionSimulationResult(TypedDict):
    """Geometry and diagnostics returned by :func:`simulate_insertion`."""

    model: Cadder
    assembly: Cadder
    pattern: TwoDDrawer
    status: Literal[
        "panel_contact",
        "kinematically_incompatible",
        "compact_nonclipping",
    ]
    reason: str
    assignment_mode: Literal["panel_sequence", "regular_masks"]
    requested_inner_deg: float
    realized_inner_deg: float
    requested_angle_reached: bool
    contact_limited: bool
    geometric_lock_acute_deg: float
    geometric_lock_obtuse_deg: float
    acute_dihedral_deg: float
    obtuse_dihedral_deg: float
    selected_combination_mask: int
    selected_bit_pattern: str
    selected_residual: float
    feasibility_tolerance: float
    num_combinations_tested: int
    num_kinematically_valid: int
    num_nonclipping: int
    num_layers: int
    side_length: float
    layer_height: float
    max_interface_error: float
    max_solve_residual: float
    minimum_panel_gap: float
    closest_panel_pair: tuple[str, str] | None
    contact_detected: bool
    clipping_detected: bool
    inner_constraint_ids: tuple[str, ...]
    outer_constraint_ids: tuple[str, ...]
    crease_loop_by_edge: dict[tuple[str, str], Literal["inner", "outer"]]
    crease_assignment_by_edge: dict[
        tuple[str, str],
        Literal[
            "contact",
            "snap",
            "inner-obtuse",
            "outer-acute",
            "outer-obtuse",
        ],
    ]
    panel_states: dict[str, InsertionPanelState]
    contact_panel_ids: frozenset[str]
    snap_panel_ids: frozenset[str]
    inner_surface_ids: frozenset[str]
    loop_dihedrals: list[LoopDihedralStats]
    mode_group_constraint_ids: tuple[tuple[str, ...], ...]
    attempts: list[CombinationAttempt]


def _canonical_edge(point_a: str, point_b: str) -> tuple[str, str]:
    return tuple(sorted((point_a, point_b)))


def _validate_angle(
    value: float,
    *,
    name: str,
    lower: float = 0.0,
    upper: float = 180.0,
) -> float:
    angle = float(value)
    if not np.isfinite(angle) or not (lower < angle < upper):
        raise ValueError(
            f"{name} must be between {lower:g} and {upper:g} degrees."
        )
    return angle


def _hexagon_lock_dihedral_pair() -> tuple[float, float]:
    """Return the analytical acute/obtuse first-contact pair for N=6."""
    return (
        HEXAGON_LOCK_ACUTE_DIHEDRAL_DEG,
        HEXAGON_LOCK_OBTUSE_DIHEDRAL_DEG,
    )


def _classify_spatial_hinges(
    model: Cadder,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[tuple[str, str], Literal["inner", "outer"]],
]:
    """Assign each unique hinge to the inner or outer spatial loop.

    The kinematic metadata is traversed with unit 0 first.  Consequently, a
    constraint whose ID contains ``u0`` is either unique to the central unit or
    is one of its shared physical hinges.  Shared hinges remain inner-owned.
    """
    inner_ids: list[str] = []
    outer_ids: list[str] = []
    crease_loop_by_edge: dict[
        tuple[str, str],
        Literal["inner", "outer"],
    ] = {}

    for constraint_id, constraint in model.constraints.items():
        if constraint.kind != "dihedral_signed_increment":
            continue
        data = constraint.data
        role: Literal["inner", "outer"] = (
            "inner"
            if constraint_id.startswith("dihedral_signed_u0_")
            else "outer"
        )
        data["spatial_loop"] = role
        if role == "inner":
            inner_ids.append(constraint_id)
        else:
            outer_ids.append(constraint_id)

        edge = _canonical_edge(data["edge_start"], data["edge_end"])
        previous = crease_loop_by_edge.get(edge)
        if previous is not None and previous != role:
            raise RuntimeError(
                f"Physical hinge {edge} was assigned to both spatial loops."
            )
        crease_loop_by_edge[edge] = role

    if len(inner_ids) != 12 or len(outer_ids) != 48:
        raise RuntimeError(
            "Unexpected two-loop hinge topology: "
            f"inner={len(inner_ids)}, outer={len(outer_ids)}; "
            "expected inner=12 and outer=48."
        )

    return (
        tuple(inner_ids),
        tuple(outer_ids),
        crease_loop_by_edge,
    )


def _classify_outer_mode_groups(
    model: Cadder,
    pattern: TwoDDrawer,
) -> tuple[tuple[str, ...], ...]:
    """Return the six independent surrounding-sector A/O groups.

    Within each hexagonal unit chain, all four hinges parallel to one crease
    direction must take the same regular post-bifurcation angle type.  Shared
    physical hinges propagate those equalities between neighboring chains.
    With the three central direction groups fixed obtuse, the remaining outer
    topology reduces to six binary groups of six hinges each.  Activating all
    six groups gives the sixfold reference state: the central unit is ``O3``
    and every surrounding unit is locally ``A2O``.
    """
    records: list[tuple[int, int, tuple, str]] = []
    for unit in pattern.hex_units:
        unit_index = int(unit["count"])
        for crease in unit["local_creases"]:
            edge_start, edge_end = crease["edge"]
            start = pattern.points[edge_start]
            end = pattern.points[edge_end]
            flat_angle = float(
                np.rad2deg(
                    np.arctan2(end.y - start.y, end.x - start.x)
                )
                % 180.0
            )
            direction_pair = int(np.rint(flat_angle / 60.0)) % 3
            physical_key = (
                _canonical_edge(edge_start, edge_end),
                crease["triangle"],
                crease["quad"],
            )
            constraint_id = (
                f"dihedral_signed_u{unit_index}_"
                f"i{crease['local_index']}_{crease['side']}"
            )
            records.append(
                (
                    unit_index,
                    direction_pair,
                    physical_key,
                    constraint_id,
                )
            )

    physical_keys = {record[2] for record in records}
    parent = {key: key for key in physical_keys}

    def find(key):
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root
            key = next_key
        return root

    def union(first_key, second_key) -> None:
        first_root = find(first_key)
        second_root = find(second_key)
        if first_root != second_root:
            parent[second_root] = first_root

    for unit_index in range(7):
        for direction_pair in range(3):
            group_keys = [
                physical_key
                for (
                    record_unit,
                    record_pair,
                    physical_key,
                    _,
                ) in records
                if (
                    record_unit == unit_index
                    and record_pair == direction_pair
                )
            ]
            for physical_key in group_keys[1:]:
                union(group_keys[0], physical_key)

    component_constraints: dict[tuple, set[str]] = {}
    for _, _, physical_key, constraint_id in records:
        if constraint_id not in model.constraints:
            continue
        root = find(physical_key)
        component_constraints.setdefault(root, set()).add(constraint_id)

    inner_roots = {
        find(physical_key)
        for unit_index, _, physical_key, _ in records
        if unit_index == 0
    }
    fixed_components = [
        constraint_ids
        for root, constraint_ids in component_constraints.items()
        if root in inner_roots
    ]
    free_components = [
        tuple(sorted(constraint_ids))
        for root, constraint_ids in component_constraints.items()
        if root not in inner_roots
    ]
    free_components.sort(key=lambda group: group[0])

    if (
        len(fixed_components) != 3
        or sorted(len(group) for group in fixed_components) != [8, 8, 8]
        or len(free_components) != 6
        or any(len(group) != 6 for group in free_components)
    ):
        raise RuntimeError(
            "Unexpected mixed-mode topology: expected three fixed inner "
            "components of eight hinges and six free outer components of "
            "six hinges."
        )

    for group_index, constraint_ids in enumerate(free_components):
        for constraint_id in constraint_ids:
            model.constraints[constraint_id].data[
                "outer_mode_group"
            ] = group_index

    return tuple(free_components)


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

    units_by_count = {
        int(unit["count"]): unit
        for unit in pattern.hex_units
    }
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
            unit_neighbors[unit_count].update(
                unit_counts - {unit_count}
            )

    unit_ring_by_count = {0: 0}
    queue: deque[int] = deque([0])
    while queue:
        unit_count = queue.popleft()
        for neighbor in sorted(unit_neighbors[unit_count]):
            if neighbor in unit_ring_by_count:
                continue
            unit_ring_by_count[neighbor] = (
                unit_ring_by_count[unit_count] + 1
            )
            queue.append(neighbor)
    missing_units = set(units_by_count) - set(unit_ring_by_count)
    if missing_units:
        raise ValueError(
            "Hex-unit insertion topology is disconnected; unreachable units: "
            f"{sorted(missing_units)}."
        )

    surface_ring = {
        surface_id: min(
            unit_ring_by_count[unit_count]
            for unit_count in unit_counts
        )
        for surface_id, unit_counts in surface_units.items()
    }
    parallelogram_ids = {
        surface_id
        for unit in pattern.hex_units
        for surface_id in unit["parallelograms"]
    }
    quad_triangles: dict[str, set[str]] = defaultdict(set)
    for unit in pattern.hex_units:
        for crease in unit["local_creases"]:
            quad_triangles[crease["quad"]].add(crease["triangle"])

    states: dict[str, InsertionPanelState] = {}

    def assign(
        surface_id: str,
        *,
        ring_index: int,
        state: PanelState,
        source: PanelStateSource,
    ) -> None:
        previous = states.get(surface_id)
        if previous is not None:
            if (
                previous["ring_index"] != ring_index
                or previous["state"] != state
                or previous["source"] != source
            ):
                raise RuntimeError(
                    "Insertion propagation attempted to rewrite panel "
                    f"'{surface_id}' from {previous} to "
                    f"ring={ring_index}, state={state}, source={source}."
                )
            return
        states[surface_id] = {
            "surface_id": surface_id,
            "ring_index": ring_index,
            "state": state,
            "source": source,
            "adjacent_triangle_ids": tuple(
                sorted(quad_triangles[surface_id])
            ),
        }

    for surface_id in sorted(parallelogram_ids):
        if surface_ring[surface_id] == 0:
            assign(
                surface_id,
                ring_index=0,
                state="contact",
                source="inner",
            )

    maximum_ring = max(unit_ring_by_count.values())
    for ring_index in range(1, maximum_ring + 1):
        ring_panels = {
            surface_id
            for surface_id in parallelogram_ids
            if surface_ring[surface_id] == ring_index
        }
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
        for surface_id in sorted(connection_panels):
            assign(
                surface_id,
                ring_index=ring_index,
                state="snap",
                source="connection",
            )

        same_triangle_panels: set[str] = set()
        for connection_id in connection_panels:
            outward_triangles = {
                triangle_id
                for triangle_id in quad_triangles[connection_id]
                if surface_ring[triangle_id] == ring_index
            }
            for triangle_id in outward_triangles:
                same_triangle_panels.update(
                    surface_id
                    for surface_id in ring_panels
                    if (
                        surface_id not in connection_panels
                        and triangle_id
                        in quad_triangles[surface_id]
                    )
                )
        for surface_id in sorted(same_triangle_panels):
            assign(
                surface_id,
                ring_index=ring_index,
                state="contact",
                source="same_outer_triangle",
            )

        remaining_panels = (
            ring_panels - connection_panels - same_triangle_panels
        )
        for surface_id in sorted(remaining_panels):
            assign(
                surface_id,
                ring_index=ring_index,
                state="snap",
                source="remaining",
            )

    if set(states) != parallelogram_ids:
        raise RuntimeError(
            "Insertion propagation did not classify every parallelogram."
        )
    return states


def _constraint_panel_ids(
    model: Cadder,
    pattern: TwoDDrawer,
) -> dict[str, str]:
    """Map each physical dihedral constraint to its parallelogram."""
    panel_by_constraint: dict[str, str] = {}
    for unit in pattern.hex_units:
        unit_count = int(unit["count"])
        for crease in unit["local_creases"]:
            constraint_id = (
                f"dihedral_signed_u{unit_count}_"
                f"i{crease['local_index']}_{crease['side']}"
            )
            if constraint_id in model.constraints:
                panel_by_constraint[constraint_id] = crease["quad"]

    dihedral_ids = {
        constraint_id
        for constraint_id, constraint in model.constraints.items()
        if constraint.kind == "dihedral_signed_increment"
    }
    if set(panel_by_constraint) != dihedral_ids:
        missing = sorted(dihedral_ids - set(panel_by_constraint))
        raise RuntimeError(
            "Could not associate every insertion crease with a panel: "
            f"{missing}."
        )
    return panel_by_constraint


def _set_panel_sequence_targets(
    model: Cadder,
    *,
    panel_states: dict[str, InsertionPanelState],
    panel_by_constraint: dict[str, str],
    contact_dihedral_deg: float,
) -> None:
    """Set contact/snap hinge targets from propagated panel metadata."""
    snap_dihedral_deg = 180.0 - contact_dihedral_deg
    for constraint_id, constraint in model.constraints.items():
        if constraint.kind != "dihedral_signed_increment":
            continue
        data = constraint.data
        surface_id = panel_by_constraint[constraint_id]
        panel = panel_states[surface_id]
        target = (
            contact_dihedral_deg
            if panel["state"] == "contact"
            else snap_dihedral_deg
        )
        data["target_kind"] = panel["state"]
        data["insertion_panel_surface"] = surface_id
        data["insertion_panel_ring"] = panel["ring_index"]
        data["target_increment"] = float(
            data["sign"] * np.deg2rad(180.0 - target)
        )


def _set_spatial_dihedral_targets(
    model: Cadder,
    *,
    inner_dihedral_deg: float,
    outer_dihedral_deg: float,
) -> None:
    """Set independent targets for the two spatial hinge groups."""
    for constraint in model.constraints.values():
        if constraint.kind != "dihedral_signed_increment":
            continue
        data = constraint.data
        target = (
            inner_dihedral_deg
            if data["spatial_loop"] == "inner"
            else outer_dihedral_deg
        )
        data["target_increment"] = float(
            data["sign"] * np.deg2rad(180.0 - target)
        )


def _set_mixed_mode_targets(
    model: Cadder,
    *,
    combination_mask: int,
    acute_dihedral_deg: float,
    obtuse_dihedral_deg: float,
) -> None:
    """Set one of the 64 structured mixed A/O combinations."""
    for constraint in model.constraints.values():
        if constraint.kind != "dihedral_signed_increment":
            continue
        data = constraint.data
        group_index = data.get("outer_mode_group")
        is_outer_acute = (
            group_index is not None
            and bool(combination_mask & (1 << group_index))
        )
        if is_outer_acute:
            target = acute_dihedral_deg
            data["target_kind"] = "outer-acute"
        elif data["spatial_loop"] == "inner":
            target = obtuse_dihedral_deg
            data["target_kind"] = "inner-obtuse"
        else:
            target = obtuse_dihedral_deg
            data["target_kind"] = "outer-obtuse"
        data["target_increment"] = float(
            data["sign"] * np.deg2rad(180.0 - target)
        )


def _measured_dihedral(model: Cadder, data: dict) -> float:
    return abs(
        model.signed_dihedral_angle(
            edge_start=data["edge_start"],
            edge_end=data["edge_end"],
            point_left=data["point_left"],
            point_right=data["point_right"],
            unit="deg",
        )
    )


def _loop_dihedral_stats(
    model: Cadder,
) -> list[LoopDihedralStats]:
    values: dict[
        Literal["inner", "outer"],
        dict[Literal["acute", "obtuse"], list[float]],
    ] = {
        "inner": {"acute": [], "obtuse": []},
        "outer": {"acute": [], "obtuse": []},
    }
    for constraint in model.constraints.values():
        if constraint.kind != "dihedral_signed_increment":
            continue
        data = constraint.data
        role = data["spatial_loop"]
        kind: Literal["acute", "obtuse"] = (
            "acute"
            if data["target_kind"] in {"outer-acute", "snap"}
            else "obtuse"
        )
        values[role][kind].append(_measured_dihedral(model, data))

    rows: list[LoopDihedralStats] = []
    for loop_index, role in (
        (0, "inner"),
        (1, "outer"),
    ):
        acute_values = values[role]["acute"]
        obtuse_values = values[role]["obtuse"]
        rows.append(
            {
                "loop_index": loop_index,
                "role": role,
                "acute_mean_deg": (
                    float(np.mean(acute_values))
                    if acute_values
                    else None
                ),
                "obtuse_mean_deg": (
                    float(np.mean(obtuse_values))
                    if obtuse_values
                    else None
                ),
                "num_acute_hinges": len(acute_values),
                "num_obtuse_hinges": len(obtuse_values),
            }
        )
    return rows


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= 1e-24:
        return float(np.linalg.norm(point - start))
    parameter = float(np.dot(point - start, direction) / length_squared)
    parameter = min(1.0, max(0.0, parameter))
    return float(np.linalg.norm(point - (start + parameter * direction)))


def _segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Return the shortest distance between two closed 3D segments."""
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    offset = first_start - second_start
    first_length_squared = float(
        np.dot(first_direction, first_direction)
    )
    second_length_squared = float(
        np.dot(second_direction, second_direction)
    )
    cross_term = float(np.dot(first_direction, second_direction))
    first_offset = float(np.dot(first_direction, offset))
    second_offset = float(np.dot(second_direction, offset))
    denominator = (
        first_length_squared * second_length_squared
        - cross_term * cross_term
    )
    epsilon = 1e-24

    first_parameter = 0.0
    second_parameter = 0.0
    first_numerator = 0.0
    second_numerator = denominator

    if first_length_squared <= epsilon:
        return _point_segment_distance(
            first_start,
            second_start,
            second_end,
        )
    if second_length_squared <= epsilon:
        return _point_segment_distance(
            second_start,
            first_start,
            first_end,
        )

    if denominator > epsilon:
        first_numerator = (
            cross_term * second_offset
            - second_length_squared * first_offset
        )
        first_parameter = first_numerator / denominator
    else:
        first_parameter = 0.0

    first_parameter = min(1.0, max(0.0, first_parameter))
    second_parameter = (
        cross_term * first_parameter + second_offset
    ) / second_length_squared

    if second_parameter < 0.0:
        second_parameter = 0.0
        first_parameter = min(
            1.0,
            max(0.0, -first_offset / first_length_squared),
        )
    elif second_parameter > 1.0:
        second_parameter = 1.0
        first_parameter = min(
            1.0,
            max(
                0.0,
                (cross_term - first_offset) / first_length_squared,
            ),
        )

    first_closest = first_start + first_parameter * first_direction
    second_closest = second_start + second_parameter * second_direction
    return float(np.linalg.norm(first_closest - second_closest))


def _point_triangle_distance(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> float:
    """Return point-to-triangle distance using Voronoi-region tests."""
    first_edge = second - first
    second_edge = third - first
    first_vector = point - first
    first_projection = float(np.dot(first_edge, first_vector))
    second_projection = float(np.dot(second_edge, first_vector))
    if first_projection <= 0.0 and second_projection <= 0.0:
        return float(np.linalg.norm(first_vector))

    second_vector = point - second
    third_projection = float(np.dot(first_edge, second_vector))
    fourth_projection = float(np.dot(second_edge, second_vector))
    if third_projection >= 0.0 and fourth_projection <= third_projection:
        return float(np.linalg.norm(second_vector))

    first_region = (
        first_projection * fourth_projection
        - third_projection * second_projection
    )
    if (
        first_region <= 0.0
        and first_projection >= 0.0
        and third_projection <= 0.0
    ):
        parameter = first_projection / (first_projection - third_projection)
        closest = first + parameter * first_edge
        return float(np.linalg.norm(point - closest))

    third_vector = point - third
    fifth_projection = float(np.dot(first_edge, third_vector))
    sixth_projection = float(np.dot(second_edge, third_vector))
    if sixth_projection >= 0.0 and fifth_projection <= sixth_projection:
        return float(np.linalg.norm(third_vector))

    second_region = (
        fifth_projection * second_projection
        - first_projection * sixth_projection
    )
    if (
        second_region <= 0.0
        and second_projection >= 0.0
        and sixth_projection <= 0.0
    ):
        parameter = second_projection / (
            second_projection - sixth_projection
        )
        closest = first + parameter * second_edge
        return float(np.linalg.norm(point - closest))

    third_region = (
        third_projection * sixth_projection
        - fifth_projection * fourth_projection
    )
    if (
        third_region <= 0.0
        and (fourth_projection - third_projection) >= 0.0
        and (fifth_projection - sixth_projection) >= 0.0
    ):
        parameter = (
            fourth_projection - third_projection
        ) / (
            fourth_projection
            - third_projection
            + fifth_projection
            - sixth_projection
        )
        closest = second + parameter * (third - second)
        return float(np.linalg.norm(point - closest))

    denominator = first_region + second_region + third_region
    if abs(denominator) <= 1e-24:
        return min(
            _point_segment_distance(point, first, second),
            _point_segment_distance(point, second, third),
            _point_segment_distance(point, third, first),
        )
    inverse_denominator = 1.0 / denominator
    second_barycentric = second_region * inverse_denominator
    third_barycentric = third_region * inverse_denominator
    closest = (
        first
        + first_edge * second_barycentric
        + second_edge * third_barycentric
    )
    return float(np.linalg.norm(point - closest))


def _segment_intersects_triangle(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    *,
    tolerance: float = 1e-10,
    proper_only: bool = False,
) -> bool:
    """Möller-Trumbore segment/triangle intersection test."""
    direction = segment_end - segment_start
    first_edge = second - first
    second_edge = third - first
    cross_direction = np.cross(direction, second_edge)
    determinant = float(np.dot(first_edge, cross_direction))
    if abs(determinant) <= tolerance:
        return False

    inverse_determinant = 1.0 / determinant
    origin_offset = segment_start - first
    first_barycentric = float(
        np.dot(origin_offset, cross_direction) * inverse_determinant
    )
    cross_offset = np.cross(origin_offset, first_edge)
    second_barycentric = float(
        np.dot(direction, cross_offset) * inverse_determinant
    )
    segment_parameter = float(
        np.dot(second_edge, cross_offset) * inverse_determinant
    )

    if proper_only:
        return (
            tolerance < segment_parameter < 1.0 - tolerance
            and tolerance < first_barycentric < 1.0 - tolerance
            and tolerance
            < second_barycentric
            < 1.0 - first_barycentric - tolerance
        )
    return (
        -tolerance <= segment_parameter <= 1.0 + tolerance
        and first_barycentric >= -tolerance
        and second_barycentric >= -tolerance
        and first_barycentric + second_barycentric <= 1.0 + tolerance
    )


def _triangle_distance(
    first_triangle: np.ndarray,
    second_triangle: np.ndarray,
) -> tuple[float, bool]:
    first_edges = (
        (first_triangle[0], first_triangle[1]),
        (first_triangle[1], first_triangle[2]),
        (first_triangle[2], first_triangle[0]),
    )
    second_edges = (
        (second_triangle[0], second_triangle[1]),
        (second_triangle[1], second_triangle[2]),
        (second_triangle[2], second_triangle[0]),
    )

    proper_intersection = any(
        _segment_intersects_triangle(
            edge_start,
            edge_end,
            *second_triangle,
            proper_only=True,
        )
        for edge_start, edge_end in first_edges
    ) or any(
        _segment_intersects_triangle(
            edge_start,
            edge_end,
            *first_triangle,
            proper_only=True,
        )
        for edge_start, edge_end in second_edges
    )

    if proper_intersection:
        return 0.0, True

    intersects = any(
        _segment_intersects_triangle(
            edge_start,
            edge_end,
            *second_triangle,
        )
        for edge_start, edge_end in first_edges
    ) or any(
        _segment_intersects_triangle(
            edge_start,
            edge_end,
            *first_triangle,
        )
        for edge_start, edge_end in second_edges
    )
    if intersects:
        return 0.0, False

    distances = [
        _point_triangle_distance(point, *second_triangle)
        for point in first_triangle
    ]
    distances.extend(
        _point_triangle_distance(point, *first_triangle)
        for point in second_triangle
    )
    distances.extend(
        _segment_segment_distance(
            first_start,
            first_end,
            second_start,
            second_end,
        )
        for first_start, first_end in first_edges
        for second_start, second_end in second_edges
    )
    return min(distances), False


def _surface_triangles(
    model: Cadder,
    surface_id: str,
) -> list[np.ndarray]:
    vertices = np.array(
        [
            model.point_array(point_id)
            for point_id in model._surface_vertices(surface_id)
        ],
        dtype=float,
    )
    if len(vertices) == 3:
        return [vertices]
    if len(vertices) == 4:
        return [vertices[[0, 1, 2]], vertices[[0, 2, 3]]]
    return [
        vertices[[0, index, index + 1]]
        for index in range(1, len(vertices) - 1)
    ]


def minimum_panel_clearance(
    model: Cadder,
) -> tuple[float, tuple[str, str] | None, bool]:
    """Return the closest non-adjacent panel gap and clipping diagnostic.

    Panels sharing a complete edge are direct hinge/boundary neighbors and
    are excluded because their edge contact is intentional.  Panels sharing
    only one vertex are still tested for transverse intersection; their
    permanent point contact is omitted only from the minimum-gap statistic.
    """
    minimum_gap = float("inf")
    closest_pair: tuple[str, str] | None = None
    clipping = False
    surface_vertices = {
        surface_id: set(model._surface_vertices(surface_id))
        for surface_id in model.surfaces
    }
    surface_triangles = {
        surface_id: _surface_triangles(model, surface_id)
        for surface_id in model.surfaces
    }

    for first_id, second_id in combinations(model.surfaces, 2):
        common_vertices = (
            surface_vertices[first_id] & surface_vertices[second_id]
        )
        if len(common_vertices) >= 2:
            continue
        pair_gap = float("inf")
        pair_clipping = False
        for first_triangle in surface_triangles[first_id]:
            for second_triangle in surface_triangles[second_id]:
                triangle_gap, proper_intersection = _triangle_distance(
                    first_triangle,
                    second_triangle,
                )
                pair_gap = min(pair_gap, triangle_gap)
                pair_clipping = pair_clipping or proper_intersection

        if not common_vertices and pair_gap < minimum_gap:
            minimum_gap = pair_gap
            closest_pair = (first_id, second_id)
        clipping = clipping or pair_clipping

    return minimum_gap, closest_pair, clipping


def _surface_polygons(
    model: Cadder,
    surface_ids: Sequence[str],
) -> list[list[np.ndarray]]:
    return [
        [
            model.point_array(point_id)
            for point_id in model._surface_vertices(surface_id)
        ]
        for surface_id in surface_ids
    ]


def _top_view_compactness(model: Cadder) -> tuple[float, float]:
    """Return rotation-independent and axis-aligned XY aspect ratios."""
    points_xy = np.array(
        [model.point_array(point_id)[:2] for point_id in model.points],
        dtype=float,
    )
    centered = points_xy - np.mean(points_xy, axis=0)
    singular_values = np.linalg.svd(
        centered,
        compute_uv=False,
    )
    if singular_values[-1] <= np.finfo(float).eps:
        pca_aspect = float("inf")
    else:
        pca_aspect = float(singular_values[0] / singular_values[-1])

    extents = np.ptp(points_xy, axis=0)
    minimum_extent = float(np.min(extents))
    if minimum_extent <= np.finfo(float).eps:
        bbox_aspect = float("inf")
    else:
        bbox_aspect = float(np.max(extents) / minimum_extent)
    return pca_aspect, bbox_aspect


def simulate_insertion(
    inner_dihedral_deg: float = DEFAULT_INNER_DIHEDRAL_DEG,
    num_layers: int = DEFAULT_NUM_LAYERS,
    side_length: float = DEFAULT_SIDE_LENGTH,
    *,
    assignment_mode: Literal[
        "panel_sequence",
        "regular_masks",
    ] = "panel_sequence",
    start_dihedral_deg: float = 175.0,
    combination_masks: Sequence[int] | None = None,
    branch_acute_start_deg: float = 88.0,
    initial_steps: int = 12,
    combination_steps: int = 3,
    panel_sequence_steps: int = 12,
    max_nfev_per_step: int = 6000,
    solve_tolerance: float = 1e-10,
    feasibility_tolerance: float = 1e-7,
    geometry_tolerance: float = 1e-6,
    contact_tolerance: float | None = None,
    contact_angle_buffer_deg: float = DEFAULT_CONTACT_ANGLE_BUFFER_DEG,
    beyond_contact_policy: Literal["search", "error", "stop"] = "search",
    verbose: bool = False,
) -> InsertionSimulationResult:
    """Fold the surrounding loop toward the requested insertion angle.

    The requested inner obtuse angle is the ``contact`` panel target.  A
    ``snap`` panel receives its supplement ``180 degrees - contact``.

    ``assignment_mode="panel_sequence"`` applies the insertion-specific,
    inner-to-outer propagation rule.  ``"regular_masks"`` retains the older
    six-sector mask search.  In regular-mask mode, mask ``111111`` produces
    the reference sixfold state; omitting ``combination_masks`` tests masks
    1 through 63.
    """
    if assignment_mode not in {"panel_sequence", "regular_masks"}:
        raise ValueError(
            "assignment_mode must be 'panel_sequence' or 'regular_masks'."
        )
    requested_inner_dihedral_deg = _validate_angle(
        inner_dihedral_deg,
        name="inner_dihedral_deg",
        lower=90.0,
    )
    start_dihedral_deg = _validate_angle(
        start_dihedral_deg,
        name="start_dihedral_deg",
    )
    if start_dihedral_deg <= requested_inner_dihedral_deg:
        raise ValueError(
            "start_dihedral_deg must be greater than inner_dihedral_deg."
        )
    requested_acute_dihedral_deg = (
        180.0 - requested_inner_dihedral_deg
    )
    if not (0.0 < requested_acute_dihedral_deg < 90.0):
        raise ValueError(
            "inner_dihedral_deg must have an acute supplementary angle."
        )
    contact_angle_buffer_deg = float(contact_angle_buffer_deg)
    if (
        not np.isfinite(contact_angle_buffer_deg)
        or contact_angle_buffer_deg <= 0.0
    ):
        raise ValueError(
            "contact_angle_buffer_deg must be a finite positive value."
        )
    (
        geometric_lock_acute_deg,
        geometric_lock_obtuse_deg,
    ) = _hexagon_lock_dihedral_pair()
    maximum_no_clipping_inner_deg = (
        geometric_lock_obtuse_deg - contact_angle_buffer_deg
    )
    if beyond_contact_policy not in {"search", "error", "stop"}:
        raise ValueError(
            "beyond_contact_policy must be 'search', 'error', or 'stop'."
        )
    requested_beyond_contact = (
        requested_inner_dihedral_deg
        > maximum_no_clipping_inner_deg + 1e-12
    )
    if (
        assignment_mode == "regular_masks"
        and requested_beyond_contact
        and beyond_contact_policy == "error"
    ):
        raise ValueError(
            "The requested inner dihedral "
            f"({requested_inner_dihedral_deg:.6f} deg) exceeds the all-acute "
            "outer mask's pre-contact target "
            f"({maximum_no_clipping_inner_deg:.6f} deg). "
            "For this rigid A2O branch its supplementary acute angle "
            f"({requested_acute_dihedral_deg:.6f} deg) lies past first "
            "panel contact. Choose a smaller REQUESTED_INNER_DIHEDRAL_DEG, "
            "or explicitly set beyond_contact_policy='stop' to fold only "
            "until contact."
        )
    contact_limited = (
        assignment_mode == "regular_masks"
        and requested_beyond_contact
        and beyond_contact_policy == "stop"
    )
    if contact_limited:
        realized_inner_dihedral_deg = maximum_no_clipping_inner_deg
    else:
        realized_inner_dihedral_deg = requested_inner_dihedral_deg
    acute_dihedral_deg = 180.0 - realized_inner_dihedral_deg
    contact_boundary_target = (
        abs(
            realized_inner_dihedral_deg
            - maximum_no_clipping_inner_deg
        )
        <= 1e-12
    )

    branch_acute_start_deg = _validate_angle(
        branch_acute_start_deg,
        name="branch_acute_start_deg",
        upper=90.0,
    )
    if branch_acute_start_deg <= acute_dihedral_deg:
        raise ValueError(
            "branch_acute_start_deg must be greater than the final acute "
            "dihedral."
        )
    side_length = float(side_length)
    if not np.isfinite(side_length) or side_length <= 0.0:
        raise ValueError("side_length must be a finite positive value.")
    if isinstance(num_layers, bool) or int(num_layers) != num_layers:
        raise ValueError("num_layers must be an integer.")
    num_layers = int(num_layers)
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")

    for name, value in (
        ("solve_tolerance", solve_tolerance),
        ("feasibility_tolerance", feasibility_tolerance),
        ("geometry_tolerance", geometry_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive value.")
    for name, value, minimum in (
        ("initial_steps", initial_steps, 2),
        ("combination_steps", combination_steps, 2),
        ("panel_sequence_steps", panel_sequence_steps, 2),
        ("max_nfev_per_step", max_nfev_per_step, 1),
    ):
        if isinstance(value, bool) or int(value) != value or value < minimum:
            raise ValueError(
                f"{name} must be an integer of at least {minimum}."
            )

    if assignment_mode == "panel_sequence":
        if combination_masks is not None:
            raise ValueError(
                "combination_masks is only valid when "
                "assignment_mode='regular_masks'."
            )
        masks = [-1]
    elif combination_masks is None:
        masks = list(range(1, 64))
    else:
        masks = []
        for raw_mask in combination_masks:
            if (
                isinstance(raw_mask, bool)
                or int(raw_mask) != raw_mask
                or not (0 <= int(raw_mask) < 64)
            ):
                raise ValueError(
                    "Each combination mask must be an integer from 0 to 63."
                )
            mask = int(raw_mask)
            if mask not in masks:
                masks.append(mask)
    if not masks:
        raise ValueError("At least one combination mask must be supplied.")

    if contact_tolerance is None:
        contact_tolerance = geometry_tolerance * max(1.0, side_length)
    contact_tolerance = float(contact_tolerance)
    if not np.isfinite(contact_tolerance) or contact_tolerance <= 0.0:
        raise ValueError("contact_tolerance must be a finite positive value.")

    pattern = TwoDDrawer(unit="mm", point_tol=1e-6)
    draw_hex_two_loops(
        pattern,
        start_point=(0.0, 0.0),
        l=side_length,
        reverse=False,
    )
    model = Cadder.from_drawer(pattern)
    kinematics = _HexagonKinematics(model)
    kinematics._add_kinematic_constraints(
        target_dihedral=start_dihedral_deg,
        unit="deg",
        fixed_triangle_surface_id="tri_0_1",
        valley_z=0.0,
        strict_unique_edges=False,
    )
    (
        inner_constraint_ids,
        outer_constraint_ids,
        crease_loop_by_edge,
    ) = _classify_spatial_hinges(model)
    mode_groups = _classify_outer_mode_groups(model, pattern)
    panel_states = classify_insertion_panel_states(pattern)
    panel_by_constraint = _constraint_panel_ids(model, pattern)

    coordinates = kinematics._initial_guess(
        mountain_height=max(1.0, 0.15 * side_length),
        valley_height=0.0,
    )
    maximum_residual = 0.0
    uniform_branch_angle = 180.0 - branch_acute_start_deg
    for step_index, target in enumerate(
        np.linspace(
            start_dihedral_deg,
            uniform_branch_angle,
            int(initial_steps),
        ),
        start=1,
    ):
        _set_spatial_dihedral_targets(
            model,
            inner_dihedral_deg=float(target),
            outer_dihedral_deg=float(target),
        )
        report = model.solve(
            X0=coordinates,
            update_model=False,
            max_nfev=int(max_nfev_per_step),
            tol=solve_tolerance,
            compute_rank=False,
        )
        maximum_residual = max(
            maximum_residual,
            report.max_abs_residual,
        )
        if report.max_abs_residual > feasibility_tolerance:
            raise RuntimeError(
                "Could not reach the uniform starting configuration: "
                f"target={target:.9g} degrees, "
                f"residual={report.max_abs_residual:.3e}."
            )
        coordinates = report.x.copy()
        model.set_coordinate_vector(coordinates)
        if verbose:
            print(
                f"[initial {step_index:02d}/{initial_steps}] "
                f"inner={target:.6f} deg, outer={target:.6f} deg, "
                f"residual={report.max_abs_residual:.3e}"
            )

    branch_coordinates = coordinates.copy()
    attempts: list[CombinationAttempt] = []
    safe_coordinates: dict[int, np.ndarray] = {}

    for combination_index, mask in enumerate(masks, start=1):
        candidate_coordinates = branch_coordinates.copy()
        candidate_report = None
        if assignment_mode == "panel_sequence":
            target_sequence = np.linspace(
                uniform_branch_angle,
                realized_inner_dihedral_deg,
                int(panel_sequence_steps),
            )
        else:
            target_sequence = 180.0 - np.linspace(
                branch_acute_start_deg,
                acute_dihedral_deg,
                int(combination_steps),
            )

        for obtuse_target in target_sequence:
            acute_target = 180.0 - float(obtuse_target)
            if assignment_mode == "panel_sequence":
                _set_panel_sequence_targets(
                    model,
                    panel_states=panel_states,
                    panel_by_constraint=panel_by_constraint,
                    contact_dihedral_deg=float(obtuse_target),
                )
            else:
                _set_mixed_mode_targets(
                    model,
                    combination_mask=mask,
                    acute_dihedral_deg=acute_target,
                    obtuse_dihedral_deg=float(obtuse_target),
                )
            candidate_report = model.solve(
                X0=candidate_coordinates,
                update_model=False,
                max_nfev=int(max_nfev_per_step),
                tol=solve_tolerance,
                compute_rank=False,
            )
            candidate_coordinates = candidate_report.x.copy()

        if candidate_report is None:
            raise RuntimeError("No combination continuation steps were run.")
        maximum_residual = max(
            maximum_residual,
            candidate_report.max_abs_residual,
        )
        kinematically_valid = (
            candidate_report.max_abs_residual <= feasibility_tolerance
        )
        panel_gap: float | None = None
        top_view_pca_aspect: float | None = None
        top_view_bbox_aspect: float | None = None
        clipping = False
        contact = False
        if kinematically_valid:
            model.set_coordinate_vector(candidate_coordinates)
            (
                top_view_pca_aspect,
                top_view_bbox_aspect,
            ) = _top_view_compactness(model)
            panel_gap, _, clipping = minimum_panel_clearance(model)
            analytical_reference_contact = (
                assignment_mode == "regular_masks"
                and contact_boundary_target
                and mask == 0b111111
            )
            contact = (
                not clipping
                and (
                    panel_gap <= contact_tolerance
                    or analytical_reference_contact
                )
            )
            if not clipping:
                safe_coordinates[mask] = candidate_coordinates.copy()

        if assignment_mode == "panel_sequence":
            bit_pattern = "panel-sequence"
            num_acute_groups = 0
            num_acute_hinges = sum(
                panel_states[panel_by_constraint[constraint_id]]["state"]
                == "snap"
                for constraint_id in outer_constraint_ids
            )
        else:
            bit_pattern = format(mask, "06b")
            num_acute_groups = mask.bit_count()
            num_acute_hinges = 6 * mask.bit_count()
        attempt: CombinationAttempt = {
            "mask": mask,
            "bit_pattern": bit_pattern,
            "num_acute_groups": num_acute_groups,
            "num_acute_hinges": num_acute_hinges,
            "max_solve_residual": candidate_report.max_abs_residual,
            "kinematically_valid": kinematically_valid,
            "minimum_panel_gap": panel_gap,
            "top_view_pca_aspect": top_view_pca_aspect,
            "top_view_bbox_aspect": top_view_bbox_aspect,
            "clipping": clipping,
            "contact": contact,
        }
        attempts.append(attempt)
        if verbose:
            gap_text = (
                f"{panel_gap:.6e}"
                if panel_gap is not None
                else "n/a"
            )
            print(
                f"[assignment {combination_index:02d}/{len(masks)}] "
                f"state={attempt['bit_pattern']}, "
                f"A-groups={attempt['num_acute_groups']}, "
                f"residual={attempt['max_solve_residual']:.3e}, "
                f"XY-aspect={top_view_pca_aspect}, "
                f"gap={gap_text}, clipping={clipping}"
            )

    valid_attempts = [
        attempt
        for attempt in attempts
        if attempt["kinematically_valid"]
    ]
    nonclipping_attempts = [
        attempt
        for attempt in valid_attempts
        if not attempt["clipping"]
        and attempt["minimum_panel_gap"] is not None
    ]
    if (
        assignment_mode == "panel_sequence"
        and not nonclipping_attempts
    ):
        attempt = attempts[0]
        raise RuntimeError(
            "The propagated contact/snap panel sequence did not produce a "
            "valid non-clipping state: "
            f"residual={attempt['max_solve_residual']:.3e}, "
            f"clipping={attempt['clipping']}."
        )
    if not nonclipping_attempts:
        status = "kinematically_incompatible"
        fallback_mask = 0
        fallback_coordinates = branch_coordinates.copy()
        for target in np.linspace(
            uniform_branch_angle,
            requested_inner_dihedral_deg,
            int(combination_steps),
        )[1:]:
            _set_spatial_dihedral_targets(
                model,
                inner_dihedral_deg=float(target),
                outer_dihedral_deg=float(target),
            )
            fallback_report = model.solve(
                X0=fallback_coordinates,
                update_model=False,
                max_nfev=int(max_nfev_per_step),
                tol=solve_tolerance,
                compute_rank=False,
            )
            fallback_coordinates = fallback_report.x.copy()
        selected_coordinates = fallback_coordinates
        selected_attempt: CombinationAttempt = {
            "mask": fallback_mask,
            "bit_pattern": "000000",
            "num_acute_groups": 0,
            "num_acute_hinges": 0,
            "max_solve_residual": fallback_report.max_abs_residual,
            "kinematically_valid": True,
            "minimum_panel_gap": None,
            "top_view_pca_aspect": None,
            "top_view_bbox_aspect": None,
            "clipping": False,
            "contact": False,
        }
        selected_acute_dihedral_deg = requested_acute_dihedral_deg
        selected_obtuse_dihedral_deg = requested_inner_dihedral_deg
        reason = (
            "No tested mixed A/O combination was both kinematically valid "
            "and non-clipping. The uniform obtuse reference is rendered."
        )
    else:
        selected_attempt = min(
            nonclipping_attempts,
            key=lambda item: (
                not item["contact"],
                round(float(item["top_view_pca_aspect"]), 6),
                abs(float(item["top_view_bbox_aspect"]) - 1.0),
                item["mask"],
            ),
        )
        selected_coordinates = safe_coordinates[selected_attempt["mask"]]
        selected_acute_dihedral_deg = acute_dihedral_deg
        selected_obtuse_dihedral_deg = realized_inner_dihedral_deg
        if selected_attempt["contact"]:
            status = "panel_contact"
            if contact_limited:
                reason = (
                    "The requested inner angle lies beyond the first panel-"
                    "contact limit. Folding stopped at the analytical A2O "
                    "lock state before panel interpenetration."
                )
            elif contact_boundary_target:
                reason = (
                    "The requested inner angle is the numerical pre-contact "
                    "A2O lock target. The angle was used exactly and the "
                    "contact buffer prevents panel interpenetration."
                )
            else:
                reason = (
                    "The selected mixed A/O combination reaches non-adjacent "
                    "panel contact without proper panel intersection."
                )
        else:
            status = "compact_nonclipping"
            if assignment_mode == "panel_sequence":
                reason = (
                    "The insertion-specific inner-to-outer contact/snap "
                    "sequence reached the exact requested angle without "
                    "panel clipping."
                )
            else:
                reason = (
                    "No tested combination reached exact contact without "
                    "clipping. The selected configuration is the valid "
                    "non-clipping combination with the most compact top-view "
                    "footprint."
                )

    model.set_coordinate_vector(selected_coordinates)
    if assignment_mode == "panel_sequence":
        _set_panel_sequence_targets(
            model,
            panel_states=panel_states,
            panel_by_constraint=panel_by_constraint,
            contact_dihedral_deg=selected_obtuse_dihedral_deg,
        )
    else:
        _set_mixed_mode_targets(
            model,
            combination_mask=selected_attempt["mask"],
            acute_dihedral_deg=selected_acute_dihedral_deg,
            obtuse_dihedral_deg=selected_obtuse_dihedral_deg,
        )
    loop_dihedrals = _loop_dihedral_stats(model)
    final_gap, final_pair, final_clipping = minimum_panel_clearance(model)
    if final_clipping:
        raise RuntimeError(
            "Internal error: the selected final configuration clips panels."
        )

    stack = stack_layers(
        model,
        num_layers=num_layers,
        tolerance=geometry_tolerance * max(1.0, side_length),
    )
    inner_surface_ids = frozenset(pattern.hex_units[0]["surfaces"])
    crease_assignment_by_edge = {}
    for constraint in model.constraints.values():
        if constraint.kind != "dihedral_signed_increment":
            continue
        data = constraint.data
        edge = _canonical_edge(data["edge_start"], data["edge_end"])
        crease_assignment_by_edge[edge] = data["target_kind"]

    return {
        "model": model,
        "assembly": stack["model"],
        "pattern": pattern,
        "status": status,
        "reason": reason,
        "assignment_mode": assignment_mode,
        "requested_inner_deg": requested_inner_dihedral_deg,
        "realized_inner_deg": selected_obtuse_dihedral_deg,
        "requested_angle_reached": (
            abs(
                selected_obtuse_dihedral_deg
                - requested_inner_dihedral_deg
            )
            <= contact_angle_buffer_deg
        ),
        "contact_limited": (
            contact_limited and selected_attempt["contact"]
        ),
        "geometric_lock_acute_deg": geometric_lock_acute_deg,
        "geometric_lock_obtuse_deg": geometric_lock_obtuse_deg,
        "acute_dihedral_deg": selected_acute_dihedral_deg,
        "obtuse_dihedral_deg": selected_obtuse_dihedral_deg,
        "selected_combination_mask": selected_attempt["mask"],
        "selected_bit_pattern": selected_attempt["bit_pattern"],
        "selected_residual": selected_attempt["max_solve_residual"],
        "feasibility_tolerance": feasibility_tolerance,
        "num_combinations_tested": len(attempts),
        "num_kinematically_valid": len(valid_attempts),
        "num_nonclipping": len(nonclipping_attempts),
        "num_layers": num_layers,
        "side_length": side_length,
        "layer_height": stack["layer_height"],
        "max_interface_error": stack["max_interface_error"],
        "max_solve_residual": selected_attempt["max_solve_residual"],
        "minimum_panel_gap": final_gap,
        "closest_panel_pair": final_pair,
        "contact_detected": selected_attempt["contact"],
        "clipping_detected": final_clipping,
        "inner_constraint_ids": inner_constraint_ids,
        "outer_constraint_ids": outer_constraint_ids,
        "crease_loop_by_edge": crease_loop_by_edge,
        "crease_assignment_by_edge": crease_assignment_by_edge,
        "panel_states": panel_states,
        "contact_panel_ids": frozenset(
            surface_id
            for surface_id, panel in panel_states.items()
            if panel["state"] == "contact"
        ),
        "snap_panel_ids": frozenset(
            surface_id
            for surface_id, panel in panel_states.items()
            if panel["state"] == "snap"
        ),
        "inner_surface_ids": inner_surface_ids,
        "loop_dihedrals": loop_dihedrals,
        "mode_group_constraint_ids": mode_groups,
        "attempts": attempts,
    }


def print_insertion_report(result: InsertionSimulationResult) -> None:
    """Print the mixed-mode search and selected loop dihedrals."""
    if result["assignment_mode"] == "panel_sequence":
        print("Spatial two-loop contact/snap insertion simulation")
        print("-------------------------------------------------")
    else:
        print("Spatial two-loop mixed A/O simulation")
        print("-------------------------------------")
    if result["assignment_mode"] == "panel_sequence":
        assignment = (
            "inner-to-outer propagated contact/snap parallelograms"
        )
    elif result["selected_combination_mask"] == 0b111111:
        assignment = "central O3; six surrounding A2O unit chains"
    elif result["selected_combination_mask"] == 0:
        assignment = "uniform O3 reference"
    else:
        assignment = "central O3; mixed outer regular A/O sectors"
    print(f"Assignment:                    {assignment}")
    print(
        "Assignment mode:               "
        f"{result['assignment_mode']}"
    )
    print(
        "Requested inner loop:          "
        f"{result['requested_inner_deg']:.6f} deg"
    )
    print(
        "Realized inner loop:           "
        f"{result['realized_inner_deg']:.6f} deg"
    )
    print(
        "Requested angle reached:       "
        f"{'yes' if result['requested_angle_reached'] else 'no'}"
    )
    print(
        "Analytical contact A / O:      "
        f"{result['geometric_lock_acute_deg']:.6f} / "
        f"{result['geometric_lock_obtuse_deg']:.6f} deg"
    )
    print(
        "Acute / obtuse pair:           "
        f"{result['acute_dihedral_deg']:.6f} / "
        f"{result['obtuse_dihedral_deg']:.6f} deg"
    )
    if result["assignment_mode"] == "panel_sequence":
        print(
            "Selected state rule:          "
            f"{result['selected_bit_pattern']}"
        )
    else:
        print(
            "Selected combination:          "
            f"{result['selected_bit_pattern']} "
            f"(mask {result['selected_combination_mask']})"
        )
    print(
        "Selected solve residual:       "
        f"{result['selected_residual']:.3e}"
    )
    print(
        "Feasibility tolerance:         "
        f"{result['feasibility_tolerance']:.3e}"
    )
    print(
        (
            "Panel sequences tested:       "
            if result["assignment_mode"] == "panel_sequence"
            else "Combinations tested:           "
        )
        + f"{result['num_combinations_tested']}"
    )
    print(
        "Kinematically valid:           "
        f"{result['num_kinematically_valid']}"
    )
    print(
        "Valid and non-clipping:        "
        f"{result['num_nonclipping']}"
    )
    print(f"Status:                        {result['status']}")
    print(f"Reason:                        {result['reason']}")
    print(
        "Panel contact reached:         "
        f"{'yes' if result['contact_detected'] else 'no'}"
    )
    print(
        "Panel clipping in drawing:     "
        f"{'yes' if result['clipping_detected'] else 'no'}"
    )
    print(
        "Minimum non-adjacent gap:      "
        f"{result['minimum_panel_gap']:.6e} mm"
    )
    print(
        "Closest panel pair:            "
        f"{result['closest_panel_pair']}"
    )
    print(f"Stacked layers:                 {result['num_layers']}")
    print(f"Layer height:                   {result['layer_height']:.6f} mm")
    print(
        "Maximum interface error:       "
        f"{result['max_interface_error']:.3e} mm"
    )
    print(
        "Selected maximum residual:     "
        f"{result['max_solve_residual']:.3e}"
    )
    print("")
    print(
        f"{'loop':6s} {'role':8s} {'acute mean':>14s} "
        f"{'A hinges':>10s} {'obtuse mean':>14s} {'O hinges':>10s}"
    )
    for row in result["loop_dihedrals"]:
        acute_text = (
            f"{row['acute_mean_deg']:.6f}"
            if row["acute_mean_deg"] is not None
            else "-"
        )
        obtuse_text = (
            f"{row['obtuse_mean_deg']:.6f}"
            if row["obtuse_mean_deg"] is not None
            else "-"
        )
        print(
            f"{row['loop_index']:<6d} {row['role']:<8s} "
            f"{acute_text:>14s} "
            f"{row['num_acute_hinges']:10d} "
            f"{obtuse_text:>14s} "
            f"{row['num_obtuse_hinges']:10d}"
        )

    if result["assignment_mode"] == "panel_sequence":
        print("")
        print("Insertion parallelogram states")
        print(
            f"{'ring':>6s} {'source':>22s} "
            f"{'contact':>10s} {'snap':>10s}"
        )
        rings = sorted(
            {
                panel["ring_index"]
                for panel in result["panel_states"].values()
            }
        )
        sources: tuple[PanelStateSource, ...] = (
            "inner",
            "connection",
            "same_outer_triangle",
            "remaining",
        )
        for ring_index in rings:
            for source in sources:
                panels = [
                    panel
                    for panel in result["panel_states"].values()
                    if (
                        panel["ring_index"] == ring_index
                        and panel["source"] == source
                    )
                ]
                if not panels:
                    continue
                print(
                    f"{ring_index:6d} {source:>22s} "
                    f"{sum(p['state'] == 'contact' for p in panels):10d} "
                    f"{sum(p['state'] == 'snap' for p in panels):10d}"
                )

    if (
        result["assignment_mode"] == "regular_masks"
        and result["attempts"]
    ):
        print("")
        ranked = sorted(
            result["attempts"],
            key=lambda attempt: (
                attempt["clipping"],
                (
                    float(attempt["top_view_pca_aspect"])
                    if attempt["top_view_pca_aspect"] is not None
                    else float("inf")
                ),
                (
                    abs(float(attempt["top_view_bbox_aspect"]) - 1.0)
                    if attempt["top_view_bbox_aspect"] is not None
                    else float("inf")
                ),
                attempt["mask"],
            ),
        )
        print(
            "Most compact non-clipping combinations "
            "(up to 12 shown)"
        )
        print(
            f"{'mask':>6s} {'pattern':>9s} {'A groups':>9s} "
            f"{'A hinges':>9s} {'residual':>12s} "
            f"{'XY aspect':>10s} {'bbox':>8s} "
            f"{'gap [mm]':>12s} {'clips':>8s}"
        )
        for attempt in ranked[:12]:
            gap_text = (
                f"{attempt['minimum_panel_gap']:.6e}"
                if attempt["minimum_panel_gap"] is not None
                else "-"
            )
            pca_text = (
                f"{attempt['top_view_pca_aspect']:.4f}"
                if attempt["top_view_pca_aspect"] is not None
                else "-"
            )
            bbox_text = (
                f"{attempt['top_view_bbox_aspect']:.4f}"
                if attempt["top_view_bbox_aspect"] is not None
                else "-"
            )
            print(
                f"{attempt['mask']:6d} "
                f"{attempt['bit_pattern']:>9s} "
                f"{attempt['num_acute_groups']:9d} "
                f"{attempt['num_acute_hinges']:9d} "
                f"{attempt['max_solve_residual']:12.3e} "
                f"{pca_text:>10s} "
                f"{bbox_text:>8s} "
                f"{gap_text:>12s} "
                f"{str(attempt['clipping']):>8s}"
            )


def draw_insertion_simulation(
    result: InsertionSimulationResult,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 240,
) -> Path | None:
    """Draw an orthographic top view of the selected one-layer state."""
    layer = result["model"]
    inner_surfaces = result["inner_surface_ids"]
    panel_color = "#dbe7e9"
    inner_panel_color = "#c7d9dc"
    acute_panel_color = "#27c8ad"
    contact_panel_color = "#f2a65a"
    snap_panel_color = "#27c8ad"
    panel_edge = "#566568"
    inner_edge = "#a9580f"
    outer_acute_edge = "#009d8f"
    outer_obtuse_edge = "#5c6b70"
    contact_edge = "#c66b16"
    snap_edge = "#009d8f"
    panel_sequence = result["assignment_mode"] == "panel_sequence"

    selected_attempt = next(
        (
            attempt
            for attempt in result["attempts"]
            if attempt["mask"] == result["selected_combination_mask"]
        ),
        {
            "mask": 0,
            "bit_pattern": "000000",
            "num_acute_groups": 0,
            "num_acute_hinges": 0,
            "max_solve_residual": result["selected_residual"],
            "kinematically_valid": True,
            "minimum_panel_gap": result["minimum_panel_gap"],
            "top_view_pca_aspect": None,
            "top_view_bbox_aspect": None,
            "clipping": False,
            "contact": False,
        },
    )

    figure, top_axis = plt.subplots(
        figsize=(9.2, 9.2),
        constrained_layout=True,
    )
    surface_records = []
    for surface_id in layer.surfaces:
        point_ids = layer._surface_vertices(surface_id)
        coordinates = np.array(
            [layer.point_array(point_id) for point_id in point_ids],
            dtype=float,
        )
        assignments = {
            result["crease_assignment_by_edge"][edge]
            for point_a, point_b in combinations(point_ids, 2)
            if (
                edge := _canonical_edge(point_a, point_b)
            ) in result["crease_assignment_by_edge"]
        }
        is_acute_triangle = (
            len(point_ids) == 3 and "outer-acute" in assignments
        )
        if panel_sequence and surface_id in result["panel_states"]:
            panel_state = result["panel_states"][surface_id]["state"]
            face_color = (
                contact_panel_color
                if panel_state == "contact"
                else snap_panel_color
            )
            alpha = 0.90
        elif is_acute_triangle:
            face_color = acute_panel_color
            alpha = 0.94
        elif surface_id in inner_surfaces:
            face_color = inner_panel_color
            alpha = 0.82
        else:
            face_color = panel_color
            alpha = 0.78
        surface_records.append(
            (
                float(np.mean(coordinates[:, 2])),
                coordinates[:, :2],
                face_color,
                alpha,
            )
        )

    minimum_z = min(record[0] for record in surface_records)
    for mean_z, polygon_xy, face_color, alpha in sorted(
        surface_records,
        key=lambda record: record[0],
    ):
        top_axis.add_collection(
            PolyCollection(
                [polygon_xy],
                facecolors=[face_color],
                edgecolors=[panel_edge],
                linewidths=0.72,
                alpha=alpha,
                zorder=10.0 + mean_z - minimum_z,
            )
        )

    for line_id in layer.lines:
        start_id, end_id, _ = layer._line_info(line_id)
        assignment = result["crease_assignment_by_edge"].get(
            _canonical_edge(start_id, end_id)
        )
        if assignment is None:
            continue
        crease_color = {
            "contact": contact_edge,
            "snap": snap_edge,
            "inner-obtuse": inner_edge,
            "outer-acute": outer_acute_edge,
            "outer-obtuse": outer_obtuse_edge,
        }[assignment]
        start = layer.point_array(start_id)
        end = layer.point_array(end_id)
        top_axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=crease_color,
            linewidth=(
                2.25
                if assignment in {"outer-acute", "snap"}
                else 1.15
                if assignment in {"inner-obtuse", "contact"}
                else 0.8
            ),
            alpha=0.98,
            solid_capstyle="round",
            zorder=200.0 + float(np.mean([start[2], end[2]])),
        )

    points_xy = np.array(
        [layer.point_array(point_id)[:2] for point_id in layer.points],
        dtype=float,
    )
    xy_min = np.min(points_xy, axis=0)
    xy_max = np.max(points_xy, axis=0)
    center = 0.5 * (xy_min + xy_max)
    span = float(np.max(xy_max - xy_min))
    margin = 0.075 * span
    top_axis.set_xlim(center[0] - 0.5 * span - margin,
                     center[0] + 0.5 * span + margin)
    top_axis.set_ylim(center[1] - 0.5 * span - margin,
                     center[1] + 0.5 * span + margin)
    top_axis.set_aspect("equal", adjustable="box")
    top_axis.axis("off")

    pca_aspect = selected_attempt["top_view_pca_aspect"]
    aspect_text = (
        f"{pca_aspect:.3f}"
        if pca_aspect is not None
        else "n/a"
    )
    if result["contact_limited"]:
        angle_text = (
            f"requested inner O = {result['requested_inner_deg']:.1f}°  →  "
            f"contact-limited O = {result['realized_inner_deg']:.3f}°"
        )
    else:
        angle_text = (
            f"requested and realized inner O = "
            f"{result['realized_inner_deg']:.3f}°"
        )
    if panel_sequence:
        state_text = (
            f"{len(result['contact_panel_ids'])} contact panels  ·  "
            f"{len(result['snap_panel_ids'])} snap panels"
        )
        title = (
            "One-layer orthographic top view — propagated panel sequence\n"
            f"{angle_text}\n"
            f"{state_text}  ·  XY aspect = {aspect_text}"
        )
        legend_handles = [
            Patch(
                facecolor=contact_panel_color,
                edgecolor=contact_edge,
                label=(
                    "Contact parallelogram "
                    f"({result['obtuse_dihedral_deg']:.1f}°)"
                ),
            ),
            Patch(
                facecolor=snap_panel_color,
                edgecolor=snap_edge,
                label=(
                    "Snap parallelogram "
                    f"({result['acute_dihedral_deg']:.1f}°)"
                ),
            ),
            Patch(
                facecolor=panel_color,
                edgecolor=panel_edge,
                label="Triangular panel",
            ),
        ]
    else:
        title = (
            "One-layer orthographic top view — mixed A/O configuration\n"
            f"{angle_text}\n"
            f"mask {result['selected_bit_pattern']}  ·  "
            f"outer = {selected_attempt['num_acute_hinges']} A / "
            f"{len(result['outer_constraint_ids']) - selected_attempt['num_acute_hinges']} O  ·  "
            f"XY aspect = {aspect_text}"
        )
        legend_handles = [
            Patch(
                facecolor=acute_panel_color,
                edgecolor=outer_acute_edge,
                label="Panels adjoining acute outer groups",
            ),
            Line2D(
                [0],
                [0],
                color=inner_edge,
                linewidth=1.5,
                label="Inner loop: obtuse",
            ),
            Line2D(
                [0],
                [0],
                color=outer_acute_edge,
                linewidth=2.25,
                label="Outer loop: acute",
            ),
            Line2D(
                [0],
                [0],
                color=outer_obtuse_edge,
                linewidth=1.0,
                label="Outer loop: obtuse",
            ),
        ]
    top_axis.set_title(
        title,
        fontsize=12.5,
        pad=15.0,
    )
    top_axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=2,
        frameon=True,
        framealpha=0.96,
        fontsize=9,
    )

    resolved_path = None
    if save_path is not None:
        resolved_path = Path(save_path).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            resolved_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(figure)
    return resolved_path


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
    layer_colors = (
        "#75858b",
        "#9fbcc2",
        "#67aeb7",
        "#25c5aa",
    )
    if num_layers > len(layer_colors):
        color_positions = np.linspace(0.25, 0.82, num_layers)
        layer_colors = tuple(
            plt.get_cmap("viridis")(position)
            for position in color_positions
        )

    figure = plt.figure(
        figsize=(11.0, 9.0),
        constrained_layout=True,
    )
    axis = figure.add_subplot(111, projection="3d")

    for layer_index in range(num_layers):
        prefix = f"layer_{layer_index}::"
        surface_ids = [
            surface_id
            for surface_id in assembly.surfaces
            if surface_id.startswith(prefix)
        ]
        axis.add_collection3d(
            Poly3DCollection(
                _surface_polygons(assembly, surface_ids),
                facecolors=layer_colors[layer_index],
                edgecolors="#3c4b50",
                linewidths=0.34,
                alpha=0.48 if layer_index < num_layers - 1 else 0.62,
            )
        )

    assembly._set_axes_equal_3d(axis)
    axis.set_xlabel("x [mm]", labelpad=8.0)
    axis.set_ylabel("y [mm]", labelpad=8.0)
    axis.set_zlabel("z [mm]", labelpad=8.0)
    axis.view_init(elev=25.0, azim=-55.0)
    axis.grid(True, alpha=0.18)
    if result["assignment_mode"] == "panel_sequence":
        configuration_name = "propagated contact/snap assembly"
    elif (
        result["selected_combination_mask"] == 0b111111
        and result["contact_detected"]
    ):
        configuration_name = "A²O contact-lock assembly"
    else:
        configuration_name = "mixed A/O assembly"
    axis.set_title(
        f"{num_layers}-layer {configuration_name} — isometric view\n"
        f"A = {result['acute_dihedral_deg']:.3f}°  ·  "
        f"O = {result['obtuse_dihedral_deg']:.3f}°  ·  "
        f"layer height = {result['layer_height']:.3f} mm  ·  "
        f"clipping = {'yes' if result['clipping_detected'] else 'no'}",
        fontsize=13,
        pad=18.0,
    )
    axis.legend(
        handles=[
            Patch(
                facecolor=layer_colors[layer_index],
                edgecolor="#3c4b50",
                alpha=0.72,
                label=(
                    f"Layer {layer_index + 1}"
                    + (" (top)" if layer_index == num_layers - 1 else "")
                ),
            )
            for layer_index in range(num_layers)
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        fontsize=9,
    )
    resolved_path = None
    if save_path is not None:
        resolved_path = Path(save_path).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            resolved_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(figure)
    return resolved_path
