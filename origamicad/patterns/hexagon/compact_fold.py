"""Find the most compact rigid fold of a hexagon pattern around its cargo.

Squeezing a folded pattern evenly from all sides drives it towards the fold
that takes the least room. With rigid panels the candidates are discrete:
every parallelogram is either at the fold angle or snapped to its supplement,
and the triangles, which only translate, must still close around every unit
chain and around the cavity (see :mod:`snap_lines`). Each snapped
parallelogram pulls its two triangles together by the same distance, so the
most compact fold is the one with the most snapped parallelograms that

* closes everywhere,
* leaves the cargo, or the panels that are held, where they are, and
* does not make panels pass through each other.

The first two conditions are linear in the snap states and the triangle
translations, which makes this a small mixed-integer program. Whether panels
intersect is checked on the exact folded shape. Each intersection found
depends only on the states of the parallelograms between the two panels, so
that combination is excluded and the program is solved again.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from ...core.clearance import _surface_triangles, _triangle_distance
from .closure import _lattice
from .metadata import hex_constraint_plan
from .snap_lines import _quad_links, _unit_quads, fold_with_snapped_panels


@dataclass
class CargoRectangle:
    """Rectangle the cavity closes on, in the folded pattern's xy plane.

    ``axis`` is the unit vector along the rectangle's first side; ``low`` and
    ``high`` are its extents along ``axis`` and along the perpendicular
    ``(-axis[1], axis[0])``. A side the cavity is open on has an infinite
    extent.
    """

    axis: np.ndarray
    low: np.ndarray
    high: np.ndarray

    def coordinates(self, xy: np.ndarray) -> np.ndarray:
        normal = np.array([-self.axis[1], self.axis[0]])
        return np.column_stack((xy @ self.axis, xy @ normal))

    def depth(self, xy: np.ndarray) -> np.ndarray:
        """How far each point lies inside the rectangle (zero if outside)."""
        uv = self.coordinates(xy)
        return np.clip(np.minimum(uv - self.low, self.high - uv).min(axis=1), 0.0, None)


@dataclass
class CompactFold:
    """Result of :func:`most_compact_fold`.

    ``coordinates`` is the model's coordinate vector for the fold and
    ``snapped`` the parallelograms at the supplementary angle. ``clipping``
    lists panel pairs that still intersect if the search ended early, and
    ``cargo_overlap`` is how far the deepest point reaches into the cargo.
    """

    snapped: frozenset[str]
    coordinates: np.ndarray
    cargo: Optional[CargoRectangle]
    resting_vertices: int
    rounds: int
    excluded_combinations: int
    clipping: list[tuple[str, str]] = field(default_factory=list)
    cargo_overlap: float = 0.0


def _pierces(starts, ends, a, b, c, tolerance=1e-10):
    """Whether each segment passes through the interior of its triangle."""
    direction = ends - starts
    ab, ac = b - a, c - a
    cross_direction = np.cross(direction, ac)
    determinant = np.einsum("ij,ij->i", ab, cross_direction)
    regular = np.abs(determinant) > tolerance
    inverse = 1.0 / np.where(regular, determinant, 1.0)
    offset = starts - a
    u = np.einsum("ij,ij->i", offset, cross_direction) * inverse
    cross_offset = np.cross(offset, ab)
    v = np.einsum("ij,ij->i", direction, cross_offset) * inverse
    t = np.einsum("ij,ij->i", ac, cross_offset) * inverse
    return (
        regular
        & (t > tolerance) & (t < 1.0 - tolerance)
        & (u > tolerance) & (u < 1.0 - tolerance)
        & (v > tolerance) & (v < 1.0 - u - tolerance)
    )


def clipping_pairs(model, minimum_gap: float = 0.0) -> list[tuple[str, str]]:
    """Return every pair of panels that pass through each other.

    Panels that share an edge are hinge neighbors and are skipped. Panels
    with no point in common also count when they are closer than
    ``minimum_gap``.
    """
    surface_ids = list(model.surfaces)
    index = {pid: i for i, pid in enumerate(model.point_ids())}
    points = model.get_coordinate_vector().reshape(-1, 3)
    corners = [[index[pid] for pid in model._surface_vertices(sid)] for sid in surface_ids]
    vertices = [set(panel) for panel in corners]
    lower = np.array([points[panel].min(axis=0) for panel in corners]) - minimum_gap
    upper = np.array([points[panel].max(axis=0) for panel in corners]) + minimum_gap
    overlap = np.all(
        (lower[:, None, :] <= upper[None, :, :]) & (lower[None, :, :] <= upper[:, None, :]),
        axis=2,
    )
    candidates = [
        (int(first), int(second))
        for first, second in zip(*np.nonzero(np.triu(overlap, k=1)))
        if len(vertices[first] & vertices[second]) < 2
    ]
    # Two panels intersect exactly when a boundary edge of one passes
    # through the other, so test every such edge against every triangle.
    segment_start, segment_end, corner_a, corner_b, corner_c, owner = [], [], [], [], [], []
    for number, (first, second) in enumerate(candidates):
        for edges_of, faces_of in ((first, second), (second, first)):
            edge_panel, face_panel = corners[edges_of], corners[faces_of]
            for start, end in zip(edge_panel, edge_panel[1:] + edge_panel[:1]):
                for k in range(1, len(face_panel) - 1):
                    segment_start.append(start)
                    segment_end.append(end)
                    corner_a.append(face_panel[0])
                    corner_b.append(face_panel[k])
                    corner_c.append(face_panel[k + 1])
                    owner.append(number)
    clipping = set()
    if owner:
        hits = _pierces(
            points[segment_start], points[segment_end],
            points[corner_a], points[corner_b], points[corner_c],
        )
        clipping = set(np.asarray(owner)[hits].tolist())
    pairs = []
    for number, (first, second) in enumerate(candidates):
        close = False
        if number not in clipping and minimum_gap > 0 and not vertices[first] & vertices[second]:
            close = min(
                _triangle_distance(a, b)[0]
                for a in _surface_triangles(model, surface_ids[first])
                for b in _surface_triangles(model, surface_ids[second])
            ) < minimum_gap
        if number in clipping or close:
            pairs.append((surface_ids[first], surface_ids[second]))
    return pairs


def cargo_rectangle(model, folded: np.ndarray) -> CargoRectangle:
    """Measure the rectangle that fits the cavity of a folded packaging pattern.

    The cavity is the block of lattice cells without a unit chain inside the
    pattern's bounding rectangle. The rectangle's first side runs along the
    folded rows of unit chains. Starting from the cavity center, its sides
    grow at the same rate until each rests on a point of the folded
    structure. A side on which the cavity is open stays unbounded.
    """
    units, side, rotation, basis, origin, cells, _ = _lattice(model)
    height = side * np.sqrt(3.0) / 2.0
    index = {pid: i for i, pid in enumerate(model.point_ids())}
    points = folded.reshape(-1, 3)[:, :2]
    flat_xy = model.get_coordinate_vector().reshape(-1, 3)[:, :2]

    def flat_center(cell):
        return rotation @ (origin + basis @ np.array(cell, dtype=float) + np.array([0.5 * side, -height]))

    # Unit chains two columns apart in a row differ by the lattice cell (1, 1).
    by_cell = {cell: unit for cell, unit in zip(cells, units)}
    row_steps = [
        points[index[by_cell[(i + 1, j + 1)]["mid"][0]]] - points[index[unit["mid"][0]]]
        for (i, j), unit in by_cell.items() if (i + 1, j + 1) in by_cell
    ]
    if not row_steps:
        raise ValueError("The pattern is too small to measure its cargo.")
    axis = np.mean(row_steps, axis=0)
    cargo = CargoRectangle(axis / np.linalg.norm(axis), np.zeros(2), np.zeros(2))

    flat_axis = rotation @ (np.array([4.5 * side, height]) / np.hypot(4.5 * side, height))
    flat_frame = CargoRectangle(flat_axis, np.zeros(2), np.zeros(2))
    present = set(cells)
    uv = flat_frame.coordinates(np.array([flat_center(cell) for cell in present]))
    low, high = uv.min(axis=0) - 1e-6 * side, uv.max(axis=0) + 1e-6 * side
    empty = np.array([
        flat_center((i, j))
        for i in range(min(i for i, _ in present) - 1, max(i for i, _ in present) + 2)
        for j in range(min(j for _, j in present) - 1, max(j for _, j in present) + 2)
        if (i, j) not in present
    ])
    empty_uv = flat_frame.coordinates(empty)
    inside = np.all((empty_uv >= low) & (empty_uv <= high), axis=1)
    if not np.any(inside):
        raise ValueError("The pattern has no cavity for a cargo.")
    # A uniform fold moves the unit chains by an exact affine map, which
    # carries the flat cavity center to the folded one.
    corners = [index[unit["mid"][0]] for unit in units]
    design = np.column_stack((flat_xy[corners], np.ones(len(corners))))
    affine = np.linalg.lstsq(design, points[corners], rcond=None)[0]
    center = np.append(empty[inside].mean(axis=0), 1.0) @ affine
    start = cargo.coordinates(center[None, :])[0]

    # Grow the rectangle from the center, all sides at the same rate. A side
    # stops at the first point that would come inside; a side that leaves the
    # structure altogether is open.
    uv = cargo.coordinates(points)
    outer_low, outer_high = uv.min(axis=0), uv.max(axis=0)
    step, tiny = side / 40.0, 1e-9 * side
    box_low, box_high = start.copy(), start.copy()
    growing = [(axis_index, sign) for axis_index in (0, 1) for sign in (-1, 1)]
    while growing:
        for axis_index, sign in list(growing):
            trial_low, trial_high = box_low.copy(), box_high.copy()
            if sign < 0:
                trial_low[axis_index] -= step
            else:
                trial_high[axis_index] += step
            inside = np.all((uv > trial_low + tiny) & (uv < trial_high - tiny), axis=1)
            if np.any(inside):
                if sign < 0:
                    box_low[axis_index] = uv[inside, axis_index].max()
                else:
                    box_high[axis_index] = uv[inside, axis_index].min()
                growing.remove((axis_index, sign))
            elif sign < 0 and trial_low[axis_index] < outer_low[axis_index]:
                box_low[axis_index] = -np.inf
                growing.remove((axis_index, sign))
            elif sign > 0 and trial_high[axis_index] > outer_high[axis_index]:
                box_high[axis_index] = np.inf
                growing.remove((axis_index, sign))
            else:
                box_low, box_high = trial_low, trial_high
    cargo.low, cargo.high = box_low, box_high
    return cargo


def most_compact_fold(
    model,
    dihedral_deg: float,
    *,
    held: Iterable[str] = (),
    required: Iterable[str] = (),
    fixed: Iterable[str] = (),
    cargo: Optional[CargoRectangle] = None,
    cargo_contact: str = "slide",
    whole_lines: bool = False,
    minimum_gap: float = 0.0,
    max_rounds: int = 200,
) -> CompactFold:
    """Return the rigid fold with the most snapped parallelograms.

    ``model`` is the flat ``Cadder`` of a hexagon pattern and is not changed.
    ``held`` parallelograms keep the fold angle and ``required`` ones must
    snap. ``fixed`` triangles stay where they are in the unsnapped fold. With
    ``cargo``, the vertices
    that rest on a side of that rectangle in the unsnapped fold keep resting
    on that side, and no other point may move into the rectangle. With
    ``cargo_contact="slide"`` those vertices may move along the side, so a
    wall can shorten along the cargo; ``"fixed"`` keeps them exactly where
    they are, which stops every snap between two of them. ``minimum_gap`` is the clearance panels with no common point
    must keep.

    With ``whole_lines`` every snap line snaps as a whole: opposite
    parallelograms of a unit chain are always in the same state. Without it a
    unit chain may also take the alternating state, in which a line changes
    state part of the way along.

    The search alternates between the mixed-integer program and the exact
    intersection check, for at most ``max_rounds`` rounds.
    """
    if cargo_contact not in ("slide", "fixed"):
        raise ValueError("cargo_contact must be 'slide' or 'fixed'.")
    fold = np.deg2rad(180.0 - float(dihedral_deg))
    cosine = float(np.cos(fold))
    point_ids = model.point_ids()
    flat = model.get_coordinate_vector().reshape(-1, 3)
    xy = {pid: flat[i, :2].copy() for i, pid in enumerate(point_ids)}
    links = _quad_links(model, xy)
    quads = list(links)
    quad_index = {quad: i for i, quad in enumerate(quads)}
    triangles = sorted({t for first, second, _ in links.values() for t in (first, second)})
    triangle_index = {t: i for i, t in enumerate(triangles)}
    n_quads, n_triangles = len(quads), len(triangles)
    size = n_quads + 2 * n_triangles
    extent = 4.0 * float(np.ptp(flat[:, :2], axis=0).max())

    rows, columns, values = [], [], []
    row = 0
    adjacency: dict[str, list[tuple[str, str]]] = {t: [] for t in triangles}
    for quad, (first, second, width) in links.items():
        adjacency[first].append((second, quad))
        adjacency[second].append((first, quad))
        for axis in (0, 1):
            rows += [row, row, row]
            columns += [
                n_quads + 2 * triangle_index[second] + axis,
                n_quads + 2 * triangle_index[first] + axis,
                quad_index[quad],
            ]
            values += [1.0, -1.0, 2.0 * cosine * width[axis]]
            row += 1
    # The closure rows are redundant around every loop and agree only to
    # rounding error, so they get a tolerance far below any real mismatch.
    slack = 1e-12 * extent
    closure = LinearConstraint(
        csr_matrix((values, (rows, columns)), shape=(row, size)), -slack, slack,
    )

    lower = np.concatenate((np.zeros(n_quads), np.full(2 * n_triangles, -extent)))
    upper = np.concatenate((np.ones(n_quads), np.full(2 * n_triangles, extent)))
    for quad in held:
        if quad in quad_index:
            upper[quad_index[quad]] = 0.0
    for quad in required:
        lower[quad_index[quad]] = 1.0
    constraints = [closure]
    if whole_lines:
        pairs = [
            (quad_index[unit_quads[k]], quad_index[unit_quads[k + 3]])
            for unit_quads in (_unit_quads(model, unit) for unit in model.hex_units)
            for k in range(3) if k in unit_quads and k + 3 in unit_quads
        ]
        if pairs:
            count = len(pairs)
            constraints.append(LinearConstraint(
                csr_matrix(
                    (np.tile([1.0, -1.0], count),
                     (np.repeat(np.arange(count), 2), np.ravel(pairs))),
                    shape=(count, size),
                ), 0.0, 0.0,
            ))
    uniform = fold_with_snapped_panels(model, dihedral_deg)
    point_row = {pid: i for i, pid in enumerate(point_ids)}
    resting = 0
    fixed = [triangle_index[t] for t in fixed]
    for position in fixed:
        base = n_quads + 2 * position
        lower[base:base + 2] = upper[base:base + 2] = 0.0
    if cargo is None:
        if not fixed:
            # The fold is only defined up to a shift; pin one triangle.
            lower[n_quads:n_quads + 2] = upper[n_quads:n_quads + 2] = 0.0
    else:
        normal = np.array([-cargo.axis[1], cargo.axis[0]])
        directions = (cargo.axis, normal)
        reach = 3.0 * float(np.mean([np.linalg.norm(link[2]) for link in links.values()]))
        tiny = 1e-7 * extent
        touching = 1e-6 * extent
        rows, columns, values, bounds_low, bounds_high = [], [], [], [], []
        row = 0
        for triangle in triangles:
            base = n_quads + 2 * triangle_index[triangle]
            for pid in model._surface_vertices(triangle):
                point = uniform.reshape(-1, 3)[point_row[pid], :2]
                uv = cargo.coordinates(point[None, :])[0]
                for axis in (0, 1):
                    other = 1 - axis
                    if not (cargo.low[other] + tiny < uv[other] < cargo.high[other] - tiny):
                        continue
                    for limit, sign in ((cargo.high[axis], 1.0), (cargo.low[axis], -1.0)):
                        distance = sign * (uv[axis] - limit)
                        if not np.isfinite(limit) or distance < -tiny or distance > reach:
                            continue
                        if distance <= touching:
                            resting += 1
                            if cargo_contact == "fixed":
                                # The vertex, and so its triangle, stays put.
                                lower[base:base + 2] = upper[base:base + 2] = 0.0
                                continue
                        rows += [row, row]
                        columns += [base, base + 1]
                        values += list(sign * directions[axis])
                        if distance <= touching:
                            # A vertex the cargo rests on keeps resting on
                            # that side; it may slide along it.
                            bounds_low.append(0.0)
                            bounds_high.append(0.0)
                        else:
                            # Any other point may not move across the side.
                            bounds_low.append(-distance)
                            bounds_high.append(np.inf)
                        row += 1
        if row:
            constraints.append(LinearConstraint(
                csr_matrix((values, (rows, columns)), shape=(row, size)), bounds_low, bounds_high,
            ))

    objective = np.concatenate((-np.ones(n_quads), np.zeros(2 * n_triangles)))
    integrality = np.concatenate((np.ones(n_quads), np.zeros(2 * n_triangles)))
    kinds = hex_constraint_plan(model).triangle_kinds

    def panel_triangles(panel):
        return [panel] if panel in kinds else list(links[panel][:2])

    def path_quads(first, second):
        """Parallelograms whose states fix where two panels are relative to each other."""
        targets = set(panel_triangles(second))
        parent = {t: None for t in panel_triangles(first)}
        queue = deque(parent)
        end = next((t for t in parent if t in targets), None)
        while queue and end is None:
            triangle = queue.popleft()
            for neighbor, quad in adjacency[triangle]:
                if neighbor not in parent:
                    parent[neighbor] = (triangle, quad)
                    if neighbor in targets:
                        end = neighbor
                        break
                    queue.append(neighbor)
        found = {panel for panel in (first, second) if panel in links}
        while end is not None and parent[end] is not None:
            end, quad = parent[end]
            found.add(quad)
        return found

    work = _copy_points(model)
    cut_rows, cut_columns, cut_values, cut_low = [], [], [], []
    seen = set()
    snapped: frozenset[str] = frozenset()
    coordinates = uniform
    remaining: list[tuple[str, str]] = []
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        problem = list(constraints)
        if cut_low:
            problem.append(LinearConstraint(
                csr_matrix((cut_values, (cut_rows, cut_columns)), shape=(len(cut_low), size)),
                cut_low, np.inf,
            ))
        solution = milp(
            objective, integrality=integrality, bounds=Bounds(lower, upper),
            constraints=problem,
        )
        if solution.x is None:
            # Presolve can wrongly report these programs as infeasible.
            solution = milp(
                objective, integrality=integrality, bounds=Bounds(lower, upper),
                constraints=problem, options={"presolve": False},
            )
        if not solution.success or solution.x is None:
            raise RuntimeError(f"No rigid fold satisfies the constraints: {solution.message}")
        snapped = frozenset(q for q in quads if solution.x[quad_index[q]] > 0.5)
        coordinates = fold_with_snapped_panels(model, dihedral_deg, snapped)
        if cargo is not None or fixed:
            # Place the exact fold where the program put it, measured at a
            # triangle that is pinned if there is one.
            pinned = np.flatnonzero((lower[n_quads::2] == 0.0) & (upper[n_quads::2] == 0.0))
            anchor = int(pinned[0]) if pinned.size else 0
            corner = point_row[model._surface_vertices(triangles[anchor])[0]]
            moved = coordinates.reshape(-1, 3)[corner, :2] - uniform.reshape(-1, 3)[corner, :2]
            shift = solution.x[n_quads + 2 * anchor:n_quads + 2 * anchor + 2] - moved
            coordinates = coordinates.reshape(-1, 3).copy()
            coordinates[:, :2] += shift
            coordinates = coordinates.ravel()
        work.set_coordinate_vector(coordinates)
        remaining = clipping_pairs(work, minimum_gap)
        added = False
        for first, second in remaining:
            combination = path_quads(first, second)
            key = frozenset((quad, quad in snapped) for quad in combination)
            if key in seen:
                continue
            seen.add(key)
            added = True
            ones = 0
            for quad in combination:
                cut_rows.append(len(cut_low))
                cut_columns.append(quad_index[quad])
                cut_values.append(-1.0 if quad in snapped else 1.0)
                ones += quad in snapped
            cut_low.append(1.0 - ones)
        if not added:
            break

    overlap = 0.0
    if cargo is not None:
        overlap = float(cargo.depth(coordinates.reshape(-1, 3)[:, :2]).max())
    return CompactFold(
        snapped, coordinates, cargo, resting, rounds, len(cut_low), remaining, overlap,
    )


def _copy_points(model):
    """A copy of the model that shares topology but owns its coordinates."""
    import copy

    work = copy.copy(model)
    work.points = copy.deepcopy(model.points)
    return work
