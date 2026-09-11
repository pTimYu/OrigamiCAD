from __future__ import annotations

import math
from collections import Counter
from functools import wraps

import numpy as np

from ...core.two_d_drawer import LineKind, TwoDDrawer
from .metadata import (
    Coordinate,
    CreaseKind,
    HexUnit,
    LocalCrease,
    PointID,
    SurfaceID,
    TriangleKind,
    indexed_creases,
    iter_local_creases,
    normalize_hex_creases,
    register_crease,
)


def _indexed_layout(function):
    @wraps(function)
    def draw(pattern: TwoDDrawer, *args, **kwargs):
        # Subclasses may override builders to mutate existing topology.
        with indexed_creases(pattern):
            if type(pattern) is not TwoDDrawer:
                return function(pattern, *args, **kwargs)
            with pattern._index_geometry():
                return function(pattern, *args, **kwargs)
    return draw


def _unit_coordinates(
    start_point: Coordinate,
    l: float,
) -> tuple[list[Coordinate], list[Coordinate]]:
    """Return the shared, ordered inner and outer coordinates of one cell."""
    x0, y0 = start_point
    h = float(l * np.sqrt(3) / 2)
    mid_coords: list[Coordinate] = [
        (x0, y0),
        (x0 + l, y0),
        (x0 + 1.5 * l, y0 - h),
        (x0 + l, y0 - 2 * h),
        (x0, y0 - 2 * h),
        (x0 - 0.5 * l, y0 - h),
    ]
    side_coords: list[Coordinate] = [
        (x0 - l, y0),
        (x0 - 0.5 * l, y0 + h),
        (x0 + 0.5 * l, y0 + h),
        (x0 + 1.5 * l, y0 + h),
        (x0 + 2 * l, y0),
        (x0 + 2.5 * l, y0 - h),
        (x0 + 2 * l, y0 - 2 * h),
        (x0 + 1.5 * l, y0 - 3 * h),
        (x0 + 0.5 * l, y0 - 3 * h),
        (x0 - 0.5 * l, y0 - 3 * h),
        (x0 - l, y0 - 2 * h),
        (x0 - 1.5 * l, y0 - h),
    ]
    return mid_coords, side_coords


@_indexed_layout
def hex_unit_chain(
    pattern: TwoDDrawer,
    start_point: Coordinate = (0.0, 0.0),
    l: float = 15.0,
    count: int = 1,
    reverse: bool = False,
) -> HexUnit:
    """
    Automatically draw one hexagon unit chain.

    Input:
        pattern:
            TwoDDrawer object.
        start_point:
            Top-left point of the inner hexagonal void.
        l:
            Side length of the inner hexagonal void.
        count:
            Unit chain index, used for point/surface names.
        reverse:
            If True, exchange every mountain crease with a valley crease.

    Output:
        Dictionary containing point IDs and surface IDs.
    """

    mid_coords, side_coords = _unit_coordinates(start_point, l)
    even_kind: CreaseKind = "valley" if reverse else "mountain"
    odd_kind: CreaseKind = "mountain" if reverse else "valley"
    crease_kinds: list[CreaseKind] = [
        even_kind if i % 2 == 0 else odd_kind
        for i in range(6)
    ]

    # ------------------------------------------------------------
    # Inner hexagonal void points
    # ------------------------------------------------------------

    mid_ids: list[PointID] = []
    for i, (x, y) in enumerate(mid_coords):
        pid = pattern.add_point(x, y, f"p{i}_{count}_mid")
        mid_ids.append(pid)

    # Inner hexagon boundary
    for i in range(6):
        pattern.add_line(
            mid_ids[i],
            mid_ids[(i + 1) % 6],
            kind="side",
        )

    # ------------------------------------------------------------
    # Outer points
    # ------------------------------------------------------------

    side_ids: list[PointID] = []
    for i, (x, y) in enumerate(side_coords):
        pid = pattern.add_point(x, y, f"p{i}_{count}_side")
        side_ids.append(pid)

    # ------------------------------------------------------------
    # Outer side lines + crease lines
    # ------------------------------------------------------------

    for i in range(6):
        # Small outer boundary segment around each corner
        pattern.add_line(
            side_ids[2 * i],
            side_ids[2 * i + 1],
            kind="side",
        )

        # Creases from outer points to inner void point
        crease_kind = crease_kinds[i]

        pattern.add_line(
            side_ids[2 * i],
            mid_ids[i],
            kind=crease_kind,
        )

        pattern.add_line(
            side_ids[2 * i + 1],
            mid_ids[i],
            kind=crease_kind,
        )

    # Longer outer boundary segments
    for i in range(5):
        pattern.add_line(
            side_ids[2 * i + 1],
            side_ids[2 * (i + 1)],
            kind="side",
        )

    pattern.add_line(
        side_ids[11],
        side_ids[0],
        kind="side",
    )

    # ------------------------------------------------------------
    # Add rigid panels / planes
    # ------------------------------------------------------------
    #
    # Each unit chain has:
    #   6 triangular panels
    #   6 parallelogram panels
    #
    # The inner hexagon is a void, so no surface is added there.
    #
    # Do NOT use auto_boundary=True here because the side/crease
    # lines have already been explicitly assigned above.
    # ------------------------------------------------------------

    triangle_ids: list[SurfaceID] = []
    parallelogram_ids: list[SurfaceID] = []

    for i in range(6):
        # Triangle panel around each inner hexagon vertex
        tri_id = pattern.add_triangle(
            mid_ids[i],
            side_ids[2 * i],
            side_ids[2 * i + 1],
            surface_id=f"tri_{count}_{i}",
            auto_boundary=False,
        )
        triangle_ids.append(tri_id)

        # Parallelogram panel along each inner hexagon edge
        #
        # Inner edge:
        #   mid_ids[i] ---- mid_ids[i+1]
        #
        # Outer corresponding edge:
        #   side_ids[2*i+1] ---- side_ids[2*i+2]
        #
        # Use modulo indexing for the closing panel.
        j = (i + 1) % 6

        quad_id = pattern.add_parallelogram(
            mid_ids[i],
            mid_ids[j],
            side_ids[(2 * i + 2) % 12],
            side_ids[2 * i + 1],
            surface_id=f"quad_{count}_{i}",
            auto_boundary=False,
        )
        parallelogram_ids.append(quad_id)

    # ------------------------------------------------------------
    # Local unit-chain topology for kinematic solver
    # ------------------------------------------------------------
    #
    # Each triangle tri_i has two crease edges:
    #
    #   edge A: mid_i -- side_{2i}
    #       adjacent to quad_{i-1}
    #
    #   edge B: mid_i -- side_{2i+1}
    #       adjacent to quad_i
    #
    # This explicit local metadata avoids ambiguous global adjacency
    # when several unit chains overlap.
    # ------------------------------------------------------------

    local_creases: list[LocalCrease] = []
    triangle_kinds: list[TriangleKind] = []

    for i in range(6):
        crease_kind = crease_kinds[i]

        triangle_kinds.append(
            {
                "surface": triangle_ids[i],
                "kind": crease_kind,
                "local_index": i,
                "unit": count,
            }
        )

        # Previous quad side
        local_creases.append(
            {
                "unit": count,
                "local_index": i,
                "edge": [mid_ids[i], side_ids[2 * i]],
                "triangle": triangle_ids[i],
                "quad": parallelogram_ids[(i - 1) % 6],
                "kind": crease_kind,
                "side": "previous_quad",
            }
        )

        # Current quad side
        local_creases.append(
            {
                "unit": count,
                "local_index": i,
                "edge": [mid_ids[i], side_ids[2 * i + 1]],
                "triangle": triangle_ids[i],
                "quad": parallelogram_ids[i],
                "kind": crease_kind,
                "side": "current_quad",
            }
        )

    return {
        "count": count,
        "mid": mid_ids,
        "side": side_ids,
        "triangles": triangle_ids,
        "parallelograms": parallelogram_ids,
        "surfaces": triangle_ids + parallelogram_ids,
        "triangle_kinds": triangle_kinds,
        "local_creases": [register_crease(pattern, crease) for crease in local_creases],
    }


def _quad_coordinate_signatures(
    start_point: Coordinate,
    l: float,
) -> list[tuple[tuple[float, float], ...]]:
    """Return the six rounded quad signatures from one coordinate table."""
    mid_coords, side_coords = _unit_coordinates(start_point, l)
    return [
        tuple(sorted((round(float(x), 10), round(float(y), 10)) for x, y in (
            mid_coords[index],
            mid_coords[(index + 1) % 6],
            side_coords[(2 * index + 2) % 12],
            side_coords[2 * index + 1],
        )))
        for index in range(6)
    ]


def _quad_coordinate_signature(
    start_point: Coordinate,
    l: float,
    index: int,
) -> tuple[tuple[float, float], ...]:
    """Return a rounded coordinate signature for one unit-chain quad."""
    return _quad_coordinate_signatures(start_point, l)[index]


def _validate_hole_punch_diameter(
    name: str,
    diameter: float,
    l: float,
) -> float:
    try:
        value = float(diameter)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative diameter.") from exc

    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative diameter.")

    maximum = float(l * np.sqrt(3) / 2)
    if value > maximum:
        raise ValueError(
            f"{name} must not exceed the side parallelogram's inscribed "
            f"circle diameter ({maximum:g})."
        )
    return value


def _add_packaging_hole_punches(
    pattern: TwoDDrawer,
    units_by_cell: dict[tuple[int, int], HexUnit],
    cell_start_points: dict[tuple[int, int], Coordinate],
    draw_cell: np.ndarray,
    l: float,
    outer_diameter: float,
    cavity_diameter: float,
    cavity_holes_enabled: bool,
) -> None:
    """Add hole punches to boundary parallelograms of a packaging grid."""
    if outer_diameter == 0 and cavity_diameter == 0:
        return

    cell_signatures = {
        cell: _quad_coordinate_signatures(start_point, l)
        for cell, start_point in cell_start_points.items()
    }
    quad_locations: dict[
        tuple[tuple[float, float], ...],
        list[tuple[int, int]],
    ] = {}
    for cell, signatures in cell_signatures.items():
        for signature in signatures:
            quad_locations.setdefault(signature, []).append(cell)

    for cell, unit in units_by_cell.items():
        for quad_index, surface_id in enumerate(unit["parallelograms"]):
            signature = cell_signatures[cell][quad_index]
            peer_cells = [
                peer_cell
                for peer_cell in quad_locations[signature]
                if peer_cell != cell
            ]

            if any(peer_cell in units_by_cell for peer_cell in peer_cells):
                continue

            cavity_boundary = any(
                not bool(draw_cell[peer_cell])
                for peer_cell in peer_cells
            )
            if cavity_boundary:
                if not cavity_holes_enabled:
                    continue
                diameter = cavity_diameter
            else:
                diameter = outer_diameter
            if diameter == 0:
                continue

            surface = pattern.surfaces[surface_id]
            center_x = sum(pattern.points[pid].x for pid in surface.vertices) / 4
            center_y = sum(pattern.points[pid].y for pid in surface.vertices) / 4
            row, col = cell
            boundary_name = "cavity" if cavity_boundary else "outer"
            pattern.add_hole_punch(
                center=(center_x, center_y),
                diameter=diameter,
                hole_id=f"hole_{boundary_name}_{row}_{col}_{quad_index}",
                surface_id=surface_id,
            )


def _rotate_packaging_to_horizon(
    pattern: TwoDDrawer,
    cavity_center: Coordinate,
) -> None:
    """Rotate the generated packaging clockwise to level its cavity."""
    angle = math.radians(-10.8934)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    center_x, center_y = cavity_center

    for point in pattern.points.values():
        relative_x = point.x - center_x
        relative_y = point.y - center_y
        point.x = center_x + cosine * relative_x - sine * relative_y
        point.y = center_y + sine * relative_x + cosine * relative_y

    for hole in pattern.hole_punches:
        hole_x, hole_y = hole["center"]
        relative_x = hole_x - center_x
        relative_y = hole_y - center_y
        hole["center"] = [
            center_x + cosine * relative_x - sine * relative_y,
            center_y + sine * relative_x + cosine * relative_y,
        ]


@_indexed_layout
def hexagon_packaging(
    pattern: TwoDDrawer,
    l: float = 15.0,
    alpha: int = 2,
    beta: int = 2,
    gamma: int = 3,
    delta: int = 4,
    start_point: Coordinate = (0.0, 0.0),
    enable_left_open: bool = False,
    enable_right_open: bool = False,
    enable_top_open: bool = False,
    enable_bot_open: bool = False,
    enable_hole_punch_outer: float = 0.0,
    enable_hole_punch_cavity: float = 0.0,
    rotate_cavity_to_horizon: bool = False,
    fill_cavity: bool = False,
    reverse: bool = False,
) -> list[HexUnit]:
    """
    Draw a packed lattice of hexagon unit chains with a rectangular cavity.

    ``alpha`` and ``beta`` are the left/right and top/bottom border thicknesses
    in unit-chain cells. ``gamma`` and ``delta`` are the cavity width and height
    in unit-chain cells. ``alpha`` and ``beta`` must be positive because they
    provide the wall thickness. The four ``enable_*_open`` flags independently
    remove the corresponding cavity-side wall. Set ``enable_hole_punch_outer``
    or ``enable_hole_punch_cavity`` to a positive diameter to add circular cut
    lines to the outer or cavity-side parallelograms, respectively. Cavity-side
    hole punches are added only when at least one cavity-side wall is opened.
    Set ``rotate_cavity_to_horizon=True`` to rotate the complete structure
    clockwise by 10.8934 degrees about the cavity center. Set
    ``fill_cavity=True`` to populate the cavity region instead of leaving it
    empty. Set ``reverse=True`` to exchange every mountain and valley crease
    label, including the local kinematic metadata.
    """

    if l <= 0:
        raise ValueError("l must be positive.")

    outer_hole_diameter = _validate_hole_punch_diameter(
        "enable_hole_punch_outer",
        enable_hole_punch_outer,
        l,
    )
    cavity_hole_diameter = _validate_hole_punch_diameter(
        "enable_hole_punch_cavity",
        enable_hole_punch_cavity,
        l,
    )

    dimensions = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
    }
    for name, value in dimensions.items():
        if int(value) != value:
            raise ValueError(f"{name} must be an integer.")
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    for name, enabled in {
        "enable_left_open": enable_left_open,
        "enable_right_open": enable_right_open,
        "enable_top_open": enable_top_open,
        "enable_bot_open": enable_bot_open,
        "rotate_cavity_to_horizon": rotate_cavity_to_horizon,
        "fill_cavity": fill_cavity,
        "reverse": reverse,
    }.items():
        if not isinstance(enabled, bool):
            raise ValueError(f"{name} must be a boolean.")

    alpha = int(alpha)
    beta = int(beta)
    gamma = int(gamma)
    delta = int(delta)

    if alpha == 0 or beta == 0:
        raise ValueError("alpha and beta must be positive wall thicknesses.")

    num_cols = 2 * alpha + gamma
    num_rows = 2 * beta + delta

    x0, y0 = start_point
    h = float(l * np.sqrt(3) / 2)

    draw_cell = np.ones((num_rows, num_cols), dtype=bool)
    cavity_col_start = 0 if enable_left_open else alpha
    cavity_col_end = num_cols if enable_right_open else alpha + gamma
    cavity_row_start = 0 if enable_top_open else beta
    cavity_row_end = num_rows if enable_bot_open else beta + delta
    if not fill_cavity:
        draw_cell[
            cavity_row_start:cavity_row_end,
            cavity_col_start:cavity_col_end,
        ] = False

    units: list[HexUnit] = []
    units_by_cell: dict[tuple[int, int], HexUnit] = {}
    cell_start_points: dict[tuple[int, int], Coordinate] = {}
    for row in range(num_rows):
        unit_x = float(x0) + 0.5 * l * row
        unit_y = float(y0) - 3.0 * h * row

        for col in range(num_cols):
            cell_start_points[(row, col)] = (unit_x, unit_y)
            if draw_cell[row, col]:
                unit = hex_unit_chain(
                    pattern,
                    start_point=(unit_x, unit_y),
                    l=l,
                    count=len(units),
                    reverse=reverse,
                )
                units.append(unit)
                units_by_cell[(row, col)] = unit

            if col % 2 == 0:
                unit_x += 2.5 * l
                unit_y -= h
            else:
                unit_x += 2.0 * l
                unit_y += 2.0 * h

    _add_packaging_hole_punches(
        pattern,
        units_by_cell,
        cell_start_points,
        draw_cell,
        l,
        outer_hole_diameter,
        cavity_hole_diameter,
        cavity_holes_enabled=any(
            (
                enable_left_open,
                enable_right_open,
                enable_top_open,
                enable_bot_open,
            )
        ),
    )

    if rotate_cavity_to_horizon:
        cavity_cells = [
            cell_start_points[(row, col)]
            for row in range(cavity_row_start, cavity_row_end)
            for col in range(cavity_col_start, cavity_col_end)
        ]
        if cavity_cells:
            cavity_center = (
                sum(x for x, _ in cavity_cells) / len(cavity_cells) + 0.5 * l,
                sum(y for _, y in cavity_cells) / len(cavity_cells) - h,
            )
        else:
            cavity_center = (float(x0), float(y0))
        _rotate_packaging_to_horizon(pattern, cavity_center)

    pattern.hex_units = units
    normalize_hex_creases(pattern)
    return pattern.hex_units


# Clearer public name. Keep ``hexagon_packaging`` as a compatibility alias for
# existing callers and for readers familiar with the original API.
build_packaging = hexagon_packaging


def _remove_loop_panels(
    pattern: TwoDDrawer,
    surface_ids: set[SurfaceID],
    original_points: set[PointID],
    original_lines: dict[str, LineKind],
) -> None:
    """Cut out panels, keeping exposed edges and discarding unused geometry."""
    removed_edges: set[tuple[str, str]] = set()
    for surface_id in surface_ids:
        surface = pattern.surfaces.pop(surface_id)
        vertices = surface.vertices
        removed_edges.update(
            tuple(sorted((start, end)))
            for start, end in zip(vertices, vertices[1:] + vertices[:1])
        )
        if pattern._geometry_indexes is not None:
            pattern._geometry_indexes[1].pop(tuple(sorted(vertices)), None)

    edge_counts = Counter(
        tuple(sorted((start, end)))
        for surface in pattern.surfaces.values()
        for start, end in zip(
            surface.vertices, surface.vertices[1:] + surface.vertices[:1]
        )
    )
    for line_id, line in list(pattern.lines.items()):
        edge = tuple(sorted((line.start, line.end)))
        if edge not in removed_edges:
            continue
        if line_id in original_lines:
            line.kind = original_lines[line_id]
            continue
        if edge_counts[edge] == 0:
            del pattern.lines[line_id]
            if pattern._geometry_indexes is not None:
                pattern._geometry_indexes[0].pop(edge, None)
        elif edge_counts[edge] == 1:
            # A crease with material on only one side is now a cut boundary.
            line.kind = "side"

    used_points = {
        point_id
        for surface in pattern.surfaces.values()
        for point_id in surface.vertices
    }
    used_points.update(
        point_id
        for line in pattern.lines.values()
        for point_id in (line.start, line.end)
    )
    for point_id in set(pattern.points) - used_points - original_points:
        del pattern.points[point_id]


@_indexed_layout
def draw_hex_loops(
    pattern: TwoDDrawer,
    n: int = 2,
    start_point: Coordinate = (0.0, 0.0),
    l: float = 15.0,
    reverse: bool = False,
    cavity_loops: int = 0,
    enable_hole_punch_outer: float = 0.0,
) -> list[HexUnit]:
    """
    Draw ``n`` concentric hexagonal loops of unit chains.

    Loop 1 contains the central unit chain.  Every later loop traces a regular
    hexagon around it using overlapping-panel placement.  Loop ``k`` contains
    ``6 * (k - 1)`` unique unit chains and has ``k`` unit-chain centers along
    each side, including the two corner centers.  The center-point hexagon for
    loop ``k >= 2`` has side length ``sqrt(7) * l * (k - 1)`` and is parallel
    to every other loop.

    ``cavity_loops`` removes the panels of the innermost loops, including
    panels shared with the next loop and strips attached to removed triangles,
    to form one connected cavity. It must be an integer with
    ``0 <= cavity_loops < n``; zero keeps the full pattern.
    For ``n=3``, values 0, 1, and 2 keep all loops, loops 2–3, and only loop 3,
    respectively. The outer size and placement stay unchanged. Exposed crease
    edges become cut boundaries, and unused lines and points are removed.

    Set ``enable_hole_punch_outer`` to a positive diameter to add a centered
    circular cut to each outer-boundary parallelogram. Zero disables holes.
    Interior and cavity-only panels are never punched. The diameter uses the
    pattern's length units and must not exceed ``sqrt(3) * l / 2``.

    Set ``reverse=True`` to exchange all mountain and valley crease labels.
    The line geometry and the local kinematic metadata are reversed together.

    The returned units retain their original counts. Units bordering a cavity
    contain only surviving panels and creases. The function also stores this
    local unit-chain metadata in:

        pattern.hex_units
    """

    try:
        integer_n = int(n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n must be a positive integer.") from exc
    if isinstance(n, bool) or integer_n != n or integer_n < 1:
        raise ValueError("n must be a positive integer.")
    cavity_error = "cavity_loops must be an integer with 0 <= cavity_loops < n."
    try:
        integer_cavity = int(cavity_loops)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(cavity_error) from exc
    if (
        isinstance(cavity_loops, bool)
        or integer_cavity != cavity_loops
        or not 0 <= integer_cavity < integer_n
    ):
        raise ValueError(cavity_error)
    if not np.isfinite(l) or l <= 0:
        raise ValueError("l must be a finite positive value.")
    outer_hole_diameter = _validate_hole_punch_diameter(
        "enable_hole_punch_outer", enable_hole_punch_outer, l,
    )

    n = integer_n
    cavity_loops = integer_cavity
    if cavity_loops:
        original_points = set(pattern.points)
        original_lines = {lid: line.kind for lid, line in pattern.lines.items()}
        original_surfaces = set(pattern.surfaces)
    x0, y0 = start_point
    h: float = float(l * np.sqrt(3) / 2)

    # First loop: central unit chain
    units: list[HexUnit] = [
        hex_unit_chain(
            pattern,
            start_point=(x0, y0),
            l=l,
            count=0,
            reverse=reverse,
        )
    ]

    # These are the six loop-2 corner offsets.  Consecutive offsets are also
    # the six directions of the lattice, so scaling and subdividing their
    # edges produces every larger, parallel center-point hexagon.
    corner_offsets = np.array(
        [
            (-0.5 * l, 3.0 * h),
            (2.0 * l, 2.0 * h),
            (2.5 * l, -1.0 * h),
            (0.5 * l, -3.0 * h),
            (-2.0 * l, -2.0 * h),
            (-2.5 * l, 1.0 * h),
        ],
        dtype=float,
    )

    for loop_number in range(2, n + 1):
        edge_segments = loop_number - 1
        loop_corners = edge_segments * corner_offsets

        for side_index in range(6):
            side_start = loop_corners[side_index]
            side_end = loop_corners[(side_index + 1) % 6]
            side_step = (side_end - side_start) / edge_segments

            # Exclude the end corner: it is the next side's start corner.
            for step_index in range(edge_segments):
                offset = side_start + step_index * side_step
                units.append(
                    hex_unit_chain(
                        pattern,
                        start_point=(
                            float(x0 + offset[0]),
                            float(y0 + offset[1]),
                        ),
                        l=l,
                        count=len(units),
                        reverse=reverse,
                    )
                )

    if outer_hole_diameter:
        # Classify the full lattice before cutting the cavity: interior quads
        # are shared by two units, while outer quads belong to just one.
        quad_uses = Counter(
            surface_id for unit in units for surface_id in unit["parallelograms"]
        )
        outer_quad_ids = [sid for sid, uses in quad_uses.items() if uses == 1]

    if cavity_loops:
        num_cavity_units = 1 + 3 * cavity_loops * (cavity_loops - 1)
        cavity_surfaces = {
            surface_id
            for unit in units[:num_cavity_units]
            for surface_id in unit["surfaces"]
        }
        # A quad attached to a removed triangle would leave a strip projecting
        # into the cavity. Remove those strips to expose the next loop's wall.
        cavity_surfaces.update(
            crease["quad"]
            for unit in units
            for crease in iter_local_creases(pattern, unit)
            if crease["triangle"] in cavity_surfaces
        )
        _remove_loop_panels(
            pattern,
            cavity_surfaces - original_surfaces,
            original_points,
            original_lines,
        )
        units = units[num_cavity_units:]
        for unit in units:
            for key in ("mid", "side"):
                unit[key] = [pid for pid in unit[key] if pid in pattern.points]
            for key in ("triangles", "parallelograms", "surfaces"):
                unit[key] = [sid for sid in unit[key] if sid in pattern.surfaces]
            unit["triangle_kinds"] = [
                triangle for triangle in unit["triangle_kinds"]
                if triangle["surface"] in pattern.surfaces
            ]
            unit["local_creases"] = [
                crease for crease in unit["local_creases"]
                if pattern.hex_creases[crease["crease"]]["triangle"] in pattern.surfaces
                and pattern.hex_creases[crease["crease"]]["quad"] in pattern.surfaces
            ]

    if outer_hole_diameter:
        for surface_id in outer_quad_ids:
            if surface_id not in pattern.surfaces:
                continue
            vertices = pattern.surfaces[surface_id].vertices
            pattern.add_hole_punch(
                center=(
                    sum(pattern.points[pid].x for pid in vertices) / 4,
                    sum(pattern.points[pid].y for pid in vertices) / 4,
                ),
                diameter=outer_hole_diameter,
                hole_id=f"hole_outer_{surface_id}",
                surface_id=surface_id,
            )

    pattern.hex_units = units
    normalize_hex_creases(pattern)
    return pattern.hex_units
