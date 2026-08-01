from __future__ import annotations

import numpy as np

from ...core.two_d_drawer import TwoDDrawer
from .metadata import (
    Coordinate,
    CreaseKind,
    HexUnit,
    LocalCrease,
    PointID,
    SurfaceID,
    TriangleKind,
)


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

    x0, y0 = start_point
    h: float = float(l * np.sqrt(3) / 2)
    even_kind: CreaseKind = "valley" if reverse else "mountain"
    odd_kind: CreaseKind = "mountain" if reverse else "valley"
    crease_kinds: list[CreaseKind] = [
        even_kind if i % 2 == 0 else odd_kind
        for i in range(6)
    ]

    # ------------------------------------------------------------
    # Inner hexagonal void points
    # ------------------------------------------------------------

    mid_coords: list[Coordinate] = [
        (x0, y0),
        (x0 + l, y0),
        (x0 + 1.5 * l, y0 - h),
        (x0 + l, y0 - 2 * h),
        (x0, y0 - 2 * h),
        (x0 - 0.5 * l, y0 - h),
    ]

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
        "local_creases": local_creases,
    }


def _quad_coordinate_signature(
    start_point: Coordinate,
    l: float,
    index: int,
) -> tuple[tuple[float, float], ...]:
    """Return a rounded coordinate signature for one unit-chain quad."""
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
    next_index = (index + 1) % 6
    vertices = (
        mid_coords[index],
        mid_coords[next_index],
        side_coords[(2 * index + 2) % 12],
        side_coords[2 * index + 1],
    )
    return tuple(
        sorted((round(float(x), 10), round(float(y), 10)) for x, y in vertices)
    )


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
) -> None:
    """Add hole punches to boundary parallelograms of a packaging grid."""
    if outer_diameter == 0 and cavity_diameter == 0:
        return

    quad_locations: dict[
        tuple[tuple[float, float], ...],
        list[tuple[tuple[int, int], int]],
    ] = {}
    for cell, start_point in cell_start_points.items():
        for quad_index in range(6):
            signature = _quad_coordinate_signature(start_point, l, quad_index)
            quad_locations.setdefault(signature, []).append((cell, quad_index))

    for cell, unit in units_by_cell.items():
        start_point = cell_start_points[cell]
        for quad_index, surface_id in enumerate(unit["parallelograms"]):
            signature = _quad_coordinate_signature(start_point, l, quad_index)
            peer_cells = [
                peer_cell
                for peer_cell, peer_quad_index in quad_locations[signature]
                if peer_cell != cell
            ]

            if any(peer_cell in units_by_cell for peer_cell in peer_cells):
                continue

            cavity_boundary = any(
                not bool(draw_cell[peer_cell])
                for peer_cell in peer_cells
            )
            diameter = cavity_diameter if cavity_boundary else outer_diameter
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
            )


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
) -> list[HexUnit]:
    """
    Draw a packed lattice of hexagon unit chains with a rectangular cavity.

    ``alpha`` and ``beta`` are the left/right and top/bottom border thicknesses
    in unit-chain cells. ``gamma`` and ``delta`` are the cavity width and height
    in unit-chain cells. ``alpha`` and ``beta`` must be positive because they
    provide the wall thickness. The four ``enable_*_open`` flags independently
    remove the corresponding cavity-side wall. Set ``enable_hole_punch_outer``
    or ``enable_hole_punch_cavity`` to a positive diameter to add circular cut
    lines to the outer or cavity-side parallelograms, respectively.
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
    )

    pattern.hex_units = units
    return units


# Clearer public name. Keep ``hexagon_packaging`` as a compatibility alias for
# existing callers and for readers familiar with the original API.
build_packaging = hexagon_packaging


def draw_hex_loops(
    pattern: TwoDDrawer,
    n: int = 2,
    start_point: Coordinate = (0.0, 0.0),
    l: float = 15.0,
    reverse: bool = False,
) -> list[HexUnit]:
    """
    Draw ``n`` concentric hexagonal loops of unit chains.

    Loop 1 contains the central unit chain.  Every later loop traces a regular
    hexagon around it using overlapping-panel placement.  Loop ``k`` contains
    ``6 * (k - 1)`` unique unit chains and has ``k`` unit-chain centers along
    each side, including the two corner centers.  The center-point hexagon for
    loop ``k >= 2`` has side length ``sqrt(7) * l * (k - 1)`` and is parallel
    to every other loop.

    Set ``reverse=True`` to exchange all mountain and valley crease labels.
    The line geometry and the local kinematic metadata are reversed together.

    The function also stores local unit-chain metadata in:

        pattern.hex_units
    """

    try:
        integer_n = int(n)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n must be a positive integer.") from exc
    if isinstance(n, bool) or integer_n != n or integer_n < 1:
        raise ValueError("n must be a positive integer.")
    if not np.isfinite(l) or l <= 0:
        raise ValueError("l must be a finite positive value.")

    n = integer_n
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

    pattern.hex_units = units
    return units
