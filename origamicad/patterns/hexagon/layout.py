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


def hexagon_packaging(
    pattern: TwoDDrawer,
    l: float = 15.0,
    alpha: int = 2,
    beta: int = 2,
    gamma: int = 3,
    delta: int = 4,
    start_point: Coordinate = (0.0, 0.0),
) -> list[HexUnit]:
    """
    Draw a packed lattice of hexagon unit chains with a rectangular cavity.

    ``alpha`` and ``beta`` are the left/right and top/bottom border thicknesses
    in unit-chain cells. ``delta`` and ``gamma`` are the cavity width and height
    in unit-chain cells.
    """

    if l <= 0:
        raise ValueError("l must be positive.")

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

    alpha = int(alpha)
    beta = int(beta)
    gamma = int(gamma)
    delta = int(delta)

    num_cols = 2 * alpha + delta
    num_rows = 2 * beta + gamma
    if num_cols <= 0 or num_rows <= 0:
        raise ValueError("The generated packaging grid must contain at least one cell.")

    x0, y0 = start_point
    h = float(l * np.sqrt(3) / 2)

    draw_cell = np.ones((num_rows, num_cols), dtype=bool)
    draw_cell[beta: beta + gamma, alpha: alpha + delta] = False

    units: list[HexUnit] = []
    for row in range(num_rows):
        unit_x = float(x0) + 0.5 * l * row
        unit_y = float(y0) - 3.0 * h * row

        for col in range(num_cols):
            if draw_cell[row, col]:
                units.append(
                    hex_unit_chain(
                        pattern,
                        start_point=(unit_x, unit_y),
                        l=l,
                        count=len(units),
                    )
                )

            if col % 2 == 0:
                unit_x += 2.5 * l
                unit_y -= h
            else:
                unit_x += 2.0 * l
                unit_y += 2.0 * h

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
