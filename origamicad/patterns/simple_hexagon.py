from ..core.two_d_drawer import TwoDDrawer
import numpy as np

def hex_unit_chain(pattern, start_point=(0, 0), l=15, count=1):
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

    Output:
        Dictionary containing point IDs and surface IDs.
    """

    x0, y0 = start_point
    h = l * np.sqrt(3) / 2

    # ------------------------------------------------------------
    # Inner hexagonal void points
    # ------------------------------------------------------------

    mid_coords = [
        (x0, y0),
        (x0 + l, y0),
        (x0 + 1.5 * l, y0 - h),
        (x0 + l, y0 - 2 * h),
        (x0, y0 - 2 * h),
        (x0 - 0.5 * l, y0 - h),
    ]

    mid_ids = []
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

    side_coords = [
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

    side_ids = []
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
        if i % 2 == 0:
            crease_kind = "mountain"
        else:
            crease_kind = "valley"

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

    triangle_ids = []
    parallelogram_ids = []

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

    local_creases = []
    triangle_kinds = []

    for i in range(6):
        crease_kind = "mountain" if i % 2 == 0 else "valley"

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

def draw_hex_two_loops(pattern, start_point=(0, 0), l=15):
    """
    Draw one central hexagon unit chain and six surrounding unit chains.

    This version uses overlapping-panel placement.

    The function also stores local unit-chain metadata in:

        pattern.hex_units
    """

    x0, y0 = start_point
    h = l * np.sqrt(3) / 2

    units = []

    # First loop: central unit chain
    unit = hex_unit_chain(
        pattern,
        start_point=(x0, y0),
        l=l,
        count=0,
    )
    units.append(unit)

    # Second loop: six surrounding unit chains
    second_loop_start_points = [
        (x0 - 0.5 * l, y0 + 3 * h),
        (x0 + 2.0 * l, y0 + 2 * h),
        (x0 + 2.5 * l, y0 - 1 * h),
        (x0 + 0.5 * l, y0 - 3 * h),
        (x0 - 2.0 * l, y0 - 2 * h),
        (x0 - 2.5 * l, y0 + 1 * h),
    ]

    for count, sp in enumerate(second_loop_start_points, start=1):
        unit = hex_unit_chain(
            pattern,
            start_point=sp,
            l=l,
            count=count,
        )
        units.append(unit)

    pattern.hex_units = units
    return units

pattern = TwoDDrawer(unit="mm", point_tol=1e-6)

draw_hex_two_loops(
    pattern,
    start_point=(0, 0),
    l=15,
)

# pattern.print_summary()

# pattern.draw(
#     show_points=False,
#     show_point_ids=False,
#     show_line_ids=False,
#     figsize=(10, 10),
# )
