"""Coordinate-independent plans for rigid panels in 2D and 3D models."""

import math
from itertools import combinations


def surface_topology(owner):
    """Snapshot public connectivity, including order and in-place edits."""
    return tuple(
        (sid, tuple(surface["vertices"] if isinstance(surface, dict) else surface.vertices))
        for sid, surface in owner.surfaces.items()
    )


def loop_area(loop, xy) -> float:
    """Signed xy area of a closed loop of point IDs (positive if counter-clockwise)."""
    area = 0.0
    for start, end in zip(loop, loop[1:] + loop[:1]):
        (x0, y0), (x1, y1) = xy[start], xy[end]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def boundary_loops(owner, xy) -> list[list[str]]:
    """Return the closed loops formed by edges that border exactly one panel.

    ``xy`` maps point IDs to flat-pattern coordinates. Each loop is a list of
    point IDs walked with the material on its left, so an outer boundary has
    positive :func:`loop_area` and a hole (a void or a cavity) negative area.
    Where several boundary edges meet at one point, the walk stays on the
    same piece of material.
    """
    directed = {}
    for _, vertices in surface_topology(owner):
        vertices = list(vertices)
        if loop_area(vertices, xy) < 0:
            vertices.reverse()
        for start, end in zip(vertices, vertices[1:] + vertices[:1]):
            directed[start, end] = True
    outgoing = {}
    for start, end in directed:
        if (end, start) not in directed:
            outgoing.setdefault(start, []).append(end)

    def angle(origin, target):
        (x0, y0), (x1, y1) = xy[origin], xy[target]
        return math.atan2(y1 - y0, x1 - x0)

    loops = []
    unused = {(start, end) for start, ends in outgoing.items() for end in ends}
    while unused:
        first = min(unused)
        loop, (previous, current) = [first[0]], first
        unused.discard(first)
        while True:
            candidates = [end for end in outgoing.get(current, ()) if (current, end) in unused]
            if current == first[0] and not candidates:
                break
            loop.append(current)
            if not candidates:
                raise ValueError(f"Open panel boundary at point '{current}'.")
            back = angle(current, previous)
            # Material lies clockwise from the edge just walked, so the next
            # boundary edge is the first one met turning clockwise.
            following = min(
                candidates,
                key=lambda end: (back - angle(current, end)) % (2.0 * math.pi) or 2.0 * math.pi,
            )
            unused.discard((current, following))
            previous, current = current, following
            if current == first[0] and not any(
                (current, end) in unused for end in outgoing.get(current, ())
            ):
                break
        loops.append(loop)
    return loops


def panel_rigidity_plan(owner):
    """Return unique (first constraint ID, p1, p2) records in insertion order.

    Generate combinations within each panel only. Canonical endpoint pairs
    merge bars shared by different panels. Names and endpoint order retain
    the former first occurrence, preserving constraint/residual ordering.
    Coordinates and lengths are excluded so the plan survives folding;
    lengths are evaluated when the numerical constraints are instantiated.
    """
    topology = surface_topology(owner)
    cached = getattr(owner, "_panel_rigidity_cache", None)
    if cached is not None and cached[0] == topology:
        return cached[1]
    pairs = {}
    for sid, vertices in topology:
        for p1, p2 in combinations(vertices, 2):
            key = tuple(sorted((p1, p2)))
            pairs.setdefault(key, (f"panel_{sid}_{p1}_{p2}", p1, p2))
    plan = tuple(pairs.values())
    owner._panel_rigidity_cache = topology, plan
    return plan
