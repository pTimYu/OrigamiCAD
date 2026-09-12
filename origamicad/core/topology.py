"""Coordinate-independent plans for rigid panels in 2D and 3D models."""

from itertools import combinations


def surface_topology(owner):
    """Snapshot public connectivity, including order and in-place edits."""
    return tuple(
        (sid, tuple(surface["vertices"] if isinstance(surface, dict) else surface.vertices))
        for sid, surface in owner.surfaces.items()
    )


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
