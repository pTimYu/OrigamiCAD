"""Keep panel cutouts attached without adding folding degrees of freedom."""

from contextlib import contextmanager

import numpy as np


@contextmanager
def panel_only_geometry(model):
    """Temporarily omit cutout vertices and restore them in their panel frames.

    Yield the full coordinate-vector indices retained for the solve, or None
    when no cutout geometry exists. The original point, line, and hole maps
    are restored even when the solve raises. Panel point objects stay shared
    so the solved configuration is preserved.
    """
    if not model.surface_holes:
        yield None
        return

    panel_points = {
        pid for sid in model.surfaces for pid in model._surface_vertices(sid)
    }
    cut_points = {
        pid for loops in model.surface_holes.values() for loop in loops for pid in loop
    } - panel_points
    attachments = []
    for surface_id, loops in model.surface_holes.items():
        vertices = model._surface_vertices(surface_id)
        anchors = (vertices[0], vertices[1], vertices[-1])
        origin, first, second = [model.point_array(pid) for pid in anchors]
        basis = np.column_stack((first - origin, second - origin))
        point_ids = list(dict.fromkeys(
            pid for loop in loops for pid in loop if pid in cut_points
        ))
        if not point_ids:
            continue
        coordinates = np.array([model.point_array(pid) for pid in point_ids])
        local, _, rank, _ = np.linalg.lstsq(basis, (coordinates - origin).T, rcond=None)
        if rank != 2:
            raise ValueError(f"Hole host surface '{surface_id}' is degenerate.")
        attachments.append((anchors, point_ids, local))

    original_points = model.points
    original_lines = model.lines
    original_holes = model.surface_holes
    coordinate_indices = np.array([
        3 * index + axis
        for index, pid in enumerate(original_points)
        if pid not in cut_points
        for axis in range(3)
    ], dtype=int)
    model.points = {
        pid: point for pid, point in original_points.items() if pid not in cut_points
    }
    model.lines = {
        lid: line for lid, line in original_lines.items()
        if not (set(model._line_info(lid)[:2]) & cut_points)
    }
    model.surface_holes = {}
    try:
        yield coordinate_indices
    finally:
        # Restore the maps before updating points so failures never leave a
        # model with missing cutout vertices or dangling face boundaries.
        model.points = original_points
        model.lines = original_lines
        model.surface_holes = original_holes
        for anchors, point_ids, local in attachments:
            origin, first, second = [model.point_array(pid) for pid in anchors]
            basis = np.column_stack((first - origin, second - origin))
            coordinates = origin + (basis @ local).T
            for pid, coordinate in zip(point_ids, coordinates):
                point = model.points[pid]
                point.x, point.y, point.z = map(float, coordinate)
