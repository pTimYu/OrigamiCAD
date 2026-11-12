"""Gaps and intersections between the panels of a folded model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cadder import Cadder

_EPSILON = 1e-24


def _point_segment_distance(point, start, end) -> float:
    direction = end - start
    length_squared = float(np.dot(direction, direction))
    if length_squared <= _EPSILON:
        return float(np.linalg.norm(point - start))
    parameter = float(np.dot(point - start, direction) / length_squared)
    parameter = min(1.0, max(0.0, parameter))
    return float(np.linalg.norm(point - (start + parameter * direction)))


def _segment_segment_distance(a0, a1, b0, b1) -> float:
    """Return the shortest distance between two closed 3D segments."""
    da, db, offset = a1 - a0, b1 - b0, a0 - b0
    aa, bb, ab = float(np.dot(da, da)), float(np.dot(db, db)), float(np.dot(da, db))
    a_offset, b_offset = float(np.dot(da, offset)), float(np.dot(db, offset))
    if aa <= _EPSILON:
        return _point_segment_distance(a0, b0, b1)
    if bb <= _EPSILON:
        return _point_segment_distance(b0, a0, a1)

    denominator = aa * bb - ab * ab
    s = (ab * b_offset - bb * a_offset) / denominator if denominator > _EPSILON else 0.0
    s = min(1.0, max(0.0, s))
    t = (ab * s + b_offset) / bb
    if t < 0.0:
        t = 0.0
        s = min(1.0, max(0.0, -a_offset / aa))
    elif t > 1.0:
        t = 1.0
        s = min(1.0, max(0.0, (ab - a_offset) / aa))
    return float(np.linalg.norm((a0 + s * da) - (b0 + t * db)))


def _point_triangle_distance(point, a, b, c) -> float:
    """Return point-to-triangle distance using Voronoi-region tests."""
    ab, ac = b - a, c - a
    ap = point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return float(np.linalg.norm(point - (a + d1 / (d1 - d3) * ab)))

    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return float(np.linalg.norm(point - (a + d2 / (d2 - d6) * ac)))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        parameter = (d4 - d3) / (d4 - d3 + d5 - d6)
        return float(np.linalg.norm(point - (b + parameter * (c - b))))

    denominator = vc + vb + va
    if abs(denominator) <= _EPSILON:
        return min(
            _point_segment_distance(point, a, b),
            _point_segment_distance(point, b, c),
            _point_segment_distance(point, c, a),
        )
    inverse = 1.0 / denominator
    return float(np.linalg.norm(point - (a + ab * (vb * inverse) + ac * (vc * inverse))))


def _segment_triangle_intersections(start, end, a, b, c, tolerance=1e-10):
    """Return (inclusive contact, proper intersection) from one Möller-Trumbore test."""
    direction = end - start
    ab, ac = b - a, c - a
    cross_direction = np.cross(direction, ac)
    determinant = float(np.dot(ab, cross_direction))
    if abs(determinant) <= tolerance:
        return False, False

    inverse = 1.0 / determinant
    origin_offset = start - a
    u = float(np.dot(origin_offset, cross_direction) * inverse)
    cross_offset = np.cross(origin_offset, ab)
    v = float(np.dot(direction, cross_offset) * inverse)
    t = float(np.dot(ac, cross_offset) * inverse)

    proper = (
        tolerance < t < 1.0 - tolerance
        and tolerance < u < 1.0 - tolerance
        and tolerance < v < 1.0 - u - tolerance
    )
    contact = (
        -tolerance <= t <= 1.0 + tolerance
        and u >= -tolerance
        and v >= -tolerance
        and u + v <= 1.0 + tolerance
    )
    return contact, proper


def _triangle_distance(first, second) -> tuple[float, bool]:
    """Return the gap between two triangles and whether they properly intersect."""
    first_edges = [(first[i], first[(i + 1) % 3]) for i in range(3)]
    second_edges = [(second[i], second[(i + 1) % 3]) for i in range(3)]

    touching = False
    for edges, triangle in ((first_edges, second), (second_edges, first)):
        for edge_start, edge_end in edges:
            contact, proper = _segment_triangle_intersections(edge_start, edge_end, *triangle)
            if proper:
                return 0.0, True
            touching = touching or contact
    if touching:
        return 0.0, False

    distances = [_point_triangle_distance(point, *second) for point in first]
    distances.extend(_point_triangle_distance(point, *first) for point in second)
    distances.extend(
        _segment_segment_distance(a0, a1, b0, b1)
        for a0, a1 in first_edges
        for b0, b1 in second_edges
    )
    return min(distances), False


def _surface_triangles(model: Cadder, surface_id: str) -> list[np.ndarray]:
    """Fan-triangulate one panel."""
    vertices = np.array(
        [model.point_array(pid) for pid in model._surface_vertices(surface_id)],
        dtype=float,
    )
    return [vertices[[0, index, index + 1]] for index in range(1, len(vertices) - 1)]


def minimum_panel_clearance(
    model: Cadder,
) -> tuple[float, tuple[str, str] | None, bool]:
    """Return the closest non-adjacent panel gap and clipping diagnostic.

    Panels sharing a complete edge are direct hinge/boundary neighbors and
    are excluded because their edge contact is intentional.  Panels sharing
    only one vertex are still tested for transverse intersection; their
    permanent point contact is omitted only from the minimum-gap statistic.

    The result is ``(gap, (surface, surface) or None, clipping)``.
    """
    surface_ids = list(model.surfaces)
    vertices = [set(model._surface_vertices(sid)) for sid in surface_ids]
    triangles = [_surface_triangles(model, sid) for sid in surface_ids]
    lower = np.array([np.min(np.vstack(parts), axis=0) for parts in triangles])
    upper = np.array([np.max(np.vstack(parts), axis=0) for parts in triangles])

    minimum_gap = float("inf")
    closest_pair: tuple[str, str] | None = None
    clipping = False
    for first in range(len(surface_ids) - 1):
        # The distance between two bounding boxes never exceeds the distance
        # between the panels inside them. A pair whose boxes are farther
        # apart than the best gap so far can neither intersect nor improve
        # that gap, so the exact test is needed only for the remaining pairs.
        separation = np.maximum(
            0.0,
            np.maximum(lower[first] - upper[first + 1:], lower[first + 1:] - upper[first]),
        )
        box_gaps = np.sqrt(np.sum(separation * separation, axis=1))
        for offset in np.flatnonzero(box_gaps <= minimum_gap):
            if box_gaps[offset] > minimum_gap:
                continue
            second = first + 1 + int(offset)
            common_vertices = vertices[first] & vertices[second]
            if len(common_vertices) >= 2:
                continue
            pair_gap = float("inf")
            for first_triangle in triangles[first]:
                for second_triangle in triangles[second]:
                    gap, proper = _triangle_distance(first_triangle, second_triangle)
                    pair_gap = min(pair_gap, gap)
                    clipping = clipping or proper
            if not common_vertices and pair_gap < minimum_gap:
                minimum_gap = pair_gap
                closest_pair = (surface_ids[first], surface_ids[second])

    return minimum_gap, closest_pair, clipping
