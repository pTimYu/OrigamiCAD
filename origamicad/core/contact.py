"""Barrier potential that keeps panels from passing through each other.

Panels are triangulated, so every contact is between a vertex and a triangle
or between two panel edges that have no point in common. Each such pair
closer than the activation distance ``d_hat`` stores the energy

    stiffness * B(d**2 / d_hat**2),    B(u) = -(u - 1)**2 * ln(u),

which is zero and smooth at ``d = d_hat`` and grows without bound as the
distance closes. This is the barrier of incremental potential contact; an
origami-specific model of the same kind is described by Zhu and Filipov.

The barrier cannot see a pair that has already crossed, so the solver must
never step across. :meth:`ContactBarrier.step_limit` returns the largest
fraction of a proposed step that provably keeps every pair apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .cadder import Cadder


def _dot(a, b):
    return np.sum(a * b, axis=-1)


# Squared distance and its gradient for each closest-feature type. ``points``
# has shape (n, k, 3). All operations are complex-differentiable, so a complex
# step through the gradient gives exact second derivatives.

def _point_point(points):
    r = points[:, 0] - points[:, 1]
    return _dot(r, r), np.concatenate((2 * r, -2 * r), axis=1)


def _point_edge(points):
    p, a, b = points[:, 0], points[:, 1], points[:, 2]
    e = b - a
    t = (_dot(p - a, e) / _dot(e, e))[:, None]
    r = p - (a + t * e)
    return _dot(r, r), np.concatenate((2 * r, -2 * (1 - t) * r, -2 * t * r), axis=1)


def _point_plane(points):
    p, a, b, c = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    v0, v1 = b - a, c - a
    n = np.cross(v0, v1)
    r = (_dot(p - a, n) / _dot(n, n))[:, None] * n
    v2 = p - r - a
    d00, d01, d11 = _dot(v0, v0), _dot(v0, v1), _dot(v1, v1)
    d20, d21 = _dot(v2, v0), _dot(v2, v1)
    denominator = d00 * d11 - d01 * d01
    wb = ((d11 * d20 - d01 * d21) / denominator)[:, None]
    wc = ((d00 * d21 - d01 * d20) / denominator)[:, None]
    return _dot(r, r), np.concatenate(
        (2 * r, -2 * (1 - wb - wc) * r, -2 * wb * r, -2 * wc * r), axis=1,
    )


def _line_line(points):
    p1, q1, p2, q2 = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, b = _dot(d1, d1), _dot(d2, d2), _dot(d1, d2)
    c, f = _dot(d1, r), _dot(d2, r)
    denominator = a * e - b * b
    s = ((b * f - c * e) / denominator)[:, None]
    t = ((a * f - b * c) / denominator)[:, None]
    diff = r + s * d1 - t * d2
    return _dot(diff, diff), np.concatenate(
        (2 * (1 - s) * diff, 2 * s * diff, -2 * (1 - t) * diff, -2 * t * diff), axis=1,
    )


_KERNELS = {"pp": _point_point, "pe": _point_edge, "pt": _point_plane, "ee": _line_line}


def _kernel_hessian(kernel, points):
    step = 1e-30
    size = 3 * points.shape[1]
    hessian = np.empty((len(points), size, size))
    for column in range(size):
        shifted = points.astype(complex)
        shifted[:, column // 3, column % 3] += 1j * step
        hessian[:, :, column] = kernel(shifted)[1].imag / step
    return 0.5 * (hessian + hessian.transpose(0, 2, 1))


def _vertex_triangle_features(points, pairs):
    """Reduce vertex-triangle pairs (n, 4) to their closest features."""
    p, a, b, c = (points[pairs[:, i]] for i in range(4))
    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    d1, d2, d3, d4 = _dot(ab, ap), _dot(ac, ap), _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    region = np.select(
        [
            (d1 <= 0) & (d2 <= 0),
            (d3 >= 0) & (d4 <= d3),
            (vc <= 0) & (d1 >= 0) & (d3 <= 0),
            (d6 >= 0) & (d5 <= d6),
            (vb <= 0) & (d2 >= 0) & (d6 <= 0),
            (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
        ],
        [0, 1, 2, 3, 4, 5],
        default=6,
    )
    columns = {0: (0, 1), 1: (0, 2), 3: (0, 3), 2: (0, 1, 2), 4: (0, 1, 3), 5: (0, 2, 3), 6: (0, 1, 2, 3)}
    kinds = {0: "pp", 1: "pp", 3: "pp", 2: "pe", 4: "pe", 5: "pe", 6: "pt"}
    return [(kinds[key], pairs[region == key][:, cols], np.flatnonzero(region == key))
            for key, cols in columns.items()]


def _edge_edge_features(points, pairs):
    """Reduce edge-edge pairs (n, 4) to their closest features."""
    p1, q1, p2, q2 = (points[pairs[:, i]] for i in range(4))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, b = _dot(d1, d1), _dot(d2, d2), _dot(d1, d2)
    c, f = _dot(d1, r), _dot(d2, r)
    denominator = a * e - b * b
    # Nearly parallel edges have no unique closest pair; start from an
    # endpoint, so that one of the point-edge cases is used instead.
    skew = denominator > 1e-6 * a * e
    s = np.where(skew, np.clip((b * f - c * e) / np.where(skew, denominator, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    low, high = t < 0, t > 1
    s = np.where(low, np.clip(-c / a, 0.0, 1.0), s)
    s = np.where(high, np.clip((b - c) / a, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)
    s_end = np.where(s <= 0, 0, np.where(s >= 1, 1, -1))
    t_end = np.where(t <= 0, 0, np.where(t >= 1, 1, -1))
    features = []
    interior = (s_end < 0) & (t_end < 0)
    features.append(("ee", pairs[interior], np.flatnonzero(interior)))
    for end in (0, 1):
        mask = (s_end == end) & (t_end < 0)
        features.append(("pe", pairs[mask][:, (end, 2, 3)], np.flatnonzero(mask)))
        mask = (t_end == end) & (s_end < 0)
        features.append(("pe", pairs[mask][:, (2 + end, 0, 1)], np.flatnonzero(mask)))
        for other in (0, 1):
            mask = (s_end == end) & (t_end == other)
            features.append(("pp", pairs[mask][:, (end, 2 + other)], np.flatnonzero(mask)))
    return features


class ContactBarrier:
    """Non-penetration energy between the panels of a model.

    ``distance`` is the activation distance in the model's length unit; it
    plays the role of the panel thickness. ``stiffness`` scales the energy.
    Pairs of primitives that share a point are structural neighbors and are
    never tested, which also leaves creases free to fold.
    """

    def __init__(self, model: Cadder, distance: float, stiffness: float):
        if distance <= 0 or stiffness <= 0:
            raise ValueError("Contact distance and stiffness must be positive.")
        index = {pid: i for i, pid in enumerate(model.point_ids())}
        triangles, edges = [], set()
        for surface_id in model.surfaces:
            vertices = [index[pid] for pid in model._surface_vertices(surface_id)]
            triangles.extend(
                (vertices[0], vertices[i], vertices[i + 1]) for i in range(1, len(vertices) - 1)
            )
            edges.update(
                tuple(sorted(edge)) for edge in zip(vertices, vertices[1:] + vertices[:1])
            )
        self.n_vars = 3 * len(index)
        self.triangles = np.array(triangles, dtype=int)
        self.edges = np.array(sorted(edges), dtype=int)
        self.vertices = np.unique(self.triangles)
        self.distance = float(distance)
        self.stiffness = float(stiffness)
        self.margin = 2.0 * self.distance
        self._candidate_X = None
        self._vertex_triangle = self._edge_edge = None
        self._X = None
        self._features = None
        self._pair_distance = None

    # Candidate pairs ------------------------------------------------------

    def _update_candidates(self, points):
        """Collect every primitive pair that could come within range soon."""
        reference = self._candidate_X
        if reference is not None and float(
            np.max(np.linalg.norm(points - reference, axis=1))
        ) < 0.25 * self.margin:
            return
        reach = self.distance + self.margin
        corners = points[self.triangles]
        centers = corners.mean(axis=1)
        radius = float(np.max(np.linalg.norm(corners - centers[:, None], axis=2)))
        hits = cKDTree(centers).query_ball_point(points[self.vertices], radius + reach)
        vertex = np.repeat(self.vertices, [len(hit) for hit in hits])
        triangle = self.triangles[np.concatenate(hits).astype(int)] if len(vertex) else np.empty((0, 3), int)
        keep = ~np.any(triangle == vertex[:, None], axis=1)
        self._vertex_triangle = np.column_stack((vertex[keep], triangle[keep]))

        ends = points[self.edges]
        middles = ends.mean(axis=1)
        half = float(np.max(np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1))) / 2
        pairs = cKDTree(middles).query_pairs(2 * half + reach, output_type="ndarray")
        first, second = self.edges[pairs[:, 0]], self.edges[pairs[:, 1]]
        keep = ~np.any(first[:, :, None] == second[:, None, :], axis=(1, 2))
        self._edge_edge = np.column_stack((first[keep], second[keep]))
        self._candidate_X = points.copy()
        self._X = None

    def _evaluate(self, X):
        """Classify all candidate pairs at ``X`` and cache the active features."""
        X = np.asarray(X, dtype=float)
        if self._X is not None and np.array_equal(X, self._X):
            return
        points = X.reshape(-1, 3)
        self._update_candidates(points)
        limit = self.distance**2
        groups: dict[str, list] = {"pp": [], "pe": [], "pt": [], "ee": []}
        pair_distance = []
        for pairs, reduce in (
            (self._vertex_triangle, _vertex_triangle_features),
            (self._edge_edge, _edge_edge_features),
        ):
            squared = np.full(len(pairs), np.inf)
            for kind, indices, rows in reduce(points, pairs):
                if not len(indices):
                    continue
                values = _KERNELS[kind](points[indices])[0]
                squared[rows] = values
                active = values < limit
                if np.any(active):
                    groups[kind].append(indices[active])
            pair_distance.append(np.sqrt(squared))
        features = {}
        for kind, parts in groups.items():
            if not parts:
                continue
            indices = np.concatenate(parts)
            if kind == "pp":
                indices = np.sort(indices, axis=1)
            elif kind == "pe":
                indices = np.column_stack((indices[:, 0], np.sort(indices[:, 1:], axis=1)))
            # Several primitive pairs can share one closest feature; count it once.
            features[kind] = np.unique(indices, axis=0)
        self._X = X.copy()
        self._features = features
        self._pair_distance = pair_distance

    # Energy ---------------------------------------------------------------

    def _terms(self, X, order):
        """Yield (indices, d2, gradient, hessian) for each active feature type."""
        self._evaluate(X)
        points = np.asarray(X, dtype=float).reshape(-1, 3)
        for kind, indices in self._features.items():
            local = points[indices]
            squared, gradient = _KERNELS[kind](local)
            hessian = _kernel_hessian(_KERNELS[kind], local) if order == 2 else None
            yield indices, squared, gradient, hessian

    def value(self, X) -> float:
        total = 0.0
        limit = self.distance**2
        for _, squared, _, _ in self._terms(X, 0):
            if np.any(squared <= 0):
                return np.inf
            u = squared / limit
            total += float(np.sum(-((u - 1) ** 2) * np.log(u)))
        return self.stiffness * total

    @staticmethod
    def _dofs(indices):
        return (3 * indices[:, :, None] + np.arange(3)).reshape(len(indices), -1)

    def gradient(self, X) -> np.ndarray:
        result = np.zeros(self.n_vars)
        limit = self.distance**2
        for indices, squared, gradient, _ in self._terms(X, 1):
            u = squared / limit
            slope = (-2 * (u - 1) * np.log(u) - (u - 1) ** 2 / u) / limit
            np.add.at(result, self._dofs(indices), self.stiffness * slope[:, None] * gradient)
        return result

    def hessian(self, X) -> csr_matrix:
        rows, columns, values = [], [], []
        limit = self.distance**2
        for indices, squared, gradient, hessian in self._terms(X, 2):
            u = squared / limit
            slope = (-2 * (u - 1) * np.log(u) - (u - 1) ** 2 / u) / limit
            curvature = (-2 * np.log(u) - 4 * (u - 1) / u + (u - 1) ** 2 / u**2) / limit**2
            block = self.stiffness * (
                curvature[:, None, None] * gradient[:, :, None] * gradient[:, None, :]
                + slope[:, None, None] * hessian
            )
            dofs = self._dofs(indices)
            size = dofs.shape[1]
            rows.append(np.repeat(dofs, size, axis=1).ravel())
            columns.append(np.tile(dofs, (1, size)).ravel())
            values.append(block.ravel())
        if not rows:
            return csr_matrix((self.n_vars, self.n_vars))
        return csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(self.n_vars, self.n_vars),
        )

    # Safe steps -----------------------------------------------------------

    def minimum_distance(self, X) -> float:
        """Smallest distance over all candidate primitive pairs."""
        self._evaluate(X)
        return float(min((np.min(d, initial=np.inf) for d in self._pair_distance), default=np.inf))

    def step_limit(self, X, step) -> float:
        """Largest fraction of ``step`` that cannot make any pair cross.

        Along a straight step every point of a primitive moves within ``dev``
        of the pair's mean displacement, where ``dev`` is the largest such
        deviation among the pair's corners. Two primitives therefore approach
        by at most ``2 * dev``, and the step is cut so that this stays below
        the pair's current distance. Pairs that are not candidates are farther
        apart than the candidate margin, which bounds the step as a whole.
        """
        self._evaluate(X)
        move = np.asarray(step, dtype=float).reshape(-1, 3)
        largest = float(np.max(np.linalg.norm(move, axis=1), initial=0.0))
        fraction = 1.0 if largest == 0 else min(1.0, 0.2 * self.margin / largest)
        for pairs, distance in zip((self._vertex_triangle, self._edge_edge), self._pair_distance):
            if not len(pairs):
                continue
            local = move[pairs]
            deviation = np.max(
                np.linalg.norm(local - local.mean(axis=1, keepdims=True), axis=2), axis=1,
            )
            moving = deviation > 0
            if np.any(moving):
                fraction = min(
                    fraction, float(np.min(0.4 * distance[moving] / deviation[moving])),
                )
        return fraction
