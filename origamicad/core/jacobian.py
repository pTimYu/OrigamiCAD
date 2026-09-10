"""Analytic derivatives of Cadder constraint residuals."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from .cadder import Cadder, Constraint


def _unit(v: np.ndarray, derivative: np.ndarray, name: str):
    """Differentiate v / ||v|| with respect to a local coordinate vector."""
    length = np.linalg.norm(v)
    if length < 1e-12:
        raise ValueError(f"{name} has near-zero length.")
    direction = v / length
    return direction, (
        derivative - np.outer(direction, direction @ derivative)
    ) / length


def _cross(a: np.ndarray, da: np.ndarray, b: np.ndarray, db: np.ndarray):
    """Product rule for a cross product; derivative columns are directions."""
    return np.cross(a, b), (
        np.cross(da.T, b).T + np.cross(a, db.T).T
    )


class JacobianBuilder:
    """Build analytic Jacobians for one constraint or the entire model.

    Rows follow ``model.residual_vector()`` (constraint insertion order, then
    residual component order). Columns follow ``model.get_coordinate_vector()``:
    x, y, z for each point in ``model.point_ids()`` order. A single-constraint
    matrix retains all model columns, including zeros for unrelated points.

    By default results are SciPy CSR matrices; ``sparse=False`` returns a dense
    NumPy array. ``X`` optionally supplies coordinates without mutating the model.
    Point order and constraints are read afresh on each call, so builders can be
    reused after model edits. Local blocks use analytic chain rules, with no
    finite differences or additional differentiation dependencies.

    Derivatives are undefined at zero-length geometric vectors and at the
    +/-pi discontinuity of a wrapped angle *residual*; these raise ValueError.
    Surface normals use the same first non-collinear triplet as the residual,
    and derivatives assume that triplet does not change locally.
    """

    def __init__(self, model: Cadder):
        self.model = model

    def build(
        self, X: np.ndarray | None = None, *, sparse: bool = True
    ) -> csr_matrix | np.ndarray:
        """Return the full (number of scalar residuals, 3N) Jacobian."""
        return self._assemble(self.model.constraints.values(), X, sparse)

    def for_constraint(
        self,
        constraint: str | Constraint,
        X: np.ndarray | None = None,
        *,
        sparse: bool = True,
    ) -> csr_matrix | np.ndarray:
        """Return one residual block's Jacobian, by constraint ID or object.

        ``add_fixed_point_constraint`` creates three ``fixed_coordinate``
        constraints; each returned ID can be passed here individually.
        """
        if isinstance(constraint, str):
            if constraint not in self.model.constraints:
                raise ValueError(f"Constraint '{constraint}' does not exist.")
            constraint = self.model.constraints[constraint]
        return self._assemble((constraint,), X, sparse)

    def _assemble(self, constraints, X, sparse):
        point_ids = self.model.point_ids()
        point_index = {pid: index for index, pid in enumerate(point_ids)}
        n_vars = 3 * len(point_ids)
        coordinates = np.asarray(
            self.model.get_coordinate_vector() if X is None else X, dtype=float
        )
        if coordinates.ndim != 1 or coordinates.size != n_vars:
            raise ValueError(f"Expected a one-dimensional coordinate vector of size {n_vars}.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Coordinate vector must contain only finite values.")
        points = dict(zip(point_ids, coordinates.reshape(-1, 3)))
        rows, cols, values = [], [], []
        row_offset = 0

        for constraint in constraints:
            local_ids, block = self._constraint_block(constraint, points)
            local_rows, local_cols = np.nonzero(block)
            global_columns = np.array([
                3 * point_index[pid] + axis
                for pid in local_ids for axis in range(3)
            ], dtype=int)
            rows.extend(row_offset + local_rows)
            cols.extend(global_columns[local_cols])
            values.extend(block[local_rows, local_cols])
            row_offset += block.shape[0]

        # CSR construction sums duplicate entries when a point appears in more
        # than one local role (e.g. two lines with a shared endpoint).
        result = csr_matrix(
            (values, (rows, cols)), shape=(row_offset, n_vars), dtype=float
        )
        result.eliminate_zeros()
        return result if sparse else result.toarray()

    def _constraint_block(self, constraint, points):
        kind = constraint.kind
        data = constraint.data
        if kind == "fixed_point":
            raise NotImplementedError(
                "fixed_point is implemented as three fixed_coordinate constraints."
            )
        if kind == "fixed_coordinate":
            block = np.zeros((1, 3))
            block[0, {"x": 0, "y": 1, "z": 2}[data["axis"]]] = 1.0
            return [data["point"]], block
        if kind in {"horizontal_surface", "surface_z_value"}:
            ids = self.model._surface_vertices(data["surface"])
            count = len(ids)
            if kind == "surface_z_value":
                block = np.zeros((count, 3 * count))
                block[np.arange(count), 3 * np.arange(count) + 2] = 1.0
            else:
                block = np.zeros((count - 1, 3 * count))
                block[:, 2] = -1.0
                block[np.arange(count - 1), 3 * np.arange(1, count) + 2] = 1.0
            return ids, block
        if kind == "bar_length":
            ids = [data["p1"], data["p2"]]
            delta = points[ids[1]] - points[ids[0]]
            length = np.linalg.norm(delta)
            if length == 0.0:
                raise ValueError("Bar-length Jacobian is undefined for coincident points.")
            direction = delta / length
            return ids, np.concatenate((-direction, direction))[None, :]
        if kind == "parallel_lines":
            ids = [data[key] for key in ("p1", "p2", "p3", "p4")]
            a, b, c, d = (points[pid] for pid in ids)
            da, db, dc, dd = np.eye(12).reshape(4, 3, 12)
            u, du = _unit(b - a, db - da, "first parallel line")
            v, dv = _unit(d - c, dd - dc, "second parallel line")
            _, block = _cross(u, du, v, dv)
            return ids, block
        if kind == "parallel_surfaces":
            first = self._surface_triplet(data["surface1"], points)
            second = self._surface_triplet(data["surface2"], points)
            ids = first + second
            a, b, c, d, e, f = (points[pid] for pid in ids)
            da, db, dc, dd, de, df = np.eye(18).reshape(6, 3, 18)
            n1, dn1 = _cross(b - a, db - da, c - a, dc - da)
            n2, dn2 = _cross(e - d, de - dd, f - d, df - dd)
            n1, dn1 = _unit(n1, dn1, "first surface normal")
            n2, dn2 = _unit(n2, dn2, "second surface normal")
            _, block = _cross(n1, dn1, n2, dn2)
            return ids, block
        if kind == "coplanar_points":
            return self._coplanar_block(data, points)
        if kind in {"dihedral_angle", "dihedral_cos", "dihedral_signed_increment"}:
            return self._dihedral_block(kind, data, points)
        raise ValueError(f"Unknown constraint kind '{kind}'.")

    def _surface_triplet(self, surface_id, points):
        # Match Cadder._non_collinear_triplet_from_surface at the supplied X,
        # which can differ from the coordinates currently stored in the model.
        vertices = self.model._surface_vertices(surface_id)
        if len(vertices) < 3:
            raise ValueError(f"Surface '{surface_id}' has fewer than 3 vertices.")
        for ids in combinations(vertices, 3):
            a, b, c = (points[pid] for pid in ids)
            if np.linalg.norm(np.cross(b - a, c - a)) > 1e-9:
                return list(ids)
        raise ValueError(f"Surface '{surface_id}' is degenerate or collinear.")

    @staticmethod
    def _coplanar_block(data, points):
        ids = [data[key] for key in ("p1", "p2", "p3", "p4")]
        origin, p2, p3, p4 = (points[pid] for pid in ids)
        a, b, c = p2 - origin, p3 - origin, p4 - origin
        la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
        denominator = la * lb * lc
        if denominator < 1e-12:
            raise ValueError("Coplanar constraint has near-degenerate point configuration.")
        residual = np.dot(np.cross(a, b), c) / denominator
        ga = np.cross(b, c) / denominator - residual * a / la**2
        gb = np.cross(c, a) / denominator - residual * b / lb**2
        gc = np.cross(a, b) / denominator - residual * c / lc**2
        return ids, np.concatenate((-(ga + gb + gc), ga, gb, gc))[None, :]

    def _dihedral_block(self, kind, data, points):
        ids = [data[key] for key in (
            "edge_start", "edge_end", "point_left", "point_right"
        )]
        a, b, c, d = (points[pid] for pid in ids)
        da, db, dc, dd = np.eye(12).reshape(4, 3, 12)
        axis, d_axis = _unit(b - a, db - da, "dihedral axis")

        def panel_direction(v, dv, name):
            along = np.dot(v, axis)
            d_along = axis @ dv + v @ d_axis
            perpendicular = v - along * axis
            d_perpendicular = dv - along * d_axis - np.outer(axis, d_along)
            return _unit(perpendicular, d_perpendicular, name)

        u, du = panel_direction(c - a, dc - da, "projected left vector")
        v, dv = panel_direction(d - a, dd - da, "projected right vector")
        cosine = np.dot(u, v)
        d_cosine = v @ du + u @ dv
        if kind == "dihedral_cos":
            return ids, d_cosine[None, :]

        cross, d_cross = _cross(u, du, v, dv)
        sine = np.dot(axis, cross)
        d_sine = cross @ d_axis + axis @ d_cross
        angle = np.arctan2(sine, cosine)
        if kind == "dihedral_signed_increment":
            difference = self.model._wrap_to_pi(angle - data["initial_angle"])
            residual = self.model._wrap_to_pi(difference - data["target_increment"])
        else:
            residual = self.model._wrap_to_pi(angle - data["target_angle"])
        if abs(abs(residual) - np.pi) <= 1e-12:
            raise ValueError("Wrapped angle residual is discontinuous at +/-pi.")
        # atan2 has a smooth local derivative across the raw +/-pi angle when
        # residual wrapping cancels that branch cut (as for a flat crease).
        gradient = (cosine * d_sine - sine * d_cosine) / (sine**2 + cosine**2)
        return ids, gradient[None, :]
