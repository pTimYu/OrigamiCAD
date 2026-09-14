"""Solve-scoped, array-based evaluation of a fixed constraint topology."""

from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


_DIHEDRALS = {"dihedral_angle", "dihedral_cos", "dihedral_signed_increment"}


def _dot(a, b):
    return np.sum(a * b, axis=-1)


def _unit_rows(v, name):
    length = np.linalg.norm(v, axis=1)
    if np.any(length < 1e-12):
        raise ValueError(f"{name} has near-zero length.")
    return v / length[:, None], length


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class CompiledConstraints:
    """Compile point indices and sparse assembly once for a solver invocation.

    Models expose mutable dictionaries, so this plan is deliberately scoped to
    one solve. Public evaluations outside that scope create a fresh plan and
    therefore observe direct edits to constraints, surfaces, and point order.
    Geometry is cached only for an identical coordinate vector; no trial point
    is ever written to the model. Uncommon constraint kinds retain their scalar
    implementations, evaluated through a coordinate view.
    """

    def __init__(self, model, builder, constraints=None):
        self.model = model
        self.builder = builder
        self.point_ids = model.point_ids()
        self.point_index = {pid: i for i, pid in enumerate(self.point_ids)}
        self.n_vars = 3 * len(self.point_ids)
        records = defaultdict(list)
        self.fallback = []
        row = 0
        constraints = model.constraints.values() if constraints is None else constraints
        for constraint in constraints:
            kind, data = constraint.kind, constraint.data
            if kind == "fixed_point":
                raise NotImplementedError(
                    "fixed_point is implemented as three fixed_coordinate constraints."
                )
            if kind == "fixed_coordinate":
                column = 3 * self.point_index[data["point"]] + {"x": 0, "y": 1, "z": 2}[data["axis"]]
                records["fixed"].append((row, [column], data["value"]))
                row += 1
            elif kind in {"horizontal_surface", "surface_z_value"}:
                vertices = model._surface_vertices(data["surface"])
                columns = [3 * self.point_index[pid] + 2 for pid in vertices]
                if kind == "horizontal_surface":
                    for column in columns[1:]:
                        records["horizontal"].append((row, [columns[0], column], 0.0))
                        row += 1
                else:
                    for column in columns:
                        records["fixed"].append((row, [column], data["z_value"]))
                        row += 1
            elif kind == "bar_length" or kind in _DIHEDRALS:
                keys = ("p1", "p2") if kind == "bar_length" else (
                    "edge_start", "edge_end", "point_left", "point_right"
                )
                indices = [self.point_index[data[key]] for key in keys]
                target = {
                    "bar_length": "length", "dihedral_cos": "target_cos",
                    "dihedral_angle": "target_angle",
                    "dihedral_signed_increment": "target_increment",
                }[kind]
                records[kind].append((row, indices, data[target], data.get("initial_angle", 0.0)))
                row += 1
            elif kind in {"parallel_lines", "coplanar_points", "parallel_surfaces"}:
                if kind == "parallel_surfaces":
                    ids = model._surface_vertices(data["surface1"]) + model._surface_vertices(data["surface2"])
                else:
                    ids = [data[key] for key in ("p1", "p2", "p3", "p4")]
                # Include all surface vertices: the active normal triplet can
                # change at a later X without changing this sparsity structure.
                ids = list(dict.fromkeys(ids))
                count = 1 if kind == "coplanar_points" else 3
                self.fallback.append((constraint, row, count, ids))
                row += count
            else:
                raise ValueError(f"Unknown constraint kind '{kind}'.")
        self.n_rows = row
        self.groups = {}
        for kind, entries in records.items():
            group = {
                "rows": np.array([entry[0] for entry in entries], dtype=int),
                "indices": np.array([entry[1] for entry in entries], dtype=int),
                "target": np.array([entry[2] for entry in entries], dtype=float),
            }
            if kind in _DIHEDRALS:
                group["initial"] = np.array([entry[3] for entry in entries], dtype=float)
            self.groups[kind] = group
        self._X = None
        self._geometry = {}
        self._structure = None

    def _coordinates(self, X):
        X = np.asarray(self.model.get_coordinate_vector() if X is None else X, dtype=float)
        if X.ndim != 1 or X.size != self.n_vars:
            raise ValueError(f"Expected a one-dimensional coordinate vector of size {self.n_vars}.")
        if not np.all(np.isfinite(X)):
            raise ValueError("Coordinate vector must contain only finite values.")
        if self._X is None or not np.array_equal(X, self._X):
            self._X = X.copy()
            self._geometry = {}
        return self._X

    def _bar_geometry(self, points, group):
        if "bar_length" not in self._geometry:
            ids = group["indices"]
            delta = points[ids[:, 1]] - points[ids[:, 0]]
            self._geometry["bar_length"] = delta, np.linalg.norm(delta, axis=1)
        return self._geometry["bar_length"]

    def _dihedral_geometry(self, kind, points, group):
        if kind not in self._geometry:
            a, b, c, d = np.moveaxis(points[group["indices"]], 1, 0)
            axis, edge_length = _unit_rows(b - a, "dihedral axis")
            p, q = c - a, d - a
            along_p, along_q = _dot(p, axis), _dot(q, axis)
            u, left_length = _unit_rows(p - along_p[:, None] * axis, "projected left vector")
            v, right_length = _unit_rows(q - along_q[:, None] * axis, "projected right vector")
            cross = np.cross(u, v)
            sine, cosine = _dot(axis, cross), _dot(u, v)
            self._geometry[kind] = (
                axis, edge_length, p, q, along_p, along_q, u, v,
                left_length, right_length, cross, sine, cosine,
            )
        return self._geometry[kind]

    @staticmethod
    def _dihedral_residual(kind, group, sine, cosine):
        if kind == "dihedral_cos":
            return cosine - group["target"]
        angle = np.arctan2(sine, cosine)
        if kind == "dihedral_signed_increment":
            angle = _wrap(angle - group["initial"])
        return _wrap(angle - group["target"])

    def residual(self, X=None):
        X = self._coordinates(X)
        points = X.reshape(-1, 3)
        result = np.empty(self.n_rows, dtype=float)
        for kind, group in self.groups.items():
            indices = group["indices"]
            if kind == "fixed":
                value = X[indices[:, 0]] - group["target"]
            elif kind == "horizontal":
                value = X[indices[:, 1]] - X[indices[:, 0]]
            elif kind == "bar_length":
                _, length = self._bar_geometry(points, group)
                value = length - group["target"]
            else:
                *_, sine, cosine = self._dihedral_geometry(kind, points, group)
                value = self._dihedral_residual(kind, group, sine, cosine)
            result[group["rows"]] = value
        if self.fallback:
            # Bind existing residual methods to a shallow model view whose point
            # reads use X. Its Point3D objects and the original model stay intact.
            view = copy.copy(self.model)
            view.point_array = lambda pid: points[self.point_index[pid]]
            for constraint, row, count, _ in self.fallback:
                result[row:row + count] = view._residual_for_constraint(constraint)
        return result

    def _compile_structure(self):
        rows, columns = [], []
        cursor = 0
        for kind, group in self.groups.items():
            indices = group["indices"]
            cols = indices if kind in {"fixed", "horizontal"} else (
                3 * indices[:, :, None] + np.arange(3)
            ).reshape(len(indices), -1)
            count = cols.size
            group["data_slice"] = slice(cursor, cursor + count)
            cursor += count
            rows.append(np.repeat(group["rows"], cols.shape[1]))
            columns.append(cols.ravel())
        self._fallback_structure = []
        for constraint, row, count, ids in self.fallback:
            cols = np.array([3 * self.point_index[pid] + axis for pid in ids for axis in range(3)])
            size = count * len(cols)
            self._fallback_structure.append((slice(cursor, cursor + size), {pid: i for i, pid in enumerate(ids)}))
            cursor += size
            rows.append(np.repeat(np.arange(row, row + count), len(cols)))
            columns.append(np.tile(cols, count))
        rows = np.concatenate(rows) if rows else np.array([], dtype=int)
        columns = np.concatenate(columns) if columns else np.array([], dtype=int)
        # Precompute COO-to-CSR accumulation, including repeated point roles.
        # Sorted unique keys give CSR order; bincount sums every duplicate role.
        keys, accumulation = np.unique(rows * self.n_vars + columns, return_inverse=True)
        csr_rows = keys // self.n_vars if self.n_vars else keys
        csr_columns = (keys % self.n_vars if self.n_vars else keys).astype(np.int32)
        indptr = np.concatenate(([0], np.cumsum(np.bincount(csr_rows, minlength=self.n_rows)))).astype(np.int32)
        self._structure = cursor, accumulation, csr_columns, indptr

    def jacobian(self, X=None, *, sparse=True):
        X = self._coordinates(X)
        points = X.reshape(-1, 3)
        if self._structure is None:
            self._compile_structure()
        size, accumulation, indices, indptr = self._structure
        values = np.empty(size, dtype=float)
        for kind, group in self.groups.items():
            if kind == "fixed":
                block = np.ones(len(group["rows"]))
            elif kind == "horizontal":
                block = np.tile([-1.0, 1.0], len(group["rows"]))
            elif kind == "bar_length":
                delta, length = self._bar_geometry(points, group)
                if np.any(length == 0.0):
                    raise ValueError("Bar-length Jacobian is undefined for coincident points.")
                direction = delta / length[:, None]
                block = np.concatenate((-direction, direction), axis=1)
            else:
                block = self._dihedral_gradient(kind, group, points)
            values[group["data_slice"]] = block.ravel()
        if self.fallback:
            point_map = dict(zip(self.point_ids, points))
            for (constraint, _, count, ids), (data_slice, local_index) in zip(self.fallback, self._fallback_structure):
                roles, local_block = self.builder._constraint_block(constraint, point_map)
                block = np.zeros((count, 3 * len(ids)))
                # Different roles may reference one point (parallel/shared lines
                # and surfaces), and their derivatives must add, never overwrite.
                for role, pid in enumerate(roles):
                    col = 3 * local_index[pid]
                    block[:, col:col + 3] += local_block[:, 3 * role:3 * role + 3]
                values[data_slice] = block.ravel()
        data = np.bincount(accumulation, weights=values, minlength=len(indices))
        # Independent arrays prevent a caller or SciPy's in-place scaling from
        # corrupting either this reusable plan or an earlier returned Jacobian.
        result = csr_matrix((data, indices.copy(), indptr.copy()), shape=(self.n_rows, self.n_vars))
        result.eliminate_zeros()
        return result if sparse else result.toarray()

    def _dihedral_gradient(self, kind, group, points):
        (axis, edge_length, p, q, along_p, along_q, u, v,
         left_length, right_length, cross, sine, cosine) = self._dihedral_geometry(kind, points, group)
        if kind == "dihedral_cos":
            gu, gv, ge = v, u, np.zeros_like(axis)
        else:
            residual = self._dihedral_residual(kind, group, sine, cosine)
            if np.any(np.abs(np.abs(residual) - np.pi) <= 1e-12):
                raise ValueError("Wrapped angle residual is discontinuous at +/-pi.")
            denominator = (sine**2 + cosine**2)[:, None]
            gu = (cosine[:, None] * np.cross(v, axis) - sine[:, None] * v) / denominator
            gv = (cosine[:, None] * np.cross(axis, u) - sine[:, None] * u) / denominator
            ge = cosine[:, None] * cross / denominator
        # Reverse-mode chain rule through panel normalization, projection onto
        # the crease-normal plane, and finally normalization of the crease.
        gp_perp = (gu - u * _dot(u, gu)[:, None]) / left_length[:, None]
        gq_perp = (gv - v * _dot(v, gv)[:, None]) / right_length[:, None]
        pe, qe = _dot(gp_perp, axis), _dot(gq_perp, axis)
        gp = gp_perp - axis * pe[:, None]
        gq = gq_perp - axis * qe[:, None]
        ge = ge - along_p[:, None] * gp_perp - p * pe[:, None] - along_q[:, None] * gq_perp - q * qe[:, None]
        edge = (ge - axis * _dot(axis, ge)[:, None]) / edge_length[:, None]
        return np.concatenate((-(edge + gp + gq), edge, gp, gq), axis=1)
