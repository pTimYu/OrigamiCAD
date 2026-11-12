from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from ._direct import solve_direct
from ._evaluation import AXIS_INDEX, CompiledConstraints
from .jacobian import JacobianBuilder
from .topology import panel_rigidity_plan
from .visualization import CadVisualizationMixin

ConstraintKind = Literal[
    "bar_length",
    "fixed_coordinate",
    "horizontal_surface",
    "surface_z_value",
    "dihedral_signed_increment",
]

@dataclass
class Point3D:
    id: str
    x: float
    y: float
    z: float


@dataclass
class Constraint:
    id: str
    kind: ConstraintKind
    data: dict

@dataclass
class SolveReport:
    """Solver outcome; rank and mobility are -1 when not computed."""

    success: bool
    message: str
    nfev: int
    cost: float
    residual_norm: float
    max_abs_residual: float
    rank: int
    mobility: int
    x: np.ndarray

class Cadder(CadVisualizationMixin):
    """
    Constraint-based kinematic model for origami/cellular structures.

    This class is responsible for:
        1. storing 3D point coordinates
        2. storing constraints
        3. evaluating constraint residuals
        4. estimating local mobility by Jacobian rank
    """

    def __init__(self, unit: str = "mm"):
        self.unit = str(unit)
        self.points: Dict[str, Point3D] = {}
        self.lines: dict = {}
        self.surfaces: dict = {}
        # Optional inner boundary loops for panel faces. Each key is a surface
        # ID and each value is a list of loops containing point IDs.
        self.surface_holes: dict[str, list[list[str]]] = {}
        self.constraints: Dict[str, Constraint] = {}

        self.hex_units = []
        self.hex_creases: dict[str, dict] = {}

        self._constraint_count = 0
        self._bar_constraint_index = None

    # ------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------

    @classmethod
    def from_drawer(cls, drawer) -> "Cadder":
        """
        Build a Cadder model from a TwoDDrawer object.

        2D points are lifted into 3D by setting z = 0.
        """
        model = cls(unit=getattr(drawer, "unit", "mm"))

        for pid, point in drawer.points.items():
            model.add_point(pid, point.x, point.y, 0.0)

        metadata = drawer.to_dict()
        model.lines = metadata["lines"]
        model.surfaces = metadata["surfaces"]
        model.surface_holes = copy.deepcopy(
            getattr(drawer, "surface_holes", {})
        )

        from ..patterns.hexagon.metadata import load_hex_metadata
        load_hex_metadata(model, metadata)

        return model

    @classmethod
    def from_3d_metadata(cls, metadata: dict) -> "Cadder":
        """
        Build a Cadder model from direct 3D metadata.
        """
        model = cls(unit=metadata.get("metadata", {}).get("unit", "mm"))

        for pid, coords in metadata["points"].items():
            if len(coords) != 3:
                raise ValueError(
                    f"Point '{pid}' must have 3 coordinates [x, y, z]."
                )

            model.add_point(pid, coords[0], coords[1], coords[2])

        model.lines = metadata.get("lines", {})
        model.surfaces = metadata.get("surfaces", {})
        model.surface_holes = copy.deepcopy(
            metadata.get("surface_holes", {})
        )
        from ..patterns.hexagon.metadata import load_hex_metadata
        load_hex_metadata(model, metadata)

        return model

    def to_2d_drawer(self, point_tol: float = 1e-9):
        """
        Project the current 3D structure to the x-y plane as a TwoDDrawer.

        The projection keeps the same point, line, and surface IDs. Points are
        copied directly instead of merged, so overlapping projected points from
        different heights remain distinct.
        """
        from .two_d_drawer import Line2D, Point2D, Surface2D, TwoDDrawer

        drawer = TwoDDrawer(unit=self.unit, point_tol=point_tol)

        for point_id, point in self.points.items():
            drawer.points[point_id] = Point2D(
                id=point_id,
                x=float(point.x),
                y=float(point.y),
            )

        for line_id in self.lines:
            start, end, kind = self._line_info(line_id)
            drawer.lines[line_id] = Line2D(
                id=line_id,
                start=start,
                end=end,
                kind=kind,
            )

        for surface_id in self.surfaces:
            drawer.surfaces[surface_id] = Surface2D(
                id=surface_id,
                vertices=self._surface_vertices(surface_id),
            )

        drawer.hex_units = copy.deepcopy(self.hex_units)
        drawer.hex_creases = copy.deepcopy(self.hex_creases)
        drawer.surface_holes = copy.deepcopy(self.surface_holes)
        drawer._point_count = len(drawer.points)
        drawer._line_count = len(drawer.lines)
        drawer._surface_count = len(drawer.surfaces)

        return drawer

    # ------------------------------------------------------------
    # Basic data handling
    # ------------------------------------------------------------

    def add_point(self, point_id: str, x: float, y: float, z: float) -> None:
        if point_id in self.points:
            raise ValueError(f"Point '{point_id}' already exists.")

        self.points[point_id] = Point3D(
            id=point_id,
            x=float(x),
            y=float(y),
            z=float(z),
        )

    def point_ids(self) -> List[str]:
        return list(self.points.keys())

    def num_points(self) -> int:
        return len(self.points)

    def num_variables(self) -> int:
        return 3 * self.num_points()

    def get_coordinate_vector(self) -> np.ndarray:
        """
        Return all point coordinates as one vector:

            X = [x0, y0, z0, x1, y1, z1, ...]
        """
        values = []

        for pid in self.point_ids():
            p = self.points[pid]
            values.extend([p.x, p.y, p.z])

        return np.array(values, dtype=float)

    def set_coordinate_vector(self, X: np.ndarray) -> None:
        """
        Update all point coordinates from one vector.
        """
        X = np.asarray(X, dtype=float)

        expected_size = self.num_variables()

        if X.size != expected_size:
            raise ValueError(
                f"Expected coordinate vector of size {expected_size}, "
                f"but got {X.size}."
            )

        for i, pid in enumerate(self.point_ids()):
            self.points[pid].x = X[3 * i + 0]
            self.points[pid].y = X[3 * i + 1]
            self.points[pid].z = X[3 * i + 2]

    def point_array(self, point_id: str) -> np.ndarray:
        """
        Return one point coordinate as numpy array [x, y, z].
        """
        if point_id not in self.points:
            raise ValueError(f"Point '{point_id}' does not exist.")

        p = self.points[point_id]
        return np.array([p.x, p.y, p.z], dtype=float)

    # ------------------------------------------------------------
    # Dihedral Angle Calculation
    # ------------------------------------------------------------

    def signed_dihedral_angle(
        self,
        edge_start: str,
        edge_end: str,
        point_left: str,
        point_right: str,
        unit: Literal["rad", "deg"] = "rad",
    ) -> float:
        """
        Compute signed angle between two panel-side vectors around a crease axis.

        The crease axis is:
            edge_start -> edge_end

        point_left:
            A point on the first panel, not on the crease.

        point_right:
            A point on the second panel, not on the crease.

        Convention:
            The returned angle is the signed rotation from the projected
            left-panel vector to the projected right-panel vector around
            the crease axis.

        Important:
            For a flat 2D crease pattern, this value may be close to +180 deg
            or -180 deg depending on vertex ordering and which side the two
            panels lie on.
        """
        a = self.point_array(edge_start)
        b = self.point_array(edge_end)
        c = self.point_array(point_left)
        d = self.point_array(point_right)

        axis = self._unit_vector(b - a, name="dihedral axis")

        u = c - a
        v = d - a

        # Project both vectors onto the plane perpendicular to the crease axis.
        u_perp = u - np.dot(u, axis) * axis
        v_perp = v - np.dot(v, axis) * axis

        u_hat = self._unit_vector(u_perp, name="projected left vector")
        v_hat = self._unit_vector(v_perp, name="projected right vector")

        sin_angle = np.dot(axis, np.cross(u_hat, v_hat))
        cos_angle = np.dot(u_hat, v_hat)

        angle_rad = float(np.arctan2(sin_angle, cos_angle))
        if unit == "deg":
            return float(np.rad2deg(angle_rad))
        if unit == "rad":
            return angle_rad
        raise ValueError("unit must be 'rad' or 'deg'.")

    # ------------------------------------------------------------
    # Constraint creation
    # ------------------------------------------------------------

    def _add_constraint(
        self,
        kind: ConstraintKind,
        data: dict,
        constraint_id: Optional[str],
        points: Iterable[str] = (),
    ) -> str:
        """Check the points and the ID, then store one constraint."""
        for pid in points:
            if pid not in self.points:
                raise ValueError(f"Point '{pid}' does not exist.")
        if constraint_id is None:
            constraint_id = f"c{self._constraint_count}"
            self._constraint_count += 1
        if constraint_id in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' already exists.")
        self.constraints[constraint_id] = Constraint(constraint_id, kind, data)
        return constraint_id

    @staticmethod
    def _canonical_point_pair(p1: str, p2: str) -> Tuple[str, str]:
        """Direction-independent key for a point pair."""
        return tuple(sorted((p1, p2)))

    def find_bar_length_constraint(self, p1: str, p2: str) -> Optional[str]:
        """Find an existing bar-length constraint between two points, in either direction."""
        target_pair = self._canonical_point_pair(p1, p2)
        if self._bar_constraint_index is not None:
            return self._bar_constraint_index.get(target_pair)
        for cid, constraint in self.constraints.items():
            if constraint.kind == "bar_length":
                data = constraint.data
                if self._canonical_point_pair(data["p1"], data["p2"]) == target_pair:
                    return cid
        return None

    @contextmanager
    def _index_bar_constraints(self):
        """Index one append-only batch; later public edits cannot stale it."""
        if self._bar_constraint_index is not None:
            yield
            return
        index = {}
        for cid, constraint in self.constraints.items():
            if constraint.kind == "bar_length":
                data = constraint.data
                index.setdefault(self._canonical_point_pair(data["p1"], data["p2"]), cid)
        self._bar_constraint_index = index
        try:
            yield
        finally:
            self._bar_constraint_index = None

    def add_bar_length_constraint(
        self,
        p1: str,
        p2: str,
        length: Optional[float] = None,
        constraint_id: Optional[str] = None,
        merge_if_duplicate: bool = True,
        length_tol: float = 1e-9,
    ) -> str:
        """Keep the distance between two points: ``||x2 - x1|| - L = 0``.

        ``length`` defaults to the current distance. A bar that already links
        the same two points is reused when ``merge_if_duplicate`` is true; a
        different length for that pair is an error either way.
        """
        if p1 == p2:
            raise ValueError("A bar-length constraint cannot use the same point twice.")
        if length is None:
            length = float(np.linalg.norm(self.point_array(p2) - self.point_array(p1)))
        if length <= 0:
            raise ValueError(f"Bar length between '{p1}' and '{p2}' must be positive.")

        existing_cid = self.find_bar_length_constraint(p1, p2)
        if existing_cid is not None:
            existing_length = self.constraints[existing_cid].data["length"]
            if abs(existing_length - length) > length_tol:
                raise ValueError(
                    f"Conflicting bar length for points '{p1}' and '{p2}'. "
                    f"Existing length = {existing_length}, new length = {length}."
                )
            if merge_if_duplicate:
                return existing_cid

        constraint_id = self._add_constraint(
            "bar_length", {"p1": p1, "p2": p2, "length": float(length)},
            constraint_id, (p1, p2),
        )
        if self._bar_constraint_index is not None:
            self._bar_constraint_index.setdefault(
                self._canonical_point_pair(p1, p2), constraint_id
            )
        return constraint_id

    def add_fixed_coordinate_constraint(
        self,
        point_id: str,
        axis: Literal["x", "y", "z"],
        value: Optional[float] = None,
        constraint_id: Optional[str] = None,
    ) -> str:
        """Fix one coordinate of one point, at its current value by default."""
        if axis not in AXIS_INDEX:
            raise ValueError("axis must be 'x', 'y', or 'z'.")
        if value is None:
            value = self.point_array(point_id)[AXIS_INDEX[axis]]
        return self._add_constraint(
            "fixed_coordinate",
            {"point": point_id, "axis": axis, "value": float(value)},
            constraint_id, (point_id,),
        )

    def add_fixed_point_constraint(
        self,
        point_id: str,
        value: Optional[Tuple[float, float, float]] = None,
    ) -> List[str]:
        """Fix x, y, and z of one point; this creates three scalar constraints."""
        if value is None:
            value = self.point_array(point_id)
        return [
            self.add_fixed_coordinate_constraint(point_id, axis, value[index])
            for axis, index in AXIS_INDEX.items()
        ]

    def add_horizontal_surface_constraint(
        self,
        surface_id: str,
        constraint_id: Optional[str] = None,
    ) -> str:
        """Keep a surface parallel to the xy-plane without fixing its height.

        For vertices v0, v1, v2, ... the residuals are ``z1 - z0``, ``z2 - z0``, ...
        """
        if len(self._surface_vertices(surface_id)) < 3:
            raise ValueError(f"Surface '{surface_id}' has fewer than 3 vertices.")
        return self._add_constraint(
            "horizontal_surface", {"surface": surface_id},
            constraint_id or f"horizontal_{surface_id}",
        )

    def add_surface_z_value_constraint(
        self,
        surface_id: str,
        z_value: float = 0.0,
        constraint_id: Optional[str] = None,
    ) -> str:
        """Fix every vertex of a surface at one z value."""
        if len(self._surface_vertices(surface_id)) < 3:
            raise ValueError(f"Surface '{surface_id}' has fewer than 3 vertices.")
        return self._add_constraint(
            "surface_z_value", {"surface": surface_id, "z_value": float(z_value)},
            constraint_id or f"z_value_{surface_id}",
        )

    def add_dihedral_signed_increment_constraint(
        self,
        edge_start: str,
        edge_end: str,
        point_left: str,
        point_right: str,
        target_increment: float,
        unit: Literal["rad", "deg"] = "rad",
        constraint_id: Optional[str] = None,
        sign: int = +1,
        crease_kind: Optional[str] = None,
    ) -> str:
        """Drive a crease by a signed change of its dihedral angle.

        The current signed angle is stored as the reference, so the residual is
        ``wrap_to_pi((current_angle - initial_angle) - target_increment)``.
        Unlike a cosine target, this distinguishes mountain from valley folds.
        """
        points = (edge_start, edge_end, point_left, point_right)
        for pid in points:
            if pid not in self.points:
                raise ValueError(f"Point '{pid}' does not exist.")
        if edge_start == edge_end:
            raise ValueError("Dihedral edge cannot have zero length.")
        return self._add_constraint(
            "dihedral_signed_increment",
            {
                "edge_start": edge_start,
                "edge_end": edge_end,
                "point_left": point_left,
                "point_right": point_right,
                "initial_angle": self.signed_dihedral_angle(*points),
                "target_increment": self._angle_to_rad(target_increment, unit=unit),
                "sign": int(sign),
                "crease_kind": crease_kind,
            },
            constraint_id,
        )

    # ------------------------------------------------------------
    # Automatic constraints from geometry
    # ------------------------------------------------------------

    def add_panel_rigidity_constraints_from_surfaces(self) -> None:
        """
        Add bar-length constraints inside each surface to make every panel rigid.

        For a triangle:
            all 3 pairwise distances are fixed.

        For a quadrilateral:
            all 6 pairwise distances are fixed.

        This is simple and robust for the first version.
        A reusable topology plan contains each endpoint pair once. An index
        of existing constraints preserves duplicate/length validation without
        scanning all previous bars for every new pair.
        """
        with self._index_bar_constraints():
            for constraint_id, p1, p2 in panel_rigidity_plan(self):
                existing = self.constraints.get(constraint_id)
                if (
                    existing is not None
                    and existing.kind == "bar_length"
                    and {existing.data["p1"], existing.data["p2"]} == {p1, p2}
                ):
                    # Repeated setup: this panel bar already holds its
                    # reference length. Re-measuring it from coordinates that
                    # an earlier solve moved would report solver error as a
                    # length conflict.
                    continue
                self.add_bar_length_constraint(
                    p1, p2, constraint_id=constraint_id, merge_if_duplicate=True
                )

    # ------------------------------------------------------------
    # Residual evaluation
    # ------------------------------------------------------------

    def residual_vector(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate all scalar constraint residuals.

        If X is provided, residuals are evaluated directly from its coordinates
        without modifying the model. During solve(), indexed constraint batches
        and geometry are shared with the analytic Jacobian.
        """
        plan = getattr(self, "_compiled_constraints", None) or CompiledConstraints(self)
        return plan.residual(X)

    # ------------------------------------------------------------
    # Jacobian and mobility
    # ------------------------------------------------------------

    def jacobian(
        self, X: Optional[np.ndarray] = None, *, sparse: bool = True
    ) -> csr_matrix | np.ndarray:
        """Build the analytic residual Jacobian; see :class:`JacobianBuilder`.

        Rows match residual_vector(), columns match get_coordinate_vector().
        Supplied coordinates are evaluated without modifying this model.
        """
        return JacobianBuilder(self).build(X, sparse=sparse)

    def constraint_jacobian(
        self,
        constraint: str | Constraint,
        X: Optional[np.ndarray] = None,
        *,
        sparse: bool = True,
    ) -> csr_matrix | np.ndarray:
        """Build one constraint's analytic Jacobian with all 3N model columns."""
        return JacobianBuilder(self).for_constraint(constraint, X, sparse=sparse)

    def constraint_rank(
        self,
        tol: float = 1e-8,
    ) -> int:
        """
        Return numerical rank of the constraint Jacobian.
        """
        J = self.jacobian(sparse=False)
        return int(np.linalg.matrix_rank(J, tol=tol))

    def mobility(
        self,
        tol: float = 1e-8,
    ) -> int:
        """
        Estimate local mobility:

            mobility = number of variables - rank(J)
        """
        rank = self.constraint_rank(tol=tol)
        return self.num_variables() - rank

    # ------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------

    def solve(
        self,
        X0: Optional[np.ndarray] = None,
        update_model: bool = True,
        tol: float = 1e-10,
        rank_tol: float = 1e-8,
        max_nfev: Optional[int] = 2000,
        verbose: int = 0,
        compute_rank: bool = False,
        x_scale: float | np.ndarray | Literal["jac"] = "jac",
        linear_solver: Literal["direct", "lsmr"] = "direct",
    ) -> SolveReport:
        """
        Solve the nonlinear constraint system ``residual_vector(X) = 0``.

        Args:
            X0:
                Initial guess. If None, use current model coordinates.
            update_model:
                If True, update self.points using the solved coordinates.
            tol:
                Stopping tolerance, applied as SciPy's ``ftol``, ``xtol``,
                and ``gtol``.
            rank_tol:
                Tolerance for Jacobian rank estimation.
            max_nfev:
                Maximum number of residual evaluations.
            verbose:
                SciPy least_squares verbosity (0 silent, 1 final report,
                2 iteration report). Used by the ``"lsmr"`` solver.
            compute_rank:
                If True, also compute the dense Jacobian rank and the
                mobility. Otherwise both report fields are -1. This is slow
                for large models; hexagon ``analyze_kinematics`` offers the
                same diagnostic separately.
            x_scale:
                Variable scales. The default "jac" adapts scales to inverse
                Jacobian column norms. Use 1.0 for unscaled variables, or
                supply positive scales in coordinate-vector order. This does
                not weight residual rows.
            linear_solver:
                "direct" (default) factorizes the sparse normal equations in
                every trust-region step, which is several times faster than
                an iterative inner solve. If a factorization fails, the solve
                restarts with "lsmr". "lsmr" uses SciPy's trust-region solver
                with LSMR settings atol=btol=1e-10 and maxiter=5000.

        Returns:
            SolveReport
        """
        if len(self.constraints) == 0:
            raise ValueError("No constraints have been added.")
        if linear_solver not in ("direct", "lsmr"):
            raise ValueError("linear_solver must be 'direct' or 'lsmr'.")

        if X0 is None:
            X0 = self.get_coordinate_vector()
        X0 = np.asarray(X0, dtype=float)
        if X0.size != self.num_variables():
            raise ValueError(
                f"Expected X0 size {self.num_variables()}, but got {X0.size}."
            )

        builder = JacobianBuilder(self)

        def residual(X):
            # Looked up on every call so an instrumented residual_vector is used.
            return self.residual_vector(X)

        with builder.compiled():
            result = None
            if linear_solver == "direct":
                result = solve_direct(
                    residual, builder.build, X0,
                    tol=tol, max_nfev=max_nfev, x_scale=x_scale,
                )
            if result is None:
                result = least_squares(
                    residual, X0, jac=builder.build,
                    xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev,
                    verbose=verbose, x_scale=x_scale,
                    tr_options={"atol": 1e-10, "btol": 1e-10, "maxiter": 5000},
                )
            residuals = self.residual_vector(result.x)

        rank = mobility = -1
        if compute_rank:
            # The solver already evaluated this Jacobian at the solution.
            rank = int(np.linalg.matrix_rank(result.jac.toarray(), tol=rank_tol))
            mobility = self.num_variables() - rank

        if update_model:
            self.set_coordinate_vector(result.x)

        return SolveReport(
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            cost=float(result.cost),
            residual_norm=float(np.linalg.norm(residuals)),
            max_abs_residual=float(np.max(np.abs(residuals))) if residuals.size else 0.0,
            rank=rank,
            mobility=mobility,
            x=result.x.copy(),
        )

    def print_solve_report(self, report: SolveReport) -> None:
        """
        Print a readable solver report.
        """
        print("Solve report")
        print("------------")
        print(f"Success:            {report.success}")
        print(f"Message:            {report.message}")
        print(f"Function evals:     {report.nfev}")
        print(f"Cost:               {report.cost}")
        print(f"Residual norm:      {report.residual_norm}")
        print(f"Max abs residual:   {report.max_abs_residual}")
        if report.rank == -1 and report.mobility == -1:
            print("Jacobian rank:      not computed")
            print("Mobility:           not computed")
            return
        print(f"Jacobian rank:      {report.rank}")
        print(f"Mobility:           {report.mobility}")

        if report.mobility == 0:
            print("Status:             locally unique")
        elif report.mobility > 0:
            print("Status:             underconstrained / mechanism remains")
        else:
            print("Status:             overconstrained or inconsistent")

    # ------------------------------------------------------------
    # File export
    # ------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the current 3D geometry as JSON-friendly metadata."""
        from ..io.cad_export import model_to_dict

        return model_to_dict(self)

    def save_json(self, filename: str) -> None:
        """Save points, lines, surfaces, and units as 3D JSON metadata."""
        from ..io.cad_export import save_json

        save_json(self, filename)

    def save_stl(self, filename: str, thickness: float = 0.0) -> None:
        """Save panel geometry as an ASCII STL mesh."""
        from ..io.cad_export import save_stl

        save_stl(self, filename, thickness=thickness)

    def save_step(
        self,
        filename: str,
        thickness: float = 0.0,
        *,
        separate_layer_parts: bool = False,
    ) -> None:
        """Save panel geometry as a faceted STEP surface or solid model.

        Use ``separate_layer_parts=True`` on a model returned by
        ``stack_layers`` to create one importable STEP part per layer.
        """
        from ..io.cad_export import save_step

        save_step(
            self,
            filename,
            thickness=thickness,
            separate_layer_parts=separate_layer_parts,
        )

    def save_cad(
        self,
        filename: str,
        thickness: float = 0.0,
        *,
        separate_layer_parts: bool = False,
    ) -> None:
        """Export based on a .json, .stl, .step, or .stp extension."""
        from ..io.cad_export import save_cad

        save_cad(
            self,
            filename,
            thickness=thickness,
            separate_layer_parts=separate_layer_parts,
        )

    def save_xy_dxf(self, filename: str, point_tol: float = 1e-9, **dxf_options):
        """Project the current 3D structure to x-y and save it as DXF.

        ``dxf_options`` are the keyword options of :meth:`TwoDDrawer.save_dxf`.
        """
        return self.to_2d_drawer(point_tol=point_tol).save_dxf(filename, **dxf_options)

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------

    @staticmethod
    def _angle_to_rad(angle: float, unit: Literal["rad", "deg"] = "rad") -> float:
        if unit == "rad":
            return float(angle)

        if unit == "deg":
            return float(np.deg2rad(angle))

        raise ValueError("unit must be 'rad' or 'deg'.")

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """
        Wrap angle to (-pi, pi].
        """
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _unit_vector(
        v: np.ndarray,
        name: str = "vector",
        eps: float = 1e-12,
    ) -> np.ndarray:
        norm = float(np.linalg.norm(v))

        if norm < eps:
            raise ValueError(f"{name} has near-zero length.")

        return v / norm

    def _orient_flat_crease_triangle_right(
        self,
        edge_start: str,
        edge_end: str,
        triangle_point: str,
        eps: float = 1e-12,
    ) -> Tuple[str, str]:
        """Orient a flat crease with the triangle on its right-hand side."""
        a = self.point_array(edge_start)
        b = self.point_array(edge_end)
        c = self.point_array(triangle_point)
        edge = b[:2] - a[:2]
        toward_triangle = c[:2] - a[:2]
        signed_area = float(
            edge[0] * toward_triangle[1]
            - edge[1] * toward_triangle[0]
        )

        if abs(signed_area) < eps:
            raise ValueError(
                f"Cannot orient crease {edge_start}-{edge_end}: the triangle "
                "point is collinear in the flat XY pattern."
            )

        if signed_area > 0.0:
            return edge_end, edge_start

        return edge_start, edge_end

    def _surface_vertices(self, surface_id: str) -> List[str]:
        """
        Return vertex IDs of a surface.

        This supports both dictionary-style surfaces and dataclass-style surfaces.
        """
        if surface_id not in self.surfaces:
            raise ValueError(f"Surface '{surface_id}' does not exist.")

        surface = self.surfaces[surface_id]

        if isinstance(surface, dict):
            return list(surface["vertices"])

        return list(surface.vertices)

    def _line_info(self, line_id: str) -> Tuple[str, str, str]:
        """
        Return start, end, kind of a line.
        """
        if line_id not in self.lines:
            raise ValueError(f"Line '{line_id}' does not exist.")

        line = self.lines[line_id]

        if isinstance(line, dict):
            return line["start"], line["end"], line["kind"]

        return line.start, line.end, line.kind

    def _non_collinear_triplet_from_surface(
        self,
        surface_id: str,
        eps: float = 1e-9,
    ) -> Tuple[str, str, str]:
        """
        Pick three non-collinear points from a surface.
        """
        vertices = self._surface_vertices(surface_id)

        if len(vertices) < 3:
            raise ValueError(f"Surface '{surface_id}' has fewer than 3 vertices.")

        for p1, p2, p3 in combinations(vertices, 3):
            x1 = self.point_array(p1)
            x2 = self.point_array(p2)
            x3 = self.point_array(p3)

            area_vector = np.cross(x2 - x1, x3 - x1)

            if np.linalg.norm(area_vector) > eps:
                return p1, p2, p3

        raise ValueError(f"Surface '{surface_id}' is degenerate or collinear.")

    @staticmethod
    def _first_non_edge_vertex(vertices: List[str], p1: str, p2: str) -> str:
        """
        Return one vertex in the surface that is not on the edge p1-p2.
        """
        for pid in vertices:
            if pid not in {p1, p2}:
                return pid

        raise ValueError(
            f"Surface {vertices} does not contain a non-edge vertex "
            f"relative to edge {p1}-{p2}."
        )
