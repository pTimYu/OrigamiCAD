from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ConstraintKind = Literal[
    "bar_length",
    "fixed_coordinate",
    "fixed_point",
    "parallel_lines",
    "dihedral_angle",
    "coplanar_points",
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
    success: bool
    message: str
    nfev: int
    cost: float
    residual_norm: float
    max_abs_residual: float
    rank: int
    mobility: int
    x: np.ndarray

class Cadder:
    """
    Constraint-based kinematic model for origami/cellular structures.

    This class is responsible for:
        1. storing 3D point coordinates
        2. storing constraints
        3. evaluating constraint residuals
        4. estimating local mobility by Jacobian rank

    It does not yet solve a folded configuration.
    """

    def __init__(self):
        self.points: Dict[str, Point3D] = {}
        self.lines: dict = {}
        self.surfaces: dict = {}
        self.constraints: Dict[str, Constraint] = {}

        self._constraint_count = 0

    # ------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------

    @classmethod
    def from_drawer(cls, drawer) -> "Cadder":
        """
        Build a Cadder model from a TwoDDrawer object.

        2D points are lifted into 3D by setting z = 0.
        """
        model = cls()

        for pid, point in drawer.points.items():
            model.add_point(pid, point.x, point.y, 0.0)

        model.lines = drawer.to_dict()["lines"]
        model.surfaces = drawer.to_dict()["surfaces"]

        return model

    @classmethod
    def from_3d_metadata(cls, metadata: dict) -> "Cadder":
        """
        Build a Cadder model from direct 3D metadata.
        """
        model = cls()

        for pid, coords in metadata["points"].items():
            if len(coords) != 3:
                raise ValueError(
                    f"Point '{pid}' must have 3 coordinates [x, y, z]."
                )

            model.add_point(pid, coords[0], coords[1], coords[2])

        model.lines = metadata.get("lines", {})
        model.surfaces = metadata.get("surfaces", {})

        return model

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

    def _point_index(self, point_id: str) -> int:
        """
        Return the integer index of a point in the coordinate vector.
        """
        if point_id not in self.points:
            raise ValueError(f"Point '{point_id}' does not exist.")

        return self.point_ids().index(point_id)

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

        return self._rad_to_angle(angle_rad, unit=unit)

    def signed_dihedral_angle_from_line(
        self,
        line_id: str,
        unit: Literal["rad", "deg"] = "rad",
    ) -> float:
        """
        Compute signed dihedral angle for a line shared by exactly two surfaces.
        """
        if line_id not in self.lines:
            raise ValueError(f"Line '{line_id}' does not exist.")

        line = self.lines[line_id]
        edge_start = line["start"]
        edge_end = line["end"]

        adjacent_surfaces = self.find_surfaces_adjacent_to_edge(
            edge_start,
            edge_end,
        )

        if len(adjacent_surfaces) != 2:
            raise ValueError(
                f"Line '{line_id}' must be shared by exactly two surfaces "
                f"to define a dihedral angle. Found {len(adjacent_surfaces)}."
            )

        s1, s2 = adjacent_surfaces

        point_left = self._first_non_edge_vertex(
            self.surfaces[s1]["vertices"],
            edge_start,
            edge_end,
        )

        point_right = self._first_non_edge_vertex(
            self.surfaces[s2]["vertices"],
            edge_start,
            edge_end,
        )

        return self.signed_dihedral_angle(
            edge_start=edge_start,
            edge_end=edge_end,
            point_left=point_left,
            point_right=point_right,
            unit=unit,
        )

    # ------------------------------------------------------------
    # Constraint creation
    # ------------------------------------------------------------

    def _new_constraint_id(self) -> str:
        cid = f"c{self._constraint_count}"
        self._constraint_count += 1
        return cid

    @staticmethod
    def _canonical_point_pair(p1: str, p2: str) -> Tuple[str, str]:
        """
        Direction-independent key for a point pair.
        """
        return tuple(sorted((p1, p2)))

    def find_bar_length_constraint(self, p1: str, p2: str) -> Optional[str]:
        """
        Find an existing bar-length constraint between two points.

        Direction does not matter.
        """
        target_pair = self._canonical_point_pair(p1, p2)

        for cid, constraint in self.constraints.items():
            if constraint.kind != "bar_length":
                continue

            data = constraint.data
            pair = self._canonical_point_pair(data["p1"], data["p2"])

            if pair == target_pair:
                return cid

        return None

    def add_bar_length_constraint(
        self,
        p1: str,
        p2: str,
        length: Optional[float] = None,
        constraint_id: Optional[str] = None,
        merge_if_duplicate: bool = True,
        length_tol: float = 1e-9,
    ) -> str:
        """
        Add distance-preservation constraint between two points.

        Residual:
            ||x2 - x1|| - L = 0

        If length is None, the current distance is used as the reference length.

        If the same bar constraint already exists, it is merged.
        """
        if p1 == p2:
            raise ValueError("A bar-length constraint cannot use the same point twice.")

        x1 = self.point_array(p1)
        x2 = self.point_array(p2)

        if length is None:
            length = float(np.linalg.norm(x2 - x1))

        if length <= 0:
            raise ValueError(
                f"Bar length between '{p1}' and '{p2}' must be positive."
            )

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

        if constraint_id is None:
            constraint_id = self._new_constraint_id()

        if constraint_id in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' already exists.")

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="bar_length",
            data={
                "p1": p1,
                "p2": p2,
                "length": float(length),
            },
        )

        return constraint_id
    
    def add_fixed_coordinate_constraint(
        self,
        point_id: str,
        axis: Literal["x", "y", "z"],
        value: Optional[float] = None,
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Fix one coordinate of one point.

        Example:
            p0.x = 0
            p0.y = 0
            p0.z = 0
        """
        p = self.point_array(point_id)

        axis_map = {
            "x": 0,
            "y": 1,
            "z": 2,
        }

        if axis not in axis_map:
            raise ValueError("axis must be 'x', 'y', or 'z'.")

        if value is None:
            value = float(p[axis_map[axis]])

        if constraint_id is None:
            constraint_id = self._new_constraint_id()

        if constraint_id in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' already exists.")

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="fixed_coordinate",
            data={
                "point": point_id,
                "axis": axis,
                "value": float(value),
            },
        )

        return constraint_id

    def add_fixed_point_constraint(
        self,
        point_id: str,
        value: Optional[Tuple[float, float, float]] = None,
    ) -> List[str]:
        """
        Fix x, y, z coordinates of one point.

        This creates three scalar constraints.
        """
        if value is None:
            p = self.point_array(point_id)
            value = (float(p[0]), float(p[1]), float(p[2]))

        cids = []
        cids.append(
            self.add_fixed_coordinate_constraint(point_id, "x", value[0])
        )
        cids.append(
            self.add_fixed_coordinate_constraint(point_id, "y", value[1])
        )
        cids.append(
            self.add_fixed_coordinate_constraint(point_id, "z", value[2])
        )

        return cids

    def add_parallel_line_constraint(
        self,
        p1: str,
        p2: str,
        p3: str,
        p4: str,
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Constrain line p1-p2 to be parallel to line p3-p4.

        Residual:
            cross(unit(p2 - p1), unit(p4 - p3)) = 0

        This gives 3 scalar residuals, but only rank 2 independently.
        That is fine because rank detection uses the Jacobian.
        """
        for pid in [p1, p2, p3, p4]:
            if pid not in self.points:
                raise ValueError(f"Point '{pid}' does not exist.")

        if p1 == p2 or p3 == p4:
            raise ValueError("Parallel-line constraint cannot use zero-length lines.")

        if constraint_id is None:
            constraint_id = self._new_constraint_id()

        if constraint_id in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' already exists.")

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="parallel_lines",
            data={
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
            },
        )

        return constraint_id

    def add_dihedral_angle_constraint(
        self,
        edge_start: str,
        edge_end: str,
        point_left: str,
        point_right: str,
        target_angle: float,
        unit: Literal["rad", "deg"] = "rad",
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Add a signed dihedral-angle constraint.

        Args:
            edge_start, edge_end:
                Two points defining the crease axis.

            point_left:
                A point on the first panel, not on the crease.

            point_right:
                A point on the second panel, not on the crease.

            target_angle:
                Target signed angle. Use unit='deg' for degrees.

        Residual:
            wrap_to_pi(current_angle - target_angle) = 0
        """
        for pid in [edge_start, edge_end, point_left, point_right]:
            if pid not in self.points:
                raise ValueError(f"Point '{pid}' does not exist.")

        if edge_start == edge_end:
            raise ValueError("Dihedral edge cannot have zero length.")

        if point_left in {edge_start, edge_end}:
            raise ValueError("point_left must not be on the crease edge.")

        if point_right in {edge_start, edge_end}:
            raise ValueError("point_right must not be on the crease edge.")

        target_angle_rad = self._angle_to_rad(target_angle, unit=unit)

        if constraint_id is None:
            constraint_id = self._new_constraint_id()

        if constraint_id in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' already exists.")

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="dihedral_angle",
            data={
                "edge_start": edge_start,
                "edge_end": edge_end,
                "point_left": point_left,
                "point_right": point_right,
                "target_angle": target_angle_rad,
            },
        )

        return constraint_id

    def add_dihedral_angle_constraint_from_line(
        self,
        line_id: str,
        target_angle: float,
        unit: Literal["rad", "deg"] = "rad",
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Add a dihedral-angle constraint using a line shared by two surfaces.

        The line must be an actual edge of exactly two surfaces.
        """
        if line_id not in self.lines:
            raise ValueError(f"Line '{line_id}' does not exist.")

        line = self.lines[line_id]
        edge_start = line["start"]
        edge_end = line["end"]

        adjacent_surfaces = self.find_surfaces_adjacent_to_edge(
            edge_start,
            edge_end,
        )

        if len(adjacent_surfaces) != 2:
            raise ValueError(
                f"Line '{line_id}' must be shared by exactly two surfaces "
                f"to define a dihedral angle. Found {len(adjacent_surfaces)}."
            )

        s1, s2 = adjacent_surfaces

        point_left = self._first_non_edge_vertex(
            self.surfaces[s1]["vertices"],
            edge_start,
            edge_end,
        )

        point_right = self._first_non_edge_vertex(
            self.surfaces[s2]["vertices"],
            edge_start,
            edge_end,
        )

        if constraint_id is None:
            constraint_id = f"dihedral_{line_id}"

        return self.add_dihedral_angle_constraint(
            edge_start=edge_start,
            edge_end=edge_end,
            point_left=point_left,
            point_right=point_right,
            target_angle=target_angle,
            unit=unit,
            constraint_id=constraint_id,
        )

    def add_dihedral_increment_constraint_from_line(
        self,
        line_id: str,
        angle_increment: float,
        unit: Literal["rad", "deg"] = "rad",
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Add a dihedral-angle constraint by incrementing the current angle.

        This is usually more convenient than specifying an absolute angle.

        Example:
            add_dihedral_increment_constraint_from_line(
                "crease_v0",
                30,
                unit="deg",
            )

        means:
            current dihedral angle + 30 deg
        """
        current_angle = self.signed_dihedral_angle_from_line(
            line_id,
            unit="rad",
        )

        angle_increment_rad = self._angle_to_rad(angle_increment, unit=unit)

        target_angle = current_angle + angle_increment_rad
        target_angle = self._wrap_to_pi(target_angle)

        if constraint_id is None:
            constraint_id = f"dihedral_increment_{line_id}"

        return self.add_dihedral_angle_constraint_from_line(
            line_id=line_id,
            target_angle=target_angle,
            unit="rad",
            constraint_id=constraint_id,
        )

    def add_coplanar_points_constraint(
        self,
        p1: str,
        p2: str,
        p3: str,
        p4: str,
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Constrain four points to be coplanar.

        Residual:
            normalized scalar triple product = 0

        This is useful for enforcing quad-panel planarity if you do not use
        full pairwise bar constraints inside a surface.
        """
        for pid in [p1, p2, p3, p4]:
            if pid not in self.points:
                raise ValueError(f"Point '{pid}' does not exist.")

        if constraint_id is None:
            constraint_id = self._new_constraint_id()

        if constraint_id in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' already exists.")

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="coplanar_points",
            data={
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
            },
        )

        return constraint_id

    def add_coplanarity_constraints_from_surfaces(self) -> None:
        """
        Add coplanarity constraints for surfaces with 4 or more vertices.

        For a quad:
            one coplanarity constraint is added.

        For n > 4:
            constraints are added relative to the first three vertices.
        """
        for surface_id, surface in self.surfaces.items():
            vertices = surface["vertices"]

            if len(vertices) < 4:
                continue

            p1, p2, p3 = vertices[0], vertices[1], vertices[2]

            for k in range(3, len(vertices)):
                p4 = vertices[k]

                self.add_coplanar_points_constraint(
                    p1,
                    p2,
                    p3,
                    p4,
                    constraint_id=f"coplanar_{surface_id}_{p4}",
                )
                
    # ------------------------------------------------------------
    # Automatic constraints from geometry
    # ------------------------------------------------------------

    def add_bar_constraints_from_lines(
        self,
        include_kinds: Optional[List[str]] = None,
    ) -> None:
        """
        Add bar-length constraints from selected line kinds.

        By default, every line becomes a bar-length constraint.
        """
        if include_kinds is None:
            include_kinds = [
                "valley",
                "mountain",
                "side",
                "rigid",
            ]

        for line_id, line in self.lines.items():
            kind = line["kind"]

            if kind not in include_kinds:
                continue

            self.add_bar_length_constraint(
                line["start"],
                line["end"],
                constraint_id=f"bar_{line_id}",
            )

    def add_reference_frame_constraints(
        self,
        p0: str,
        p1: str,
        p2: str,
    ) -> None:
        """
        Remove global rigid-body motion using three non-collinear points.

        Constraints:
            p0: fix x, y, z
            p1: fix y, z
            p2: fix z

        Total scalar constraints:
            3 + 2 + 1 = 6

        This removes 3 translations and 3 rotations.
        """
        self.add_fixed_coordinate_constraint(p0, "x")
        self.add_fixed_coordinate_constraint(p0, "y")
        self.add_fixed_coordinate_constraint(p0, "z")

        self.add_fixed_coordinate_constraint(p1, "y")
        self.add_fixed_coordinate_constraint(p1, "z")

        self.add_fixed_coordinate_constraint(p2, "z")

    def add_panel_rigidity_constraints_from_surfaces(self) -> None:
        """
        Add bar-length constraints inside each surface to make every panel rigid.

        For a triangle:
            all 3 pairwise distances are fixed.

        For a quadrilateral:
            all 6 pairwise distances are fixed.

        This is simple and robust for the first version.
        Duplicate bars are automatically merged by add_bar_length_constraint().
        """
        for surface_id, surface in self.surfaces.items():
            vertices = surface["vertices"]

            n = len(vertices)

            for i in range(n):
                for j in range(i + 1, n):
                    p1 = vertices[i]
                    p2 = vertices[j]

                    self.add_bar_length_constraint(
                        p1,
                        p2,
                        constraint_id=f"panel_{surface_id}_{p1}_{p2}",
                        merge_if_duplicate=True,
                    )

    # ------------------------------------------------------------
    # Residual evaluation
    # ------------------------------------------------------------

    def residual_vector(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate all scalar constraint residuals.

        If X is provided, residuals are evaluated at X without permanently
        modifying the model.
        """
        old_X = None

        if X is not None:
            old_X = self.get_coordinate_vector()
            self.set_coordinate_vector(X)

        residuals = []

        for constraint in self.constraints.values():
            if constraint.kind == "bar_length":
                residuals.append(
                    self._residual_bar_length(constraint.data)
                )

            elif constraint.kind == "fixed_coordinate":
                residuals.append(
                    self._residual_fixed_coordinate(constraint.data)
                )

            elif constraint.kind == "parallel_lines":
                residuals.extend(
                    self._residual_parallel_lines(constraint.data)
                )

            elif constraint.kind == "dihedral_angle":
                residuals.append(
                    self._residual_dihedral_angle(constraint.data)
                )

            elif constraint.kind == "coplanar_points":
                residuals.append(
                    self._residual_coplanar_points(constraint.data)
                )

            elif constraint.kind == "fixed_point":
                raise NotImplementedError(
                    "fixed_point is implemented as three fixed_coordinate constraints."
                )

            else:
                raise ValueError(
                    f"Unknown constraint kind '{constraint.kind}'."
                )

        if X is not None:
            self.set_coordinate_vector(old_X)

        return np.array(residuals, dtype=float)

    def _residual_bar_length(self, data: dict) -> float:
        p1 = self.point_array(data["p1"])
        p2 = self.point_array(data["p2"])
        length = data["length"]

        return float(np.linalg.norm(p2 - p1) - length)

    def _residual_fixed_coordinate(self, data: dict) -> float:
        point = self.point_array(data["point"])
        axis = data["axis"]
        value = data["value"]

        axis_map = {
            "x": 0,
            "y": 1,
            "z": 2,
        }

        return float(point[axis_map[axis]] - value)

    def _residual_parallel_lines(self, data: dict) -> np.ndarray:
        p1 = self.point_array(data["p1"])
        p2 = self.point_array(data["p2"])
        p3 = self.point_array(data["p3"])
        p4 = self.point_array(data["p4"])

        u = self._unit_vector(p2 - p1, name="first parallel line")
        v = self._unit_vector(p4 - p3, name="second parallel line")

        return np.cross(u, v)

    def _residual_dihedral_angle(self, data: dict) -> float:
        current_angle = self.signed_dihedral_angle(
            edge_start=data["edge_start"],
            edge_end=data["edge_end"],
            point_left=data["point_left"],
            point_right=data["point_right"],
            unit="rad",
        )

        target_angle = data["target_angle"]

        return self._wrap_to_pi(current_angle - target_angle)

    def _residual_coplanar_points(self, data: dict) -> float:
        p1 = self.point_array(data["p1"])
        p2 = self.point_array(data["p2"])
        p3 = self.point_array(data["p3"])
        p4 = self.point_array(data["p4"])

        a = p2 - p1
        b = p3 - p1
        c = p4 - p1

        numerator = float(np.dot(np.cross(a, b), c))

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
            * np.linalg.norm(c)
        )

        if denominator < 1e-12:
            raise ValueError(
                "Coplanar constraint has near-degenerate point configuration."
            )

        return float(numerator / denominator)

    # ------------------------------------------------------------
    # Jacobian and mobility
    # ------------------------------------------------------------

    def numerical_jacobian(
        self,
        X: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        Compute numerical Jacobian of residuals with respect to coordinates.

        J[i, j] = d residual_i / d X_j
        """
        if X is None:
            X = self.get_coordinate_vector()

        X = np.asarray(X, dtype=float)
        f0 = self.residual_vector(X)

        J = np.zeros((f0.size, X.size), dtype=float)

        for j in range(X.size):
            X_plus = X.copy()
            X_minus = X.copy()

            X_plus[j] += eps
            X_minus[j] -= eps

            f_plus = self.residual_vector(X_plus)
            f_minus = self.residual_vector(X_minus)

            J[:, j] = (f_plus - f_minus) / (2.0 * eps)

        return J

    def constraint_rank(
        self,
        tol: float = 1e-8,
    ) -> int:
        """
        Return numerical rank of the constraint Jacobian.
        """
        J = self.numerical_jacobian()
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

    def is_locally_unique(
        self,
        tol: float = 1e-8,
    ) -> bool:
        """
        Return True if local mobility is zero.
        """
        return self.mobility(tol=tol) == 0

    def print_constraint_summary(self, tol: float = 1e-8) -> None:
        """
        Print basic constraint information.
        """
        J = self.numerical_jacobian()
        rank = np.linalg.matrix_rank(J, tol=tol)
        mobility = self.num_variables() - rank

        print("Constraint summary")
        print("------------------")
        print(f"Points:              {self.num_points()}")
        print(f"Variables:           {self.num_variables()}")
        print(f"Scalar constraints:  {len(self.constraints)}")
        print(f"Jacobian rank:       {rank}")
        print(f"Mobility:            {mobility}")

        if mobility == 0:
            print("Status:              locally unique")
        elif mobility > 0:
            print("Status:              underconstrained / mechanism remains")
        else:
            print("Status:              overconstrained or inconsistent")

    # ------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------

    def solve(
        self,
        X0: Optional[np.ndarray] = None,
        update_model: bool = True,
        tol: float = 1e-10,
        rank_tol: float = 1e-8,
        max_nfev: int = 2000,
        verbose: int = 0,
    ) -> SolveReport:
        """
        Solve the nonlinear constraint system.

        The solver finds X such that:

            residual_vector(X) ≈ 0

        Args:
            X0:
                Initial guess. If None, use current model coordinates.
            update_model:
                If True, update self.points using the solved coordinates.
            tol:
                Tolerance passed to scipy least_squares.
            rank_tol:
                Tolerance for Jacobian rank estimation.
            max_nfev:
                Maximum number of function evaluations.
            verbose:
                scipy least_squares verbosity.
                0 -> silent
                1 -> final report
                2 -> iteration report

        Returns:
            SolveReport
        """
        if len(self.constraints) == 0:
            raise ValueError("No constraints have been added.")

        if X0 is None:
            X0 = self.get_coordinate_vector()

        X0 = np.asarray(X0, dtype=float)

        if X0.size != self.num_variables():
            raise ValueError(
                f"Expected X0 size {self.num_variables()}, but got {X0.size}."
            )

        result = least_squares(
            fun=lambda X: self.residual_vector(X),
            x0=X0,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=max_nfev,
            verbose=verbose,
        )

        residuals = self.residual_vector(result.x)
        residual_norm = float(np.linalg.norm(residuals))
        max_abs_residual = float(np.max(np.abs(residuals))) if residuals.size else 0.0

        J = self.numerical_jacobian(result.x)
        rank = int(np.linalg.matrix_rank(J, tol=rank_tol))
        mobility = self.num_variables() - rank

        if update_model:
            self.set_coordinate_vector(result.x)

        return SolveReport(
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            cost=float(result.cost),
            residual_norm=residual_norm,
            max_abs_residual=max_abs_residual,
            rank=rank,
            mobility=mobility,
            x=result.x.copy(),
        )

    def solve_from_perturbed_state(
        self,
        perturbation_scale: float = 1e-3,
        seed: Optional[int] = None,
        update_model: bool = True,
        **solve_kwargs,
    ) -> SolveReport:
        """
        Solve after applying a small random perturbation to the current coordinates.

        This is useful for testing whether the constraints pull the system back
        to a valid configuration.

        Args:
            perturbation_scale:
                Size of random coordinate perturbation.
            seed:
                Random seed.
            update_model:
                Whether to update the model after solving.
            solve_kwargs:
                Extra arguments passed to solve().
        """
        rng = np.random.default_rng(seed)
        X = self.get_coordinate_vector()
        X0 = X + perturbation_scale * rng.standard_normal(X.shape)

        return self.solve(
            X0=X0,
            update_model=update_model,
            **solve_kwargs,
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
        print(f"Jacobian rank:      {report.rank}")
        print(f"Mobility:           {report.mobility}")

        if report.mobility == 0:
            print("Status:             locally unique")
        elif report.mobility > 0:
            print("Status:             underconstrained / mechanism remains")
        else:
            print("Status:             overconstrained or inconsistent")

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    def draw(
        self,
        show_points: bool = True,
        show_point_ids: bool = True,
        show_line_ids: bool = False,
        show_surface_ids: bool = False,
        show_surfaces: bool = True,
        equal_axis: bool = True,
        figsize: Tuple[float, float] = (8, 7),
        view: Tuple[float, float] = (25, -60),
    ) -> None:
        """
        Draw the current 3D configuration using matplotlib.

        Args:
            show_points:
                If True, draw point markers.
            show_point_ids:
                If True, show point IDs.
            show_line_ids:
                If True, show line IDs.
            show_surface_ids:
                If True, show surface IDs.
            show_surfaces:
                If True, draw translucent panel surfaces.
            equal_axis:
                If True, use equal scaling for x, y, z.
            figsize:
                Matplotlib figure size.
            view:
                3D view angle as (elevation, azimuth).
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Draw surfaces first
        if show_surfaces:
            for surface_id, surface in self.surfaces.items():
                vertices = surface["vertices"]

                coords = [
                    self.point_array(pid)
                    for pid in vertices
                ]

                poly = Poly3DCollection(
                    [coords],
                    alpha=0.18,
                    facecolor="lightgray",
                    edgecolor="black",
                    linewidth=0.8,
                )
                ax.add_collection3d(poly)

                if show_surface_ids:
                    center = np.mean(np.array(coords), axis=0)
                    ax.text(
                        center[0],
                        center[1],
                        center[2],
                        surface_id,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        # Draw lines
        for line_id, line in self.lines.items():
            p0 = self.point_array(line["start"])
            p1 = self.point_array(line["end"])

            style = self._line_style_3d(line["kind"])

            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                **style,
            )

            if show_line_ids:
                mid = 0.5 * (p0 + p1)
                ax.text(
                    mid[0],
                    mid[1],
                    mid[2],
                    line_id,
                    fontsize=8,
                )

        # Draw points
        if show_points:
            for point_id, point in self.points.items():
                ax.scatter(
                    point.x,
                    point.y,
                    point.z,
                    s=25,
                    color="black",
                )

                if show_point_ids:
                    ax.text(
                        point.x,
                        point.y,
                        point.z,
                        f" {point_id}",
                        fontsize=8,
                    )

        if equal_axis:
            self._set_axes_equal_3d(ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.view_init(elev=view[0], azim=view[1])
        ax.grid(True, alpha=0.3)

        plt.show()

    @staticmethod
    def _line_style_3d(kind: str) -> dict:
        """
        Visual convention:
            valley       -> blue dashed
            mountain     -> red dash-dot
            side         -> black solid thick
            rigid        -> black solid thin
            construction -> gray dotted
        """
        if kind == "valley":
            return {"color": "blue", "linestyle": "--", "linewidth": 1.8}

        if kind == "mountain":
            return {"color": "red", "linestyle": "-.", "linewidth": 1.8}

        if kind == "side":
            return {"color": "black", "linestyle": "-", "linewidth": 2.2}

        if kind == "rigid":
            return {"color": "black", "linestyle": "-", "linewidth": 1.0}

        if kind == "construction":
            return {"color": "gray", "linestyle": ":", "linewidth": 1.0}

        raise ValueError(f"Unknown line kind: {kind}")

    def _set_axes_equal_3d(self, ax) -> None:
        """
        Make x, y, z axes have equal scale.

        Matplotlib 3D plots do not use equal axis scaling by default.
        This helper prevents geometric distortion.
        """
        xs = [p.x for p in self.points.values()]
        ys = [p.y for p in self.points.values()]
        zs = [p.z for p in self.points.values()]

        if not xs:
            return

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        z_mid = 0.5 * (z_min + z_max)

        max_range = max(
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )

        if max_range == 0:
            max_range = 1.0

        radius = 0.5 * max_range

        ax.set_xlim(x_mid - radius, x_mid + radius)
        ax.set_ylim(y_mid - radius, y_mid + radius)
        ax.set_zlim(z_mid - radius, z_mid + radius)

        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------

    @staticmethod
    def _angle_to_rad(angle: float, unit: Literal["rad", "deg"] = "rad") -> float:
        """
        Convert angle to radians.
        """
        if unit == "rad":
            return float(angle)

        if unit == "deg":
            return float(np.deg2rad(angle))

        raise ValueError("unit must be 'rad' or 'deg'.")

    @staticmethod
    def _rad_to_angle(angle_rad: float, unit: Literal["rad", "deg"] = "rad") -> float:
        """
        Convert radians to requested unit.
        """
        if unit == "rad":
            return float(angle_rad)

        if unit == "deg":
            return float(np.rad2deg(angle_rad))

        raise ValueError("unit must be 'rad' or 'deg'.")

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """
        Wrap angle to (-pi, pi].

        This prevents angle residuals from jumping by 2*pi.
        """
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _unit_vector(v: np.ndarray, name: str = "vector", eps: float = 1e-12) -> np.ndarray:
        """
        Return normalized vector.

        Raises an error if the vector is too small.
        """
        norm = float(np.linalg.norm(v))

        if norm < eps:
            raise ValueError(f"{name} has near-zero length.")

        return v / norm

    @staticmethod
    def _surface_has_edge(vertices: List[str], p1: str, p2: str) -> bool:
        """
        Check whether a surface contains the edge p1-p2.

        The edge is treated as direction-independent.
        """
        n = len(vertices)

        for i in range(n):
            a = vertices[i]
            b = vertices[(i + 1) % n]

            if {a, b} == {p1, p2}:
                return True

        return False

    @staticmethod
    def _first_non_edge_vertex(vertices: List[str], p1: str, p2: str) -> str:
        """
        Return the first vertex in a surface that is not one of the edge vertices.
        """
        for pid in vertices:
            if pid not in {p1, p2}:
                return pid

        raise ValueError(
            f"Surface {vertices} does not contain a non-edge vertex "
            f"relative to edge {p1}-{p2}."
        )

    def find_surfaces_adjacent_to_edge(self, p1: str, p2: str) -> List[str]:
        """
        Find surfaces that contain the edge p1-p2.

        Returns:
            List of surface IDs.
        """
        adjacent = []

        for surface_id, surface in self.surfaces.items():
            vertices = surface["vertices"]

            if self._surface_has_edge(vertices, p1, p2):
                adjacent.append(surface_id)

        return adjacent