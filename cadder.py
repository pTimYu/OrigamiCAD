from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple
import numpy as np


ConstraintKind = Literal[
    "bar_length",
    "fixed_coordinate",
    "fixed_point",
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
    # Constraint creation
    # ------------------------------------------------------------

    def _new_constraint_id(self) -> str:
        cid = f"c{self._constraint_count}"
        self._constraint_count += 1
        return cid

    def add_bar_length_constraint(
        self,
        p1: str,
        p2: str,
        length: Optional[float] = None,
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Add distance-preservation constraint between two points.

        Residual:
            ||x2 - x1||^2 - L^2 = 0

        If length is None, the current distance is used as the reference length.
        """
        x1 = self.point_array(p1)
        x2 = self.point_array(p2)

        if length is None:
            length = float(np.linalg.norm(x2 - x1))

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

        return float(np.dot(p2 - p1, p2 - p1) - length**2)

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