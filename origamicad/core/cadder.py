from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal, Tuple
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations

from .simple_hexagon_mixin import SimpleHexagonMixin
from .visualization import CadVisualizationMixin

ConstraintKind = Literal[
    "bar_length",
    "fixed_coordinate",
    "fixed_point",
    "parallel_lines",
    "parallel_surfaces",
    "dihedral_angle",
    "dihedral_cos",
    "dihedral_signed_increment",
    "coplanar_points",
    "horizontal_surface",
    "surface_z_value",
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

class Cadder(CadVisualizationMixin, SimpleHexagonMixin):
    """
    Constraint-based kinematic model for origami/cellular structures.

    This class is responsible for:
        1. storing 3D point coordinates
        2. storing constraints
        3. evaluating constraint residuals
        4. estimating local mobility by Jacobian rank

    It does not yet solve a folded configuration.
    """

    def __init__(self, unit: str = "mm"):
        self.unit = str(unit)
        self.points: Dict[str, Point3D] = {}
        self.lines: dict = {}
        self.surfaces: dict = {}
        self.constraints: Dict[str, Constraint] = {}

        self.hex_units = []

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
        model = cls(unit=getattr(drawer, "unit", "mm"))

        for pid, point in drawer.points.items():
            model.add_point(pid, point.x, point.y, 0.0)

        model.lines = drawer.to_dict()["lines"]
        model.surfaces = drawer.to_dict()["surfaces"]

        model.hex_units = getattr(drawer, "hex_units", [])

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
        model.hex_units = metadata.get("hex_units", [])

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
        drawer._point_count = len(drawer.points)
        drawer._line_count = len(drawer.lines)
        drawer._surface_count = len(drawer.surfaces)

        return drawer

    def project_to_xy(self, point_tol: float = 1e-9):
        """Alias for to_2d_drawer()."""
        return self.to_2d_drawer(point_tol=point_tol)

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

    def add_fixed_surface_constraint(self, surface_id: str) -> List[str]:
        """
        Fix one surface as the reference panel.

        This removes global rigid-body motion by fixing three non-collinear
        points on the surface:

            p0: x, y, z fixed
            p1: y, z fixed
            p2: z fixed

        Total scalar constraints:
            6
        """
        p0, p1, p2 = self._non_collinear_triplet_from_surface(surface_id)

        cids = []

        cids.append(self.add_fixed_coordinate_constraint(p0, "x"))
        cids.append(self.add_fixed_coordinate_constraint(p0, "y"))
        cids.append(self.add_fixed_coordinate_constraint(p0, "z"))

        cids.append(self.add_fixed_coordinate_constraint(p1, "y"))
        cids.append(self.add_fixed_coordinate_constraint(p1, "z"))

        cids.append(self.add_fixed_coordinate_constraint(p2, "z"))

        return cids

    def add_parallel_surface_constraint(
        self,
        surface1_id: str,
        surface2_id: str,
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Constrain two surfaces to be parallel.

        Residual:
            cross(n1, n2) = 0

        This gives 3 residual components, but rank is usually 2.
        """
        self.surface_normal(surface1_id)
        self.surface_normal(surface2_id)

        if constraint_id is None:
            constraint_id = f"parallel_surface_{surface1_id}_{surface2_id}"

        if constraint_id in self.constraints:
            raise ValueError(f"Constraint '{constraint_id}' already exists.")

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="parallel_surfaces",
            data={
                "surface1": surface1_id,
                "surface2": surface2_id,
            },
        )

        return constraint_id

    def add_parallel_triangle_surface_constraints(
        self,
        reference_surface_id: Optional[str] = None,
    ) -> List[str]:
        """
        Make all triangular surfaces parallel to one reference triangle surface.
        """
        triangle_ids = [
            sid for sid in self.surfaces.keys()
            if len(self._surface_vertices(sid)) == 3
        ]

        if len(triangle_ids) < 2:
            return []

        if reference_surface_id is None:
            reference_surface_id = triangle_ids[0]

        if reference_surface_id not in triangle_ids:
            raise ValueError(
                f"Reference surface '{reference_surface_id}' is not a triangle surface."
            )

        cids = []

        for sid in triangle_ids:
            if sid == reference_surface_id:
                continue

            cid = self.add_parallel_surface_constraint(
                reference_surface_id,
                sid,
                constraint_id=f"parallel_triangles_{reference_surface_id}_{sid}",
            )
            cids.append(cid)

        return cids

    def add_mountain_valley_dihedral_constraints(
        self,
        target_dihedral: float = 110.0,
        unit: Literal["rad", "deg"] = "deg",
        valley_sign: int = +1,
        only_triangle_quad: bool = True,
    ) -> List[str]:
        """
        Add dihedral constraints to all mountain/valley crease lines.

        Interpretation:
            target_dihedral = unsigned obtuse dihedral angle between panels.

        Example:
            target_dihedral = 110 deg

        Since flat state is treated as 180 deg, the folding increment is:

            fold_amount = 180 deg - 110 deg = 70 deg

        Valley and mountain creases receive opposite signed increments.

        Args:
            target_dihedral:
                Desired obtuse dihedral angle.
            unit:
                'deg' or 'rad'.
            valley_sign:
                +1 means valley uses positive signed rotation.
                -1 flips the convention.
            only_triangle_quad:
                If True, only apply to crease edges shared by one triangle
                and one quadrilateral surface.
        """
        theta = self._angle_to_rad(target_dihedral, unit=unit)

        if not (0.0 < theta < np.pi):
            raise ValueError("target_dihedral must be between 0 and 180 degrees.")

        fold_amount = np.pi - theta

        if valley_sign not in {+1, -1}:
            raise ValueError("valley_sign must be +1 or -1.")

        cids = []

        for line_id in self.lines.keys():
            edge_start, edge_end, kind = self._line_info(line_id)

            if kind not in {"valley", "mountain"}:
                continue

            if only_triangle_quad:
                pairs = self.find_triangle_quad_pairs_adjacent_to_edge(
                    edge_start,
                    edge_end,
                )
            else:
                adjacent = self.find_surfaces_adjacent_to_edge(
                    edge_start,
                    edge_end,
                )

                if len(adjacent) != 2:
                    continue

                pairs = [(adjacent[0], adjacent[1])]

            if len(pairs) == 0:
                continue

            crease_sign = valley_sign if kind == "valley" else -valley_sign

            for pair_index, (tri_id, quad_id) in enumerate(pairs):
                tri_vertices = self._surface_vertices(tri_id)
                quad_vertices = self._surface_vertices(quad_id)

                triangle_point = self._first_non_edge_vertex(
                    tri_vertices,
                    edge_start,
                    edge_end,
                )

                quad_point = self._first_non_edge_vertex(
                    quad_vertices,
                    edge_start,
                    edge_end,
                )

                current_angle = self.signed_dihedral_angle(
                    edge_start=edge_start,
                    edge_end=edge_end,
                    point_left=triangle_point,
                    point_right=quad_point,
                    unit="rad",
                )

                target_angle = current_angle + crease_sign * fold_amount
                target_angle = self._wrap_to_pi(target_angle)

                cid = self.add_dihedral_angle_constraint(
                    edge_start=edge_start,
                    edge_end=edge_end,
                    point_left=triangle_point,
                    point_right=quad_point,
                    target_angle=target_angle,
                    unit="rad",
                    constraint_id=(
                        f"dihedral_{line_id}_{tri_id}_{quad_id}_{pair_index}"
                    ),
                )

                cids.append(cid)

        return cids

    def add_horizontal_surface_constraint(
        self,
        surface_id: str,
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Constrain a surface to be horizontal.

        For vertices v0, v1, v2, ...
        residuals are:

            z1 - z0 = 0
            z2 - z0 = 0
            ...

        This keeps the surface parallel to the xy-plane, but does not fix
        its height.
        """
        vertices = self._surface_vertices(surface_id)

        if len(vertices) < 3:
            raise ValueError(f"Surface '{surface_id}' has fewer than 3 vertices.")

        if constraint_id is None:
            constraint_id = f"horizontal_{surface_id}"

        if constraint_id in self.constraints:
            return constraint_id

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="horizontal_surface",
            data={
                "surface": surface_id,
            },
        )

        return constraint_id

    def add_surface_z_value_constraint(
        self,
        surface_id: str,
        z_value: float = 0.0,
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Fix all vertices of a surface to a given z value.

        This is stronger than horizontal_surface.

        For your current interpretation:
            valley triangle -> z = 0
            mountain triangle -> horizontal only, z not fixed
        """
        vertices = self._surface_vertices(surface_id)

        if len(vertices) < 3:
            raise ValueError(f"Surface '{surface_id}' has fewer than 3 vertices.")

        if constraint_id is None:
            constraint_id = f"z_value_{surface_id}"

        if constraint_id in self.constraints:
            return constraint_id

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="surface_z_value",
            data={
                "surface": surface_id,
                "z_value": float(z_value),
            },
        )

        return constraint_id

    def add_dihedral_cos_constraint(
        self,
        edge_start: str,
        edge_end: str,
        point_left: str,
        point_right: str,
        target_angle: float,
        unit: Literal["rad", "deg"] = "deg",
        constraint_id: Optional[str] = None,
    ) -> str:
        """
        Constrain the dihedral angle magnitude.

        Residual:
            cos(current_angle) - cos(target_angle) = 0

        Example:
            target_angle = 110 deg
        """
        for pid in [edge_start, edge_end, point_left, point_right]:
            if pid not in self.points:
                raise ValueError(f"Point '{pid}' does not exist.")

        if unit == "deg":
            target_angle_rad = float(np.deg2rad(target_angle))
        elif unit == "rad":
            target_angle_rad = float(target_angle)
        else:
            raise ValueError("unit must be 'deg' or 'rad'.")

        if not (0.0 < target_angle_rad < np.pi):
            raise ValueError("target_angle must be between 0 and 180 degrees.")

        if constraint_id is None:
            constraint_id = self._new_constraint_id()

        if constraint_id in self.constraints:
            return constraint_id

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="dihedral_cos",
            data={
                "edge_start": edge_start,
                "edge_end": edge_end,
                "point_left": point_left,
                "point_right": point_right,
                "target_cos": float(np.cos(target_angle_rad)),
            },
        )

        return constraint_id

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
        """
        Add a signed dihedral increment constraint.

        The constraint stores the current signed dihedral angle as the
        reference flat/initial angle.

        Residual:
            wrap_to_pi((current_angle - initial_angle) - target_increment) = 0

        This is better than cos(dihedral) because it can distinguish the
        mountain/valley branch.
        """
        for pid in [edge_start, edge_end, point_left, point_right]:
            if pid not in self.points:
                raise ValueError(f"Point '{pid}' does not exist.")

        if edge_start == edge_end:
            raise ValueError("Dihedral edge cannot have zero length.")

        target_increment_rad = self._angle_to_rad(target_increment, unit=unit)

        initial_angle = self.signed_dihedral_angle(
            edge_start=edge_start,
            edge_end=edge_end,
            point_left=point_left,
            point_right=point_right,
            unit="rad",
        )

        if constraint_id is None:
            constraint_id = self._new_constraint_id()

        if constraint_id in self.constraints:
            return constraint_id

        self.constraints[constraint_id] = Constraint(
            id=constraint_id,
            kind="dihedral_signed_increment",
            data={
                "edge_start": edge_start,
                "edge_end": edge_end,
                "point_left": point_left,
                "point_right": point_right,
                "initial_angle": float(initial_angle),
                "target_increment": float(target_increment_rad),
                "sign": int(sign),
                "crease_kind": crease_kind,
            },
        )

        return constraint_id

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

        try:
            residuals = [
                self._residual_for_constraint(constraint)
                for constraint in self.constraints.values()
            ]
        finally:
            if old_X is not None:
                self.set_coordinate_vector(old_X)

        if not residuals:
            return np.array([], dtype=float)

        return np.concatenate(residuals)

    def _residual_for_constraint(self, constraint: Constraint) -> np.ndarray:
        if constraint.kind == "fixed_point":
            raise NotImplementedError(
                "fixed_point is implemented as three fixed_coordinate constraints."
            )

        handlers = {
            "bar_length": self._residual_bar_length,
            "fixed_coordinate": self._residual_fixed_coordinate,
            "parallel_lines": self._residual_parallel_lines,
            "parallel_surfaces": self._residual_parallel_surfaces,
            "dihedral_angle": self._residual_dihedral_angle,
            "dihedral_cos": self._residual_dihedral_cos,
            "dihedral_signed_increment": self._residual_dihedral_signed_increment,
            "coplanar_points": self._residual_coplanar_points,
            "horizontal_surface": self._residual_horizontal_surface,
            "surface_z_value": self._residual_surface_z_value,
        }

        try:
            value = handlers[constraint.kind](constraint.data)
        except KeyError:
            raise ValueError(f"Unknown constraint kind '{constraint.kind}'.") from None

        return np.asarray(value, dtype=float).reshape(-1)

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

    def _residual_parallel_surfaces(self, data: dict) -> np.ndarray:
        n1 = self.surface_normal(data["surface1"])
        n2 = self.surface_normal(data["surface2"])

        return np.cross(n1, n2)

    def _residual_horizontal_surface(self, data: dict) -> np.ndarray:
        surface_id = data["surface"]
        vertices = self._surface_vertices(surface_id)

        z0 = self.point_array(vertices[0])[2]

        residuals = []

        for pid in vertices[1:]:
            zi = self.point_array(pid)[2]
            residuals.append(zi - z0)

        return np.array(residuals, dtype=float)

    def _residual_surface_z_value(self, data: dict) -> np.ndarray:
        surface_id = data["surface"]
        z_value = data["z_value"]

        vertices = self._surface_vertices(surface_id)

        residuals = []

        for pid in vertices:
            zi = self.point_array(pid)[2]
            residuals.append(zi - z_value)

        return np.array(residuals, dtype=float)

    def _residual_dihedral_cos(self, data: dict) -> float:
        current_cos = self.dihedral_cos(
            edge_start=data["edge_start"],
            edge_end=data["edge_end"],
            point_left=data["point_left"],
            point_right=data["point_right"],
        )

        return float(current_cos - data["target_cos"])

    def _residual_dihedral_signed_increment(self, data: dict) -> float:
        current_angle = self.signed_dihedral_angle(
            edge_start=data["edge_start"],
            edge_end=data["edge_end"],
            point_left=data["point_left"],
            point_right=data["point_right"],
            unit="rad",
        )

        initial_angle = data["initial_angle"]
        target_increment = data["target_increment"]

        actual_increment = self._wrap_to_pi(current_angle - initial_angle)

        return self._wrap_to_pi(actual_increment - target_increment)

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
        residuals = self.residual_vector()
        J = self.numerical_jacobian()
        rank = np.linalg.matrix_rank(J, tol=tol)
        mobility = self.num_variables() - rank

        print("Constraint summary")
        print("------------------")
        print(f"Points:              {self.num_points()}")
        print(f"Variables:           {self.num_variables()}")
        print(f"Constraint objects:  {len(self.constraints)}")
        print(f"Scalar residuals:    {residuals.size}")
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

        # SciPy already computed this Jacobian while solving. Recomputing it
        # coordinate-by-coordinate made every continuation step need hundreds
        # of redundant residual evaluations.
        rank = int(np.linalg.matrix_rank(result.jac, tol=rank_tol))
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

    def save_step(self, filename: str, thickness: float = 0.0) -> None:
        """Save panel geometry as a faceted STEP surface or solid model."""
        from ..io.cad_export import save_step

        save_step(self, filename, thickness=thickness)

    def save_cad(self, filename: str, thickness: float = 0.0) -> None:
        """Export based on a .json, .stl, .step, or .stp extension."""
        from ..io.cad_export import save_cad

        save_cad(self, filename, thickness=thickness)

    def save_xy_dxf(
        self,
        filename: str,
        include_creases: bool = True,
        crease_style: Literal["solid", "dashed"] = "dashed",
        include_construction: bool = False,
        include_rigid: bool = True,
        include_side: bool = True,
        point_tol: float = 1e-9,
    ):
        """
        Project the current 3D structure to x-y and save it as DXF.
        """
        return self.to_2d_drawer(point_tol=point_tol).save_dxf(
            filename,
            include_creases=include_creases,
            crease_style=crease_style,
            include_construction=include_construction,
            include_rigid=include_rigid,
            include_side=include_side,
        )

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
    def _rad_to_angle(
        angle: float,
        unit: Literal["rad", "deg"] = "rad",
    ) -> float:
        if unit == "rad":
            return float(angle)

        if unit == "deg":
            return float(np.rad2deg(angle))

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

    def surface_normal(self, surface_id: str) -> np.ndarray:
        """
        Return unit normal vector of a surface.

        The normal direction depends on vertex ordering, but parallel constraints
        only use cross(n1, n2) = 0, so sign is not important there.
        """
        p1, p2, p3 = self._non_collinear_triplet_from_surface(surface_id)

        x1 = self.point_array(p1)
        x2 = self.point_array(p2)
        x3 = self.point_array(p3)

        normal = np.cross(x2 - x1, x3 - x1)

        return self._unit_vector(normal, name=f"surface normal of {surface_id}")

    @staticmethod
    def _surface_has_edge(vertices: List[str], p1: str, p2: str) -> bool:
        """
        Check whether a surface contains edge p1-p2.

        The edge is direction-independent.
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
        Return one vertex in the surface that is not on the edge p1-p2.
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
        Find all surfaces containing edge p1-p2.
        """
        adjacent = []

        for surface_id in self.surfaces.keys():
            vertices = self._surface_vertices(surface_id)

            if self._surface_has_edge(vertices, p1, p2):
                adjacent.append(surface_id)

        return adjacent

    def find_triangle_quad_pairs_adjacent_to_edge(
        self,
        p1: str,
        p2: str,
    ) -> List[Tuple[str, str]]:
        """
        Find triangle-parallelogram surface pairs adjacent to edge p1-p2.

        Returns:
            List of (triangle_surface_id, quad_surface_id).
        """
        adjacent = self.find_surfaces_adjacent_to_edge(p1, p2)

        triangle_surfaces = [
            sid for sid in adjacent
            if len(self._surface_vertices(sid)) == 3
        ]

        quad_surfaces = [
            sid for sid in adjacent
            if len(self._surface_vertices(sid)) == 4
        ]

        pairs = []

        for tri_id in triangle_surfaces:
            for quad_id in quad_surfaces:
                pairs.append((tri_id, quad_id))

        return pairs

    def dihedral_cos(
        self,
        edge_start: str,
        edge_end: str,
        point_left: str,
        point_right: str,
    ) -> float:
        """
        Compute cosine of the dihedral angle.

        This enforces the obtuse angle magnitude, e.g. 110 degrees,
        without relying on signed angle convention.

        This is more stable for the first implementation.
        """
        a = self.point_array(edge_start)
        b = self.point_array(edge_end)
        c = self.point_array(point_left)
        d = self.point_array(point_right)

        axis = self._unit_vector(b - a, name="dihedral axis")

        u = c - a
        v = d - a

        u_perp = u - np.dot(u, axis) * axis
        v_perp = v - np.dot(v, axis) * axis

        u_hat = self._unit_vector(u_perp, name="projected left vector")
        v_hat = self._unit_vector(v_perp, name="projected right vector")

        return float(np.dot(u_hat, v_hat))
