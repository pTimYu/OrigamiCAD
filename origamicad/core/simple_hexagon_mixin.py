from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np


class SimpleHexagonMixin:
    """Simple-hexagon specific setup, continuation, and diagnostics."""

    def _triangle_surface_ids(self) -> list[str]:
        return [
            sid for sid in self.surfaces
            if len(self._surface_vertices(sid)) == 3
        ]

    def add_simple_hexagon_dihedral_constraints_from_metadata(
        self,
        target_dihedral: float = 110.0,
        unit: Literal["rad", "deg"] = "deg",
        valley_sign: int = +1,
    ) -> dict:
        if not self.hex_units:
            raise ValueError(
                "No hex-unit metadata found. Make sure simple_hexagon.py stores "
                "pattern.hex_units before creating the Cadder model."
            )

        if valley_sign not in {+1, -1}:
            raise ValueError("valley_sign must be +1 or -1.")

        theta = self._angle_to_rad(target_dihedral, unit=unit)
        if not (0.0 < theta < np.pi):
            raise ValueError("target_dihedral must be between 0 and 180 degrees.")

        added, skipped, duplicate = [], [], []
        used_keys = set()
        fold_amount = np.pi - theta

        for unit_data in self.hex_units:
            unit_count = unit_data.get("count", "unknown")

            for crease_data in unit_data.get("local_creases", []):
                edge_start, edge_end = crease_data["edge"]
                tri_id = crease_data["triangle"]
                quad_id = crease_data["quad"]
                local_index = crease_data["local_index"]
                side = crease_data["side"]
                crease_kind = crease_data["kind"]
                label = (unit_count, local_index, side)

                if tri_id not in self.surfaces:
                    skipped.append((*label, "missing triangle", tri_id))
                    continue
                if quad_id not in self.surfaces:
                    skipped.append((*label, "missing quad", quad_id))
                    continue
                if edge_start not in self.points or edge_end not in self.points:
                    skipped.append((*label, "missing edge points"))
                    continue

                key = (tuple(sorted((edge_start, edge_end))), tri_id, quad_id)
                if key in used_keys:
                    duplicate.append((*label, tri_id, quad_id))
                    continue
                used_keys.add(key)

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
                edge_start, edge_end = self._orient_flat_crease_triangle_right(
                    edge_start,
                    edge_end,
                    triangle_point,
                )

                if crease_kind not in {"valley", "mountain"}:
                    skipped.append((*label, "not mountain/valley", crease_kind))
                    continue

                sign = valley_sign if crease_kind == "valley" else -valley_sign
                added.append(
                    self.add_dihedral_signed_increment_constraint(
                        edge_start=edge_start,
                        edge_end=edge_end,
                        point_left=triangle_point,
                        point_right=quad_point,
                        target_increment=sign * fold_amount,
                        unit="rad",
                        sign=sign,
                        crease_kind=crease_kind,
                        constraint_id=f"dihedral_signed_u{unit_count}_i{local_index}_{side}",
                    )
                )

        return {
            "num_added": len(added),
            "num_skipped": len(skipped),
            "num_duplicate": len(duplicate),
            "added": added,
            "skipped": skipped,
            "duplicate": duplicate,
        }

    def add_simple_hexagon_kinematic_constraints(
        self,
        target_dihedral: float = 110.0,
        unit: Literal["rad", "deg"] = "deg",
        fixed_triangle_surface_id: Optional[str] = None,
        valley_z: float = 0.0,
        strict_unique_edges: bool = False,
    ) -> dict:
        self.add_panel_rigidity_constraints_from_surfaces()

        triangle_ids = self._triangle_surface_ids()
        if not triangle_ids:
            raise ValueError("No triangle surfaces found.")

        valley_triangle_ids = [
            sid for sid in triangle_ids
            if self.triangle_crease_kind(sid) == "valley"
        ]

        fixed_triangle_surface_id = fixed_triangle_surface_id or (
            valley_triangle_ids[0] if valley_triangle_ids else triangle_ids[0]
        )
        if fixed_triangle_surface_id not in triangle_ids:
            raise ValueError(
                f"Fixed surface '{fixed_triangle_surface_id}' is not a triangle."
            )

        fixed_kind = self.triangle_crease_kind(fixed_triangle_surface_id)
        if valley_triangle_ids and fixed_kind != "valley":
            raise ValueError(
                f"Fixed surface '{fixed_triangle_surface_id}' is {fixed_kind}, "
                "but valley panels are pinned to valley_z. Choose a valley "
                f"triangle such as '{valley_triangle_ids[0]}'."
            )

        self.add_fixed_surface_constraint(fixed_triangle_surface_id)

        for tri_id in triangle_ids:
            self.add_horizontal_surface_constraint(
                tri_id,
                constraint_id=f"horizontal_{tri_id}",
            )
            if self.triangle_crease_kind(tri_id) == "valley":
                self.add_surface_z_value_constraint(
                    tri_id,
                    z_value=valley_z,
                    constraint_id=f"valley_z_{tri_id}",
                )

        dihedral_info = self.add_simple_hexagon_dihedral_constraints_from_metadata(
            target_dihedral=target_dihedral,
            unit=unit,
        )

        if strict_unique_edges and (
            dihedral_info["skipped"] or dihedral_info["duplicate"]
        ):
            raise ValueError(
                "Simple-hexagon crease metadata is not unique: "
                f"{dihedral_info['num_skipped']} skipped and "
                f"{dihedral_info['num_duplicate']} duplicate entries."
            )

        return {
            "fixed_triangle": fixed_triangle_surface_id,
            "num_dihedral_constraints": len(dihedral_info["added"]),
            "num_skipped_crease_edges": len(dihedral_info["skipped"]),
            "num_duplicate_dihedral_constraints": dihedral_info["num_duplicate"],
            "skipped_crease_edges": dihedral_info["skipped"],
            "duplicate_dihedral_constraints": dihedral_info["duplicate"],
        }

    def find_line_by_points(self, p1: str, p2: str) -> Optional[str]:
        target = {p1, p2}

        for line_id in self.lines:
            start, end, _ = self._line_info(line_id)
            if {start, end} == target:
                return line_id

        return None

    def triangle_crease_kind(self, surface_id: str) -> Optional[str]:
        vertices = self._surface_vertices(surface_id)
        if len(vertices) != 3:
            raise ValueError(f"Surface '{surface_id}' is not a triangle.")

        crease_kinds = []
        for i, a in enumerate(vertices):
            b = vertices[(i + 1) % 3]
            line_id = self.find_line_by_points(a, b)
            if line_id is None:
                continue

            _, _, kind = self._line_info(line_id)
            if kind in {"valley", "mountain"}:
                crease_kinds.append(kind)

        unique_kinds = set(crease_kinds)
        if not unique_kinds:
            return None
        if len(unique_kinds) == 1:
            return crease_kinds[0]

        raise ValueError(
            f"Triangle surface '{surface_id}' has mixed crease kinds: "
            f"{crease_kinds}."
        )

    def find_unique_triangle_quad_pair_adjacent_to_edge(
        self,
        p1: str,
        p2: str,
    ) -> Optional[Tuple[str, str]]:
        adjacent = self.find_surfaces_adjacent_to_edge(p1, p2)
        triangle_ids = [
            sid for sid in adjacent
            if len(self._surface_vertices(sid)) == 3
        ]
        quad_ids = [
            sid for sid in adjacent
            if len(self._surface_vertices(sid)) == 4
        ]

        if len(triangle_ids) == 1 and len(quad_ids) == 1:
            return triangle_ids[0], quad_ids[0]
        return None

    def simple_hexagon_initial_guess(
        self,
        mountain_height: float = 5.0,
        valley_height: float = 0.0,
    ) -> np.ndarray:
        X0 = self.get_coordinate_vector().copy()
        point_to_index = {pid: i for i, pid in enumerate(self.point_ids())}
        proposed_z = {}

        for surface_id in self._triangle_surface_ids():
            kind = self.triangle_crease_kind(surface_id)
            if kind not in {"valley", "mountain"}:
                continue

            z_target = valley_height if kind == "valley" else mountain_height
            for pid in self._surface_vertices(surface_id):
                if pid in proposed_z and abs(proposed_z[pid] - z_target) > 1e-9:
                    raise ValueError(
                        f"Point '{pid}' is shared by triangle panels that request "
                        f"different initial heights: {proposed_z[pid]} and "
                        f"{z_target}."
                    )
                proposed_z[pid] = z_target

        for pid, z in proposed_z.items():
            X0[3 * point_to_index[pid] + 2] = z

        return X0

    def print_simple_hexagon_metadata_summary(self) -> None:
        print("Simple hexagon metadata summary")
        print("-------------------------------")

        if not self.hex_units:
            print("No hex-unit metadata found.")
            return

        totals = {"triangles": 0, "quads": 0, "creases": 0}
        for unit_data in self.hex_units:
            unit_count = unit_data.get("count", "unknown")
            n_tri = len(unit_data.get("triangles", []))
            n_quad = len(unit_data.get("parallelograms", []))
            n_creases = len(unit_data.get("local_creases", []))

            totals["triangles"] += n_tri
            totals["quads"] += n_quad
            totals["creases"] += n_creases

            print(
                f"Unit {unit_count}: "
                f"triangles={n_tri}, quads={n_quad}, local creases={n_creases}"
            )

        print("")
        print(f"Total units:         {len(self.hex_units)}")
        print(f"Total triangles:     {totals['triangles']}")
        print(f"Total quads:         {totals['quads']}")
        print(f"Total local creases: {totals['creases']}")

    def update_simple_hexagon_dihedral_target(
        self,
        target_dihedral: float,
        unit: Literal["rad", "deg"] = "deg",
    ) -> None:
        theta = self._angle_to_rad(target_dihedral, unit=unit)
        if not (0.0 < theta < np.pi):
            raise ValueError("target_dihedral must be between 0 and 180 degrees.")

        fold_amount = np.pi - theta
        for constraint in self.constraints.values():
            if constraint.kind == "dihedral_signed_increment":
                constraint.data["target_increment"] = float(
                    constraint.data["sign"] * fold_amount
                )

    def solve_simple_hexagon_continuation(
        self,
        final_dihedral: float = 110.0,
        start_dihedral: float = 175.0,
        steps: int = 14,
        unit: Literal["rad", "deg"] = "deg",
        X0: Optional[np.ndarray] = None,
        max_nfev_per_step: int = 5000,
        tol: float = 1e-10,
        residual_warning_tol: float = 1e-5,
        verbose: bool = False,
    ):
        if steps < 2:
            raise ValueError("steps must be at least 2.")
        if unit not in {"deg", "rad"}:
            raise ValueError("unit must be 'deg' or 'rad'.")

        X = self.get_coordinate_vector() if X0 is None else np.asarray(X0, dtype=float)
        last_report = None

        for k, theta in enumerate(np.linspace(start_dihedral, final_dihedral, steps)):
            self.update_simple_hexagon_dihedral_target(theta, unit=unit)
            report = self.solve(
                X0=X,
                update_model=True,
                max_nfev=max_nfev_per_step,
                tol=tol,
                compute_rank=(k == steps - 1),
            )
            X = report.x.copy()
            last_report = report

            if verbose:
                print(
                    f"[step {k + 1:02d}/{steps}] "
                    f"target_dihedral={theta:.3f} {unit}, "
                    f"max_residual={report.max_abs_residual:.3e}, "
                    f"success={report.success}"
                )
                if report.max_abs_residual > residual_warning_tol:
                    print(
                        "  warning: residual is still large at this step; "
                        "continuing anyway."
                    )

        return last_report

    def solve_simple_hexagon_kinematics(
        self,
        final_dihedral: float = 110.0,
        start_dihedral: float = 175.0,
        steps: int = 14,
        unit: Literal["rad", "deg"] = "deg",
        fixed_triangle_surface_id: Optional[str] = None,
        valley_z: float = 0.0,
        strict_unique_edges: bool = False,
        mountain_height: float = 5.0,
        valley_height: float = 0.0,
        X0: Optional[np.ndarray] = None,
        max_nfev_per_step: int = 5000,
        tol: float = 1e-10,
        residual_warning_tol: float = 1e-5,
        verbose: bool = True,
        print_metadata_summary: bool = False,
        print_constraint_info: bool = False,
        print_solve_report: bool = False,
        print_dihedral_status: bool = False,
        print_residual_warning: bool = False,
        dihedral_status_max_items: int = 20,
    ) -> dict:
        """
        Set up and solve a simple-hexagon model in one front-layer call.

        The same start_dihedral is used to initialize the dihedral constraints
        and to start continuation, so the setup target and solver start target
        cannot drift apart accidentally.
        """
        if print_metadata_summary:
            self.print_simple_hexagon_metadata_summary()

        constraint_info = self.add_simple_hexagon_kinematic_constraints(
            target_dihedral=start_dihedral,
            unit=unit,
            fixed_triangle_surface_id=fixed_triangle_surface_id,
            valley_z=valley_z,
            strict_unique_edges=strict_unique_edges,
        )

        if print_constraint_info:
            print(constraint_info)

        if X0 is None:
            X0 = self.simple_hexagon_initial_guess(
                mountain_height=mountain_height,
                valley_height=valley_height,
            )

        report = self.solve_simple_hexagon_continuation(
            final_dihedral=final_dihedral,
            start_dihedral=start_dihedral,
            steps=steps,
            unit=unit,
            X0=X0,
            max_nfev_per_step=max_nfev_per_step,
            tol=tol,
            residual_warning_tol=residual_warning_tol,
            verbose=verbose,
        )

        if print_solve_report:
            self.print_solve_report(report)
        if print_dihedral_status:
            self.print_dihedral_signed_status(
                max_items=dihedral_status_max_items,
                unit=unit,
            )
        if print_residual_warning and report.max_abs_residual > residual_warning_tol:
            print("WARNING: constraints are not sufficiently satisfied.")

        return {
            "constraint_info": constraint_info,
            "report": report,
        }

    def print_dihedral_signed_status(
        self,
        max_items: int = 20,
        unit: Literal["rad", "deg"] = "deg",
    ) -> None:
        rows = []

        for cid, constraint in self.constraints.items():
            if constraint.kind != "dihedral_signed_increment":
                continue

            data = constraint.data
            current_angle = self.signed_dihedral_angle(
                edge_start=data["edge_start"],
                edge_end=data["edge_end"],
                point_left=data["point_left"],
                point_right=data["point_right"],
                unit="rad",
            )
            actual_increment = self._wrap_to_pi(
                current_angle - data["initial_angle"]
            )
            target_increment = data["target_increment"]
            residual = self._wrap_to_pi(actual_increment - target_increment)

            values = (
                abs(current_angle),
                np.pi - abs(target_increment),
                actual_increment,
                residual,
            )
            if unit == "deg":
                values = tuple(np.rad2deg(v) for v in values)

            rows.append(
                (
                    abs(residual),
                    cid,
                    data.get("crease_kind"),
                    *values,
                )
            )

        rows.sort(reverse=True, key=lambda x: x[0])
        print("Dihedral status")
        print("----------------")
        print(
            f"{'constraint':45s} {'kind':10s} "
            f"{'dihedral':>12s} {'target':>12s} "
            f"{'signed fold':>12s} {'error':>12s}"
        )

        for _, cid, kind, dihedral, target, fold, residual in rows[:max_items]:
            print(
                f"{cid:45s} {str(kind):10s} "
                f"{dihedral:12.4f} {target:12.4f} "
                f"{fold:12.4f} {residual:12.4f}"
            )
