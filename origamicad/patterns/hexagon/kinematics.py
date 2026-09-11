from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ...core.panel_holes import panel_only_geometry
from .metadata import normalize_hex_creases

if TYPE_CHECKING:
    from ...core.cadder import Cadder


class _HexagonKinematics:
    """Pattern-specific kinematic operations for a hexagon model."""

    def __init__(self, model: Cadder):
        self._model = model

    def __getattr__(self, name: str) -> Any:
        """Delegate generic geometry and solver operations to the Cadder."""
        return getattr(self._model, name)

    def _triangle_surface_ids(self) -> list[str]:
        return [
            sid for sid in self.surfaces
            if len(self._surface_vertices(sid)) == 3
        ]

    def _add_dihedral_constraints_from_metadata(
        self,
        target_dihedral: float = 110.0,
        unit: Literal["rad", "deg"] = "deg",
        valley_sign: int = +1,
        _plan=None,
    ) -> dict:
        if not self.hex_units:
            raise ValueError(
                "No hex-unit metadata found. Build the pattern with "
                "draw_hex_loops() or build_packaging() before creating "
                "the Cadder model."
            )

        if valley_sign not in {+1, -1}:
            raise ValueError("valley_sign must be +1 or -1.")

        theta = self._angle_to_rad(target_dihedral, unit=unit)
        if not (0.0 < theta < np.pi):
            raise ValueError("target_dihedral must be between 0 and 180 degrees.")

        entries, skipped, shared = self._crease_plan() if _plan is None else _plan
        added = []
        fold_amount = np.pi - theta

        for crease_data, label in entries:
            unit_count, local_index, side = label
            edge_start, edge_end = crease_data["edge"]
            triangle_point = self._first_non_edge_vertex(
                self._surface_vertices(crease_data["triangle"]), edge_start, edge_end)
            quad_point = self._first_non_edge_vertex(
                self._surface_vertices(crease_data["quad"]), edge_start, edge_end)
            edge_start, edge_end = self._orient_flat_crease_triangle_right(
                edge_start, edge_end, triangle_point)
            crease_kind = crease_data["kind"]
            sign = valley_sign if crease_kind == "valley" else -valley_sign
            added.append(self.add_dihedral_signed_increment_constraint(
                edge_start=edge_start, edge_end=edge_end,
                point_left=triangle_point, point_right=quad_point,
                target_increment=sign * fold_amount, unit="rad", sign=sign,
                crease_kind=crease_kind,
                constraint_id=f"dihedral_signed_u{unit_count}_i{local_index}_{side}",
            ))

        return {
            "num_added": len(added),
            "num_skipped": len(skipped),
            "num_shared": len(shared),
            "shared": shared,
            # Retained keys now describe actual duplicate equations, not sharing.
            "num_duplicate": 0,
            "added": added,
            "skipped": skipped,
            "duplicate": [],
        }

    def _crease_plan(self):
        """Validate local ownership, then visit each physical hinge once."""
        if not self.hex_units:
            raise ValueError("No hex-unit metadata found. Build a hexagon pattern first.")
        normalize_hex_creases(self._model)
        owners = {}
        for unit in self.hex_units:
            for ref in unit.get("local_creases", []):
                owners.setdefault(ref["crease"], []).append(
                    (unit["count"], ref["local_index"], ref["side"]))
        line_kinds = {tuple(sorted((a, b))): kind
                      for a, b, kind in (self._line_info(lid) for lid in self.lines)}
        entries, skipped, shared = [], [], []
        edge_panels = {}
        for cid, labels in owners.items():
            crease = self.hex_creases[cid]
            a, b = crease["edge"]
            tri, quad = crease["triangle"], crease["quad"]
            missing = None
            if tri not in self.surfaces:
                missing = "missing triangle", tri
            elif quad not in self.surfaces:
                missing = "missing quad", quad
            elif a not in self.points or b not in self.points:
                missing = ("missing edge points",)
            if missing:
                skipped.extend((*label, *missing) for label in labels)
                continue
            edge = tuple(sorted((a, b)))
            for sid, size in ((tri, 3), (quad, 4)):
                vertices = self._surface_vertices(sid)
                boundary = {tuple(sorted((p, vertices[(i + 1) % len(vertices)])))
                            for i, p in enumerate(vertices)}
                if len(vertices) != size or edge not in boundary:
                    raise ValueError(f"Crease '{cid}' is not a boundary edge of panel '{sid}'.")
            if edge in edge_panels and edge_panels[edge] != (tri, quad):
                raise ValueError(f"Conflicting adjacent panels for crease edge {edge}.")
            edge_panels[edge] = tri, quad
            if (line_kinds.get(edge) in {"mountain", "valley"}
                    and line_kinds[edge] != crease["kind"]):
                raise ValueError(f"Conflicting mountain/valley assignment for crease '{cid}'.")
            entries.append((crease, labels[0]))
            shared.extend((*label, tri, quad) for label in labels[1:])
        return entries, skipped, shared

    def _add_kinematic_constraints(
        self,
        target_dihedral: float = 110.0,
        unit: Literal["rad", "deg"] = "deg",
        fixed_triangle_surface_id: Optional[str] = None,
        valley_z: float = 0.0,
        strict_unique_edges: bool = False,
    ) -> dict:
        plan = self._crease_plan()
        if strict_unique_edges and plan[1]:
            raise ValueError(f"Invalid hexagon crease metadata: {len(plan[1])} skipped references.")
        self.add_panel_rigidity_constraints_from_surfaces()

        triangle_ids = self._triangle_surface_ids()
        if not triangle_ids:
            raise ValueError("No triangle surfaces found.")

        valley_triangle_ids = [
            sid for sid in triangle_ids
            if self._triangle_crease_kind(sid) == "valley"
        ]

        fixed_triangle_surface_id = fixed_triangle_surface_id or (
            valley_triangle_ids[0] if valley_triangle_ids else triangle_ids[0]
        )
        if fixed_triangle_surface_id not in triangle_ids:
            raise ValueError(
                f"Fixed surface '{fixed_triangle_surface_id}' is not a triangle."
            )

        fixed_kind = self._triangle_crease_kind(fixed_triangle_surface_id)
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
            if self._triangle_crease_kind(tri_id) == "valley":
                self.add_surface_z_value_constraint(
                    tri_id,
                    z_value=valley_z,
                    constraint_id=f"valley_z_{tri_id}",
                )

        dihedral_info = self._add_dihedral_constraints_from_metadata(
            target_dihedral=target_dihedral,
            unit=unit,
            _plan=plan,
        )

        return {
            "fixed_triangle": fixed_triangle_surface_id,
            "num_dihedral_constraints": len(dihedral_info["added"]),
            "num_skipped_crease_edges": len(dihedral_info["skipped"]),
            "num_shared_crease_references": dihedral_info["num_shared"],
            "shared_crease_references": dihedral_info["shared"],
            "num_duplicate_dihedral_constraints": dihedral_info["num_duplicate"],
            "skipped_crease_edges": dihedral_info["skipped"],
            "duplicate_dihedral_constraints": dihedral_info["duplicate"],
        }

    def _find_line_by_points(self, p1: str, p2: str) -> Optional[str]:
        target = {p1, p2}

        for line_id in self.lines:
            start, end, _ = self._line_info(line_id)
            if {start, end} == target:
                return line_id

        return None

    def _triangle_crease_kind(self, surface_id: str) -> Optional[str]:
        vertices = self._surface_vertices(surface_id)
        if len(vertices) != 3:
            raise ValueError(f"Surface '{surface_id}' is not a triangle.")

        crease_kinds = []
        for i, a in enumerate(vertices):
            b = vertices[(i + 1) % 3]
            line_id = self._find_line_by_points(a, b)
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

    def _initial_guess(
        self,
        mountain_height: float,
        valley_height: float = 0.0,
    ) -> np.ndarray:
        X0 = self.get_coordinate_vector().copy()
        point_to_index = {pid: i for i, pid in enumerate(self.point_ids())}
        proposed_z = {}

        for surface_id in self._triangle_surface_ids():
            kind = self._triangle_crease_kind(surface_id)
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

    def _automatic_mountain_height(
        self,
        dihedral_angle: float,
        unit: Literal["rad", "deg"] = "deg",
        valley_height: float = 0.0,
    ) -> float:
        """Return the initial mountain elevation from the hexagon geometry.

        For a hexagon with side length ``d`` and dihedral angle ``theta``, the
        mountain-to-valley height is

            h = sqrt(3) / 2 * d * sin(theta).

        The side length is measured from the middle-hexagon edges (using the
        surviving panels for units cut by a cavity), so callers do not need
        to repeat the pattern's ``l`` parameter.
        """
        if not self.hex_units:
            raise ValueError(
                "Cannot calculate mountain height without hex-unit metadata."
            )

        side_lengths = []
        for unit_data in self.hex_units:
            mid_point_ids = list(unit_data.get("mid", []))
            if any(point_id not in self.points for point_id in mid_point_ids):
                raise ValueError(
                    "Hex-unit metadata references a missing middle-hexagon point."
                )

            if len(mid_point_ids) == 6:
                mid_edges = list(zip(
                    mid_point_ids, mid_point_ids[1:] + mid_point_ids[:1]
                ))
            else:
                # Cavity units have gaps in their middle hexagon. Closing the
                # shortened point list would measure diagonals across the gap.
                mid_edges = []
                for surface_id in unit_data.get("parallelograms", []):
                    vertices = self._surface_vertices(surface_id)
                    mid_edges.extend(
                        (start, end)
                        for start, end in zip(vertices, vertices[1:] + vertices[:1])
                        if start in mid_point_ids and end in mid_point_ids
                    )
                if not mid_edges:
                    raise ValueError(
                        "Each hex unit must provide middle-hexagon edges to "
                        "calculate its side length."
                    )
            side_lengths.extend(
                float(
                    np.linalg.norm(
                        self.point_array(end) - self.point_array(start)
                    )
                )
                for start, end in mid_edges
            )

        side_length = float(np.mean(side_lengths))
        length_tolerance = 1e-6 * max(1.0, side_length)
        if not np.isfinite(side_length) or side_length <= length_tolerance:
            raise ValueError("The measured hexagon side length must be positive.")
        maximum_length_error = max(
            abs(length - side_length)
            for length in side_lengths
        )
        if maximum_length_error > length_tolerance:
            raise ValueError(
                "Hex-unit side lengths are inconsistent; one value of d cannot "
                "be used in the automatic height formula."
            )

        theta = self._angle_to_rad(dihedral_angle, unit=unit)
        if not (0.0 < theta < np.pi):
            raise ValueError("dihedral_angle must be between 0 and 180 degrees.")

        layer_height = float(
            np.sqrt(3.0) / 2.0 * side_length * np.sin(theta)
        )
        return float(valley_height) + layer_height

    def _print_metadata_summary(self) -> None:
        print("Hexagon metadata summary")
        print("------------------------")

        if not self.hex_units:
            print("No hex-unit metadata found.")
            return

        normalize_hex_creases(self._model)
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
        print(f"Triangle references: {totals['triangles']}")
        print(f"Quad references:     {totals['quads']}")
        print(f"Local crease refs:   {totals['creases']}")
        print(f"Physical creases:    {len(self.hex_creases)}")

    def _update_dihedral_target(
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

    def _solve_continuation(
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
        adaptive_tolerance: bool = True,
    ):
        if steps < 2:
            raise ValueError("steps must be at least 2.")
        if unit not in {"deg", "rad"}:
            raise ValueError("unit must be 'deg' or 'rad'.")

        X = self.get_coordinate_vector() if X0 is None else np.asarray(X0, dtype=float)
        last_report = None

        for k, theta in enumerate(np.linspace(start_dihedral, final_dihedral, steps)):
            self._update_dihedral_target(theta, unit=unit)
            report = self.solve(
                X0=X,
                update_model=True,
                max_nfev=max_nfev_per_step,
                tol=tol,
                compute_rank=False,
                adaptive_tolerance=adaptive_tolerance,
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

    def solve_kinematics(
        self,
        final_dihedral: float = 110.0,
        start_dihedral: float = 175.0,
        steps: int = 14,
        unit: Literal["rad", "deg"] = "deg",
        fixed_triangle_surface_id: Optional[str] = None,
        valley_z: float = 0.0,
        strict_unique_edges: bool = False,
        mountain_height: Optional[float] = None,
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
        adaptive_tolerance: bool = True,
    ) -> dict:
        """
        Set up and solve a simple-hexagon model in one front-layer call.

        Rank and mobility are not calculated during continuation. Call
        ``analyze_kinematics(model)`` separately when these diagnostics are
        needed. The returned SolveReport uses -1 for both uncomputed fields.

        The same start_dihedral is used to initialize the dihedral constraints
        and to start continuation, so the setup target and solver start target
        cannot drift apart accidentally. If ``mountain_height`` is omitted,
        its initial value is calculated as
        ``valley_height + sqrt(3) / 2 * d * sin(start_dihedral)``, where ``d``
        is measured directly from the hexagon pattern.
        """
        if print_metadata_summary:
            self._print_metadata_summary()

        constraint_info = self._add_kinematic_constraints(
            target_dihedral=start_dihedral,
            unit=unit,
            fixed_triangle_surface_id=fixed_triangle_surface_id,
            valley_z=valley_z,
            strict_unique_edges=strict_unique_edges,
        )

        if print_constraint_info:
            print(constraint_info)

        if X0 is None:
            if mountain_height is None:
                mountain_height = self._automatic_mountain_height(
                    dihedral_angle=start_dihedral,
                    unit=unit,
                    valley_height=valley_height,
                )
            X0 = self._initial_guess(
                mountain_height=mountain_height,
                valley_height=valley_height,
            )

        report = self._solve_continuation(
            final_dihedral=final_dihedral,
            start_dihedral=start_dihedral,
            steps=steps,
            unit=unit,
            X0=X0,
            max_nfev_per_step=max_nfev_per_step,
            tol=tol,
            residual_warning_tol=residual_warning_tol,
            verbose=verbose,
            adaptive_tolerance=adaptive_tolerance,
        )

        if print_solve_report:
            self.print_solve_report(report)
        if print_dihedral_status:
            self._print_dihedral_status(
                max_items=dihedral_status_max_items,
                unit=unit,
            )
        if print_residual_warning and report.max_abs_residual > residual_warning_tol:
            print("WARNING: constraints are not sufficiently satisfied.")

        return {
            "constraint_info": constraint_info,
            "report": report,
        }

    def _print_dihedral_status(
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


def solve_kinematics(
    model: Cadder,
    final_dihedral: float = 110.0,
    start_dihedral: float = 175.0,
    steps: int = 14,
    unit: Literal["rad", "deg"] = "deg",
    fixed_triangle_surface_id: Optional[str] = None,
    valley_z: float = 0.0,
    strict_unique_edges: bool = False,
    mountain_height: Optional[float] = None,
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
    adaptive_tolerance: bool = True,
) -> dict:
    """Add hexagon constraints and solve a generated pattern in 3D.

    Hole contours move with their host panels and add no independent solver
    variables. ``report.x`` includes the restored hole points in the model's
    original point order.

    Rank and mobility are not calculated, including at the final continuation
    step. Their SolveReport fields are -1 (not computed). Use
    ``analyze_kinematics(model)`` separately to evaluate them at the solved
    configuration; that diagnostic does not perform another solve.

    Sparse inner-solver accuracy adapts automatically while ``tol`` remains
    the outer stopping tolerance. Set ``adaptive_tolerance=False`` to use
    SciPy's fixed inner accuracy. The automatic initial guess already keeps
    the flat pattern's XY coordinates and sets triangle heights only.

    Shared unit references to a physical crease are valid, including with
    ``strict_unique_edges=True``. Strict mode rejects missing geometry;
    conflicting crease definitions always raise before adding constraints.
    ``constraint_info`` reports sharing via ``num_shared_crease_references``.
    The older duplicate-constraint fields remain present and are zero/empty.
    """
    full_size = model.num_variables()
    with panel_only_geometry(model) as coordinate_indices:
        if coordinate_indices is not None and X0 is not None:
            X0 = np.asarray(X0, dtype=float)
            if X0.size == full_size:
                X0 = X0.ravel()[coordinate_indices]
            elif X0.size != model.num_variables():
                raise ValueError(
                    f"Expected X0 size {full_size} (with holes) or "
                    f"{model.num_variables()} (panel points), but got {X0.size}."
                )
        result = _HexagonKinematics(model).solve_kinematics(
            final_dihedral=final_dihedral,
            start_dihedral=start_dihedral,
            steps=steps,
            unit=unit,
            fixed_triangle_surface_id=fixed_triangle_surface_id,
            valley_z=valley_z,
            strict_unique_edges=strict_unique_edges,
            mountain_height=mountain_height,
            valley_height=valley_height,
            X0=X0,
            max_nfev_per_step=max_nfev_per_step,
            tol=tol,
            residual_warning_tol=residual_warning_tol,
            verbose=verbose,
            print_metadata_summary=print_metadata_summary,
            print_constraint_info=print_constraint_info,
            print_solve_report=print_solve_report,
            print_dihedral_status=print_dihedral_status,
            print_residual_warning=print_residual_warning,
            dihedral_status_max_items=dihedral_status_max_items,
            adaptive_tolerance=adaptive_tolerance,
        )
    if coordinate_indices is not None:
        result["report"].x = model.get_coordinate_vector()
    return result


def analyze_kinematics(model: Cadder, tol: float = 1e-8) -> dict[str, int]:
    """Calculate rank and local mobility at the model's current configuration.

    Build constraints first, normally by calling ``solve_kinematics``. This
    explicit diagnostic builds the analytical Jacobian and computes its dense
    numerical rank, which can require substantial time and memory for large
    models. ``tol`` is the absolute singular-value cutoff for rank.

    Return ``rank``, ``mobility``, ``num_variables``, and ``num_residuals``.
    Mobility is the number of variables minus rank. Hole-contour-only points
    are excluded, matching the folding solver's degrees of freedom. Other
    unconstrained points remain included. The model and any previous
    SolveReport are left unchanged; no optimization is run.
    """
    jacobian = model.jacobian()
    if model.surface_holes:
        panel_points = {
            pid for sid in model.surfaces for pid in model._surface_vertices(sid)
        }
        cut_points = {
            pid for loops in model.surface_holes.values() for loop in loops
            for pid in loop
        } - panel_points
        columns = [
            3 * index + axis
            for index, pid in enumerate(model.point_ids()) if pid not in cut_points
            for axis in range(3)
        ]
        jacobian = jacobian[:, columns]

    num_residuals, num_variables = jacobian.shape
    rank = (
        int(np.linalg.matrix_rank(jacobian.toarray(), tol=tol))
        if num_residuals and num_variables else 0
    )
    return {
        "rank": rank,
        "mobility": num_variables - rank,
        "num_variables": num_variables,
        "num_residuals": num_residuals,
    }


# Compatibility name for code that imported the old solver operation directly.
solve_simple_hexagon_kinematics = solve_kinematics
