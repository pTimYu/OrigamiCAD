from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ...core._tangent import predict_dihedral_tangent
from ...core.panel_holes import panel_only_geometry
from .metadata import hex_constraint_plan, normalize_hex_creases

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
        """Read the ordered physical hinges prepared by the 2D metadata."""
        if not self.hex_units:
            raise ValueError("No hex-unit metadata found. Build a hexagon pattern first.")
        plan = hex_constraint_plan(self._model)
        return list(plan.crease_entries), list(plan.skipped), list(plan.shared)

    def _triangle_metadata(self):
        plan = hex_constraint_plan(self._model)
        missing = set(plan.triangles) - plan.triangle_kinds.keys()
        if missing:
            raise ValueError(f"Missing triangle_kinds metadata for {sorted(missing)}. Rebuild the 2D metadata.")
        return plan.triangles, plan.triangle_kinds

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

        triangle_ids, triangle_kinds = self._triangle_metadata()
        if not triangle_ids:
            raise ValueError("No triangle surfaces found.")

        valley_triangle_ids = [
            sid for sid in triangle_ids
            if triangle_kinds[sid] == "valley"
        ]

        fixed_triangle_surface_id = fixed_triangle_surface_id or (
            valley_triangle_ids[0] if valley_triangle_ids else triangle_ids[0]
        )
        if fixed_triangle_surface_id not in triangle_ids:
            raise ValueError(
                f"Fixed surface '{fixed_triangle_surface_id}' is not a triangle."
            )

        fixed_kind = triangle_kinds[fixed_triangle_surface_id]
        if valley_triangle_ids and fixed_kind != "valley":
            raise ValueError(
                f"Fixed surface '{fixed_triangle_surface_id}' is {fixed_kind}, "
                "but valley panels are pinned to valley_z. Choose a valley "
                f"triangle such as '{valley_triangle_ids[0]}'."
            )

        self.add_panel_rigidity_constraints_from_surfaces()
        frame_constraint_ids = self.add_fixed_surface_constraint(fixed_triangle_surface_id)

        pinned_z = {}
        for tri_id in triangle_ids:
            if triangle_kinds[tri_id] == "valley":
                cid = self.add_surface_z_value_constraint(
                    tri_id,
                    z_value=valley_z,
                    constraint_id=f"valley_z_{tri_id}",
                )
                constraint = self.constraints[cid]
                if (constraint.kind == "surface_z_value"
                        and constraint.data["surface"] == tri_id):
                    # Equal absolute vertex heights already imply every
                    # horizontal-surface equation for this triangle.
                    for pid in self._surface_vertices(tri_id):
                        pinned_z[pid] = constraint.data["z_value"]
                    continue
            self.add_horizontal_surface_constraint(
                tri_id,
                constraint_id=f"horizontal_{tri_id}",
            )

        for cid in frame_constraint_ids:
            data = self.constraints[cid].data
            if (data["axis"] == "z" and data["point"] in pinned_z
                    and data["value"] == pinned_z[data["point"]]):
                # Only remove exact duplicates created for this frame.
                # Different target heights must retain their conflict.
                del self.constraints[cid]

        dihedral_info = self._add_dihedral_constraints_from_metadata(
            target_dihedral=target_dihedral,
            unit=unit,
            _plan=plan,
        )

        return {
            "fixed_triangle": fixed_triangle_surface_id,
            "num_dihedral_constraints": len(dihedral_info["added"]),
            "dihedral_constraint_ids": tuple(dihedral_info["added"]),
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
        triangle_ids, triangle_kinds = self._triangle_metadata()
        if surface_id not in triangle_ids:
            raise ValueError(f"Surface '{surface_id}' is not a triangle.")
        return triangle_kinds[surface_id]

    def _initial_guess(
        self,
        mountain_height: float,
        valley_height: float = 0.0,
    ) -> np.ndarray:
        X0 = self.get_coordinate_vector().copy()
        point_to_index = {pid: i for i, pid in enumerate(self.point_ids())}
        proposed_z = {}

        triangle_ids, triangle_kinds = self._triangle_metadata()
        for surface_id in triangle_ids:
            kind = triangle_kinds[surface_id]
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

    def solve_kinematics(
        self,
        final_dihedral: float = 110.0,
        unit: Literal["rad", "deg"] = "deg",
        fixed_triangle_surface_id: Optional[str] = None,
        valley_z: float = 0.0,
        strict_unique_edges: bool = False,
        mountain_height: Optional[float] = None,
        valley_height: float = 0.0,
        X0: Optional[np.ndarray] = None,
        max_nfev: int = 5000,
        tol: float = 1e-10,
        residual_warning_tol: float = 1e-5,
        verbose: bool = True,
        print_metadata_summary: bool = False,
        print_constraint_info: bool = False,
        print_solve_report: bool = False,
        print_dihedral_status: bool = False,
        print_residual_warning: bool = False,
        dihedral_status_max_items: int = 20,
        x_scale: float | np.ndarray | Literal["jac"] = "jac",
        steps: int = 3,
    ) -> dict:
        """
        Solve using tangent prediction followed by nonlinear refinement.

        Rank and mobility are not calculated during solving. Call
        ``analyze_kinematics(model)`` separately when these diagnostics are
        needed. The returned SolveReport uses -1 for both uncomputed fields.

        Positive ``steps`` take that many equal tangent increments from the
        flat 180-degree coordinates to the requested angle before nonlinear
        refinement. There is no nonlinear starting-pose solve. ``steps=0``
        uses the original direct solve and its exact target-angle height guess.
        For positive steps, ``X0`` may supply the flat starting coordinates;
        otherwise the model's current coordinates are used.
        """
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
            raise ValueError("steps must be a nonnegative integer.")
        if max_nfev is not None and (
            isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev < 1
        ):
            raise ValueError("max_nfev must be a positive integer or None.")
        theta = self._angle_to_rad(final_dihedral, unit=unit)
        if not (0.0 < theta < np.pi):
            raise ValueError("final_dihedral must be between 0 and 180 degrees.")

        if print_metadata_summary:
            self._print_metadata_summary()

        constraint_info = self._add_kinematic_constraints(
            target_dihedral=final_dihedral,
            unit=unit,
            fixed_triangle_surface_id=fixed_triangle_surface_id,
            valley_z=valley_z,
            strict_unique_edges=strict_unique_edges,
        )

        if print_constraint_info:
            print(constraint_info)

        def endpoint_guess():
            if X0 is not None:
                return np.asarray(X0, dtype=float).copy()
            height = mountain_height
            if height is None:
                height = self._automatic_mountain_height(
                    dihedral_angle=theta, unit="rad", valley_height=valley_height,
                )
            return self._initial_guess(
                mountain_height=height,
                valley_height=valley_height,
            )

        hybrid = {
            "requested_steps": int(steps), "completed_steps": 0,
            "initial_nfev": 0, "refinement_nfev": 0,
            "linear_iterations": 0, "fallback_reason": None,
        }
        guess = None
        if steps:
            start_theta = np.pi
            driven_ids = constraint_info["dihedral_constraint_ids"]
            final_targets = {
                cid: self.constraints[cid].data["target_increment"] for cid in driven_ids
            }
            try:
                for cid in driven_ids:
                    data = self.constraints[cid].data
                    data["target_increment"] = float(data["sign"] * (np.pi - start_theta))
                start_coordinates = (
                    np.asarray(X0, dtype=float).copy()
                    if X0 is not None else self.get_coordinate_vector().copy()
                )
                prediction = predict_dihedral_tangent(
                    self._model, np.linspace(start_theta, theta, int(steps) + 1),
                    X0=start_coordinates, constraint_ids=driven_ids,
                    linear_tol=1e-10, maxiter=5000, x_scale=x_scale,
                )
                hybrid["completed_steps"] = prediction.completed_steps
                hybrid["linear_iterations"] = prediction.linear_iterations
                if prediction.success:
                    guess = prediction.x
                else:
                    hybrid["fallback_reason"] = prediction.failure_reason
            finally:
                # A failed predictor must never leave the public model
                # configured for an intermediate angle.
                for cid, target in final_targets.items():
                    self.constraints[cid].data["target_increment"] = target

        if guess is None:
            guess = endpoint_guess()
        report = self.solve(
            X0=guess,
            update_model=True,
            max_nfev=max_nfev,
            tol=tol,
            compute_rank=False,
            x_scale=x_scale,
        )
        hybrid["refinement_nfev"] = report.nfev

        if verbose:
            print(
                f"[solve] target_dihedral={final_dihedral:.3f} {unit}, "
                f"max_residual={report.max_abs_residual:.3e}, "
                f"success={report.success}"
            )
        if print_solve_report:
            self.print_solve_report(report)
        if print_dihedral_status:
            self._print_dihedral_status(
                max_items=dihedral_status_max_items,
                unit=unit,
            )
        if (verbose or print_residual_warning) and report.max_abs_residual > residual_warning_tol:
            print("WARNING: constraints are not sufficiently satisfied.")

        return {
            "constraint_info": constraint_info,
            "report": report,
            "hybrid": hybrid,
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
    unit: Literal["rad", "deg"] = "deg",
    fixed_triangle_surface_id: Optional[str] = None,
    valley_z: float = 0.0,
    strict_unique_edges: bool = False,
    mountain_height: Optional[float] = None,
    valley_height: float = 0.0,
    X0: Optional[np.ndarray] = None,
    max_nfev: int = 5000,
    tol: float = 1e-10,
    residual_warning_tol: float = 1e-5,
    verbose: bool = True,
    print_metadata_summary: bool = False,
    print_constraint_info: bool = False,
    print_solve_report: bool = False,
    print_dihedral_status: bool = False,
    print_residual_warning: bool = False,
    dihedral_status_max_items: int = 20,
    x_scale: float | np.ndarray | Literal["jac"] = "jac",
    steps: int = 3,
) -> dict:
    """Add hexagon constraints and solve using hybrid tangent/nonlinear steps.

    ``steps`` is a nonnegative integer, defaulting to 3. Positive values take
    ``steps`` equal tangent increments from the flat 180-degree coordinates
    to ``final_dihedral``, then refine with the nonlinear solver. No nonlinear
    starting-pose solve is performed. ``steps=0`` uses the original direct
    nonlinear solve at the final angle. If tangent prediction fails, the
    endpoint solve falls back to its target-angle height guess.

    Hole contours move with their host panels and add no independent solver
    variables. ``report.x`` includes the restored hole points in the model's
    original point order.

    Rank and mobility are not calculated. Their SolveReport fields are -1
    (not computed). Use
    ``analyze_kinematics(model)`` separately to evaluate them at the solved
    configuration; that diagnostic does not perform another solve.

    Sparse nonlinear refinement uses fixed LSMR settings ``atol=btol=1e-10``
    and ``maxiter=5000``. ``tol`` remains the outer stopping tolerance. For
    ``steps=0``, the automatic initial guess keeps the flat pattern's XY
    coordinates and sets triangle heights using
    ``valley_height + sqrt(3) / 2 * d * sin(final_dihedral)`` for mountain
    height. ``d`` is measured from the pattern and ``unit`` applies to
    ``final_dihedral``. ``mountain_height`` overrides this formula. For
    positive steps, ``X0`` supplies the flat 180-degree coordinates when
    provided; otherwise the model's current coordinates are used.

    ``max_nfev`` (default 5000) is the evaluation budget for final refinement.
    ``result["hybrid"]`` records requested/completed tangent steps, refinement
    evaluations, linear iterations, and any fallback reason. Its compatibility
    field ``initial_nfev`` is always zero. Tangent LSMR iterations do not
    consume ``max_nfev``.

    ``x_scale`` defaults to ``"jac"`` to scale variables by inverse Jacobian
    column norms. Use ``1.0`` to disable scaling, or positive per-coordinate
    scales in full-model or panel-only point order. The same scaling applies
    to tangent prediction. Scaling does not change the residual equations
    or their weights.

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
        if coordinate_indices is not None and np.ndim(x_scale) > 0:
            x_scale = np.asarray(x_scale, dtype=float)
            if x_scale.size == full_size:
                x_scale = x_scale.ravel()[coordinate_indices]
            elif x_scale.size != model.num_variables():
                raise ValueError(
                    f"Expected x_scale size {full_size} (with holes) or "
                    f"{model.num_variables()} (panel points), but got {x_scale.size}."
                )
        result = _HexagonKinematics(model).solve_kinematics(
            final_dihedral=final_dihedral,
            unit=unit,
            fixed_triangle_surface_id=fixed_triangle_surface_id,
            valley_z=valley_z,
            strict_unique_edges=strict_unique_edges,
            mountain_height=mountain_height,
            valley_height=valley_height,
            X0=X0,
            max_nfev=max_nfev,
            tol=tol,
            residual_warning_tol=residual_warning_tol,
            verbose=verbose,
            print_metadata_summary=print_metadata_summary,
            print_constraint_info=print_constraint_info,
            print_solve_report=print_solve_report,
            print_dihedral_status=print_dihedral_status,
            print_residual_warning=print_residual_warning,
            dihedral_status_max_items=dihedral_status_max_items,
            x_scale=x_scale,
            steps=steps,
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
