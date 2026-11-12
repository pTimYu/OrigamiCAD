from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from ...core._evaluation import CompiledConstraints
from ...core._tangent import predict_dihedral_tangent
from ...core.panel_holes import cut_point_ids, panel_only_geometry
from .metadata import hex_constraint_plan, normalize_hex_creases

if TYPE_CHECKING:
    from ...core.cadder import Cadder


def dihedral_constraint_id(unit_count: int, local_index: int, side: str) -> str:
    """Name of the hinge constraint for one unit-local crease."""
    return f"dihedral_signed_u{unit_count}_i{local_index}_{side}"


class _HexagonKinematics:
    """Pattern-specific kinematic operations for a hexagon model."""

    def __init__(self, model: Cadder):
        self._model = model

    def __getattr__(self, name: str) -> Any:
        """Delegate generic geometry and solver operations to the Cadder."""
        return getattr(self._model, name)

    def _fold_angle(
        self,
        value: float,
        unit: Literal["rad", "deg"],
        name: str,
    ) -> float:
        """Return a dihedral angle in radians, strictly between 0 and pi."""
        theta = self._angle_to_rad(value, unit=unit)
        if not (0.0 < theta < np.pi):
            raise ValueError(f"{name} must be between 0 and 180 degrees.")
        return theta

    def _require_hex_units(self) -> None:
        if not self.hex_units:
            raise ValueError(
                "No hex-unit metadata found. Build the pattern with "
                "draw_hex_loops() or build_packaging() before creating "
                "the Cadder model."
            )

    def _triangle_metadata(self):
        plan = hex_constraint_plan(self._model)
        missing = set(plan.triangles) - plan.triangle_kinds.keys()
        if missing:
            raise ValueError(f"Missing triangle_kinds metadata for {sorted(missing)}. Rebuild the 2D metadata.")
        return plan.triangles, plan.triangle_kinds

    def _add_dihedral_constraints(self, fold_amount: float, entries) -> list[str]:
        """Add or retarget one signed hinge per physical crease."""
        constraint_ids = []
        for crease, (unit_count, local_index, side) in entries:
            constraint_id = dihedral_constraint_id(unit_count, local_index, side)
            constraint_ids.append(constraint_id)
            existing = self.constraints.get(constraint_id)
            if existing is not None and existing.kind == "dihedral_signed_increment":
                # Repeated setup on a model that may already be folded: keep
                # the hinge's flat reference angle and orientation, and move
                # only its target to the newly requested angle.
                existing.data["target_increment"] = float(
                    existing.data["sign"] * fold_amount
                )
                continue
            edge_start, edge_end = crease["edge"]
            triangle_point = self._first_non_edge_vertex(
                self._surface_vertices(crease["triangle"]), edge_start, edge_end)
            quad_point = self._first_non_edge_vertex(
                self._surface_vertices(crease["quad"]), edge_start, edge_end)
            edge_start, edge_end = self._orient_flat_crease_triangle_right(
                edge_start, edge_end, triangle_point)
            sign = +1 if crease["kind"] == "valley" else -1
            self.add_dihedral_signed_increment_constraint(
                edge_start=edge_start, edge_end=edge_end,
                point_left=triangle_point, point_right=quad_point,
                target_increment=sign * fold_amount, unit="rad", sign=sign,
                crease_kind=crease["kind"], constraint_id=constraint_id,
            )
        return constraint_ids

    def _add_frame_constraints(self, surface_id: str) -> None:
        """Pin the reference triangle once to remove rigid-body motion.

        The IDs name the triangle, so repeated setup keeps the existing frame
        (and its original coordinates) instead of stacking a second one. A
        different reference triangle replaces the previous frame.
        """
        prefix = f"hex_frame_{surface_id}_"
        previous = [cid for cid in self.constraints if cid.startswith("hex_frame_")]
        if any(cid.startswith(prefix) for cid in previous):
            return
        for cid in previous:
            del self.constraints[cid]
        p0, p1, p2 = self._non_collinear_triplet_from_surface(surface_id)
        for pid, axis in (
            (p0, "x"), (p0, "y"), (p0, "z"), (p1, "y"), (p1, "z"), (p2, "z"),
        ):
            self.add_fixed_coordinate_constraint(
                pid, axis, constraint_id=f"{prefix}{pid}_{axis}"
            )

    def _add_kinematic_constraints(
        self,
        target_dihedral: float = 110.0,
        unit: Literal["rad", "deg"] = "deg",
        fixed_triangle_surface_id: Optional[str] = None,
        valley_z: float = 0.0,
        strict_unique_edges: bool = False,
    ) -> dict:
        self._require_hex_units()
        fold_amount = np.pi - self._fold_angle(target_dihedral, unit, "target_dihedral")
        plan = hex_constraint_plan(self._model)
        if strict_unique_edges and plan.skipped:
            raise ValueError(f"Invalid hexagon crease metadata: {len(plan.skipped)} skipped references.")

        triangle_ids, triangle_kinds = self._triangle_metadata()
        if not triangle_ids:
            raise ValueError("No triangle surfaces found.")
        valley_triangle_ids = [
            sid for sid in triangle_ids if triangle_kinds[sid] == "valley"
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
        self._add_frame_constraints(fixed_triangle_surface_id)

        pinned_points = set()
        for tri_id in triangle_ids:
            if triangle_kinds[tri_id] == "valley":
                constraint_id = f"valley_z_{tri_id}"
                constraint = self.constraints.get(constraint_id)
                if constraint is None:
                    self.add_surface_z_value_constraint(
                        tri_id, z_value=valley_z, constraint_id=constraint_id,
                    )
                    constraint = self.constraints[constraint_id]
                    constraint.data["from_valley_z"] = True
                elif constraint.data.get("from_valley_z"):
                    # Repeated setup: follow a changed valley_z. A height
                    # constraint the caller added under this ID is theirs
                    # and keeps its own value.
                    constraint.data["z_value"] = float(valley_z)
                if (constraint.kind == "surface_z_value"
                        and constraint.data["surface"] == tri_id):
                    # Equal absolute vertex heights already imply every
                    # horizontal-surface equation for this triangle.
                    pinned_points.update(self._surface_vertices(tri_id))
                    continue
            if f"horizontal_{tri_id}" not in self.constraints:
                self.add_horizontal_surface_constraint(tri_id)

        for cid in [cid for cid in self.constraints if cid.startswith("hex_frame_")]:
            data = self.constraints[cid].data
            if data["axis"] == "z" and data["point"] in pinned_points:
                # The frame only removes rigid-body motion, and a pinned
                # height already does that for this coordinate. Keeping the
                # frame's own height as well would fight valley_z whenever
                # the two differ.
                del self.constraints[cid]

        dihedral_ids = self._add_dihedral_constraints(fold_amount, plan.crease_entries)
        return {
            "fixed_triangle": fixed_triangle_surface_id,
            "num_dihedral_constraints": len(dihedral_ids),
            "dihedral_constraint_ids": tuple(dihedral_ids),
            "num_skipped_crease_edges": len(plan.skipped),
            "skipped_crease_edges": list(plan.skipped),
            "num_shared_crease_references": len(plan.shared),
            "shared_crease_references": list(plan.shared),
        }

    def _update_dihedral_target(
        self,
        target_dihedral: float,
        unit: Literal["rad", "deg"] = "deg",
    ) -> None:
        fold_amount = np.pi - self._fold_angle(target_dihedral, unit, "target_dihedral")
        for constraint in self.constraints.values():
            if constraint.kind == "dihedral_signed_increment":
                constraint.data["target_increment"] = float(
                    constraint.data["sign"] * fold_amount
                )

    def _starting_fold_angle(
        self,
        constraint_ids,
        coordinates: np.ndarray,
    ) -> Optional[float]:
        """Return the dihedral angle the hinges share at ``coordinates``.

        A model that has never been folded is at pi. A model that an earlier
        solve already folded is at that solve's angle, which is where tangent
        prediction has to start. Return None when the hinges disagree, because
        no single starting angle describes such a state.
        """
        constraints = [self.constraints[cid] for cid in constraint_ids]
        if not constraints:
            return float(np.pi)
        try:
            residual = CompiledConstraints(self._model, constraints).residual(coordinates)
        except ValueError:
            # Unusable coordinates are reported by the predictor itself.
            return float(np.pi)
        targets = np.array([c.data["target_increment"] for c in constraints])
        signs = np.array([c.data["sign"] for c in constraints], dtype=float)
        increments = (residual + targets + np.pi) % (2.0 * np.pi) - np.pi
        angles = np.pi - signs * increments
        if float(np.ptp(angles)) > 1e-5:
            return None
        angle = float(np.mean(angles))
        if abs(angle - np.pi) <= 1e-9:
            return float(np.pi)
        return angle if 0.0 < angle < np.pi else None

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
                float(np.linalg.norm(self.point_array(end) - self.point_array(start)))
                for start, end in mid_edges
            )

        side_length = float(np.mean(side_lengths))
        length_tolerance = 1e-6 * max(1.0, side_length)
        if not np.isfinite(side_length) or side_length <= length_tolerance:
            raise ValueError("The measured hexagon side length must be positive.")
        if max(abs(length - side_length) for length in side_lengths) > length_tolerance:
            raise ValueError(
                "Hex-unit side lengths are inconsistent; one value of d cannot "
                "be used in the automatic height formula."
            )

        theta = self._fold_angle(dihedral_angle, unit, "dihedral_angle")
        return float(valley_height) + float(
            np.sqrt(3.0) / 2.0 * side_length * np.sin(theta)
        )

    def _print_metadata_summary(self) -> None:
        print("Hexagon metadata summary")
        print("------------------------")
        if not self.hex_units:
            print("No hex-unit metadata found.")
            return

        normalize_hex_creases(self._model)
        totals = dict.fromkeys(("triangles", "parallelograms", "local_creases"), 0)
        for unit_data in self.hex_units:
            counts = {key: len(unit_data.get(key, [])) for key in totals}
            for key, count in counts.items():
                totals[key] += count
            print(
                f"Unit {unit_data.get('count', 'unknown')}: "
                f"triangles={counts['triangles']}, quads={counts['parallelograms']}, "
                f"local creases={counts['local_creases']}"
            )

        print("")
        print(f"Total units:         {len(self.hex_units)}")
        print(f"Triangle references: {totals['triangles']}")
        print(f"Quad references:     {totals['parallelograms']}")
        print(f"Local crease refs:   {totals['local_creases']}")
        print(f"Physical creases:    {len(self.hex_creases)}")

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
                data["edge_start"], data["edge_end"],
                data["point_left"], data["point_right"],
            )
            actual_increment = self._wrap_to_pi(current_angle - data["initial_angle"])
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
            rows.append((abs(residual), cid, data.get("crease_kind"), *values))

        rows.sort(reverse=True, key=lambda row: row[0])
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


def _panel_vector(name, value, coordinate_indices, full_size, panel_size):
    """Reduce a full-model vector to panel points; accept one already reduced."""
    value = np.asarray(value, dtype=float)
    if value.size == full_size:
        return value.ravel()[coordinate_indices]
    if value.size != panel_size:
        raise ValueError(
            f"Expected {name} size {full_size} (with holes) or "
            f"{panel_size} (panel points), but got {value.size}."
        )
    return value


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
    max_nfev: Optional[int] = 5000,
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
    linear_solver: Literal["direct", "lsmr"] = "direct",
) -> dict:
    """Add hexagon constraints and solve using hybrid tangent/nonlinear steps.

    ``steps`` is a nonnegative integer, defaulting to 3. Positive values take
    ``steps`` equal tangent increments from the flat 180-degree coordinates
    to ``final_dihedral``, then refine with the nonlinear solver. No nonlinear
    starting-pose solve is performed. ``steps=0`` solves directly at the final
    angle from a height guess. If tangent prediction fails, the endpoint solve
    falls back to that guess.

    A solved model can be passed again with a different ``final_dihedral``.
    Its existing constraints are retargeted instead of being added a second
    time, and the tangent steps start from the angle the model is currently
    folded to rather than from 180 degrees. If the hinges of the starting
    coordinates are not at one common angle, prediction is skipped and
    ``result["hybrid"]["fallback_reason"]`` says so.

    ``valley_z`` is the height the valley panels are pinned to. It need not
    match the pattern's current height, and a later call with a different
    value moves the panels there.

    Hole contours move with their host panels and add no independent solver
    variables. ``report.x`` includes the restored hole points in the model's
    original point order.

    Rank and mobility are not calculated. Their SolveReport fields are -1
    (not computed). Use ``analyze_kinematics(model)`` separately to evaluate
    them at the solved configuration; that diagnostic does not perform
    another solve.

    ``linear_solver="direct"`` (the default) uses sparse direct
    factorizations for both the tangent steps and the refinement. ``"lsmr"``
    uses the iterative LSMR solver for both, with ``atol=btol=1e-10`` and
    ``maxiter=5000``, inside SciPy's trust-region solver for the refinement.
    ``tol`` is the outer stopping tolerance of either. For ``steps=0``, the
    automatic initial guess
    keeps the flat pattern's XY coordinates and sets triangle heights using
    ``valley_height + sqrt(3) / 2 * d * sin(final_dihedral)`` for mountain
    height. ``d`` is measured from the pattern and ``unit`` applies to
    ``final_dihedral``. ``mountain_height`` overrides this formula. For
    positive steps, ``X0`` supplies the starting coordinates when provided;
    otherwise the model's current coordinates are used.

    ``max_nfev`` (default 5000) is the evaluation budget for final refinement.
    ``result["hybrid"]`` records requested/completed tangent steps, refinement
    evaluations, tangent LSMR iterations (zero with the direct solver), and
    any fallback reason. Tangent steps do not consume ``max_nfev``.

    ``x_scale`` defaults to ``"jac"`` to scale variables by inverse Jacobian
    column norms. Use ``1.0`` to disable scaling, or positive per-coordinate
    scales in full-model or panel-only point order. The same scaling applies
    to tangent prediction. Scaling does not change the residual equations
    or their weights.

    Shared unit references to a physical crease are valid, including with
    ``strict_unique_edges=True``. Strict mode rejects missing geometry;
    conflicting crease definitions always raise before adding constraints.
    ``constraint_info`` reports sharing via ``num_shared_crease_references``.
    """
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    if max_nfev is not None and (
        isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or max_nfev < 1
    ):
        raise ValueError("max_nfev must be a positive integer or None.")
    kinematics = _HexagonKinematics(model)
    theta = kinematics._fold_angle(final_dihedral, unit, "final_dihedral")

    full_size = model.num_variables()
    with panel_only_geometry(model) as coordinate_indices:
        if coordinate_indices is not None:
            sizes = (coordinate_indices, full_size, model.num_variables())
            if X0 is not None:
                X0 = _panel_vector("X0", X0, *sizes)
            if np.ndim(x_scale) > 0:
                x_scale = _panel_vector("x_scale", x_scale, *sizes)

        if print_metadata_summary:
            kinematics._print_metadata_summary()
        constraint_info = kinematics._add_kinematic_constraints(
            target_dihedral=final_dihedral,
            unit=unit,
            fixed_triangle_surface_id=fixed_triangle_surface_id,
            valley_z=valley_z,
            strict_unique_edges=strict_unique_edges,
        )
        if print_constraint_info:
            print(constraint_info)

        hybrid = {
            "requested_steps": int(steps), "completed_steps": 0,
            "refinement_nfev": 0, "linear_iterations": 0, "fallback_reason": None,
        }
        guess = None
        if steps:
            driven_ids = constraint_info["dihedral_constraint_ids"]
            final_targets = {
                cid: model.constraints[cid].data["target_increment"] for cid in driven_ids
            }
            start_coordinates = (
                np.asarray(X0, dtype=float).copy()
                if X0 is not None else model.get_coordinate_vector().copy()
            )
            # pi for a flat model; the previous angle when re-solving a model
            # that is already folded.
            start_theta = kinematics._starting_fold_angle(driven_ids, start_coordinates)
        if steps and start_theta is None:
            hybrid["fallback_reason"] = (
                "The starting coordinates are not at one uniform dihedral "
                "angle, so tangent prediction was skipped."
            )
        elif steps:
            try:
                for cid in driven_ids:
                    data = model.constraints[cid].data
                    data["target_increment"] = float(data["sign"] * (np.pi - start_theta))
                prediction = predict_dihedral_tangent(
                    model, np.linspace(start_theta, theta, int(steps) + 1),
                    X0=start_coordinates, constraint_ids=driven_ids,
                    linear_tol=1e-10, maxiter=5000, x_scale=x_scale,
                    linear_solver=linear_solver,
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
                    model.constraints[cid].data["target_increment"] = target

        if guess is None and X0 is not None:
            guess = np.asarray(X0, dtype=float).copy()
        elif guess is None:
            height = mountain_height
            if height is None:
                height = kinematics._automatic_mountain_height(
                    dihedral_angle=theta, unit="rad", valley_height=valley_height,
                )
            guess = kinematics._initial_guess(
                mountain_height=height, valley_height=valley_height,
            )
        report = model.solve(
            X0=guess,
            update_model=True,
            max_nfev=max_nfev,
            tol=tol,
            compute_rank=False,
            x_scale=x_scale,
            linear_solver=linear_solver,
        )
        hybrid["refinement_nfev"] = report.nfev

        if verbose:
            print(
                f"[solve] target_dihedral={final_dihedral:.3f} {unit}, "
                f"max_residual={report.max_abs_residual:.3e}, "
                f"success={report.success}"
            )
        if print_solve_report:
            model.print_solve_report(report)
        if print_dihedral_status:
            kinematics._print_dihedral_status(
                max_items=dihedral_status_max_items, unit=unit,
            )
        if (verbose or print_residual_warning) and report.max_abs_residual > residual_warning_tol:
            print("WARNING: constraints are not sufficiently satisfied.")

    if coordinate_indices is not None:
        report.x = model.get_coordinate_vector()
    return {"constraint_info": constraint_info, "report": report, "hybrid": hybrid}


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
        cut_points = cut_point_ids(model)
        jacobian = jacobian[:, [
            3 * index + axis
            for index, pid in enumerate(model.point_ids()) if pid not in cut_points
            for axis in range(3)
        ]]

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
