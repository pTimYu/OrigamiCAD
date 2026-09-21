"""Sparse first-order dihedral prediction for hybrid kinematics solves."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
from scipy.sparse.linalg import lsmr

from .jacobian import JacobianBuilder

if TYPE_CHECKING:
    from .cadder import Cadder


@dataclass
class TangentPrediction:
    """Last completed Euler state and diagnostics for attempted linear solves.

    Success means that all linear solves terminated normally. Finite tangent
    steps still accumulate constraint drift, which the nonlinear solver must
    correct. Iteration counts include an attempted step that failed.
    """

    x: np.ndarray
    completed_steps: int
    linear_iterations: int
    failure_reason: str | None

    @property
    def success(self) -> bool:
        return self.failure_reason is None


def predict_dihedral_tangent(
    model: Cadder,
    angles_rad: Iterable[float],
    *,
    X0: np.ndarray | None = None,
    constraint_ids: Iterable[str] | None = None,
    linear_tol: float = 1e-10,
    maxiter: int = 5000,
    x_scale: float | np.ndarray | Literal["jac"] = "jac",
) -> TangentPrediction:
    """Predict coordinates along absolute unsigned dihedral angles in radians.

    The first angle describes ``X0`` (or the model coordinates when omitted)
    and configured targets. Each driven constraint must use
    ``sign * (pi - theta)`` as its
    signed-increment target. ``constraint_ids`` selects the driven signed
    constraints; when omitted all signed-increment constraints are driven.
    Other constraint targets remain fixed. The caller supplies a feasible
    starting state and performs final nonlinear refinement. The first angle
    may equal pi for an exactly flat starting state; subsequent angles must
    remain strictly below pi.

    Each forward Euler step solves ``J dx = b`` using the compiled analytic
    sparse Jacobian. Driven rows have ``b = -sign * dtheta``; all other rows
    are zero. There is no residual correction. Column scaling accepts positive
    scalar or per-coordinate scales. ``"jac"`` uses inverse column norms,
    retaining their running maxima as in SciPy's least-squares solver.

    Model coordinates are never changed. Signed targets in the model and the
    temporary compiled plan advance through completed steps. If a derivative
    or linear solve fails, the returned coordinates and targets correspond to
    the last completed step. Invalid arguments raise ValueError. A final
    nonlinear solve is still needed to assess endpoint geometric feasibility.
    """
    try:
        angles = np.asarray(list(angles_rad), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("angles_rad must be a one-dimensional sequence of angles.") from error
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("angles_rad must contain at least the starting angle.")
    if (not np.all(np.isfinite(angles)) or angles[0] <= 0 or angles[0] > np.pi
            or np.any((angles[1:] <= 0) | (angles[1:] >= np.pi))):
        raise ValueError(
            "angles_rad must be finite and positive; only the first angle may equal pi."
        )
    try:
        tolerance = float(linear_tol)
    except (TypeError, ValueError) as error:
        raise ValueError("linear_tol must be finite and positive.") from error
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("linear_tol must be finite and positive.")
    if not isinstance(maxiter, Integral) or isinstance(maxiter, bool) or maxiter <= 0:
        raise ValueError("maxiter must be a positive integer.")

    try:
        x = np.asarray(model.get_coordinate_vector() if X0 is None else X0, dtype=float).copy()
    except (TypeError, ValueError) as error:
        raise ValueError("X0 must be a one-dimensional coordinate vector.") from error
    if x.ndim != 1 or x.size == 0 or x.size != model.num_variables():
        raise ValueError(f"X0 must be a one-dimensional vector of size {model.num_variables()}.")
    if X0 is not None and not np.all(np.isfinite(x)):
        raise ValueError("X0 must contain only finite values.")
    if isinstance(x_scale, str):
        if x_scale != "jac":
            raise ValueError("x_scale must be 'jac' or finite positive coordinate scales.")
        fixed_scale = None
    else:
        try:
            fixed_scale = np.asarray(x_scale, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("x_scale must be 'jac' or finite positive coordinate scales.") from error
        if fixed_scale.ndim == 0:
            fixed_scale = np.full(x.shape, float(fixed_scale))
        if fixed_scale.shape != x.shape:
            raise ValueError(f"x_scale must be a scalar or have shape {x.shape}.")
        if not np.all(np.isfinite(fixed_scale)) or np.any(fixed_scale <= 0):
            raise ValueError("x_scale must contain finite positive values.")

    signed_constraints = [
        (key, constraint) for key, constraint in model.constraints.items()
        if constraint.kind == "dihedral_signed_increment"
    ]
    if constraint_ids is None:
        selected_ids = {key for key, _ in signed_constraints}
    else:
        if isinstance(constraint_ids, (str, bytes)):
            raise ValueError("constraint_ids must be a sequence of constraint identifiers.")
        try:
            requested_ids = list(constraint_ids)
        except TypeError as error:
            raise ValueError("constraint_ids must be a sequence of constraint identifiers.") from error
        if not all(isinstance(key, str) for key in requested_ids):
            raise ValueError("constraint_ids must contain string identifiers.")
        selected_ids = set(requested_ids)
        if len(selected_ids) != len(requested_ids):
            raise ValueError("constraint_ids must not contain duplicate identifiers.")
        for key in requested_ids:
            if key not in model.constraints:
                raise ValueError(f"Constraint '{key}' does not exist.")
            if model.constraints[key].kind != "dihedral_signed_increment":
                raise ValueError(f"Constraint '{key}' must be a signed-increment dihedral.")
    driven_positions = np.asarray([
        index for index, (key, _) in enumerate(signed_constraints) if key in selected_ids
    ], dtype=int)
    driven = [constraint for key, constraint in signed_constraints if key in selected_ids]
    if not driven:
        raise ValueError("The model must contain signed-increment dihedral constraints.")
    signs = np.asarray([constraint.data["sign"] for constraint in driven], dtype=float)
    if not np.all(np.isin(signs, [-1.0, 1.0])):
        raise ValueError("Each driven dihedral sign must be -1 or +1.")
    targets = np.asarray([constraint.data["target_increment"] for constraint in driven], dtype=float)
    if not np.allclose(targets, signs * (np.pi - angles[0]), rtol=0, atol=1e-10):
        raise ValueError("Model dihedral targets must be configured at angles_rad[0].")
    if not np.all(np.isfinite(x)):
        return TangentPrediction(x, 0, 0, "The starting coordinates contain nonfinite values.")

    completed_steps = linear_iterations = 0
    failure_reason = None
    running_column_norms = None
    builder = JacobianBuilder(model)
    step = 0
    try:
        with builder.compiled(), np.errstate(over="raise", invalid="raise", divide="raise"):
            plan = builder._compiled
            group = plan.groups["dihedral_signed_increment"]
            rhs = np.zeros(plan.n_rows, dtype=float)
            for step, (theta, next_theta) in enumerate(zip(angles[:-1], angles[1:]), start=1):
                jacobian = builder.build(x, sparse=True)
                if not np.all(np.isfinite(jacobian.data)):
                    failure_reason = f"Step {step}: the Jacobian contains nonfinite values."
                    break
                if fixed_scale is None:
                    norms = np.sqrt(np.asarray(jacobian.power(2).sum(axis=0)).ravel())
                    if running_column_norms is None:
                        norms[norms == 0] = 1.0
                        running_column_norms = norms
                    else:
                        running_column_norms = np.maximum(running_column_norms, norms)
                    scale = 1.0 / running_column_norms
                else:
                    scale = fixed_scale
                scaled_jacobian = jacobian.multiply(scale).tocsr()
                rhs[group["rows"][driven_positions]] = -signs * (next_theta - theta)
                scaled_dx, status, iterations, *_ = lsmr(
                    scaled_jacobian, rhs, atol=tolerance, btol=tolerance,
                    maxiter=int(maxiter),
                )
                linear_iterations += int(iterations)
                dx = scale * scaled_dx
                if not np.all(np.isfinite(dx)):
                    failure_reason = f"Step {step}: LSMR returned a nonfinite displacement."
                    break
                if status not in {0, 1, 2, 4, 5}:
                    description = {
                        3: "condition-number limit", 6: "machine-precision condition limit",
                        7: "iteration budget exhausted",
                    }.get(int(status), "unrecognized termination")
                    failure_reason = f"Step {step}: LSMR status {status} ({description})."
                    break
                next_x = x + dx
                if not np.all(np.isfinite(next_x)):
                    failure_reason = f"Step {step}: the coordinate update is nonfinite."
                    break
                next_targets = signs * (np.pi - next_theta)
                group["target"][driven_positions] = next_targets
                for constraint, target in zip(driven, next_targets):
                    constraint.data["target_increment"] = float(target)
                x = next_x
                completed_steps += 1
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        context = f"Step {step}" if step else "Predictor setup"
        failure_reason = f"{context}: {type(error).__name__}: {error}"

    return TangentPrediction(x, completed_steps, linear_iterations, failure_reason)
