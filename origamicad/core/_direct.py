"""Dogleg Gauss-Newton least squares using sparse direct linear solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import splu

_MESSAGES = {
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
}
# Keeps the scaled normal matrix factorizable when the Jacobian has
# dependent or empty columns (unconstrained points, flat creases).
_REGULARIZATION = 1e-12


def solve_normal_equations(J_scaled, gradient: np.ndarray) -> np.ndarray | None:
    """Solve ``(J^T J + eps I) y = gradient`` with one sparse factorization.

    ``J_scaled`` should have columns of at most unit norm, so that the small
    regularization only decides directions the Jacobian leaves undetermined;
    there it selects the minimum-norm solution, as LSMR does. Return None if
    the factorization fails or the solution is not finite.
    """
    normal = (J_scaled.T @ J_scaled).tocsc()
    try:
        solution = splu(
            normal + _REGULARIZATION * identity(normal.shape[0], format="csc")
        ).solve(gradient)
    except (RuntimeError, ValueError):
        return None
    return solution if np.all(np.isfinite(solution)) else None


@dataclass
class DirectResult:
    """The fields of SciPy's ``OptimizeResult`` that :meth:`Cadder.solve` reads."""

    x: np.ndarray
    cost: float
    jac: csr_matrix
    nfev: int
    status: int

    @property
    def success(self) -> bool:
        return self.status > 0

    @property
    def message(self) -> str:
        return _MESSAGES[self.status]


def solve_direct(
    residual: Callable[[np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray], csr_matrix],
    x0: np.ndarray,
    *,
    tol: float,
    max_nfev: int | None,
    x_scale: float | np.ndarray | Literal["jac"] = "jac",
) -> DirectResult | None:
    """Minimize ``0.5 * ||residual(x)||^2`` from ``x0``.

    Each iteration factorizes the scaled normal matrix once and takes a
    Powell dogleg step inside a trust region, so a good starting point
    converges at the Gauss-Newton rate without an inner iterative solve.
    ``tol`` is used for the same ``ftol``, ``xtol``, and ``gtol`` tests as
    SciPy's trust-region solver, and ``x_scale`` has the same meaning.

    Return None when a factorization or step is not finite; the caller then
    falls back to the iterative solver from the same starting point.
    """
    x = np.array(x0, dtype=float)
    if max_nfev is None:
        max_nfev = 100 * x.size
    jac_scaling = isinstance(x_scale, str)
    if jac_scaling:
        if x_scale != "jac":
            raise ValueError("`x_scale` must be 'jac' or array_like with positive numbers.")
    else:
        scale = np.broadcast_to(np.asarray(x_scale, dtype=float), x.shape)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("`x_scale` must be 'jac' or array_like with positive numbers.")

    f = residual(x)
    nfev = 1
    cost = 0.5 * float(f @ f)
    J = jacobian(x)
    column_norms = None
    radius = None
    status = 0

    while True:
        gradient = J.T @ f
        if float(np.max(np.abs(gradient), initial=0.0)) < tol:
            status = 1
            break
        if nfev >= max_nfev:
            break

        if jac_scaling:
            norms = np.sqrt(np.asarray(J.multiply(J).sum(axis=0)).ravel())
            norms[norms == 0] = 1.0
            column_norms = norms if column_norms is None else np.maximum(column_norms, norms)
            scale = 1.0 / column_norms
        if radius is None:
            radius = float(np.linalg.norm(x / scale)) or 1.0

        # Work in scaled variables x = scale * y, where the trust region is round.
        J_scaled = J.multiply(scale).tocsc()
        g_scaled = scale * gradient
        newton = solve_normal_equations(J_scaled, -g_scaled)
        if newton is None:
            return None
        newton_norm = float(np.linalg.norm(newton))
        curvature = float(np.sum((J_scaled @ g_scaled) ** 2))
        cauchy = -(float(g_scaled @ g_scaled) / curvature) * g_scaled if curvature > 0 else 0.0 * g_scaled
        cauchy_norm = float(np.linalg.norm(cauchy))

        accepted = False
        while nfev < max_nfev:
            if newton_norm <= radius:
                step = newton
            elif cauchy_norm >= radius:
                step = (radius / cauchy_norm) * cauchy
            else:
                # Along the dogleg from the Cauchy point toward the Newton
                # point, stop where the path leaves the trust region.
                leg = newton - cauchy
                a, b = float(leg @ leg), float(cauchy @ leg)
                c = cauchy_norm**2 - radius**2
                step = cauchy + ((-b + np.sqrt(b * b - a * c)) / a) * leg
            step_norm = float(np.linalg.norm(step))
            x_new = x + scale * step
            try:
                f_new = residual(x_new)
                cost_new = 0.5 * float(f_new @ f_new)
            except ValueError:
                # Degenerate trial geometry: treat it as a failed step.
                f_new, cost_new = None, np.inf
            nfev += 1

            model_step = J_scaled @ step
            predicted = -(float(g_scaled @ step) + 0.5 * float(model_step @ model_step))
            actual = cost - cost_new
            ratio = actual / predicted if predicted > 0 else 0.0
            if ratio < 0.25:
                radius = 0.25 * step_norm
            elif ratio > 0.75 and step_norm >= 0.95 * radius:
                radius *= 2.0

            if actual < tol * cost and ratio > 0.25:
                status = 2
            elif step_norm < tol * (tol + float(np.linalg.norm(x / scale))):
                status = 3
            if actual > 0:
                x, f, cost, accepted = x_new, f_new, cost_new, True
            if accepted or status:
                break

        if status or not accepted:
            break
        J = jacobian(x)

    return DirectResult(x=x, cost=cost, jac=J, nfev=nfev, status=status)
