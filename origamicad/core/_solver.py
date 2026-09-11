"""Adaptive accuracy for the sparse least-squares linear subproblems."""

import numpy as np


def adaptive_sparse_solve(solver, *, tol, max_nfev, **kwargs):
    """Warm-start progressively more accurate LSMR solves within one budget.

    SciPy's outer tolerances do not set LSMR's inner accuracy. Give its
    default 1e-6 accuracy eight residual evaluations, then reduce atol/btol
    by 100 if convergence still needs work. The accuracy floor is
    max(10 * eps, min(1e-6, tol)). At that floor, spend the
    remaining budget without further restarts.

    An ftol/xtol exit before gtol is met also triggers tightening. A gtol
    exit, or a terminal exit at the accuracy floor, keeps SciPy's outcome.
    This controls inner accuracy; it does not relax the user's outer tol
    or reinterpret optimizer success as an absolute residual guarantee.

    Restarts use public least_squares options, including on SciPy versions
    without an iteration callback. Their initial evaluations count toward
    the original budget. No dense Jacobian or rank calculation is needed.
    """
    budget = 100 * np.size(kwargs["x0"]) if max_nfev is None else max_nfev
    inner_tol = 1e-6
    floor = max(10 * np.finfo(float).eps, min(inner_tol, tol))
    nfev = 0
    njev = 0

    while True:
        at_floor = inner_tol <= floor
        remaining = budget - nfev
        result = solver(
            **kwargs,
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=remaining if at_floor else min(8, remaining),
            tr_options={"atol": inner_tol, "btol": inner_tol},
        )
        nfev += result.nfev
        njev += result.njev or 0

        # Negative status includes a caller's callback cancellation. Never
        # restart a cancelled solve or continue beyond the shared budget.
        if (result.status < 0 or result.optimality < tol or at_floor
                or nfev >= budget):
            result.nfev = nfev
            result.njev = njev
            return result

        kwargs["x0"] = result.x
        inner_tol = max(floor, inner_tol * 0.01)
