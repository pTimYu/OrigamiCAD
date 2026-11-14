"""Stable equilibria of a :class:`MechanicalModel` under a prescribed load.

The equilibrium of the potential energy ``E(X) = U(X) + load * W(X)`` on the
hard constraints ``c(X) = 0`` satisfies

    grad E + Jc^T mu = 0,    c = 0.

:func:`solve_equilibrium` finds one by a damped Newton iteration on this
system. The damping ``sigma`` is an inverse pseudo-time step: a large value
gives a short step down the energy gradient, and zero gives a Newton step.
Steps are only accepted when the energy decreases, so from a state that has
lost stability the iteration descends to the next stable state, which is the
path an overdamped structure takes when it snaps under a dead load.

:func:`ramp_load` raises the load in adaptive increments and records the
states it passes through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.linalg import eigh, null_space
from scipy.sparse import bmat, diags, identity
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .mechanics import MechanicalModel

_DENSE_STABILITY_LIMIT = 1500


@dataclass
class EquilibriumResult:
    """Outcome of one equilibrium solve.

    ``residual`` is the largest component of ``grad E + Jc^T mu`` and
    ``violation`` the largest hard-constraint residual at ``x``.
    """

    x: np.ndarray
    multipliers: np.ndarray
    energy: float
    converged: bool
    iterations: int
    residual: float
    violation: float
    message: str


@dataclass
class LoadStep:
    """One accepted state of a load ramp."""

    load: float
    x: np.ndarray
    multipliers: np.ndarray
    energy: float
    elastic_energy: float
    load_potential: float
    lowest_eigenvalue: float
    iterations: int
    signature: object = None
    changed: bool = False
    branch_switched: bool = False


@dataclass
class LoadPath:
    """Accepted states of :func:`ramp_load` and the reason it ended."""

    steps: list[LoadStep] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def final(self) -> LoadStep:
        return self.steps[-1]


def _factorize(hessian, jacobian, damping, scale, softening):
    """Factorize the damped saddle-point matrix of one Newton step."""
    n, m = hessian.shape[0], jacobian.shape[0]
    block = hessian + damping * diags(scale) if damping else hessian
    if m:
        matrix = bmat(
            [[block, jacobian.T], [jacobian, -softening * identity(m)]], format="csc",
        )
    else:
        matrix = block.tocsc()
    return splu(matrix), n


def _restore(mech, lu, n, X, tolerance, iterations=8):
    """Pull ``X`` back onto the hard constraints with a frozen factorization."""
    for _ in range(iterations):
        violation = mech.constraint(X)
        if not violation.size or float(np.max(np.abs(violation))) <= tolerance:
            return X, True
        correction = lu.solve(np.concatenate((np.zeros(n), -violation)))[:n]
        if not np.all(np.isfinite(correction)):
            return X, False
        X = X + correction
    violation = mech.constraint(X)
    return X, float(np.max(np.abs(violation), initial=0.0)) <= tolerance


def solve_equilibrium(
    mech: MechanicalModel,
    X0: np.ndarray,
    load: float = 0.0,
    *,
    multipliers: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    constraint_tol: float = 1e-10,
    max_iterations: int = 3000,
    initial_damping: float = 0.0,
    verbose: bool = False,
) -> EquilibriumResult:
    """Descend from ``X0`` to an equilibrium at the given load factor.

    ``tol`` bounds the largest component of ``grad E + Jc^T mu`` relative to
    the largest elastic force at the start; ``constraint_tol`` bounds the hard
    constraint residuals. ``multipliers`` is an optional starting estimate,
    normally those of the previous load step. ``initial_damping`` greater
    than zero starts with short gradient-like steps, which is what a state
    that was just pushed off an unstable equilibrium needs.
    """
    n, m = mech.n_vars, mech.n_hard
    X = np.array(X0, dtype=float)
    mu = np.zeros(m) if multipliers is None else np.array(multipliers, dtype=float)
    scale = mech.stiffness_scale(X)
    # Only decides multipliers the constraints leave undetermined (redundant
    # caller constraints); small enough not to loosen the others.
    softening = 1e-13 / float(np.mean(scale))
    damping = float(initial_damping)

    # Start on the constraint surface, using the elastic scale as the metric.
    lu, _ = _factorize(diags(scale).tocsr(), mech.constraint_jacobian(X), 0.0, scale, softening)
    X, feasible = _restore(mech, lu, n, X, constraint_tol, iterations=30)
    if not feasible:
        return EquilibriumResult(
            X, mu, mech.energy(X, load), False, 0, np.inf,
            float(np.max(np.abs(mech.constraint(X)), initial=0.0)),
            "The starting state could not be placed on the hard constraints.",
        )

    energy = mech.energy(X, load)
    force_scale = max(
        float(np.max(np.abs(mech.elastic.jacobian(X).T @ (
            mech.stiffness * np.abs(mech.elastic.residual(X))
        )), initial=0.0)),
        abs(load) * max(
            (float(np.max(np.abs(p.gradient(X)), initial=0.0)) for p in mech.loads),
            default=0.0,
        ),
        1e-12,
    )
    # Forces cannot be resolved below the rounding error of stiffness times
    # size, which matters when nothing is strained and nothing is loaded.
    extent = float(np.ptp(X.reshape(-1, 3), axis=0).max()) or 1.0
    target = max(tol * force_scale, 1e-13 * float(np.max(scale)) * extent)
    energy_noise = 1e-13 * max(1.0, abs(energy))
    residual = np.inf
    message = "The maximum number of iterations was reached."
    converged = False
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        gradient = mech.gradient(X, load)
        jacobian = mech.constraint_jacobian(X)
        violation = mech.constraint(X) if m else np.zeros(0)
        hessian = mech.hessian(X, load, mu if m else None)
        accepted = False
        while True:
            try:
                lu, _ = _factorize(hessian, jacobian, damping, scale, softening)
                solution = lu.solve(np.concatenate((-gradient, -violation)))
            except RuntimeError:
                solution = None
            if solution is None or not np.all(np.isfinite(solution)):
                damping = max(10.0 * damping, 1e-6)
                if damping > 1e12:
                    break
                continue
            step, mu_new = solution[:n], solution[n:]
            residual = float(np.max(np.abs(
                gradient + (jacobian.T @ mu_new if m else 0.0)
            ), initial=0.0))
            if residual <= target:
                mu = mu_new
                converged, accepted = True, True
                message = "Converged to an equilibrium."
                break
            # Contact may only allow part of the step; the quadratic model
            # is then evaluated for the shortened step.
            fraction = mech.step_limit(X, step)
            if fraction < 1.0:
                fraction *= 0.9
            curvature = float(step @ (hessian @ step))
            damped = curvature + damping * float(step @ (scale * step))
            predicted = fraction * damped - 0.5 * fraction**2 * curvature
            trial_energy = np.inf
            if predicted > -100.0 * energy_noise:
                try:
                    trial, feasible = _restore(
                        mech, lu, n, X + fraction * step, constraint_tol,
                    )
                    if feasible and mech.step_limit(X, trial - X) >= 1.0:
                        trial_energy = mech.energy(trial, load)
                except ValueError:
                    # Degenerate trial geometry: treat it as a rejected step.
                    trial_energy = np.inf
            actual = energy - trial_energy
            if verbose:
                print(
                    f"[equilibrium {iteration}] residual={residual:.3e}, "
                    f"damping={damping:.3e}, predicted={predicted:.3e}, actual={actual:.3e}"
                )
            negligible = (
                abs(predicted) <= 100.0 * energy_noise
                and abs(actual) <= 100.0 * energy_noise
            )
            if negligible or (predicted > 0.0 and actual >= 1e-4 * predicted):
                if actual >= 0.75 * predicted:
                    damping = 0.0 if damping < 1e-9 else 0.2 * damping
                X, mu, energy, accepted = trial, mu_new, trial_energy, True
                break
            damping = max(4.0 * damping, 1e-6)
            if damping > 1e12:
                break
        if converged:
            break
        if not accepted:
            message = "No energy-decreasing step was found."
            break

    violation = float(np.max(np.abs(mech.constraint(X)), initial=0.0))
    return EquilibriumResult(
        X, mu, energy, converged, iteration, residual, violation, message,
    )


def lowest_modes(
    mech: MechanicalModel,
    X: np.ndarray,
    load: float,
    multipliers: np.ndarray,
    count: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Smallest eigenvalues, and their modes, of the constrained stiffness.

    The stiffness is the Hessian of the Lagrangian restricted to motions the
    hard constraints allow. A negative eigenvalue means the equilibrium is
    unstable. Small models use a dense null-space projection; large ones use
    a shift-invert iteration on the stiffness with the constraints penalized.
    """
    hessian = mech.hessian(X, load, multipliers if mech.n_hard else None)
    jacobian = mech.constraint_jacobian(X)
    n = mech.n_vars
    if n <= _DENSE_STABILITY_LIMIT:
        try:
            basis = null_space(jacobian.toarray()) if mech.n_hard else np.eye(n)
        except np.linalg.LinAlgError:
            basis = None
        if basis is not None:
            if basis.shape[1] == 0:
                return np.array([np.inf]), np.zeros((n, 1))
            values, vectors = eigh(basis.T @ (hessian @ basis))
            count = min(count, values.size)
            return values[:count], basis @ vectors[:, :count]

    scale = float(np.mean(np.abs(hessian.diagonal()))) or 1.0
    row_norm = float(jacobian.multiply(jacobian).sum(axis=1).mean()) or 1.0
    penalized = (hessian + (1e7 * scale / row_norm) * (jacobian.T @ jacobian)).tocsc()
    shift = -1e-3 * scale
    lu = splu(penalized - shift * identity(n, format="csc"))
    values, vectors = eigsh(
        penalized, k=min(count, n - 2), sigma=shift, which="LM",
        OPinv=LinearOperator((n, n), matvec=lu.solve),
    )
    order = np.argsort(values)
    return values[order], vectors[:, order]


def _leave_unstable_state(mech, result, load, mode, length, tol, constraint_tol):
    """Step off an unstable equilibrium along its critical mode, both ways."""
    mode = mode / float(np.max(np.abs(mode)))
    # A fixed sign convention keeps the choice reproducible when both
    # directions lead to states of equal energy.
    if mode[int(np.argmax(np.abs(mode)))] < 0:
        mode = -mode
    best = None
    for direction in (1.0, -1.0):
        push = direction * length * mode
        push = 0.9 * mech.step_limit(result.x, push) * push
        candidate = solve_equilibrium(
            mech, result.x + push, load,
            multipliers=result.multipliers, tol=tol, constraint_tol=constraint_tol,
            initial_damping=1e-2,
        )
        if candidate.converged and (
            best is None or candidate.energy < best.energy - 1e-10 * max(1.0, abs(best.energy))
        ):
            best = candidate
    return best


def ramp_load(
    mech: MechanicalModel,
    X0: np.ndarray,
    *,
    load_max: float,
    initial_step: float,
    min_step: Optional[float] = None,
    max_step: Optional[float] = None,
    signature: Optional[Callable[[np.ndarray], object]] = None,
    stop: Optional[Callable[[LoadStep], Optional[str]]] = None,
    change_resolution: float = 0.02,
    perturbation: float = 1e-3,
    stability_tol: float = 1e-8,
    tol: float = 1e-9,
    constraint_tol: float = 1e-10,
    max_steps: int = 2000,
    verbose: bool = False,
) -> LoadPath:
    """Raise the load from zero to ``load_max`` through stable equilibria.

    ``signature`` maps a state to a hashable description of its discrete
    configuration, for example which creases are acute. A step whose
    signature differs from the previous one contains a snap; the increment is
    then halved until it is below ``change_resolution`` times the load (or
    ``min_step``), which locates the load at which the snap happens. ``stop``
    may end the ramp by returning a reason for a newly accepted step.

    Every accepted state is checked for stability. An unstable equilibrium is
    left along its critical mode, in the direction of lower energy;
    ``perturbation`` is the size of that push in the model's length unit.
    """
    if load_max <= 0 or initial_step <= 0:
        raise ValueError("load_max and initial_step must be positive.")
    min_step = initial_step * 1e-4 if min_step is None else float(min_step)
    max_step = load_max if max_step is None else float(max_step)
    path = LoadPath()

    def settle(start, load, multipliers):
        """Solve at one load, leaving any unstable equilibrium that is found."""
        result = solve_equilibrium(
            mech, start, load, multipliers=multipliers, tol=tol, constraint_tol=constraint_tol,
        )
        switched = False
        lowest = np.nan
        for _ in range(6):
            if not result.converged:
                return None, lowest, switched
            values, modes = lowest_modes(mech, result.x, load, result.multipliers)
            lowest = float(values[0])
            reference = float(np.mean(mech.stiffness_scale(result.x)))
            if lowest >= -stability_tol * reference:
                return result, lowest, switched
            moved = _leave_unstable_state(
                mech, result, load, modes[:, 0], perturbation, tol, constraint_tol,
            )
            if moved is None:
                return None, lowest, switched
            result, switched = moved, True
        return None, lowest, switched

    def record(result, load, lowest, switched, previous_signature):
        current = signature(result.x) if signature else None
        step = LoadStep(
            load=load, x=result.x, multipliers=result.multipliers, energy=result.energy,
            elastic_energy=mech.elastic_energy(result.x),
            load_potential=mech.load_potential(result.x),
            lowest_eigenvalue=lowest, iterations=result.iterations,
            signature=current,
            changed=previous_signature is not None and current != previous_signature,
            branch_switched=switched,
        )
        return step

    result, lowest, switched = settle(X0, 0.0, None)
    if result is None:
        path.stop_reason = "No stable equilibrium was found for the unloaded structure."
        return path
    path.steps.append(record(result, 0.0, lowest, switched, None))

    increment = min(initial_step, load_max)
    while len(path.steps) < max_steps:
        last = path.final
        if last.load >= load_max:
            path.stop_reason = "The maximum load was reached."
            break
        load = min(last.load + increment, load_max)
        result, lowest, switched = settle(last.x, load, last.multipliers)
        if result is None:
            if increment <= min_step:
                path.stop_reason = (
                    f"No stable equilibrium was found beyond load {last.load:.6g}."
                )
                break
            increment = max(0.5 * increment, min_step)
            continue
        step = record(result, load, lowest, switched, last.signature)
        if step.changed and increment > max(min_step, change_resolution * load):
            increment = max(0.5 * increment, min_step)
            continue
        path.steps.append(step)
        if verbose:
            print(
                f"[load {step.load:.6g}] energy={step.energy:.6g}, "
                f"iterations={step.iterations}, lowest eigenvalue={step.lowest_eigenvalue:.3e}"
                + (", configuration changed" if step.changed else "")
            )
        reason = stop(step) if stop else None
        if reason:
            path.stop_reason = reason
            break
        if not step.changed and result.iterations <= 12:
            increment = min(1.5 * increment, max_step)
    else:
        path.stop_reason = "The maximum number of load steps was reached."
    return path
