"""Elastic energy, loads, and hard constraints of a bar-and-hinge model.

A :class:`MechanicalModel` splits a set of ``Cadder`` constraints in two:

* *hard* constraints are satisfied exactly, ``c(X) = 0``;
* *elastic* elements store the energy ``U(X) = 1/2 * sum(k_i * r_i(X)**2)``,
  where ``r_i`` is the same residual a hard constraint of that kind would
  have: a bar's change of length, or a crease's deviation from its rest angle.

Loads are potentials ``W(X)`` scaled by one load factor, and further
potentials ``V(X)`` such as panel contact act at full strength, so the total
potential energy is ``U(X) + V(X) + load * W(X)``. Residuals, Jacobians, and
second derivatives of the elements come from :class:`CompiledConstraints`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags

from ._evaluation import CompiledConstraints

if TYPE_CHECKING:
    from .cadder import Cadder, Constraint


class AreaPotential:
    """Enclosed xy area of a closed loop of points: uniform in-plane pressure.

    With a load factor ``p`` the potential ``p * A`` pushes every loop edge
    inward with a force ``p * edge length`` along its in-plane normal, shared
    equally between the edge's two points. ``point_indices`` are positions in
    the model's point order, listed around the loop in either direction; the
    area is measured positive in the given configuration.
    """

    def __init__(self, point_indices: Sequence[int], n_vars: int, X: np.ndarray):
        loop = np.asarray(point_indices, dtype=int)
        if loop.ndim != 1 or loop.size < 3:
            raise ValueError("An area loop needs at least three points.")
        self.n_vars = int(n_vars)
        self._x, self._y = 3 * loop, 3 * loop + 1
        self._sign = 1.0
        if self.value(X) < 0.0:
            self._sign = -1.0
        x, y = self._x, self._y
        x_next, y_next = np.roll(x, -1), np.roll(y, -1)
        half = np.full(loop.size, 0.5 * self._sign)
        rows = np.concatenate((x, y_next, x_next, y))
        columns = np.concatenate((y_next, x, y, x_next))
        self._hessian = csr_matrix(
            (np.concatenate((half, half, -half, -half)), (rows, columns)),
            shape=(self.n_vars, self.n_vars),
        )

    def value(self, X: np.ndarray) -> float:
        x, y = X[self._x], X[self._y]
        return float(0.5 * self._sign * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def gradient(self, X: np.ndarray) -> np.ndarray:
        x, y = X[self._x], X[self._y]
        gradient = np.zeros(self.n_vars)
        np.add.at(gradient, self._x, 0.5 * self._sign * (np.roll(y, -1) - np.roll(y, 1)))
        np.add.at(gradient, self._y, 0.5 * self._sign * (np.roll(x, 1) - np.roll(x, -1)))
        return gradient

    def hessian(self, X: np.ndarray) -> csr_matrix:
        return self._hessian


class DeadLoad:
    """Constant forces: the potential is ``-force . X``."""

    def __init__(self, force: np.ndarray):
        self.force = np.asarray(force, dtype=float).ravel()
        self._hessian = csr_matrix((self.force.size, self.force.size))

    def value(self, X: np.ndarray) -> float:
        return -float(self.force @ X)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        return -self.force

    def hessian(self, X: np.ndarray) -> csr_matrix:
        return self._hessian


class MechanicalModel:
    """Total potential energy and hard constraints over a model's coordinates.

    ``stiffness`` holds one value per elastic element; each element must be a
    single-residual constraint (a bar, a crease, or one fixed coordinate).
    The point order, and therefore the coordinate vector, is the model's own.
    """

    def __init__(
        self,
        model: Cadder,
        hard: Iterable[Constraint],
        elastic: Iterable[Constraint],
        stiffness: Sequence[float],
        loads: Iterable = (),
        potentials: Iterable = (),
    ):
        self.model = model
        self.hard = CompiledConstraints(model, list(hard))
        self.elastic = CompiledConstraints(model, list(elastic))
        self.stiffness = np.asarray(stiffness, dtype=float).ravel()
        if self.stiffness.size != self.elastic.n_rows:
            raise ValueError(
                f"Expected {self.elastic.n_rows} stiffness values, one per "
                f"elastic residual, but got {self.stiffness.size}."
            )
        if np.any(self.stiffness < 0) or not np.all(np.isfinite(self.stiffness)):
            raise ValueError("Stiffness values must be finite and nonnegative.")
        self.loads = list(loads)
        self.potentials = list(potentials)
        self.n_vars = self.hard.n_vars
        self.n_hard = self.hard.n_rows
        self._stiffness_matrix = diags(self.stiffness)

    def elastic_energy(self, X: np.ndarray) -> float:
        residual = self.elastic.residual(X)
        return 0.5 * float(self.stiffness @ (residual * residual))

    def load_potential(self, X: np.ndarray) -> float:
        return float(sum(load.value(X) for load in self.loads))

    def energy(self, X: np.ndarray, load: float = 0.0) -> float:
        return (
            self.elastic_energy(X)
            + sum(potential.value(X) for potential in self.potentials)
            + load * self.load_potential(X)
        )

    def gradient(self, X: np.ndarray, load: float = 0.0) -> np.ndarray:
        residual = self.elastic.residual(X)
        gradient = self.elastic.jacobian(X).T @ (self.stiffness * residual)
        for potential in self.loads:
            gradient = gradient + load * potential.gradient(X)
        for potential in self.potentials:
            gradient = gradient + potential.gradient(X)
        return gradient

    def hessian(self, X: np.ndarray, load: float = 0.0, multipliers=None) -> csr_matrix:
        """Hessian of the energy, plus the hard-constraint curvature if given.

        With ``multipliers`` this is the Hessian of the Lagrangian
        ``energy + multipliers . c``, which is the stiffness that decides
        stability on the constraint surface.
        """
        residual = self.elastic.residual(X)
        jacobian = self.elastic.jacobian(X)
        hessian = (
            jacobian.T @ self._stiffness_matrix @ jacobian
            + self.elastic.weighted_hessian(self.stiffness * residual, X)
        )
        for potential in self.loads:
            hessian = hessian + load * potential.hessian(X)
        for potential in self.potentials:
            hessian = hessian + potential.hessian(X)
        if multipliers is not None and self.n_hard:
            hessian = hessian + self.hard.weighted_hessian(multipliers, X)
        return csr_matrix(hessian)

    def stiffness_scale(self, X: np.ndarray) -> np.ndarray:
        """Positive diagonal that measures each coordinate's elastic stiffness."""
        jacobian = self.elastic.jacobian(X)
        diagonal = np.asarray(
            jacobian.multiply(jacobian).T @ self.stiffness
        ).ravel()
        largest = float(diagonal.max(initial=0.0)) or 1.0
        return np.maximum(diagonal, 1e-6 * largest)

    def step_limit(self, X: np.ndarray, step: np.ndarray) -> float:
        """Largest fraction of ``step`` every potential allows from ``X``."""
        return min(
            (
                potential.step_limit(X, step)
                for potential in self.potentials if hasattr(potential, "step_limit")
            ),
            default=1.0,
        )

    def constraint(self, X: np.ndarray) -> np.ndarray:
        return self.hard.residual(X)

    def constraint_jacobian(self, X: np.ndarray) -> csr_matrix:
        return self.hard.jacobian(X)
