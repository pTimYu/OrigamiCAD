"""Analytic derivatives of Cadder constraint residuals."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from ._evaluation import CompiledConstraints

if TYPE_CHECKING:
    from .cadder import Cadder, Constraint


class JacobianBuilder:
    """Build analytic Jacobians for one constraint or the entire model.

    Rows follow ``model.residual_vector()`` (constraint insertion order, then
    residual component order). Columns follow ``model.get_coordinate_vector()``:
    x, y, z for each point in ``model.point_ids()`` order. A single-constraint
    matrix retains all model columns, including zeros for unrelated points.

    By default results are SciPy CSR matrices; ``sparse=False`` returns a dense
    NumPy array. ``X`` optionally supplies coordinates without mutating the model.
    Point order and constraints are read afresh on each public call, so builders
    can be reused after model edits. Within ``compiled()``, topology and sparse
    assembly are reused until that solve scope exits; structural edits must wait
    until afterward.

    Derivatives are undefined at zero-length geometric vectors and at the
    +/-pi discontinuity of a wrapped angle *residual*; these raise ValueError.
    """

    def __init__(self, model: Cadder):
        self.model = model
        self._compiled = None

    @contextmanager
    def compiled(self):
        """Reuse topology and geometry for one solve with no structural edits.

        Residual calls through the model share this plan, including overridden
        ``residual_vector`` methods used for instrumentation. Restore any outer
        scope even if evaluation or the solver raises an exception.
        """
        previous_builder = self._compiled
        previous_model = getattr(self.model, "_compiled_constraints", None)
        plan = CompiledConstraints(self.model)
        self._compiled = plan
        self.model._compiled_constraints = plan
        try:
            yield self
        finally:
            self._compiled = previous_builder
            if previous_model is None:
                del self.model._compiled_constraints
            else:
                self.model._compiled_constraints = previous_model

    def build(
        self, X: np.ndarray | None = None, *, sparse: bool = True
    ) -> csr_matrix | np.ndarray:
        """Return the full (number of scalar residuals, 3N) Jacobian."""
        plan = self._compiled or CompiledConstraints(self.model)
        return plan.jacobian(X, sparse=sparse)

    def for_constraint(
        self,
        constraint: str | Constraint,
        X: np.ndarray | None = None,
        *,
        sparse: bool = True,
    ) -> csr_matrix | np.ndarray:
        """Return one residual block's Jacobian, by constraint ID or object.

        ``add_fixed_point_constraint`` creates three ``fixed_coordinate``
        constraints; each returned ID can be passed here individually.
        """
        if isinstance(constraint, str):
            if constraint not in self.model.constraints:
                raise ValueError(f"Constraint '{constraint}' does not exist.")
            constraint = self.model.constraints[constraint]
        return CompiledConstraints(self.model, [constraint]).jacobian(X, sparse=sparse)
