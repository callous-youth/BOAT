from abc import abstractmethod
from typing import List
from boat_torch.operation_registry import register_class

importlib = __import__("importlib")


@register_class
class HyperGradient(object):
    """
    Base class for computing hyper-gradients of upper-level variables in bilevel optimization problems.

    This class provides an abstract interface for hyper-gradient computation that can be extended for specific methods such as Conjugate Gradient, Finite Differentiation, or First-Order Approximation.

    Parameters
    ----------
    ll_objective : callable
        The lower-level objective function of the bilevel optimization problem.

    ul_objective : callable
        The upper-level objective function of the bilevel optimization problem.

    ul_model : torch.nn.Module
        The upper-level model of the bilevel optimization problem.

    ll_model : torch.nn.Module
        The lower-level model of the bilevel optimization problem.

    ll_var : List[torch.Tensor]
        A list of variables optimized with the lower-level objective.

    ul_var : List[torch.Tensor]
        A list of variables optimized with the upper-level objective.

    solver_config : dict
        Dictionary containing configurations for the solver.
    """

    def __init__(
        self,
        ll_objective,
        ul_objective,
        ul_model,
        ll_model,
        ll_var,
        ul_var,
        solver_config,
    ):
        self.ll_objective = ll_objective
        self.ul_objective = ul_objective
        self.ul_model = ul_model
        self.ll_model = ll_model
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.solver_config = solver_config

    @abstractmethod
    def compute_gradients(self, **kwargs):
        pass
