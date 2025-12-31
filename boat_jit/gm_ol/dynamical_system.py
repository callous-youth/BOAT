import abc
from typing import List, Dict
from boat_jit.utils import DynamicalSystemRules, ResultStore

importlib = __import__("importlib")

from boat_jit.operation_registry import register_class


@register_class
class DynamicalSystem(object):
    def __init__(
        self, ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
    ) -> None:
        """
        Abstract class for defining lower-level optimization procedures in Bilevel Optimization (BLO).

        Parameters
        ----------
        ll_objective : Callable
            The lower-level objective function of the BLO problem.
        ul_objective : Callable
            The upper-level objective function of the BLO problem.
        ll_model : jittor.Module
            The lower-level model of the BLO problem.
        ul_model : jittor.Module
            The upper-level model of the BLO problem.
        lower_loop : int
            The number of iterations for lower-level optimization.
        solver_config : Dict[str, Any]
            A dictionary containing solver configurations. It includes details about optimization algorithms, hyperparameter settings, and additional configurations required for solving the BLO problem.
        """

        self.ll_objective = ll_objective
        self.ul_objective = ul_objective
        self.lower_loop = lower_loop
        self.ul_model = ul_model
        self.ll_model = ll_model
        self.solver_config = solver_config

    @abc.abstractmethod
    def optimize(self, **kwargs):
        pass
