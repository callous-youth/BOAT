from jittor import Module
from ..higher_jit.patch import _MonkeyPatchBase
from ..higher_jit.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable

from boat_jit.operation_registry import register_class
from boat_jit.dynamic_ol.dynamical_system import DynamicalSystem


@register_class
class DI(DynamicalSystem):
    """
    Implements the lower-level optimization procedure of the Dynamic Initialization (DI) _`[1]`.

    Parameters
    ----------
        :param ll_objective: The lower-level objective of the BLO problem.
        :type ll_objective: callable
        :param ul_objective: The upper-level objective of the BLO problem.
        :type ul_objective: callable
        :param ll_model: The lower-level model of the BLO problem.
        :type ll_model: Jittor.Module
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: Jittor.Module
        :param lower_loop: Number of iterations for lower-level optimization.
        :type lower_loop: int
        :param solver_config: Dictionary containing solver configurations.
        :type solver_config: dict


    References
    ----------
    _`[1]` R. Liu, Y. Liu, S. Zeng, and J. Zhang, "Towards Gradient-based Bilevel
     Optimization with Non-convex Followers and Beyond", in NeurIPS, 2021.
    """

    def __init__(
        self,
        ll_objective: Callable,
        ul_objective: Callable,
        ll_model: Module,
        ul_model: Module,
        lower_loop: int,
        solver_config: Dict[str, Any],
    ):

        super(DI, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )

    def optimize(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        auxiliary_opt: DifferentiableOptimizer,
        current_iter: int,
        next_operation: str = None,
        **kwargs
    ):
        """
        Execute the lower-level optimization procedure with the data from feed_dict and patched models.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param auxiliary_model: A patched lower model wrapped by the `higher` library.
            It serves as the lower-level model for optimization.
        :type auxiliary_model: _MonkeyPatchBase

        :param auxiliary_opt: A patched optimizer for the lower-level model,
            wrapped by the `higher` library. This optimizer allows for differentiable optimization.
        :type auxiliary_opt: DifferentiableOptimizer

        :param current_iter: The current iteration number of the optimization process.
        :type current_iter: int

        :param next_operation: The next operation to be executed in the optimization process.
        :type next_operation: str

        :param kwargs: Additional arguments for the optimization process.
        :type kwargs: dict

        :returns: None
        """
        assert next_operation is not None, "Next operation should be defined."
        return {
            "ll_feed_dict": ll_feed_dict,
            "ul_feed_dict": ul_feed_dict,
            "auxiliary_model": auxiliary_model,
            "auxiliary_opt": auxiliary_opt,
            "current_iter": current_iter,
            **kwargs,
        }
