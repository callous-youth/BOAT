from jittor import Module
from typing import List, Callable, Dict
from ..higher_jit.patch import _MonkeyPatchBase
from boat_jit.operation_registry import register_class
from boat_jit.hyper_ol.hyper_gradient import HyperGradient


@register_class
class RGT(HyperGradient):
    """
    Calculation of the hyper gradient of the upper-level variables with Reverse Gradient Truncation (RGT) _`[1]`.

    Parameters
    ----------
        :param ll_objective: The lower-level objective of the BLO problem.
        :type ll_objective: callable
        :param ul_objective: The upper-level objective of the BLO problem.
        :type ul_objective: callable
        :param ll_model: The lower-level model of the BLO problem.
        :type ll_model: torch.nn.Module
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: torch.nn.Module
        :param ll_var: List of variables optimized with the lower-level objective.
        :type ll_var: List
        :param ul_var:  of variables optimized with the upper-level objective.
        :type ul_var: List
        :param solver_config: Dictionary containing solver configurations.
        :type solver_config: dict

    References
    ----------
    _`[1]` Shaban A, Cheng C A, Hatch N, et al. Truncated back-propagation for bilevel optimization[C]. In AISTATS,2019.
    """

    def __init__(
            self,
            ll_objective: Callable,
            ul_objective: Callable,
            ll_model: Module,
            ul_model: Module,
            ll_var: List,
            ul_var: List,
            solver_config: Dict,
    ):
        super(RGT, self).__init__(ll_objective, ul_objective, ul_model, ll_model, ll_var, ul_var, solver_config)
        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]
        self.truncate_iter = solver_config["RGT"]["truncate_iter"]

    def compute_gradients(
            self,
            ll_feed_dict: Dict,
            ul_feed_dict: Dict,
            auxiliary_model: _MonkeyPatchBase,
            max_loss_iter: int = 0,
            hyper_gradient_finished: bool = False,
            next_operation: str = None,
            **kwargs
    ):
        """
        Compute the hyper-gradients of the upper-level variables with the data from feed_dict and patched models.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param auxiliary_model: A patched lower model wrapped by the `higher` library.
            It serves as the lower-level model for optimization.
        :type auxiliary_model: _MonkeyPatchBase

        :param max_loss_iter: The number of iteration used for backpropagation.
        :type max_loss_iter: int

        :param next_operation: The next operator for the calculation of the hypergradient.
        :type next_operation: str

        :param hyper_gradient_finished: A boolean flag indicating whether the hypergradient computation is finished.
        :type  hyper_gradient_finished: bool

        :returns: the current upper-level objective
        """
        assert hyper_gradient_finished is False, "Hypergradient computation should not be finished"
        assert (
            self.truncate_iter > 0
        ), "With RGT operation, 'truncate_iter' should be greater than 0"
        assert next_operation is not None, "Next operation should be defined"
        lower_model_params = kwargs.get("lower_model_params", list(auxiliary_model.parameters(time=max_loss_iter)))
        return {'ll_feed_dict': ll_feed_dict, 'ul_feed_dict': ul_feed_dict, 'auxiliary_model': auxiliary_model,
                'max_loss_iter': max_loss_iter, 'hyper_gradient_finished': False, 'lower_model_params':lower_model_params, **kwargs}
