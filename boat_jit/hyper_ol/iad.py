import jittor as jit
from .hyper_gradient import HyperGradient
from jittor import Module
from typing import List, Callable, Dict
from ..higher_jit.patch import _MonkeyPatchBase
from boat_jit.utils.op_utils import update_tensor_grads


class IAD(HyperGradient):
    """
    Calculation of the hyper gradient of the upper-level variables with Initialization-based Auto Differentiation (IAD) _`[1]``.

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
    _`[1]` Finn C, Abbeel P, Levine S. Model-agnostic meta-learning for fast
    adaptation of deep networks[C]. in ICML, 2017.
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
        super(IAD, self).__init__(ul_objective, ul_model, ll_model, ll_var, ul_var)

    def compute_gradients(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        max_loss_iter: int = 0,
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

        :returns: the current upper-level objective
        """

        ul_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        grads_upper = jit.grad(ul_loss, list(auxiliary_model.parameters(time=0)))
        update_tensor_grads(self.ul_var, grads_upper)

        return ul_loss
