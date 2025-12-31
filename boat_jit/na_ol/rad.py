import jittor as jit
from jittor import Module
from typing import List, Callable, Dict
from ..higher_jit.patch import _MonkeyPatchBase
from boat_jit.utils.op_utils import update_tensor_grads

from boat_jit.operation_registry import register_class
from boat_jit.na_ol.hyper_gradient import HyperGradient


@register_class
class RAD(HyperGradient):
    """
    Computes the hyper-gradient of the upper-level variables using Reverse Auto Differentiation (RAD) [1].

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
    ll_var : List[jittor.Var]
        List of variables optimized with the lower-level objective.
    ul_var : List[jittor.Var]
        List of variables optimized with the upper-level objective.
    solver_config : Dict[str, Any]
        Dictionary containing solver configurations, including optional dynamic operation settings.

    References
    ----------
    [1] Franceschi, Luca, et al. "Forward and reverse gradient-based hyperparameter optimization." in ICML, 2017.
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
        super(RAD, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )
        self.dynamic_initialization = "DI" in solver_config["gm_op"]

    def compute_gradients(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        max_loss_iter: int = 0,
        next_operation: str = None,
        **kwargs
    ):
        """
        Compute the hyper-gradients of the upper-level variables using the provided data and patched models.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization.
            Typically includes training data, targets, and other information required to compute the lower-level objective.

        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization.
            Typically includes validation data, targets, and other information required to compute the upper-level objective.

        auxiliary_model : _MonkeyPatchBase
            A patched lower model wrapped by the `higher` library. It serves as the lower-level model for optimization.

        max_loss_iter : int, optional
            The number of iterations used for backpropagation. Default is 0.

        next_operation : str, optional
            The next operator for the calculation of the hypergradient. Default is None.

        **kwargs : dict
            Additional keyword arguments passed to the method.


        Returns
        -------
        Dict
            A dictionary containing the upper-level objective and the status of hypergradient computation.
        """
        assert next_operation is None, "RAD does not support any further operations."
        lower_model_params = kwargs.get(
            "lower_model_params", list(auxiliary_model.parameters())
        )
        upper_loss = self.ul_objective(
            ul_feed_dict, self.ul_model, auxiliary_model, params=lower_model_params
        )
        grads_upper = jit.grad(
            upper_loss, self.ul_var, retain_graph=self.dynamic_initialization
        )
        update_tensor_grads(self.ul_var, grads_upper)

        if self.dynamic_initialization:
            grads_lower = jit.grad(upper_loss, list(auxiliary_model.parameters(time=0)))
            update_tensor_grads(self.ll_var, grads_lower)

        return {"upper_loss": upper_loss.item(), "hyper_gradient_finished": True}
