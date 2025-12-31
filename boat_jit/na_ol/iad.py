import jittor as jit
from jittor import Module
from typing import List, Callable, Dict
from boat_jit.higher_jit.patch import _MonkeyPatchBase
from boat_jit.utils.op_utils import update_tensor_grads

from boat_jit.operation_registry import register_class
from boat_jit.na_ol.hyper_gradient import HyperGradient


@register_class
class IAD(HyperGradient):
    """
    Computes the hyper-gradient of the upper-level variables using Initialization-based Auto Differentiation (IAD) [1].

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
        Dictionary containing solver configurations.

    References
    ----------
    [1] Finn C., Abbeel P., Levine S., "Model-agnostic meta-learning for fast adaptation of deep networks", in ICML, 2017.
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
        super(IAD, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )

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

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization. It typically includes training data, targets, and other information required to compute the LL objective.
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization. It typically includes validation data, targets, and other information required to compute the UL objective.
        auxiliary_model : _MonkeyPatchBase
            A patched lower model wrapped by the `higher` library. It serves as the lower-level model for optimization.
        max_loss_iter : int
            The number of iterations used for backpropagation.
        next_operation : str
            The next operator for the calculation of the hypergradient.
        hyper_gradient_finished : bool
            A boolean flag indicating whether the hypergradient computation is finished.

        Returns
        -------
        Dict
            A dictionary containing the upper-level objective and the status of hypergradient computation.
        """

        if next_operation is not None:
            lower_model_params = kwargs.get(
                "lower_model_params", list(auxiliary_model.parameters())
            )
            hparams = list(auxiliary_model.parameters(time=0))
            return {
                "ll_feed_dict": ll_feed_dict,
                "ul_feed_dict": ul_feed_dict,
                "auxiliary_model": auxiliary_model,
                "max_loss_iter": max_loss_iter,
                "hyper_gradient_finished": hyper_gradient_finished,
                "hparams": hparams,
                "lower_model_params": lower_model_params,
                **kwargs,
            }
        else:
            lower_model_params = kwargs.get(
                "lower_model_params", list(auxiliary_model.parameters())
            )
            ul_loss = self.ul_objective(
                ul_feed_dict, self.ul_model, auxiliary_model, params=lower_model_params
            )
            grads_upper = jit.grad(ul_loss, list(auxiliary_model.parameters(time=0)))
            update_tensor_grads(self.ul_var, grads_upper)
            return {"upper_loss": ul_loss.item(), "hyper_gradient_finished": True}
