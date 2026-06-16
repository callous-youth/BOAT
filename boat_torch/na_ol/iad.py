import torch
from torch.nn import Module
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase
from boat_torch.utils.op_utils import update_tensor_grads

from boat_torch.operation_registry import register_class
from boat_torch.na_ol.hyper_gradient import HyperGradient


@register_class
class IAD(HyperGradient):
    """
    Implements the optimization procedure of the Naive Gradient Descent (NGD) [1].

    Parameters
    ----------
    ll_objective : Callable
        The lower-level objective function of the BLO problem.
    ul_objective : Callable
        The upper-level objective function of the BLO problem.
    ll_model : torch.nn.Module
        The lower-level model of the BLO problem.
    ul_model : torch.nn.Module
        The upper-level model of the BLO problem.
    lower_loop : int
        The number of iterations for lower-level optimization.
    solver_config : Dict[str, Any]
        A dictionary containing configurations for the solver. Expected keys include:

        - "lower_level_opt" (torch.optim.Optimizer): The optimizer for the lower-level model.
        - "na_op" (List[str]): A list of hyper-gradient operations to apply, such as "PTT" or "FOA".
        - "RGT" (Dict): Configuration for Truncated Gradient Iteration (RGT):
            - "truncate_iter" (int): The number of iterations to truncate the gradient computation.

    Attributes
    ----------
    truncate_max_loss_iter : bool
        Indicates whether to truncate based on a maximum loss iteration (enabled if "PTT" is in `na_op`).
    truncate_iters : int
        The number of iterations for gradient truncation, derived from `solver_config["RGT"]["truncate_iter"]`.
    ll_opt : torch.optim.Optimizer
        The optimizer used for the lower-level model.
    foa : bool
        Indicates whether First-Order Approximation (FOA) is applied, based on `na_op` configuration.

    References
    ----------
    [1] L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, "Bilevel
        programming for hyperparameter optimization and meta-learning", in ICML, 2018.
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
        self.solver_config["copy_last_param"] = False

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
        Compute the hyper-gradients of the upper-level variables using the data from feed_dict and patched models.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization. It typically includes training data, targets, and other information required to compute the LL objective.

        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization. It typically includes validation data, targets, and other information required to compute the UL objective.

        auxiliary_model : _MonkeyPatchBase
            A patched lower-level model wrapped by the `higher` library. It serves as the lower-level model for optimization.

        max_loss_iter : int
            The number of iterations used for backpropagation.

        next_operation : str
            The next operator for the calculation of the hypergradient.

        hyper_gradient_finished : bool
            A boolean flag indicating whether the hypergradient computation is finished.

        Returns
        -------
        float
            The current upper-level objective.

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
            grads_upper = torch.autograd.grad(
                ul_loss, list(auxiliary_model.parameters(time=0)), allow_unused=True
            )
            update_tensor_grads(self.ul_var, grads_upper)
            return {"upper_loss": ul_loss.item(), "hyper_gradient_finished": True}
