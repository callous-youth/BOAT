import torch
from .hyper_gradient import HyperGradient
from torch.nn import Module
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase
from boat.utils.op_utils import update_tensor_grads


class IAD(HyperGradient):
    """
    Implements the optimization procedure of the Naive Gradient Descent (NGD) [1].

    Parameters
    ----------
    :param ll_objective: The lower-level objective function of the BLO problem.
    :type ll_objective: Callable
    :param ul_objective: The upper-level objective function of the BLO problem.
    :type ul_objective: Callable
    :param ll_model: The lower-level model of the BLO problem.
    :type ll_model: torch.nn.Module
    :param ul_model: The upper-level model of the BLO problem.
    :type ul_model: torch.nn.Module
    :param lower_loop: The number of iterations for lower-level optimization.
    :type lower_loop: int
    :param solver_config: A dictionary containing configurations for the solver. Expected keys include:
        - "lower_level_opt" (torch.optim.Optimizer): The optimizer for the lower-level model.
        - "hyper_op" (List[str]): A list of hyper-gradient operations to apply, such as "PTT" or "FOA".
        - "RGT" (Dict): Configuration for Truncated Gradient Iteration (RGT):
            - "truncate_iter" (int): The number of iterations to truncate the gradient computation.
    :type solver_config: Dict[str, Any]

    Attributes
    ----------
    :attribute truncate_max_loss_iter: Indicates whether to truncate based on a maximum loss iteration 
        (enabled if "PTT" is in `hyper_op`).
    :type truncate_max_loss_iter: bool
    :attribute truncate_iters: The number of iterations for gradient truncation, derived from `solver_config["RGT"]["truncate_iter"]`.
    :type truncate_iters: int
    :attribute ll_opt: The optimizer used for the lower-level model.
    :type ll_opt: torch.optim.Optimizer
    :attribute foa: Indicates whether First-Order Approximation (FOA) is applied, based on `hyper_op` configuration.
    :type foa: bool

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
            return {"upper_loss": ul_loss, "hyper_gradient_finished": True}
