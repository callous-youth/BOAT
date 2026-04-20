from boat_torch.utils.op_utils import (
    grad_unused_zero,
    update_tensor_grads,
    copy_parameter_from_list,
)
import numpy
import torch
from torch.nn import Module
import copy
from typing import Dict, Any, Callable, List
from boat_torch.operation_registry import register_class
from boat_torch.gm_ol.dynamical_system import DynamicalSystem


@register_class
class MESO(DynamicalSystem):
    """
    Implements the optimization procedure of Moreau Envelope based Single-loop Method (MESO) [1].

    Parameters
    ----------
    ll_objective : Callable
        The lower-level objective of the BLO problem.
    ul_objective : Callable
        The upper-level objective of the BLO problem.
    ll_model : torch.nn.Module
        The lower-level model of the BLO problem.
    ul_model : torch.nn.Module
        The upper-level model of the BLO problem.
    ll_var : List[torch.Tensor]
        The list of lower-level variables of the BLO problem.
    ul_var : List[torch.Tensor]
        The list of upper-level variables of the BLO problem.
    lower_loop : int
        Number of iterations for lower-level optimization.
    solver_config : Dict[str, Any]
        A dictionary containing solver configurations. Expected keys include:

        - "lower_level_opt": The optimizer for the lower-level model.
        - "MESO" (Dict): A dictionary containing the following keys:
            - "eta": Learning rate for the MESO optimization procedure.
            - "gamma_1": Regularization parameter for the MESO algorithm.
            - "c0": Initial constant for the update steps.
            - "y_hat_lr": Learning rate for optimizing the surrogate variable `y_hat`.

    References
    ----------
    [1] Liu R, Liu Z, Yao W, et al. "Moreau Envelope for Nonconvex Bi-Level Optimization: A Single-loop and Hessian-free Solution Strategy," ICML, 2024.
    """

    def __init__(
        self,
        ll_objective: Callable,
        lower_loop: int,
        ul_model: Module,
        ul_objective: Callable,
        ll_model: Module,
        ll_var: List,
        ul_var: List,
        solver_config: Dict[str, Any],
    ):
        super(MESO, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_loop = lower_loop
        self.eta = solver_config["MESO"]["eta"]
        self.gamma_1 = solver_config["MESO"]["gamma_1"]
        self.c0 = solver_config["MESO"]["c0"]
        self.y_hat = copy.deepcopy(self.ll_model)
        self.y_hat_opt = torch.optim.SGD(
            self.y_hat.parameters(), lr=solver_config["MESO"]["y_hat_lr"], momentum=0.9
        )

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        """
        Executes the optimization procedure using the provided data and model configurations.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization. Typically includes training data or parameters for the lower-level objective.
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization. Usually includes parameters or configurations for the upper-level objective.
        current_iter : int
            The current iteration count of the optimization process, used for tracking progress or adjusting optimization parameters.

        Returns
        -------
        Dict
            A dictionary containing the upper-level objective and the status of hypergradient computation.
        """

        if current_iter == 0:
            ck = 0.2
        else:
            ck = numpy.power(current_iter + 1, 0.25) * self.c0

        theta_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.y_hat)

        grad_theta_parmaters = grad_unused_zero(
            theta_loss, list(self.y_hat.parameters())
        )

        errs = []
        for a, b in zip(list(self.y_hat.parameters()), list(self.ll_var)):
            diff = a - b
            errs.append(diff)
        vs_param = []
        for v0, gt, err in zip(
            list(self.y_hat.parameters()), grad_theta_parmaters, errs
        ):
            vs_param.append(v0 - self.eta * (gt + self.gamma_1 * err))  # upate \theta

        copy_parameter_from_list(self.y_hat, vs_param)

        reg = 0
        for param1, param2 in zip(list(self.ll_var), vs_param):
            diff = param1 - param2
            reg += torch.norm(diff, p=2) ** 2
        lower_loss = (
            (1 / ck) * self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
            + self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            - 0.5 * self.gamma_1 * reg
        )

        self.ll_opt.zero_grad()
        grad_y_parmaters = grad_unused_zero(lower_loss, list(self.ll_var))

        update_tensor_grads(self.ll_var, grad_y_parmaters)

        self.ll_opt.step()

        upper_loss = (
            (1 / ck) * self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
            + self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            - self.ll_objective(ll_feed_dict, self.ul_model, self.y_hat)
        )
        grad_x_parmaters = grad_unused_zero(upper_loss, self.ul_var)
        update_tensor_grads(self.ul_var, grad_x_parmaters)
        return {"upper_loss": upper_loss.item()}
