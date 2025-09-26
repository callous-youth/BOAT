from boat_torch.utils.op_utils import (
    grad_unused_zero,
    require_model_grad,
    update_tensor_grads,
)

import torch
from torch.nn import Module
import copy
from typing import Dict, Any, Callable, List
from boat_torch.operation_registry import register_class
from boat_torch.dynamic_ol.dynamical_system import DynamicalSystem


@register_class
class PGDM(DynamicalSystem):
    """
    Implements the optimization procedure of Penalty-based Gradient Descent Method (PGDM) [1].

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
        - "PGDM" (Dict): A dictionary containing the following keys:
            - "y_hat_lr": Learning rate for optimizing the surrogate variable `y_hat`.
            - "gamma_init": Initial value of the hyperparameter `gamma`.
            - "gamma_max": Maximum value of the hyperparameter `gamma`.
            - "gamma_argmax_step": Step size of the hyperparameter `gamma`.


    References
    ----------
    [1] Shen H, Chen T. "On penalty-based bilevel gradient descent method," in ICML, 2023.
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
        super(PGDM, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_hat_lr = float(solver_config["PGDM"]["y_hat_lr"])
        self.gamma_init = solver_config["PGDM"]["gamma_init"]
        self.gamma_max = solver_config["PGDM"]["gamma_max"]
        self.gamma_argmax_step = solver_config["PGDM"]["gamma_argmax_step"]
        self.gam = self.gamma_init
        self.device = solver_config["device"]

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

        y_hat = copy.deepcopy(self.ll_model).to(self.device)
        y_hat_opt = torch.optim.SGD(list(y_hat.parameters()), lr=self.y_hat_lr)

        if self.gamma_init > self.gamma_max:
            self.gamma_max = self.gamma_init
            print(
                "Initial gamma is larger than max gamma, proceeding with gamma_max=gamma_init."
            )
        step_gam = (self.gamma_max - self.gamma_init) / self.gamma_argmax_step
        lr_decay = min(1 / (self.gam + 1e-8), 1)
        require_model_grad(y_hat)
        for y_itr in range(self.lower_loop):
            y_hat_opt.zero_grad()
            tr_loss = self.ll_objective(ll_feed_dict, self.ul_model, y_hat)
            grads_hat = grad_unused_zero(tr_loss, y_hat.parameters())
            update_tensor_grads(list(y_hat.parameters()), grads_hat)
            y_hat_opt.step()

        self.ll_opt.zero_grad()
        F_y = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
        loss = lr_decay * (
            F_y
            + self.gam
            * (
                self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
                - self.ll_objective(ll_feed_dict, self.ul_model, y_hat)
            )
        )
        loss.backward()
        self.gam += step_gam
        self.gam = min(self.gamma_max, self.gam)
        self.ll_opt.step()
        return {"upper_loss": F_y.item()}
