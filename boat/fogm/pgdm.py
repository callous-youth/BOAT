from ..dynamic_ol.dynamical_system import DynamicalSystem
from boat.utils.op_utils import grad_unused_zero,require_model_grad,update_tensor_grads,stop_model_grad

import torch
from torch.nn import Module
from torch.optim import Optimizer
import copy
from typing import Dict, Any, Callable,List


class PGDM(DynamicalSystem):
    """
    Implements the optimization procedure of Penalty based Gradient Descent Method (PGDM) _`[1]`.

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
    :param ll_var: The list of lower-level variables of the BLO problem.
    :type ll_var: List
    :param ul_var: The list of upper-level variables of the BLO problem.
    :type ul_var: List
    :param lower_loop: Number of iterations for lower-level optimization.
    :type lower_loop: int
    :param solver_config: Dictionary containing solver configurations.
    :type solver_config: dict


    References
    ----------
    _`[1]` Shen H, Chen T. On penalty-based bilevel gradient descent method[C]. In ICML, 2023.
    """
    def __init__(
            self,
            ll_objective: Callable,
            lower_loop: int,
            ul_model: Module,
            ul_objective: Callable,
            ll_model: Module,
            ll_opt: Optimizer,
            ll_var: List,
            ul_var: List,
            solver_config: Dict[str, Any]
    ):
        super(PGDM, self).__init__(ll_objective, lower_loop, ul_model, ll_model)
        self.ul_objective = ul_objective
        self.ll_opt = ll_opt
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_hat_lr = float(solver_config['PGDM']['y_hat_lr'])
        self.gamma_init = solver_config['PGDM']["gamma_init"]
        self.gamma_max = solver_config['PGDM']["gamma_max"]
        self.gamma_argmax_step = solver_config['PGDM']["gamma_argmax_step"]
        self.gam = self.gamma_init
        self.device = solver_config["device"]

    def optimize(
            self,
            ll_feed_dict: Dict,
            ul_feed_dict: Dict,
            current_iter: int
    ):
        """
        Execute the optimization procedure with the data from feed_dict.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param current_iter: The current iteration number of the optimization process.
        :type current_iter: int

        :returns: None
        """
        y_hat = copy.deepcopy(self.ll_model).to(self.device)
        y_hat_opt = torch.optim.SGD(list(y_hat.parameters()), lr=self.y_hat_lr)

        if self.gamma_init > self.gamma_max:
            self.gamma_max = self.gamma_init
            print('Initial gamma is larger than max gamma, proceeding with gamma_max=gamma_init.')
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
        loss = lr_decay * (F_y + self.gam * (self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model) - self.ll_objective(ll_feed_dict, self.ul_model, y_hat)))
        loss.backward()
        self.gam += step_gam
        self.gam = min(self.gamma_max, self.gam)
        self.ll_opt.step()
        return F_y
