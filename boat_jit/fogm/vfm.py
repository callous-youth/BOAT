from boat_jit.utils.op_utils import (
    grad_unused_zero,
    require_model_grad,
    update_tensor_grads,
    stop_model_grad,
    manual_update,
)

import jittor as jit
from jittor import Module
import copy
from typing import Dict, Any, Callable, List
from boat_jit.operation_registry import register_class
from boat_jit.dynamic_ol.dynamical_system import DynamicalSystem


@register_class
class VFM(DynamicalSystem):
    """
    Implements the optimization procedure of Value-function based First-Order Method (VFM) _`[1]`.

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
    _`[1]` R. Liu, X. Liu, X. Yuan, S. Zeng and J. Zhang, "A Value-Function-based
    Interior-point Method for Non-convex Bi-level Optimization", in ICML, 2021.
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
        super(VFM, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_hat_lr = float(solver_config["VFM"]["y_hat_lr"])
        self.eta = solver_config["VFM"]["eta"]
        self.u1 = solver_config["VFM"]["u1"]
        self.device = solver_config["device"]

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
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
        y_hat = copy.deepcopy(self.ll_model)
        y_hat_opt = jit.optim.SGD(y_hat.parameters(), lr=self.y_hat_lr, momentum=0.9)
        n_params_y = sum([p.numel() for p in self.ll_model.parameters()])
        n_params_x = sum([p.numel() for p in self.ul_model.parameters()])
        delta_f = jit.zeros((n_params_x + n_params_y,))
        delta_F = jit.zeros((n_params_x + n_params_y,))

        def g_x_xhat_w(y, y_hat, x):
            loss = self.ll_objective(ll_feed_dict, x, y) - self.ll_objective(
                ll_feed_dict, x, y_hat
            )
            grad_y = grad_unused_zero(loss, list(y.parameters()), retain_graph=True)
            grad_x = grad_unused_zero(loss, list(x.parameters()))
            return loss, grad_y, grad_x

        require_model_grad(y_hat)
        for y_itr in range(self.lower_loop):
            # y_hat_opt.zero_grad()
            tr_loss = self.ll_objective(ll_feed_dict, self.ul_model, y_hat)
            grads_hat = grad_unused_zero(tr_loss, y_hat.parameters())
            update_tensor_grads(list(y_hat.parameters()), grads_hat)
            # y_hat_opt.step()
            manual_update(y_hat_opt, list(y_hat.parameters()))
        F_y = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)

        grad_F_y = grad_unused_zero(
            F_y, list(self.ll_model.parameters()), retain_graph=True
        )
        grad_F_x = grad_unused_zero(F_y, list(self.ul_model.parameters()))
        stop_model_grad(y_hat)
        loss, gy, gx_minus_gx_k = g_x_xhat_w(self.ll_model, y_hat, self.ul_model)
        delta_F[:n_params_y].update(
            jit.concat([fc_param.flatten().clone() for fc_param in grad_F_y])
        )
        delta_f[:n_params_y].update(
            jit.concat([fc_param.flatten().clone() for fc_param in gy])
        )

        delta_F[n_params_y:].update(
            jit.concat([fc_param.flatten().clone() for fc_param in grad_F_x])
        )
        delta_f[n_params_y:].update(
            jit.concat([fc_param.flatten().clone() for fc_param in gx_minus_gx_k])
        )

        # Compute squared norm of delta_f
        norm_dq = (delta_f * delta_f).sum()

        # Compute dot product
        dot = (delta_F * delta_f).sum()

        scaling_factor = jit.nn.relu((self.u1 * loss - dot) / (norm_dq + 1e-8))  #
        d = delta_F + scaling_factor * delta_f

        y_grad = []
        x_grad = []
        all_numel = 0
        for _, param in enumerate(self.ll_model.parameters()):
            sliced = d[all_numel : all_numel + param.numel()]
            reshaped = sliced.reshape(tuple(param.shape))
            y_grad.append(reshaped.clone())
            all_numel += param.numel()

        for _, param in enumerate(self.ul_model.parameters()):
            sliced = d[all_numel : all_numel + param.numel()]
            reshaped = sliced.reshape(tuple(param.shape))
            x_grad.append(reshaped.clone())
            all_numel += param.numel()

        update_tensor_grads(self.ll_var, y_grad)
        update_tensor_grads(self.ul_var, x_grad)
        manual_update(self.ll_opt, list(self.ll_var))
        return F_y
