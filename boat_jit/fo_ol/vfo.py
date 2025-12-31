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
from boat_jit.gm_ol.dynamical_system import DynamicalSystem


@register_class
class VFO(DynamicalSystem):
    """
    Implements the optimization procedure of Value-function based First-Order Method (VFO) [1].

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
        A list of lower-level variables of the BLO problem.
    ul_var : List[jittor.Var]
        A list of upper-level variables of the BLO problem.
    lower_loop : int
        The number of iterations for lower-level optimization.
    solver_config : Dict[str, Any]
        A dictionary containing configurations for the solver. Expected keys include:

        - "lower_level_opt" (jittor.optim.Optimizer): Optimizer for the lower-level model.
        - "VFO" (Dict): Configuration for the VFO algorithm:
            - "y_hat_lr" (float): Learning rate for optimizing the surrogate variable `y_hat`.
            - "eta" (float): Step size for value-function updates.
            - "u1" (float): Hyperparameter controlling the penalty in the value function.
        - "device" (str): Device on which computations are performed, e.g., "cpu" or "cuda".

    References
    ----------
    [1] R. Liu, X. Liu, X. Yuan, S. Zeng and J. Zhang, "A Value-Function-based
        Interior-point Method for Non-convex Bi-level Optimization," in ICML, 2021.
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
        super(VFO, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_hat_lr = float(solver_config["VFO"]["y_hat_lr"])
        self.eta = solver_config["VFO"]["eta"]
        self.u1 = solver_config["VFO"]["u1"]
        self.device = solver_config["device"]

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        """
        Execute the optimization procedure with the data from feed_dict.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        current_iter : int
            The current iteration number of the optimization process.

        Returns
        -------
        Dict
            A dictionary containing the upper-level objective and the status of hypergradient computation.
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
            tr_loss = self.ll_objective(ll_feed_dict, self.ul_model, y_hat)
            grads_hat = grad_unused_zero(tr_loss, y_hat.parameters())
            update_tensor_grads(list(y_hat.parameters()), grads_hat)
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
        return {"upper_loss": F_y.item()}
