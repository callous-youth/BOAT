from boat_jit.utils.op_utils import l2_reg
from boat_jit.utils.op_utils import (
    update_grads,
    update_tensor_grads,
    grad_unused_zero,
    manual_update,
)
import jittor as jit
from jittor import Module
import copy
from typing import Dict, Any, Callable, List

from boat_jit.operation_registry import register_class
from boat_jit.gm_ol.dynamical_system import DynamicalSystem


@register_class
class VSO(DynamicalSystem):
    """
    Implements the optimization procedure of Value-function based Sequential Method (VSO) [1].

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
        - "VSO" (Dict): Configuration for the VSO algorithm:
            - "z_loop" (int): Number of iterations for optimizing the auxiliary variable `z`.
            - "ll_l2_reg" (float): L2 regularization coefficient for the lower-level model.
            - "ul_l2_reg" (float): L2 regularization coefficient for the upper-level model.
            - "ul_ln_reg" (float): Logarithmic regularization coefficient for the upper-level model.
            - "reg_decay" (float): Decay rate for the regularization coefficients.
            - "z_lr" (float): Learning rate for optimizing the auxiliary variable `z`.
        - "device" (str): Device on which computations are performed, e.g., "cpu" or "cuda".

    References
    ----------
    [1] Liu B, Ye M, Wright S, et al. "BOME! Bilevel Optimization Made Easy: A Simple First-Order Approach,"
        in NeurIPS, 2022.
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
        super(VSO, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_loop = lower_loop
        self.z_loop = solver_config["VSO"]["z_loop"]
        self.ll_l2_reg = solver_config["VSO"]["ll_l2_reg"]
        self.ul_l2_reg = solver_config["VSO"]["ul_l2_reg"]
        self.ul_ln_reg = solver_config["VSO"]["ul_ln_reg"]
        self.reg_decay = float(solver_config["VSO"]["reg_decay"])
        self.z_lr = solver_config["VSO"]["z_lr"]

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

        reg_decay = self.reg_decay * current_iter + 1
        for z_idx in range(self.z_loop):
            self.ll_opt.zero_grad()
            loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.parameters())
            loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            loss_z = loss_z_ + loss_l2_z
            grads = grad_unused_zero(loss_z, list(self.ll_model.parameters()))
            update_grads(grads, self.ll_model)
            manual_update(self.ll_opt, list(self.ll_model.parameters()))

        auxiliary_model = copy.deepcopy(self.ll_model)
        auxiliary_opt = jit.nn.SGD(auxiliary_model.parameters(), lr=self.z_lr)

        with jit.no_grad():
            loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.parameters())
            loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            loss_z = loss_z_ + loss_l2_z

        for y_idx in range(self.y_loop):
            loss_y_f_ = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
            loss_y_ = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
            loss_l2_y = l2_reg(auxiliary_model.parameters())
            loss_l2_y = self.ul_l2_reg / reg_decay * loss_l2_y
            loss_ln = jit.log(loss_y_f_.item() + loss_z.item() - loss_y_f_.item())
            loss_ln = self.ul_ln_reg / reg_decay * loss_ln
            loss_y = loss_y_ - loss_ln + loss_l2_y
            grads = jit.grad(loss_y, auxiliary_model.parameters())
            update_grads(grads, auxiliary_model)
            manual_update(auxiliary_opt, list(auxiliary_model.parameters()))

        with jit.no_grad():
            loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.parameters())
            loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            loss_z = loss_z_ + loss_l2_z

            loss_y_f_ = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
            loss_ln = (
                self.ul_ln_reg
                / reg_decay
                * jit.log(loss_y_f_.item() + loss_z.item() - loss_y_f_.item())
            )

        loss_x_ = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        loss_x = loss_x_ - loss_ln
        grads = jit.grad(loss_x, self.ul_model.parameters())
        update_tensor_grads(self.ul_var, grads)
        return {"upper_loss": loss_x_.item()}
