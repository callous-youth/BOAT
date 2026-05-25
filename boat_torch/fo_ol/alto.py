from boat_torch.utils.op_utils import (
    grad_unused_zero,
    update_tensor_grads,
)
import torch
from torch.nn import Module
from typing import Dict, Any, Callable, List
from boat_torch.operation_registry import register_class
from boat_torch.gm_ol.dynamical_system import DynamicalSystem


@register_class
class ALTO(DynamicalSystem):
    """
    Implements a simple Alternating Optimization (ALT) procedure for bi-level optimization.
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
        super(ALTO, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        """
        Executes the simple alternating optimization procedure.
        """
        # Lower-level update
        self.ll_opt.zero_grad()
        lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        grad_y_parameters = grad_unused_zero(lower_loss, list(self.ll_var))
        update_tensor_grads(self.ll_var, grad_y_parameters)
        self.ll_opt.step()

        # Upper-level update
        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
        grad_x_parameters = grad_unused_zero(upper_loss, self.ul_var)
        update_tensor_grads(self.ul_var, grad_x_parameters)
        
        return {"upper_loss": upper_loss.item()}
