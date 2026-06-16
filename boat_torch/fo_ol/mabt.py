from typing import Any, Callable, Dict, Iterable, List

import torch
from torch.nn import Module

from boat_torch.gm_ol.dynamical_system import DynamicalSystem
from boat_torch.operation_registry import register_class
from boat_torch.utils.op_utils import grad_unused_zero, update_tensor_grads


@register_class
class MABT(DynamicalSystem):
    """
    Manifold Anchored Bilevel Transfer (MABT).

    This first-order implementation treats lower-level variables as the current
    task state and writes the meta-gradient to matching upper-level variables.
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
        super(MABT, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        config = solver_config.get("MABT", {})
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = list(ll_var)
        self.ul_var = list(ul_var)
        self.gap_lambda = float(config.get("lambda", config.get("gap_lambda", 1.0)))
        self.sigma = float(config.get("sigma", 0.01))
        self.lower_step_size = config.get("lower_step_size", None)
        self.use_sign_lower_step = bool(config.get("use_sign_lower_step", False))
        self.maximize = bool(config.get("maximize", True))
        self.sync_lower_from_upper = bool(config.get("sync_lower_from_upper", True))
        self.projection = config.get("projection", None)

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        self._check_meta_shapes()

        if self.sync_lower_from_upper:
            self._copy_params(self.ul_var, self.ll_var)
            self._project_params(self.ll_var)

        lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        lower_grads = grad_unused_zero(lower_loss, self.ll_var)
        self._lower_step(lower_grads)
        self._project_params(self.ll_var)

        lower_loss_next = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        gap_base_grads = grad_unused_zero(lower_loss_next, self.ll_var)

        current_lower = [param.detach().clone() for param in self.ll_var]
        self._set_probe_params(current_lower, gap_base_grads)

        lower_loss_probe = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        gap_probe_grads = grad_unused_zero(lower_loss_probe, self.ll_var)

        self._copy_params(current_lower, self.ll_var)

        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
        risk_grads = grad_unused_zero(upper_loss, self.ll_var)

        ascent_direction = [
            risk_grad / max(self.gap_lambda, 1e-8) - (probe_grad - base_grad)
            for risk_grad, probe_grad, base_grad in zip(
                risk_grads, gap_probe_grads, gap_base_grads
            )
        ]
        meta_grads = [-grad if self.maximize else grad for grad in ascent_direction]
        update_tensor_grads(self.ul_var, meta_grads)

        return {"upper_loss": upper_loss.item()}

    def _check_meta_shapes(self):
        if len(self.ll_var) != len(self.ul_var):
            raise ValueError(
                "MABT expects lower_level_var and upper_level_var to have the "
                "same structure."
            )
        for ll_param, ul_param in zip(self.ll_var, self.ul_var):
            if ll_param.shape != ul_param.shape:
                raise ValueError(
                    "MABT expects matching lower/upper variable shapes. "
                    f"Got lower {tuple(ll_param.shape)} and upper {tuple(ul_param.shape)}."
                )

    def _lower_step(self, grads: Iterable[torch.Tensor]):
        if self.lower_step_size is not None:
            step_size = float(self.lower_step_size)
            with torch.no_grad():
                for param, grad in zip(self.ll_var, grads):
                    step_grad = grad.sign() if self.use_sign_lower_step else grad
                    param.sub_(step_size * step_grad)
            return

        self.ll_opt.zero_grad()
        update_tensor_grads(self.ll_var, grads)
        self.ll_opt.step()
        self.ll_opt.zero_grad()

    def _set_probe_params(
        self, base_params: List[torch.Tensor], base_grads: Iterable[torch.Tensor]
    ):
        with torch.no_grad():
            for param, base_param, grad in zip(self.ll_var, base_params, base_grads):
                param.copy_(base_param + self.sigma * grad)
        self._project_params(self.ll_var)

    def _project_params(self, params: List[torch.Tensor]):
        if self.projection is None:
            return
        projected = self._call_projection(params)
        if projected is None:
            return
        with torch.no_grad():
            for param, projected_param in zip(params, projected):
                param.copy_(projected_param)

    def _call_projection(self, params: List[torch.Tensor]):
        try:
            return self.projection(params)
        except TypeError:
            return self.projection(params=params)

    @staticmethod
    def _copy_params(source: Iterable[torch.Tensor], target: Iterable[torch.Tensor]):
        with torch.no_grad():
            for src, dst in zip(source, target):
                dst.copy_(src)
