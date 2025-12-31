from .dynamical_system import DynamicalSystem
import jittor as jit
from jittor import Module
from ..higher_jit.patch import _MonkeyPatchBase
from ..higher_jit.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from ..utils.op_utils import (
    update_tensor_grads,
    grad_unused_zero,
    list_tensor_norm,
    list_tensor_matmul,
    custom_grad,
    manual_update,
)

from boat_jit.operation_registry import register_class
from boat_jit.gm_ol.dynamical_system import DynamicalSystem


@register_class
class DM(DynamicalSystem):
    """
    Implements the lower-level optimization procedure for Dual Multiplier (DM) [1].

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
    lower_loop : int
        The number of iterations for the lower-level optimization process.
    solver_config : Dict[str, Any]
        A dictionary containing configurations for the optimization solver, including
        hyperparameters and specific settings for NGD, GDA, and DM.

    References
    ----------
    [1] Liu R, Liu Y, Yao W, et al., "Averaged method of multipliers for bi-level optimization without lower-level
        strong convexity," ICML, 2023.
    """

    def __init__(
        self,
        ll_objective: Callable,
        lower_loop: int,
        ul_model: Module,
        ul_objective: Callable,
        ll_model: Module,
        solver_config: Dict[str, Any],
    ):
        super(DM, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.truncate_max_loss_iter = "PTT" in solver_config["na_op"]
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.truncate_iters = solver_config["RGT"]["truncate_iter"] if "RGT" in solver_config["na_op"] else 0
        self.ll_opt = solver_config["lower_level_opt"]
        self.ul_opt = solver_config["upper_level_opt"]
        self.auxiliary_v = solver_config["DM"]["auxiliary_v"]
        self.auxiliary_v_opt = solver_config["DM"]["auxiliary_v_opt"]
        self.auxiliary_v_lr = solver_config["DM"]["auxiliary_v_lr"]
        self.tau = solver_config["DM"]["tau"]
        self.p = solver_config["DM"]["p"]
        self.mu0 = solver_config["DM"]["mu0"]
        self.eta = solver_config["DM"]["eta0"]
        self.strategy = solver_config["DM"]["strategy"]
        self.na_op = solver_config["na_op"]
        self.gda_loss = solver_config.get("gda_loss", None)

    def optimize(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        auxiliary_opt: DifferentiableOptimizer,
        current_iter: int,
        next_operation: str = None,
        **kwargs
    ):
        """
        Executes the lower-level optimization procedure with support for NGD, GDA, and RAD operations.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization. Typically includes:
                - "data" : Training input data.
                - "target" : Training target data (optional, depending on the task).
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization. Typically includes:
                - "data" : Validation input data.
                - "target" : Validation target data (optional, depending on the task).
        auxiliary_model : _MonkeyPatchBase
            A patched lower model wrapped by the `higher` library. Used for differentiable optimization.
        auxiliary_opt : DifferentiableOptimizer
            A patched optimizer for the lower-level model, wrapped by the `higher` library. Enables differentiable optimization steps.
        current_iter : int
            The current iteration number in the optimization process.
        next_operation : str, optional
            Specifies the next operation in the optimization process. Must be `None` for NGD. (default: None)
        kwargs : dict
            Additional keyword arguments for the optimization process.

        Returns
        -------
        int
            Returns `-1` upon successful completion of the optimization process.

        Notes
        -----
        - For GDA operations, this method supports three strategies: 's1', 's2', and 's3'.
        - When using RAD in `na_op`, a higher-order gradient adjustment is applied to the auxiliary variables.
        - Ensure that `next_operation` is `None` for NGD, as it does not support additional operations.

        Raises
        ------
        AssertionError
            If `next_operation` is not `None` for NGD or if an unsupported strategy is specified for GDA.
        """
        assert next_operation is None, "NGD does not support next_operation"
        if "gda_loss" in kwargs:
            gda_loss = kwargs["gda_loss"]
            assert self.strategy in [
                "s1",
                "s2",
                "s3",
            ], "Three strategies are supported for DM operation, including ['s1','s2','s3']."
            if self.strategy == "s1":
                self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
                self.eta = (
                    (current_iter + 1) ** (-0.5 * self.tau)
                    * self.alpha**2
                    * self.ll_opt.defaults["lr"]
                )
                x_lr = (
                    (current_iter + 1) ** (-1.5 * self.tau)
                    * self.alpha**7
                    * self.ll_opt.defaults["lr"]
                )
            elif self.strategy == "s2":
                self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
                self.eta = (
                    (current_iter + 1) ** (-0.5 * self.tau)
                    * self.alpha
                    * self.ll_opt.defaults["lr"]
                )
                x_lr = (
                    (current_iter + 1) ** (-1.5 * self.tau)
                    * self.alpha**5
                    * self.ll_opt.defaults["lr"]
                )
            elif self.strategy == "s3":
                self.alpha = self.mu0 * 1 / (current_iter + 1) ** (1 / self.p)
                self.eta = (current_iter + 1) ** (
                    -0.5 * self.tau
                ) * self.ll_opt.defaults["lr"]
                x_lr = (
                    (current_iter + 1) ** (-1.5 * self.tau)
                    * self.alpha**3
                    * self.ll_opt.defaults["lr"]
                )
            for params in self.ul_opt.param_groups:
                params["lr"] = x_lr
        else:
            gda_loss = None
            assert (
                self.strategy == "s1"
            ), "Only 's1' strategy is supported for DM without GDA operation."

            x_lr = (
                self.ul_opt.defaults["lr"]
                * (current_iter + 1) ** (-self.tau)
                * self.ll_opt.defaults["lr"]
            )
            eta = (
                self.eta
                * (current_iter + 1) ** (-0.5 * self.tau)
                * self.ll_opt.defaults["lr"]
            )
            for params in self.auxiliary_v_opt.param_groups:
                params["lr"] = eta
            for params in self.ul_opt.param_groups:
                params["lr"] = x_lr

        if gda_loss is not None:
            ll_feed_dict["alpha"] = self.alpha
            loss_full = self.gda_loss(
                ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
            )
        else:
            loss_full = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)

        grad_y_temp = jit.grad(
            loss_full, list(auxiliary_model.parameters()), retain_graph=True
        )
        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
        grad_outer_params = grad_unused_zero(
            upper_loss, list(auxiliary_model.parameters()), retain_graph=True
        )
        grads_phi_params = grad_unused_zero(
            loss_full, list(auxiliary_model.parameters()), retain_graph=True
        )
        grads = custom_grad(
            grads_phi_params,
            list(self.ul_model.parameters()),
            self.auxiliary_v,
            retain_graph=True,
        )  # dx (dy f) v
        grad_outer_hparams = grad_unused_zero(
            upper_loss, list(self.ul_model.parameters())
        )

        if "RAD" in self.na_op:
            vsp = custom_grad(
                grads_phi_params,
                list(auxiliary_model.parameters()),
                grad_outputs=self.auxiliary_v,
            )  # dy (dy f) v=d2y f v

            for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params):
                v0._custom_grad = v - gow
            update_tensor_grads(list(self.ll_model.parameters()), grad_y_temp)

            manual_update(self.ll_opt, list(self.ll_model.parameters()))

            manual_update(self.auxiliary_v_opt, self.auxiliary_v)

            grads = [
                -g + v if g is not None else v
                for g, v in zip(grads, grad_outer_hparams)
            ]
            update_tensor_grads(list(self.ul_model.parameters()), grads)

        else:
            vsp = custom_grad(
                grads_phi_params,
                list(auxiliary_model.parameters()),
                grad_outputs=self.auxiliary_v,
            )

            tem = [v - gow for v, gow in zip(vsp, grad_outer_params)]

            ita_u = list_tensor_norm(tem) ** 2
            grad_tem = custom_grad(
                grads_phi_params, list(auxiliary_model.parameters()), grad_outputs=tem
            )
            ita_l = list_tensor_matmul(tem, grad_tem)

            ita = ita_u / (ita_l + 1e-12)

            self.auxiliary_v = [
                v0 - ita * v + ita * gow
                for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params)
            ]

            vsp = custom_grad(
                grads_phi_params,
                list(auxiliary_model.parameters()),
                grad_outputs=self.auxiliary_v,
            )

            for v0, v, gow in zip(self.auxiliary_v, vsp, grad_outer_params):
                v0._custom_grad = v - gow

            update_tensor_grads(list(self.ll_model.parameters()), grad_y_temp)
            manual_update(self.ll_opt, list(self.ll_model.parameters()))

            grads = [
                -g + v if g is not None else v
                for g, v in zip(grads, grad_outer_hparams)
            ]
            update_tensor_grads(list(self.ul_model.parameters()), grads)

        return -1
