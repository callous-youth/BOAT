from boat_ms.utils.op_utils import require_model_grad

import mindspore as ms
from mindspore import nn, ops
import copy
from typing import Dict, Any, Callable, List

from boat_ms.operation_registry import register_class
from boat_ms.dynamic_ol.dynamical_system import DynamicalSystem


@register_class
class PGDM(DynamicalSystem):
    """
    Implements the optimization procedure of Moreau Envelope based Single-loop Method (MESM) [1].

    Parameters
    ----------
    ll_objective : Callable
        The lower-level objective of the BLO problem.

    ul_objective : Callable
        The upper-level objective of the BLO problem.

    ll_model : mindspore.nn.Cell
        The lower-level model of the BLO problem.

    ul_model : mindspore.nn.Cell
        The upper-level model of the BLO problem.

    ll_var : List[mindspore.Tensor]
        The list of lower-level variables of the BLO problem.

    ul_var : List[mindspore.Tensor]
        The list of upper-level variables of the BLO problem.

    lower_loop : int
        Number of iterations for lower-level optimization.

    solver_config : Dict[str, Any]
        A dictionary containing solver configurations. Expected keys include:

        - "lower_level_opt": The optimizer for the lower-level model.
        - "MESM" (Dict): A dictionary containing the following keys:
            - "eta": Learning rate for the MESM optimization procedure.
            - "gamma_1": Regularization parameter for the MESM algorithm.
            - "c0": Initial constant for the update steps.
            - "y_hat_lr": Learning rate for optimizing the surrogate variable `y_hat`.

    References
    ----------
    [1] Liu R, Liu Z, Yao W, et al. "Moreau Envelope for Nonconvex Bi-Level Optimization:
        A Single-loop and Hessian-free Solution Strategy," ICML, 2024.
    """
    def __init__(
        self,
        ll_objective: Callable,
        lower_loop: int,
        ul_model: nn.Cell,
        ul_objective: Callable,
        ll_model: nn.Cell,
        ll_opt: nn.Optimizer,
        ll_var: List,
        ul_var: List,
        solver_config: Dict[str, Any],
    ):
        super(PGDM, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_opt = ll_opt
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_hat_lr = float(solver_config["PGDM"]["y_hat_lr"])
        self.gamma_init = solver_config["PGDM"]["gamma_init"]
        self.gamma_max = solver_config["PGDM"]["gamma_max"]
        self.gamma_argmax_step = solver_config["PGDM"]["gamma_argmax_step"]
        self.gam = self.gamma_init
        self.device = ms.context.get_context("device_target")

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        """
        Implements the optimization procedure of Penalty-based Gradient Descent Method (PGDM) [1].

        Parameters
        ----------
        :param ll_objective: The lower-level objective of the BLO problem.
        :type ll_objective: Callable
        :param ul_objective: The upper-level objective of the BLO problem.
        :type ul_objective: Callable
        :param ll_model: The lower-level model of the BLO problem.
        :type ll_model: mindspore.nn.Cell
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: mindspore.nn.Cell
        :param ll_var: The list of lower-level variables of the BLO problem.
        :type ll_var: List[mindspore.Tensor]
        :param ul_var: The list of upper-level variables of the BLO problem.
        :type ul_var: List[mindspore.Tensor]
        :param lower_loop: Number of iterations for lower-level optimization.
        :type lower_loop: int
        :param solver_config: Dictionary containing solver configurations.
        :type solver_config: Dict[str, Any]

        References
        ----------
        [1] Shen H, Chen T. "On penalty-based bilevel gradient descent method," in ICML, 2023.
        """
        y_hat = copy.deepcopy(self.ll_model)
        y_hat_opt = nn.SGD(
            y_hat.trainable_params(), learning_rate=self.y_hat_lr, momentum=0.9
        )
        if self.gamma_init > self.gamma_max:
            self.gamma_max = self.gamma_init
            print(
                "Initial gamma is larger than max gamma, proceeding with gamma_max=gamma_init."
            )

        step_gam = (self.gamma_max - self.gamma_init) / self.gamma_argmax_step
        lr_decay = min(1 / (self.gam + 1e-8), 1)

        require_model_grad(y_hat)

        # Lower-level optimization loop
        for y_itr in range(self.lower_loop):
            # Zero gradients
            for param in y_hat.trainable_params():
                param.set_data(ms.numpy.zeros_like(param.data))

            # Compute gradients
            grad_fn = ops.GradOperation(get_by_list=True)(
                self.ll_objective, y_hat.trainable_params()
            )
            grads_hat = grad_fn(ll_feed_dict, self.ul_model, y_hat)

            y_hat_opt(grads_hat)

        def loss_fn():
            F_y = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
            ll_loss_current = self.ll_objective(
                ll_feed_dict, self.ul_model, self.ll_model
            )
            ll_loss_hat = self.ll_objective(ll_feed_dict, self.ul_model, y_hat)
            loss = lr_decay * (F_y + self.gam * (ll_loss_current - ll_loss_hat))
            return loss

        def compute_and_update_grads(loss_fn, ll_model, ul_model, ll_opt, ul_opt):
            grad_fn_ll = ops.GradOperation(get_by_list=True)(
                loss_fn, ll_model.trainable_params()
            )
            ll_grads = grad_fn_ll()
            ll_opt(ll_grads)
            grad_fn_ul = ops.GradOperation(get_by_list=True)(
                loss_fn, ul_model.trainable_params()
            )
            ul_grads = grad_fn_ul()
            ul_opt(ul_grads)

        compute_and_update_grads(
            loss_fn, self.ll_model, self.ul_model, self.ll_opt, self.ul_opt
        )
        # Compute upper-level objective
        F_y = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)

        # Update gamma
        self.gam += step_gam
        self.gam = min(self.gamma_max, self.gam)

        return F_y
