from boat_ms.utils.op_utils import copy_parameter_from_list
import numpy as np
from mindspore import nn, Tensor, ops
from typing import Dict, Any, Callable, List
import copy

from boat_ms.operation_registry import register_class
from boat_ms.gm_ol.dynamical_system import DynamicalSystem


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
        ul_model: nn.Cell,
        ul_objective: Callable,
        ll_model: nn.Cell,
        ll_var: List,
        ul_var: List,
        solver_config: Dict[str, Any],
    ):
        super(MESO, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )

        self.ll_opt = solver_config["lower_level_opt"]
        self.ul_opt = solver_config["upper_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_loop = lower_loop
        self.eta = solver_config["MESO"]["eta"]
        self.gamma_1 = solver_config["MESO"]["gamma_1"]
        self.c0 = solver_config["MESO"]["c0"]
        self.y_hat = copy.deepcopy(self.ll_model)
        self.y_hat_opt = nn.SGD(
            self.y_hat.trainable_params(),
            learning_rate=solver_config["MESO"]["y_hat_lr"],
            momentum=0.9,
        )

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        """
        Execute the optimization procedure with the data from feed_dict.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization. It typically includes training data, targets, and other information required to compute the LL objective.
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization. It typically includes validation data, targets, and other information required to compute the UL objective.
        current_iter : int
            The current iteration number of the optimization process.

        Returns
        -------
        The upper-level loss.
        """

        if current_iter == 0:
            ck = 0.2
        else:
            ck = np.power(current_iter + 1, 0.25) * self.c0

        grad_fn = ops.GradOperation(get_by_list=True)
        grad_theta_parameters = grad_fn(
            self.ll_objective, self.y_hat.trainable_params()
        )(ll_feed_dict, self.ul_model, self.y_hat)

        errs = []
        for a, b in zip(
            list(self.y_hat.trainable_params()), list(self.ll_model.trainable_params())
        ):
            diff = a - b
            errs.append(diff)
        vs_param = []
        for v0, gt, err in zip(
            list(self.y_hat.trainable_params()), grad_theta_parameters, errs
        ):
            vs_param.append(v0 - self.eta * (gt + self.gamma_1 * err))  # Update θ
        copy_parameter_from_list(self.y_hat, vs_param)

        reg = 0
        for param1, param2 in zip(list(self.ll_model.trainable_params()), vs_param):
            diff = param1 - param2
            reg += ops.norm(diff, ord=2) ** 2

        lower_loss = (
            (1 / ck) * self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
            + self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            - 0.5 * self.gamma_1 * reg
        )

        def wrapped_ll_objective():
            return self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)

        grad_fn_lower = ops.GradOperation(get_by_list=True)
        grad_y_parameters = grad_fn_lower(
            wrapped_ll_objective, self.ll_model.trainable_params()
        )()

        for param, grad in zip(self.ll_model.trainable_params(), grad_y_parameters):
            param.set_data(param - self.ll_opt.learning_rate * grad)

        def wrapped_ul_objective():
            return (
                (1 / ck) * self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
                + self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
                - self.ll_objective(ll_feed_dict, self.ul_model, self.y_hat)
            )

        grad_fn_upper = ops.GradOperation(get_by_list=True)
        grad_x_parameters = grad_fn_upper(
            wrapped_ul_objective, self.ul_model.trainable_params()
        )()
        self.ul_opt(grad_x_parameters)
        return lower_loss.item()
