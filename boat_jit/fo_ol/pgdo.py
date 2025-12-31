from boat_jit.utils.op_utils import (
    grad_unused_zero,
    require_model_grad,
    update_tensor_grads,
    manual_update,
    l2_reg,
)
import jittor as jit
from jittor import Module
import copy
from typing import Dict, Any, Callable, List

from boat_jit.operation_registry import register_class
from boat_jit.gm_ol.dynamical_system import DynamicalSystem


@register_class
class PGDO(DynamicalSystem):
    """
    Implements the optimization procedure of Penalty-based Gradient Descent Method (PGDO) [1].

    Parameters
    ----------
    ll_objective : Callable
        The lower-level objective of the BLO problem.
    ul_objective : Callable
        The upper-level objective of the BLO problem.
    ll_model : jittor.Module
        The lower-level model of the BLO problem.
    ul_model : jittor.Module
        The upper-level model of the BLO problem.
    ll_var : List[jittor.Var]
        The list of lower-level variables of the BLO problem.
    ul_var : List[jittor.Var]
        The list of upper-level variables of the BLO problem.
    lower_loop : int
        Number of iterations for lower-level optimization.
    solver_config : Dict[str, Any]
        A dictionary containing solver configurations. Expected keys include:

        - "lower_level_opt": The optimizer for the lower-level model.
        - "PGDO" (Dict): A dictionary containing the following keys:
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
        super(PGDO, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.y_hat_lr = float(solver_config["PGDO"]["y_hat_lr"])
        self.gamma_init = solver_config["PGDO"]["gamma_init"]
        self.gamma_max = solver_config["PGDO"]["gamma_max"]
        self.gamma_argmax_step = solver_config["PGDO"]["gamma_argmax_step"]
        self.gam = self.gamma_init
        self.device = solver_config["device"]
        self.y_hat = copy.deepcopy(self.ll_model)
        self.y_hat_opt = jit.optim.SGD(list(self.y_hat.parameters()), lr=self.y_hat_lr)
        self.updata_y_ahead = solver_config["PGDO"]["updata_y_ahead"]
        self.penalty = solver_config["PGDO"]["penalty"]

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


        if self.gamma_init > self.gamma_max:
            self.gamma_max = self.gamma_init
            print(
                "Initial gamma is larger than max gamma, proceeding with gamma_max=gamma_init."
            )
        step_gam = (self.gamma_max - self.gamma_init) / self.gamma_argmax_step
        lr_decay = min(1 / (self.gam + 1e-8), 1)

        if self.updata_y_ahead: # meta_learning need
            for y_itr in range(self.lower_loop):
                self.ll_opt.zero_grad()
                tr_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
                grads_hat = grad_unused_zero(tr_loss, list(self.ll_model.parameters()))
                update_tensor_grads(list(self.ll_model.parameters()), grads_hat)
                manual_update(self.ll_opt, list(self.ll_model.parameters()))


        require_model_grad(self.y_hat)
        for y_itr in range(self.lower_loop):
            tr_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.y_hat)
            grads_hat = grad_unused_zero(tr_loss, self.y_hat.parameters())
            update_tensor_grads(list(self.y_hat.parameters()), grads_hat)
            manual_update(self.y_hat_opt, list(self.y_hat.parameters()))

        F_y = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)
        assert self.penalty in ["difference", "gradient"], "Set 'penalty' properly."
        if self.penalty == "difference":
            loss = lr_decay * (
                    F_y
                    + self.gam
                    * (
                            self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
                            - self.ll_objective(ll_feed_dict, self.ul_model, self.y_hat)
                    )
            )
        elif self.penalty == "gradient":
            tr_loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            g_y = grad_unused_zero(tr_loss, list(self.ll_model.parameters()))
            loss = lr_decay * (
                    F_y
                    + self.gam
                    * l2_reg(g_y)
            )

        grads_lower = grad_unused_zero(loss, self.ll_var)
        update_tensor_grads(self.ll_var, grads_lower)
        grads_upper = grad_unused_zero(loss, self.ul_var)
        update_tensor_grads(self.ul_var, grads_upper)
        self.gam += step_gam
        self.gam = min(self.gamma_max, self.gam)
        manual_update(self.ll_opt, list(self.ll_var))
        return {"upper_loss": F_y.item()}
