import mindspore as ms
from mindspore import nn, ops, numpy as mnp
from typing import Dict, Any, Callable, List
import copy

from boat_ms.operation_registry import register_class
from boat_ms.gm_ol.dynamical_system import DynamicalSystem


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
    ll_model : mindspore.nn.Cell
        The lower-level model of the BLO problem.
    ul_model : mindspore.nn.Cell
        The upper-level model of the BLO problem.
    ll_var : List[mindspore.Tensor]
        A list of lower-level variables of the BLO problem.
    ul_var : List[mindspore.Tensor]
        A list of upper-level variables of the BLO problem.
    lower_loop : int
        The number of iterations for lower-level optimization.
    solver_config : Dict[str, Any]
        A dictionary containing configurations for the solver. Expected keys include:

        - "lower_level_opt" (mindspore.nn.optim.Optimizer): Optimizer for the lower-level model.
        - "VFO" (Dict): Configuration for the VFO algorithm:
            - "y_hat_lr" (float): Learning rate for optimizing the surrogate variable `y_hat`.
            - "eta" (float): Step size for value-function updates.
            - "u1" (float): Hyperparameter controlling the penalty in the value function.
        - "device" (str): Device on which computations are performed, e.g., "cpu" or "cuda".

    References
    ----------
    [1] R. Liu, X. Liu, X. Yuan, S. Zeng and J. Zhang, "A Value-Function-based Interior-point Method for Non-convex Bi-level Optimization," in ICML, 2021.
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
        super(VFO, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_opt = solver_config["lower_level_opt"]
        self.ul_opt = solver_config["upper_level_opt"]
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
            Dictionary containing the lower-level data used for optimization. It typically includes training data, targets, and other information required to compute the LL objective.
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization. It typically includes validation data, targets, and other information required to compute the UL objective.
        current_iter : int
            The current iteration number of the optimization process.

        Returns
        -------
        The upper-level loss.
        """

        y_hat = copy.deepcopy(self.ll_model)
        y_hat_opt = nn.SGD(
            y_hat.trainable_params(), learning_rate=self.y_hat_lr, momentum=0.9
        )
        n_params_y = sum([p.size for p in self.ll_model.trainable_params()])
        n_params_x = sum([p.size for p in self.ul_model.trainable_params()])
        delta_f = mnp.zeros((n_params_x + n_params_y), dtype=ms.float32)
        delta_F = mnp.zeros((n_params_x + n_params_y), dtype=ms.float32)

        def g_x_xhat_w(y, y_hat, x):
            loss = self.ll_objective(ll_feed_dict, x, y) - self.ll_objective(
                ll_feed_dict, x, y_hat
            )
            grad_y = ops.GradOperation(get_by_list=True)(
                self.ll_objective, y.trainable_params()
            )(ll_feed_dict, x, y)
            grad_x = ops.GradOperation(get_by_list=True)(
                self.ll_objective, x.trainable_params()
            )(ll_feed_dict, x, y)
            return loss, grad_y, grad_x

        for y_itr in range(self.lower_loop):
            for param in y_hat.trainable_params():
                param.set_data(mnp.zeros_like(param.data))

            grad_fn = ops.GradOperation(get_by_list=True)(
                self.ll_objective, y_hat.trainable_params()
            )
            grads_hat = grad_fn(ll_feed_dict, self.ul_model, y_hat)

            y_hat_opt(grads_hat)

        F_y = self.ul_objective(ul_feed_dict, self.ul_model, self.ll_model)

        grad_F_y = ops.GradOperation(get_by_list=True)(
            self.ul_objective, self.ll_model.trainable_params()
        )(ul_feed_dict, self.ul_model, self.ll_model)
        grad_F_x = ops.GradOperation(get_by_list=True)(
            self.ul_objective, self.ul_model.trainable_params()
        )(ul_feed_dict, self.ul_model, self.ll_model)

        for param in y_hat.trainable_params():
            param.requires_grad = False

        loss, gy, gx_minus_gx_k = g_x_xhat_w(self.ll_model, y_hat, self.ul_model)

        delta_F[:n_params_y] = mnp.concatenate([p.view(-1) for p in grad_F_y]).astype(
            ms.float32
        )
        delta_f[:n_params_y] = mnp.concatenate([p.view(-1) for p in gy]).astype(
            ms.float32
        )
        delta_F[n_params_y:] = mnp.concatenate([p.view(-1) for p in grad_F_x]).astype(
            ms.float32
        )
        delta_f[n_params_y:] = mnp.concatenate(
            [p.view(-1) for p in gx_minus_gx_k]
        ).astype(ms.float32)

        norm_dq = (ops.norm(delta_f) ** 2).astype(ms.float32)
        dot = ops.ReduceSum()(delta_F * delta_f)
        correction = ops.ReLU()((self.u1 * loss - dot) / (norm_dq + 1e-8))
        d = delta_F + correction * delta_f
        y_grad = []
        x_grad = []
        all_numel = 0

        for param in self.ll_model.trainable_params():
            grad_slice = d[all_numel : all_numel + param.size].reshape(param.shape)
            y_grad.append(ms.Tensor(grad_slice, dtype=ms.float32))
            all_numel += param.size

        for param in self.ul_model.trainable_params():
            grad_slice = d[all_numel : all_numel + param.size].reshape(param.shape)
            x_grad.append(ms.Tensor(grad_slice, dtype=ms.float32))
            all_numel += param.size

        for param, grad in zip(self.ll_model.trainable_params(), y_grad):
            new_param = param - self.y_hat_lr * grad
            param.set_data(new_param)

        for param, grad in zip(self.ul_model.trainable_params(), x_grad):
            new_param = param - self.y_hat_lr * grad
            param.set_data(new_param)

        return F_y.item()
