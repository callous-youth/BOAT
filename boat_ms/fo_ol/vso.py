from boat_ms.utils.op_utils import l2_reg
import mindspore as ms
from mindspore import nn, ops, Tensor
import copy
from typing import Dict, Any, Callable, List
from boat_ms.operation_registry import register_class
from boat_ms.gm_ol.dynamical_system import DynamicalSystem


@register_class
class VSO(DynamicalSystem):
    """
    Value-function based Sequential Method (VSO) [Liu et al., NeurIPS 2022]
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
        super(VSO, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.ll_var = ll_var
        self.ul_var = ul_var
        self.ll_opt = solver_config["lower_level_opt"]
        self.ul_opt = solver_config["upper_level_opt"]
        self.y_loop = lower_loop
        self.z_loop = solver_config["VSO"]["z_loop"]
        self.ll_l2_reg = solver_config["VSO"]["ll_l2_reg"]
        self.ul_l2_reg = solver_config["VSO"]["ul_l2_reg"]
        self.ul_ln_reg = solver_config["VSO"]["ul_ln_reg"]
        self.reg_decay = float(solver_config["VSO"]["reg_decay"])
        self.z_lr = solver_config["VSO"]["z_lr"]

    def optimize(self, ll_feed_dict: Dict, ul_feed_dict: Dict, current_iter: int):
        reg_decay = self.reg_decay * current_iter + 1

        # -------- z loop --------
        for _ in range(self.z_loop):
            def compute_loss_z():
                loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.trainable_params())
                loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
                return loss_z_ + loss_l2_z

            grad_fn = ops.GradOperation(get_by_list=True)(compute_loss_z, self.ll_model.trainable_params())
            grads = grad_fn()
            self.ll_opt(grads)

        # -------- auxiliary model --------
        auxiliary_model = copy.deepcopy(self.ll_model)
        auxiliary_opt = nn.SGD(auxiliary_model.trainable_params(), learning_rate=self.z_lr)

        # loss_z
        loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.trainable_params())
        loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        loss_z = loss_z_ + loss_l2_z

        # -------- y loop --------
        for _ in range(self.y_loop):
            def compute_loss_y():
                loss_y_f_ = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
                loss_y_ = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
                loss_l2_y = self.ul_l2_reg / reg_decay * l2_reg(auxiliary_model.trainable_params())
                loss_ln = ops.log(loss_y_f_ + loss_z - loss_y_f_ + 1e-8)  # ✅ 避免 log(0)
                loss_ln = self.ul_ln_reg / reg_decay * loss_ln
                return loss_y_ - loss_ln + loss_l2_y

            grad_fn = ops.GradOperation(get_by_list=True)(compute_loss_y, auxiliary_model.trainable_params())
            grads = grad_fn()
            auxiliary_opt(grads)

        # -------- x step --------
        def compute_loss_x():
            loss_l2_z = self.ll_l2_reg / reg_decay * l2_reg(self.ll_model.trainable_params())
            loss_z_ = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
            loss_z = loss_z_ + loss_l2_z
            loss_y_f_ = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)
            loss_ln = self.ul_ln_reg / reg_decay * ops.log(loss_y_f_ + loss_z - loss_y_f_ + 1e-8)
            loss_x_ = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
            return loss_x_ - loss_ln

        grad_fn = ops.GradOperation(get_by_list=True)(compute_loss_x, self.ul_model.trainable_params())
        ul_grads = grad_fn()
        self.ul_opt(ul_grads)

        loss_x_value = compute_loss_x()
        return float(loss_x_value.asnumpy())
