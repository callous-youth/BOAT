from jittor import Module
from ..higher_jit.patch import _MonkeyPatchBase
from ..higher_jit.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable
from ..utils.op_utils import stop_grads
import jittor
from boat_jit.operation_registry import register_class
from boat_jit.gm_ol.dynamical_system import DynamicalSystem


@register_class
class NGD(DynamicalSystem):
    """
    Implements the optimization procedure of the Naive Gradient Descent (NGD) [1].

    Jittor version aligned with Torch version.

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
        The number of iterations for lower-level optimization.
    solver_config : Dict[str, Any]
        A dictionary containing configurations for the solver.
        Keys include::

            - "lower_level_opt": optimizer for LL
            - "na_op": list of hyper-gradient ops
            - "RGT": {"truncate_iter": int}

    References
    ----------
    [1] Franceschi et al., ICML 2018
    """


    def __init__(
        self,
        ll_objective: Callable,
        ul_objective: Callable,
        ll_model: Module,
        ul_model: Module,
        lower_loop: int,
        solver_config: Dict[str, Any],
    ):
        super(NGD, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.truncate_max_loss_iter = "PTT" in solver_config["na_op"]
        self.truncate_iters = solver_config["RGT"]["truncate_iter"] if "RGT" in solver_config["na_op"] else 0
        self.ll_opt = solver_config["lower_level_opt"]
        self.foa = "FOA" in solver_config["na_op"]

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

        if "gda_loss" in kwargs:
            gda_loss = kwargs["gda_loss"]
            alpha = kwargs["alpha"]
            alpha_decay = kwargs["alpha_decay"]
        else:
            gda_loss = None

        # ----------------- Truncate with RGT -----------------
        if self.truncate_iters > 0:
            ll_backup = [x.clone().stop_grad() for x in self.ll_model.parameters()]
            for _ in range(self.truncate_iters):
                if gda_loss is not None:
                    ll_feed_dict["alpha"] = alpha
                    loss_f = gda_loss(
                        ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                    )
                    alpha *= alpha_decay
                else:
                    loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)

                # naive gd step
                self.ll_opt.step(loss_f)

            # reco
            with jittor.no_grad():
                for x, y in zip(self.ll_model.parameters(), auxiliary_model.parameters()):
                    y.update(x.clone())
                for x, y in zip(ll_backup, self.ll_model.parameters()):
                    y.update(x.clone())
            del ll_backup

        # ----------------- Truncate with PTT -----------------
        if self.truncate_max_loss_iter:
            ul_loss_list = []
            for _ in range(self.lower_loop):
                if gda_loss is not None:
                    ll_feed_dict["alpha"] = alpha
                    loss_f = gda_loss(
                        ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                    )
                    alpha *= alpha_decay
                else:
                    loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)

                auxiliary_opt.step(loss_f)
                with jittor.no_grad():
                    upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model)
                    ul_loss_list.append(float(upper_loss.item()))
            ll_step_with_max_ul_loss = ul_loss_list.index(max(ul_loss_list))
            return ll_step_with_max_ul_loss + 1

        # ----------------- Standard lower-level loop -----------------
        for _ in range(self.lower_loop - self.truncate_iters):
            if gda_loss is not None:
                ll_feed_dict["alpha"] = alpha
                loss_f = gda_loss(
                    ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model
                )
                alpha *= alpha_decay
            else:
                loss_f = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model)

            auxiliary_opt.step(loss_f, grad_callback=stop_grads if self.foa else None)


        if next_operation is None:
            return -1
        else:
            return {
                "ll_feed_dict": ll_feed_dict,
                "ul_feed_dict": ul_feed_dict,
                "auxiliary_model": auxiliary_model,
                "auxiliary_opt": auxiliary_opt,
                "current_iter": current_iter,
                **kwargs,
            }
