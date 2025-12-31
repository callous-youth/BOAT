import time
from typing import Dict, Any, Callable, List, Tuple
import mindspore as ms
from mindspore import Tensor, grad
import matplotlib.pyplot as plt
import os
import json
import importlib

from mindspore import context

from boat_ms.operation_registry import get_registered_operation


def _load_loss_function(loss_config: Dict[str, Any]) -> Callable:
    """
    Dynamically load a loss function from the provided configuration.

    :param loss_config: Dictionary with keys:
        - "function": Path to the loss function (e.g., "module.path.to_function").
        - "params": Parameters to be passed to the loss function.
    :type loss_config: Dict[str, Any]

    :returns: Loaded loss function ready for use.
    :rtype: Callable
    """
    module_name, func_name = loss_config["function"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)

    # Return a wrapper function that can accept both positional and keyword arguments
    return lambda *args, **kwargs: func(
        *args, **{**loss_config.get("params", {}), **kwargs}
    )


class Problem:
    """
    Enhanced bi-level optimization problem class supporting flexible loss functions and operation configurations.
    """

    def __init__(self, config: Dict[str, Any], loss_config: Dict[str, Any]):
        """
        Initialize the Problem instance.

        :param config: Configuration dictionary for the optimization setup.
            - "fo_op": First Order Gradient based Method (optional), e.g., ["VSO"], ["VFO"], ["MESO"].
            - "gm_op": List of gradient mapping operations (optional), e.g., ["NGD"], ["NGD", "GDA"], ["NGD", "GDA", "DI"].
            - "na_op": Hyper-optimization method (optional), e.g., ["RAD"], ["RAD", "PTT"], ["IAD", "NS", "PTT"].
            - "lower_level_loss": Configuration for the lower-level loss function based on the json file configuration.
            - "upper_level_loss": Configuration for the upper-level loss function based on the json file configuration.
            - "lower_level_model": The lower-level model to be optimized.
            - "upper_level_model": The upper-level model to be optimized.
            - "lower_level_var": Variables in the lower-level model.
            - "upper_level_var": Variables in the upper-level model.
            - "device": Device target (MindSpore uses context internally; kept for compatibility).
            - "fogo_batch_input": bool, align with torch version for batch-wise FOGO optimize.
            - "return_grad": bool, if True, return grads of upper vars instead of stepping optimizer.
            - "loss_log_path": str, path to save json losses and curve.
        :type config: Dict[str, Any]

        :param loss_config: Loss function configuration dictionary.
            - "lower_level_loss": Configuration for the lower-level loss function.
            - "upper_level_loss": Configuration for the upper-level loss function.
            - "gda_loss": Configuration for GDA loss function (optional).
        :type loss_config: Dict[str, Any]
        """
        self._fo_op = config["fo_op"]
        self._ll_model = config["lower_level_model"]
        self._ul_model = config["upper_level_model"]
        self._ll_var = list(config["lower_level_var"])
        self._ul_var = list(config["upper_level_var"])
        self.boat_configs = config

        # optional: dynamic ops specific loss
        self.boat_configs["gda_loss"] = (
            _load_loss_function(loss_config["gda_loss"])
            if ("gm_op" in config and config["gm_op"] is not None and "GDA" in config["gm_op"])
            else None
        )

        self._ll_loss = _load_loss_function(loss_config["lower_level_loss"])
        self._ul_loss = _load_loss_function(loss_config["upper_level_loss"])

        self._ll_solver = None
        self._ul_solver = None

        # Note: In MindSpore, Optimizer.step() is usually bound to a TrainingCell. Keep references if needed by solver.
        self._lower_opt = config.get("lower_level_opt", None)
        self._upper_opt = config.get("upper_level_opt", None)

        self._lower_init_opt = None
        self._fo_op_solver = None
        self._lower_loop = self.boat_configs.get("lower_iters", 10)

        self._log_results_dict = {}
        self._device = context.get_context("device_target")
        self.loss_log_path = config["loss_log_path"]
        self.loss_history: List[Dict[str, float]] = []

        # feature flags aligned with torch version
        self._fogo_batch_input = bool(self.boat_configs.get("fogo_batch_input", False))
        self._return_grad = bool(self.boat_configs.get("return_grad", False))

    def build_ll_solver(self):
        """
        Configure the lower-level solver.
        """
        self.boat_configs["ll_opt"] = self._lower_opt
        self._lower_loop = self.boat_configs.get("lower_iters", 10)
        # For MindSpore path we currently support FOGO-family via registered op
        self._fo_op_solver = get_registered_operation(
            "%s" % self.boat_configs["fo_op"]
        )(
            ll_objective=self._ll_loss,
            lower_loop=self._lower_loop,
            ul_model=self._ul_model,
            ul_objective=self._ul_loss,
            ll_model=self._ll_model,
            ll_var=self._ll_var,
            ul_var=self._ul_var,
            solver_config=self.boat_configs,
        )

        return self

    def build_ul_solver(self):
        """
        Placeholder for UL solver configuration (FOGO manages updates internally in MindSpore setting).
        """
        assert (
            self.boat_configs["fo_op"] is not None
        ), "Choose FOGO based methods from ['VSO','VFO','MESO'] or set 'gm_ol' and 'hyper_ol' properly."
        return self

    def _compute_current_losses(
        self,
        ll_fd: Dict[str, Tensor],
        ul_fd: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        ll_loss = self._ll_loss(ll_fd, self._ul_model, self._ll_model)
        ul_loss = self._ul_loss(ul_fd, self._ul_model, self._ll_model)
        return ll_loss, ul_loss

    def _compute_ul_grads(
        self,
        ul_fd: Dict[str, Tensor],
        ll_model,
        ul_model,
        ul_vars: List[Tensor],
    ):
        """
        Return gradients of upper-level loss w.r.t. upper-level variables.
        """
        # build a zero-arg closure to keep signatures clean for ms.grad
        def ul_loss_closure():
            return self._ul_loss(ul_fd, ul_model, ll_model)

        grad_fn = grad(ul_loss_closure, weights=tuple(ul_vars))
        grads = grad_fn()
        # mindspore.grad returns a structure matching weights; normalize to list
        if isinstance(grads, (tuple, list)):
            return list(grads)
        return [grads]

    def run_iter(
        self,
        ll_feed_dict: Dict[str, Tensor],
        ul_feed_dict: Dict[str, Tensor],
        current_iter: int,
    ) -> tuple:
        """
        Run a single iteration of the bi-level optimization process.

        :returns: (results, run_time)
        """
        self._log_results_dict["upper_loss"] = []
        start_time = time.perf_counter()

        # --- FOGO path (MindSpore registered op handles updates internally) ---
        if self._fo_op_solver is not None:
            if self._fogo_batch_input:
                # batch-wise optimize (zip two lists)
                for batch_ll_fd, batch_ul_fd in zip(ll_feed_dict, ul_feed_dict):
                    self._log_results_dict["upper_loss"].append(
                        self._fo_op_solver.optimize(batch_ll_fd, batch_ul_fd, current_iter)
                    )
            else:
                self._log_results_dict["upper_loss"].append(
                    self._fo_op_solver.optimize(ll_feed_dict, ul_feed_dict, current_iter)
                )
        run_time = time.perf_counter() - start_time

        # normalize fd for loss/grad logging
        if isinstance(ll_feed_dict, list):
            ll_fd = ll_feed_dict[0]
            ul_fd = ul_feed_dict[0]
        else:
            ll_fd = ll_feed_dict
            ul_fd = ul_feed_dict

        # --- return_grad behavior aligned with torch version ---
        if self._return_grad:
            # compute current losses (for logging)
            ll_loss, ul_loss = self._compute_current_losses(ll_fd, ul_fd)
            print(f"ll_loss: {float(ll_loss.asnumpy())}  ul_loss: {float(ul_loss.asnumpy())}")
            self.save_losses(current_iter=current_iter, ll_loss=ll_loss, ul_loss=ul_loss)

            # return grads of UL variables
            ul_grads = self._compute_ul_grads(
                ul_fd=ul_fd,
                ll_model=self._ll_model,
                ul_model=self._ul_model,
                ul_vars=self._ul_var,
            )
            return ul_grads, run_time

        # --- default: record losses (solver internally updated params if needed) ---
        ll_loss, ul_loss = self._compute_current_losses(ll_fd, ul_fd)
        print(f"ll_loss: {float(ll_loss.asnumpy())}  ul_loss: {float(ul_loss.asnumpy())}")
        self.save_losses(current_iter=current_iter, ll_loss=ll_loss, ul_loss=ul_loss)

        return self._log_results_dict["upper_loss"], run_time

    def plot_losses(self):
        iters = [x["iter"] for x in self.loss_history]
        ll_losses = [x["ll_loss"] for x in self.loss_history]
        ul_losses = [x["ul_loss"] for x in self.loss_history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(iters, ll_losses, label="Lower-level Loss", color="blue")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Lower-level Loss")
        axes[0].legend(loc="upper left")
        axes[0].grid(True)

        axes[1].plot(iters, ul_losses, label="Upper-level Loss", color="orange")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Upper-level Loss")
        axes[1].legend(loc="upper left")
        axes[1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(self.loss_log_path), "loss_curve.png")
        plt.savefig(save_path)
        plt.close()

    def save_losses(self, current_iter, ll_loss, ul_loss):
        """
        Save the losses to a JSON file and update the loss history.
        """
        # MindSpore Tensor -> python float
        ll_val = float(ll_loss.asnumpy())
        ul_val = float(ul_loss.asnumpy())

        self.loss_history.append({
            "iter": current_iter,
            "ll_loss": ll_val,
            "ul_loss": ul_val
        })

        with open(self.loss_log_path, "w") as f:
            json.dump(self.loss_history, f)
