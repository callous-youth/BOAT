import time
from typing import Dict, Any, Callable
import mindspore as ms
from mindspore import Tensor
from mindspore.nn.optim import Optimizer
import matplotlib.pyplot as plt
import os
import json


import importlib
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
            - "fo_gm": First Order Gradient based Method (optional), e.g., ["VSM"], ["VFM"], ["MESM"].
            - "dynamic_op": List of dynamic operations (optional), e.g., ["NGD"], ["NGD", "GDA"], ["NGD", "GDA", "DI"].
            - "hyper_op": Hyper-optimization method (optional), e.g., ["RAD"], ["RAD", "PTT"], ["IAD", "NS", "PTT"].
            - "lower_level_loss": Configuration for the lower-level loss function based on the json file configuration.
            - "upper_level_loss": Configuration for the upper-level loss function based on the json file configuration.
            - "lower_level_model": The lower-level model to be optimized.
            - "upper_level_model": The upper-level model to be optimized.s
            - "lower_level_var": Variables in the lower-level model.
            - "upper_level_var": Variables in the upper-level model.
            - "device": Device configuration (e.g., "cpu", "cuda").
        :type config: Dict[str, Any]

        :param loss_config: Loss function configuration dictionary.
            - "lower_level_loss": Configuration for the lower-level loss function.
            - "upper_level_loss": Configuration for the upper-level loss function.
            - "GDA_loss": Configuration for GDA loss function (optional).
        :type loss_config: Dict[str, Any]

        :returns: None
        """
        self._fo_gm = config["fo_gm"]
        self._ll_model = config["lower_level_model"]
        self._ul_model = config["upper_level_model"]
        self._ll_var = list(config["lower_level_var"])
        self._ul_var = list(config["upper_level_var"])
        self.boat_configs = config
        self.boat_configs["gda_loss"] = (
            _load_loss_function(loss_config["gda_loss"])
            if "GDA" in config["dynamic_op"]
            else None
        )
        self._ll_loss = _load_loss_function(loss_config["lower_level_loss"])
        self._ul_loss = _load_loss_function(loss_config["upper_level_loss"])
        self._ll_solver = None
        self._ul_solver = None
        self._lower_opt = config["lower_level_opt"]
        self._upper_opt = config["upper_level_opt"]
        self._lower_init_opt = None
        self._fo_gm_solver = None
        self._lower_loop = None
        self._log_results_dict = {}
        self._device = ms.context.get_context("device_target")
        self.loss_log_path = config["loss_log_path"]
        self.loss_history = []

    def build_ll_solver(self):
        """
        Configure the lower-level solver.

        :returns: None
        """
        self.boat_configs["ll_opt"] = self._lower_opt
        self._lower_loop = self.boat_configs.get("lower_iters", 10)
        self._fo_gm_solver = get_registered_operation(
            "%s" % self.boat_configs["fo_gm"]
        )(
            ll_objective=self._ll_loss,
            ul_objective=self._ul_loss,
            ll_model=self._ll_model,
            ul_model=self._ul_model,
            lower_loop=self._lower_loop,
            ll_var=self._ll_var,
            ul_var=self._ul_var,
            solver_config=self.boat_configs,
        )
        return self

    def build_ul_solver(self):
        """
        Configure the lower-level solver.

        :returns: None
        """
        assert (
            self.boat_configs["fo_gm"] is not None
        ), "Choose FOGM based methods from ['VSM','VFM','MESM'] or set 'dynamic_ol' and 'hyper_ol' properly."

        return self

    def run_iter(
        self,
        ll_feed_dict: Dict[str, Tensor],
        ul_feed_dict: Dict[str, Tensor],
        current_iter: int,
    ) -> tuple:
        """
        Run a single iteration of the bi-level optimization process.

        :param ll_feed_dict: Dictionary containing the real-time data and parameters
            fed for the construction of the lower-level (LL) objective.

            Example::

                {
                    "image": train_images,
                    "text": train_texts,
                    "target": train_labels  # Optional
                }

        :type ll_feed_dict: Dict[str, Tensor]

        :param ul_feed_dict: Dictionary containing the real-time data and parameters
            fed for the construction of the upper-level (UL) objective.

            Example::

                {
                    "image": val_images,
                    "text": val_texts,
                    "target": val_labels  # Optional
                }

        :type ul_feed_dict: Dict[str, Tensor]

        :param current_iter: The current iteration number.
        :type current_iter: int

        :notes:
            - When `accumulate_grad` is set to True, you need to pack the data of
              each batch based on the format above.
            - In that case, pass `ll_feed_dict` and `ul_feed_dict` as lists of
              dictionaries, i.e., `[Dict[str, Tensor]]`.

        :returns: A tuple containing:
            - **loss** (*float*): The loss value for the current iteration.
            - **run_time** (*float*): The total time taken for the iteration.

        :rtype: tuple
        """
        self._log_results_dict["upper_loss"] = []
        start_time = time.perf_counter()
        self._log_results_dict["upper_loss"].append(
            self._fo_gm_solver.optimize(ll_feed_dict, ul_feed_dict, current_iter)
        )
        run_time = time.perf_counter() - start_time

        if isinstance(ll_feed_dict, list):
            ll_fd = ll_feed_dict[0]
            ul_fd = ul_feed_dict[0]
        else:
            ll_fd = ll_feed_dict
            ul_fd = ul_feed_dict

        ll_loss = self._ll_loss(ll_fd, self._ul_model, self._ll_model)
        ul_loss = self._ul_loss(ul_fd, self._ul_model, self._ll_model)
        print(f"ll_loss: {ll_loss.item()}  ul_loss: {ul_loss.item()}")
        self.save_losses(current_iter=current_iter, ll_loss=ll_loss, ul_loss=ul_loss)

        return self._log_results_dict["upper_loss"], run_time

    def plot_losses(self):
        iters = [x["iter"] for x in self.loss_history]
        ll_losses = [x["ll_loss"] for x in self.loss_history]
        ul_losses = [x["ul_loss"] for x in self.loss_history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图 - 下层 loss
        axes[0].plot(iters, ll_losses, label="Lower-level Loss", color="blue")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Lower-level Loss")
        axes[0].legend(loc="upper left")  # 图例在左上角
        axes[0].grid(True)

        # 右图 - 上层 loss
        axes[1].plot(iters, ul_losses, label="Upper-level Loss", color="orange")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Upper-level Loss")
        axes[1].legend(loc="upper left")  # 图例在左上角
        axes[1].grid(True)

        # 保存
        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(self.loss_log_path), "loss_curve.png")
        plt.savefig(save_path)
        plt.close()

    def save_losses(self, current_iter, ll_loss, ul_loss):
        """
        Save the losses to a JSON file and update the loss history.
        :param current_iter:iteration number
        :param ll_loss:lower loss
        :param ul_loss:upper loss
        :return: None
        """
        self.loss_history.append({
            "iter": current_iter,
            "ll_loss": float(ll_loss.item()),
            "ul_loss": float(ul_loss.item())
        })

        # 追加写入文件（每次迭代都更新）
        with open(self.loss_log_path, "w") as f:
            json.dump(self.loss_history, f)
