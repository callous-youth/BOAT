import time
import copy
from typing import Dict, Any, Callable
from boat_torch.utils.op_utils import copy_parameter_from_list, average_grad

import torch
from torch import Tensor
import higher

from boat_torch.operation_registry import get_registered_operation
from boat_torch.dynamic_ol import makes_functional_dynamical_system
from boat_torch.hyper_ol import makes_functional_hyper_operation
import matplotlib.pyplot as plt
import os
import json

importlib = __import__("importlib")


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
            - "upper_level_model": The upper-level model to be optimized.
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
        self._dynamic_op = config["dynamic_op"]
        self._hyper_op = config["hyper_op"]
        self._ll_model = config["lower_level_model"]
        self._ul_model = config["upper_level_model"]
        self._ll_var = config["lower_level_var"]
        self._ul_var = config["upper_level_var"]
        self._lower_opt = config["lower_level_opt"]
        self._upper_opt = config["upper_level_opt"]
        self._ll_loss = _load_loss_function(loss_config["lower_level_loss"])
        self._ul_loss = _load_loss_function(loss_config["upper_level_loss"])
        self.boat_configs = config
        self._lower_loop = config.get("lower_iters", 10)
        self._log_results = []
        self._device = torch.device(config["device"])
        self._ll_solver = None
        self._ul_solver = None
        self._lower_init_opt = None
        self._fo_gm_solver = None
        self._track_opt_traj = False
        if config["dynamic_op"] is not None:
            if "GDA" in config["dynamic_op"]:
                assert (
                    loss_config.get("gda_loss", None) is not None
                ), "Set the 'gda_loss' in loss_config properly."
                self.boat_configs["gda_loss"] = _load_loss_function(
                    loss_config["gda_loss"]
                )
        self.loss_log_path = config["loss_log_path"]
        self.loss_history = []

    def build_ll_solver(self):
        """
        Configure the lower-level solver.

        :returns: None
        """
        if self.boat_configs["fo_gm"] is None:
            assert (self.boat_configs["dynamic_op"] is not None) and (
                self.boat_configs["hyper_op"] is not None
            ), "Set 'dynamic_op' and 'hyper_op' properly."
            self.check_status()
            sorted_ops = sorted([op.upper() for op in self._dynamic_op])
            self._ll_solver = makes_functional_dynamical_system(
                custom_order=sorted_ops,
                ll_objective=self._ll_loss,
                ul_objective=self._ul_loss,
                ll_model=self._ll_model,
                ul_model=self._ul_model,
                lower_loop=self._lower_loop,
                solver_config=self.boat_configs,
            )
            if "DI" in self.boat_configs["dynamic_op"]:
                self._lower_init_opt = copy.deepcopy(self._lower_opt)
                for _ in range(len(self._lower_init_opt.param_groups)):
                    self._lower_init_opt.param_groups[_]["params"] = (
                        self._lower_opt.param_groups[_]["params"]
                    )
                    self._lower_init_opt.param_groups[_]["lr"] = self.boat_configs[
                        "DI"
                    ]["lr"]

        else:
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
        if self.boat_configs["fo_gm"] is None:
            assert (
                self.boat_configs["hyper_op"] is not None
            ), "Choose FOGM based methods from ['VSM','VFM','MESM', 'PGDM'] or set 'dynamic_ol' and 'hyper_ol' properly. Currently, fo_gm ={} is not None".format(
                self.boat_configs["fo_gm"]
            )
            sorted_ops = sorted([op.upper() for op in self._hyper_op])
            if "DM" not in self._dynamic_op:
                self._ul_solver = makes_functional_hyper_operation(
                    custom_order=sorted_ops,
                    ul_objective=self._ul_loss,
                    ll_objective=self._ll_loss,
                    ll_model=self._ll_model,
                    ul_model=self._ul_model,
                    ll_var=self._ll_var,
                    ul_var=self._ul_var,
                    solver_config=self.boat_configs,
                )
        else:
            assert (
                self.boat_configs["hyper_op"] is None
            ), "Choose FOGM based methods from ['VSM','VFM','MESM', 'PGDM'] or set 'dynamic_ol' and 'hyper_ol' properly. Currently, hyper_op ={} is not None".format(
                self.boat_configs["hyper_op"]
            )
            self._ul_solver = None
        return self

    def run_iter(
        self,
        ll_feed_dict: Dict[str, Tensor],
        ul_feed_dict: Dict[str, Tensor],
        current_iter: int,
    ) -> tuple:
        """
        Run a single iteration of the bi-level optimization process.

        :param ll_feed_dict: Dictionary containing the real-time data and parameters fed for the construction of the lower-level (LL) objective.

            Example::

                {
                    "image": train_images,
                    "text": train_texts,
                    "target": train_labels  # Optional
                }

        :type ll_feed_dict: Dict[str, Tensor]

        :param ul_feed_dict: Dictionary containing the real-time data and parameters fed for the construction of the upper-level (UL) objective.

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
            - When `accumulate_grad` is set to True, you need to pack the data of each batch based on the format above.
            - In that case, pass `ll_feed_dict` and `ul_feed_dict` as lists of dictionaries, i.e., `[Dict[str, Tensor]]`.

        :returns: A tuple containing:
            - **loss** (*float*): The loss value for the current iteration.
            - **run_time** (*float*): The total time taken for the iteration.

        :rtype: tuple
        """

        if self.boat_configs["fo_gm"] is not None:
            start_time = time.perf_counter()
            self._log_results.append(
                self._fo_gm_solver.optimize(ll_feed_dict, ul_feed_dict, current_iter)
            )
            run_time = time.perf_counter() - start_time
        else:
            run_time = 0
            if self.boat_configs["accumulate_grad"]:
                for batch_ll_feed_dict, batch_ul_feed_dict in zip(
                    ll_feed_dict, ul_feed_dict
                ):
                    with higher.innerloop_ctx(
                        self._ll_model,
                        self._lower_opt,
                        copy_initial_weights=False,
                        device=self._device,
                        track_higher_grads=self._track_opt_traj,
                    ) as (auxiliary_model, auxiliary_opt):
                        forward_time = time.perf_counter()
                        dynamic_results = self._ll_solver.optimize(
                            ll_feed_dict=batch_ll_feed_dict,
                            ul_feed_dict=batch_ul_feed_dict,
                            auxiliary_model=auxiliary_model,
                            auxiliary_opt=auxiliary_opt,
                            current_iter=current_iter,
                        )
                        self._log_results.append(dynamic_results)
                        max_loss_iter = list(dynamic_results[-1].values())[-1]
                        forward_time = time.perf_counter() - forward_time
                        backward_time = time.perf_counter()
                        self._log_results.append(
                            self._ul_solver.compute_gradients(
                                ll_feed_dict=batch_ll_feed_dict,
                                ul_feed_dict=batch_ll_feed_dict,
                                auxiliary_model=auxiliary_model,
                                max_loss_iter=max_loss_iter,
                            )
                        )
                        backward_time = time.perf_counter() - backward_time
                    run_time += forward_time + backward_time
                average_grad(self._ul_model, len(ll_feed_dict))
            else:
                with higher.innerloop_ctx(
                    self._ll_model,
                    self._lower_opt,
                    copy_initial_weights=False,
                    device=self._device,
                    track_higher_grads=self._track_opt_traj,
                ) as (auxiliary_model, auxiliary_opt):
                    forward_time = time.perf_counter()
                    dynamic_results = self._ll_solver.optimize(
                        ll_feed_dict=ll_feed_dict,
                        ul_feed_dict=ul_feed_dict,
                        auxiliary_model=auxiliary_model,
                        auxiliary_opt=auxiliary_opt,
                        current_iter=current_iter,
                    )
                    max_loss_iter = list(dynamic_results[-1].values())[-1]
                    forward_time = time.perf_counter() - forward_time
                    print("forward_time", forward_time)
                    backward_time = time.perf_counter()
                    if self._ul_solver is not None:
                        self._log_results.append(
                            self._ul_solver.compute_gradients(
                                ll_feed_dict=ll_feed_dict,
                                ul_feed_dict=ul_feed_dict,
                                auxiliary_model=auxiliary_model,
                                max_loss_iter=max_loss_iter,
                            )
                        )
                    backward_time = time.perf_counter() - backward_time
                    print("backward_time", backward_time)
                    if self.boat_configs["copy_last_param"]:
                        copy_parameter_from_list(
                            self._ll_model,
                            list(auxiliary_model.parameters(time=-1)),
                        )
                if "DI" in self.boat_configs["dynamic_op"]:
                    self._lower_init_opt.step()
                    self._lower_init_opt.zero_grad()
                run_time = forward_time + backward_time

        if isinstance(ll_feed_dict, list):
            ll_fd = ll_feed_dict[0]
            ul_fd = ul_feed_dict[0]
        else:
            ll_fd = ll_feed_dict
            ul_fd = ul_feed_dict

        if not self.boat_configs["return_grad"]:
            self._upper_opt.step()
            self._upper_opt.zero_grad()
        else:
            ll_loss = self._ll_loss(ll_fd, self._ul_model, self._ll_model)
            ul_loss = self._ul_loss(ul_fd, self._ul_model, self._ll_model)
            print(f"ll_loss: {ll_loss.item()}  ul_loss: {ul_loss.item()}")
            self.save_losses(current_iter = current_iter, ll_loss = ll_loss, ul_loss = ul_loss)
            return [var.grad for var in list(self._ul_var)], run_time

        ll_loss = self._ll_loss(ll_fd, self._ul_model, self._ll_model)
        ul_loss = self._ul_loss(ul_fd, self._ul_model, self._ll_model)
        print(f"ll_loss: {ll_loss.item()}  ul_loss: {ul_loss.item()}")
        self.save_losses(current_iter = current_iter, ll_loss = ll_loss, ul_loss = ul_loss)

        return self._log_results, run_time

    def set_track_trajectory(self, track_traj=True):
        self._track_opt_traj = track_traj

    def check_status(self):
        if any(item in self._hyper_op for item in ["PTT", "IAD", "RAD"]):
            self.set_track_trajectory(True)
        if "DM" in self.boat_configs["dynamic_op"]:
            assert (self.boat_configs["hyper_op"] == ["RAD"]) or (
                self.boat_configs["hyper_op"] == ["CG"]
            ), "When 'DM' is chosen, set the 'truncate_iter' properly."
        if "RGT" in self.boat_configs["hyper_op"]:
            assert (
                self.boat_configs["RGT"]["truncate_iter"] > 0
            ), "When 'RGT' is chosen, set the 'truncate_iter' properly ."
        if self.boat_configs["accumulate_grad"]:
            assert (
                "IAD" in self.boat_configs["hyper_op"]
            ), "When using 'accumulate_grad', only 'IAD' based methods are supported."
        if self.boat_configs["GDA"]["alpha_init"] > 0.0:
            assert (
                0.0 < self.boat_configs["GDA"]["alpha_decay"] <= 1.0
            ), "Parameter 'alpha_decay' used in method BDA should be in the interval (0,1)."
        if "FD" in self._hyper_op:
            assert (
                self.boat_configs["RGT"]["truncate_iter"] == 0
            ), "One-stage method doesn't need trajectory truncation."

        def check_model_structure(base_model, meta_model):
            for param1, param2 in zip(base_model.parameters(), meta_model.parameters()):
                if (
                    (param1.shape != param2.shape)
                    or (param1.dtype != param2.dtype)
                    or (param1.device != param2.device)
                ):
                    return False
            return True

        if "IAD" in self._hyper_op:
            assert check_model_structure(self._ll_model, self._ul_model), (
                "With IAD or FOA operation, 'upper_level_model' and 'lower_level_model' have the same structure, "
                "and 'lower_level_var' and 'upper_level_var' are the same group of variables."
            )
        assert (("DI" in self._dynamic_op) ^ ("IAD" in self._hyper_op)) or (
            ("DI" not in self._dynamic_op) and ("IAD" not in self._hyper_op)
        ), "Only one of the 'PTT' and 'RGT' methods could be chosen."
        assert (
            0.0 <= self.boat_configs["GDA"]["alpha_init"] <= 1.0
        ), "Parameter 'alpha' used in method BDA should be in the interval (0,1)."
        assert (
            self.boat_configs["RGT"]["truncate_iter"] < self.boat_configs["lower_iters"]
        ), "The value of 'truncate_iter' shouldn't be greater than 'lower_loop'."

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