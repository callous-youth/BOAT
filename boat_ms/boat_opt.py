import time
from typing import Dict, Any, Callable
try:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Tensor
    from mindspore.nn.optim import Optimizer
except ImportError as e:
    missing_module = str(e).split()[-1]  # 提取缺失模块的名称
    print(f"Error: The required module '{missing_module}' is not installed.")
    print("Please run the following command to install all required dependencies:")
    print("pip install -r requirements.txt")
    raise

import importlib
fo_gms = importlib.import_module("boat_ms.fogm")


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
    return lambda *args, **kwargs: func(*args, **{**loss_config.get("params", {}), **kwargs})


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
            - "device": Device configuration (e.g., "CPU", "GPU").
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
        self._ll_var = list(config["lower_level_var"])
        self._ul_var = list(config["upper_level_var"])
        self.boat_configs = config
        self.boat_configs["gda_loss"] = _load_loss_function(loss_config["gda_loss"]) \
            if 'GDA' in config["dynamic_op"] else None
        self._ll_loss = _load_loss_function(loss_config["lower_level_loss"])
        self._ul_loss = _load_loss_function(loss_config["upper_level_loss"])
        self._ll_solver = None
        self._ul_solver = None
        self._lower_opt = None
        self._upper_opt = None
        self._lower_init_opt = None
        self._fo_gm_solver = None
        self._lower_loop = None
        self._log_results_dict = {}
        self._device = ms.context.get_context("device_target")

    def build_ll_solver(self, lower_opt: Optimizer):
        """
        Configure the lower-level solver.

        :param lower_opt: The optimizer to use for the lower-level variables initialized (defined in the 'config["lower_level_var"]').
        :type lower_opt: Optimizer

        :returns: None
        """
        self._lower_opt = lower_opt
        self.boat_configs['ll_opt'] = self._lower_opt
        self._lower_loop = self.boat_configs.get("lower_iters", 10)
        self._fo_gm_solver = getattr(
            fo_gms, "%s" % self.boat_configs['fo_gm']
        )(ll_objective=self._ll_loss,
          ul_objective=self._ul_loss,
          ll_model=self._ll_model,
          ul_model=self._ul_model,
          lower_loop=self._lower_loop,
          ll_opt=self._lower_opt,
          ll_var=self._ll_var,
          ul_var=self._ul_var,
          solver_config=self.boat_configs)
        return self

    def build_ul_solver(self, upper_opt: Optimizer):
        """
        Configure the lower-level solver.

        :param upper_opt: The optimizer to use for the lower-level variables initialized (defined in the 'config["lower_level_var"]').
        :type upper_opt: Optimizer

        :returns: None
        """
        self._upper_opt = upper_opt
        setattr(self._fo_gm_solver, 'ul_opt', upper_opt)
        assert self.boat_configs['fo_gm'] is not None, \
            "Choose FOGM based methods from ['VSM','VFM','MESM'] or set 'dynamic_ol' and 'hyper_ol' properly."

        return self

    def run_iter(self, ll_feed_dict: Dict[str, Tensor], ul_feed_dict: Dict[str, Tensor], current_iter: int) -> tuple:
        """
        Run a single iteration of the bi-level optimization process.

        :param ll_feed_dict: Dictionary containing the real-time data and parameters fed for the construction of the lower-level (LL) objective.
            Example:
                {
                    "image": train_images,
                    "text": train_texts,
                    "target": train_labels  # Optional
                }
        :type ll_feed_dict: Dict[str, Tensor]

        :param ul_feed_dict: Dictionary containing the real-time data and parameters fed for the construction of the upper-level (UL) objective.
            Example:
                {
                    "image": val_images,
                    "text": val_texts,
                    "target": val_labels  # Optional
                }
        :type ul_feed_dict: Dict[str, Tensor]

        :param current_iter: The current iteration number.
        :type current_iter: int

        :returns: A tuple containing:
            - loss (float): The loss value for the current iteration.
            - run_time (float): The total time taken for the iteration.
        :rtype: tuple
        """
        self._log_results_dict['upper_loss'] = []
        start_time = time.perf_counter()
        self._log_results_dict['upper_loss'].append(
            self._fo_gm_solver.optimize(ll_feed_dict, ul_feed_dict, current_iter))
        run_time = time.perf_counter() - start_time

        return self._log_results_dict['upper_loss'], run_time


    def check_status(self):
        """
        Check the validity of the optimization setup and configuration.

        :raises AssertionError: If any configuration constraints are violated.
        """
        if "DM" in self.boat_configs["dynamic_op"]:
            assert (self.boat_configs["hyper_op"] == ["RAD"]) or (self.boat_configs["hyper_op"] == ["CG"]), \
                "When 'DM' is chosen, set the 'truncate_iter' properly."
        if "RGT" in self.boat_configs["hyper_op"]:
            assert self.boat_configs['RGT']["truncate_iter"] > 0, \
                "When 'RGT' is chosen, set the 'truncate_iter' properly."
        if self.boat_configs['accumulate_grad']:
            assert "IAD" in self.boat_configs['hyper_op'], \
                "When using 'accumulate_grad', only 'IAD' based methods are supported."
        if self.boat_configs['GDA']["alpha_init"] > 0.0:
            assert (0.0 < self.boat_configs['GDA']["alpha_decay"] <= 1.0), \
                "Parameter 'alpha_decay' used in method BDA should be in the interval (0,1)."
        if 'FD' in self._hyper_op:
            assert self.boat_configs['RGT']["truncate_iter"] == 0, \
                "One-stage method doesn't need trajectory truncation."

        def check_model_structure(base_model, meta_model):
            for param1, param2 in zip(base_model.parameters(), meta_model.parameters()):
                if (param1.shape != param2.shape) or (param1.dtype != param2.dtype) or (param1.device != param2.device):
                    return False
            return True

        if "IAD" in self._hyper_op:
            assert check_model_structure(self._ll_model, self._ul_model), \
                ("With IAD or FOA operation, 'upper_level_model' and 'lower_level_model' have the same structure, "
                 "and 'lower_level_var' and 'upper_level_var' are the same group of variables.")
        assert (("DI" in self._dynamic_op) ^ ("IAD" in self._hyper_op)) or (
                ("DI" not in self._dynamic_op) and ("IAD" not in self._hyper_op)), \
            "Only one of the 'PTT' and 'RGT' methods could be chosen."
        assert (0.0 <= self.boat_configs['GDA']["alpha_init"] <= 1.0), \
            "Parameter 'alpha' used in method BDA should be in the interval (0,1)."
        assert (self.boat_configs['RGT']["truncate_iter"] < self.boat_configs['lower_iters']), \
            "The value of 'truncate_iter' shouldn't be greater than 'lower_loop'."


