import torch
from torch.nn import Module
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase

from boat_torch.operation_registry import register_class
from boat_torch.na_ol.hyper_gradient import HyperGradient


@register_class
class RGT(HyperGradient):
    """
    Computes the hyper-gradient of the upper-level variables using Reverse Gradient Truncation (RGT) [1].

    Parameters
    ----------
    ll_objective : Callable
        The lower-level objective function of the BLO problem.
    ul_objective : Callable
        The upper-level objective function of the BLO problem.
    ll_model : torch.nn.Module
        The lower-level model of the BLO problem.
    ul_model : torch.nn.Module
        The upper-level model of the BLO problem.
    ll_var : List[torch.Tensor]
        List of variables optimized with the lower-level objective.
    ul_var : List[torch.Tensor]
        List of variables optimized with the upper-level objective.
    solver_config : Dict[str, Any]
        Dictionary containing solver configurations, including the hyper-gradient operations and truncation settings.

    References
    ----------
    [1] Shaban A., Cheng C.A., Hatch N., et al. "Truncated back-propagation for bilevel optimization," in AISTATS, 2019.
    """

    def __init__(
        self,
        ll_objective: Callable,
        ul_objective: Callable,
        ll_model: Module,
        ul_model: Module,
        ll_var: List,
        ul_var: List,
        solver_config: Dict,
    ):
        super(RGT, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )
        self.truncate_max_loss_iter = "PTT" in solver_config["na_op"]
        self.truncate_iter = solver_config["RGT"]["truncate_iter"]

    def compute_gradients(
        self,
        ll_feed_dict: Dict,
        ul_feed_dict: Dict,
        auxiliary_model: _MonkeyPatchBase,
        max_loss_iter: int = 0,
        hyper_gradient_finished: bool = False,
        next_operation: str = None,
        **kwargs
    ):
        """
        Computes the hyper-gradients of the upper-level variables using the data from feed_dict and patched models.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization. Typically includes training data, targets, and other information required to compute the lower-level objective.

        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization. Typically includes validation data, targets, and other information required to compute the upper-level objective.

        auxiliary_model : _MonkeyPatchBase
            A patched lower model wrapped by the `higher` library. It serves as the lower-level model for optimization.

        max_loss_iter : int, optional
            The number of iterations used for backpropagation. Default is 0.

        next_operation : str, optional
            The next operator for the calculation of the hypergradient. Default is None.

        hyper_gradient_finished : bool, optional
            A boolean flag indicating whether the hypergradient computation is finished. Default is False.

        **kwargs : dict
            Additional keyword arguments passed to the method.

        Returns
        -------
        Dict
            A dictionary containing:
            - `ll_feed_dict`: The lower-level feed dictionary.
            - `ul_feed_dict`: The upper-level feed dictionary.
            - `auxiliary_model`: The patched lower model.
            - `max_loss_iter`: The number of iterations for backpropagation.
            - `hyper_gradient_finished`: Flag indicating if the hypergradient computation is finished.
            - `lower_model_params`: Parameters of the lower model.
        """
        assert (
            hyper_gradient_finished is False
        ), "Hypergradient computation should not be finished"
        assert (
            self.truncate_iter > 0
        ), "With RGT operation, 'truncate_iter' should be greater than 0"
        assert next_operation is not None, "Next operation should be defined"
        lower_model_params = kwargs.get(
            "lower_model_params", list(auxiliary_model.parameters(time=max_loss_iter))
        )
        return {
            "ll_feed_dict": ll_feed_dict,
            "ul_feed_dict": ul_feed_dict,
            "auxiliary_model": auxiliary_model,
            "max_loss_iter": max_loss_iter,
            "hyper_gradient_finished": False,
            "lower_model_params": lower_model_params,
            **kwargs,
        }
