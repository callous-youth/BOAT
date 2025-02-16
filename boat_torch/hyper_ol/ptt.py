import torch
from torch.nn import Module
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase

from boat_torch.operation_registry import register_class
from boat_torch.hyper_ol.hyper_gradient import HyperGradient


@register_class
class PTT(HyperGradient):
    """
    Computes the hyper-gradient of the upper-level variables using Pessimistic Trajectory Truncation (PTT) [1].

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
        Dictionary containing solver configurations, including:
        - "hyper_op" (List[str]): Indicates if PTT is used in the hyper-gradient operations.

    References
    ----------
    [1] Liu R., Liu Y., Zeng S., et al. "Towards gradient-based bilevel optimization with non-convex followers and beyond," in NeurIPS, 2021.
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
        super(PTT, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )
        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]

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
        Compute the hyper-gradients of the upper-level variables with the data from feed_dict and patched models.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.

        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.

        auxiliary_model : _MonkeyPatchBase
            A patched lower model wrapped by the `higher` library.
            It serves as the lower-level model for optimization.

        max_loss_iter : int, optional
            The number of iterations used for backpropagation, by default 0.

        next_operation : str, optional
            The next operator for the calculation of the hypergradient, by default None.

        hyper_gradient_finished : bool, optional
            A boolean flag indicating whether the hypergradient computation is finished, by default False.

        Returns
        -------
        Dict
            A dictionary containing updated feed_dict, auxiliary model, and gradient computation results.
        """
        assert (
            hyper_gradient_finished is False
        ), "Hypergradient computation should not be finished"
        assert self.truncate_max_loss_iter and (
            max_loss_iter > 0
        ), "With PTT operation, 'max_loss_iter' should be greater than 0"
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
