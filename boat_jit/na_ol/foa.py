from jittor import Module
from typing import List, Callable, Dict
from ..higher_jit.patch import _MonkeyPatchBase

from boat_jit.operation_registry import register_class
from boat_jit.na_ol.hyper_gradient import HyperGradient


@register_class
class FOA(HyperGradient):
    """
    Computes the hyper-gradient of the upper-level variables using First-Order Approximation (FOA) [1], leveraging Initialization-based Auto Differentiation (IAD) [2].

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
    ll_var : List[jittor.Var]
        List of variables optimized with the lower-level objective.
    ul_var : List[jittor.Var]
        List of variables optimized with the upper-level objective.
    solver_config : Dict[str, Any]
        Dictionary containing solver configurations.

    References
    ----------
    [1] Nichol A., "On first-order meta-learning algorithms," arXiv preprint arXiv:1803.02999, 2018.
    [2] Finn C., Abbeel P., Levine S., "Model-agnostic meta-learning for fast adaptation of deep networks", in ICML, 2017.
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
        super(FOA, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )

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
        Compute the hyper-gradients of the upper-level variables using the data from feed_dict and patched models.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.

        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.

        auxiliary_model : _MonkeyPatchBase
            A patched lower-level model wrapped by the `higher` library.
            It serves as the lower-level model for optimization.

        max_loss_iter : int, optional
            The number of iterations used for backpropagation, by default 0.

        hyper_gradient_finished : bool, optional
            A boolean flag indicating whether the hypergradient computation is finished, by default False.

        next_operation : str, optional
            The next operator for the calculation of the hypergradient, by default None.

        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Dict
            A dictionary containing information required for the next step in the hypergradient computation,
            including the feed dictionaries, auxiliary model, iteration count, and other optional arguments.

        Raises
        ------
        AssertionError
            If `next_operation` is not defined or if `hyper_gradient_finished` is True.
        """
        assert next_operation is None, "FOA does not support next_operation"
        assert (
            hyper_gradient_finished is False
        ), "Hypergradient computation should not be finished"
        return {
            "ll_feed_dict": ll_feed_dict,
            "ul_feed_dict": ul_feed_dict,
            "auxiliary_model": auxiliary_model,
            "max_loss_iter": max_loss_iter,
            "hyper_gradient_finished": False,
            **kwargs,
        }
