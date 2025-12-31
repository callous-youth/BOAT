from jittor import Module
from typing import Callable
from ..higher_jit.patch import _MonkeyPatchBase
from ..higher_jit.optim import DifferentiableOptimizer
from typing import Dict, Any, Callable

from boat_jit.operation_registry import register_class
from boat_jit.gm_ol.dynamical_system import DynamicalSystem


@register_class
class GDA(DynamicalSystem):
    """
    Implements the optimization procedure of the Gradient Descent Aggregation (GDA) [1].

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
        A dictionary containing configurations for the solver. Expected keys include:

        - "GDA" (Dict): Configuration for the GDA algorithm:
            - "alpha_init" (float): Initial learning rate for the GDA updates.
            - "alpha_decay" (float): Decay rate for the learning rate.
        - "gda_loss" (Callable): The loss function used in the GDA optimization.

    References
    ----------
    [1] R. Liu, P. Mu, X. Yuan, S. Zeng, and J. Zhang, "A generic first-order algorithmic framework for bi-level programming beyond lower-level singleton", in ICML, 2020.
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
        super(GDA, self).__init__(
            ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config
        )
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.gda_loss = solver_config["gda_loss"]

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
        """
        Execute the lower-level optimization procedure using provided data, models, and patched optimizers.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization.
            Typically includes training data, targets, and other information required to compute the LL objective.
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization.
            Typically includes validation data, targets, and other information required to compute the UL objective.
        auxiliary_model : _MonkeyPatchBase
            A patched lower model wrapped by the `higher` library.
            This model is used for differentiable optimization in the lower-level procedure.
        auxiliary_opt : DifferentiableOptimizer
            A patched optimizer for the lower-level model, wrapped by the `higher` library.
            Allows differentiable optimization.
        current_iter : int
            The current iteration number of the optimization process.
        next_operation : str, optional
            Specifies the next operation to be executed in the optimization pipeline.
            Default is None.
        **kwargs : dict
            Additional parameters required for the optimization procedure.

        Returns
        -------
        dict
            A dictionary containing:
                - "ll_feed_dict" : Dict
                    Lower-level feed dictionary.
                - "ul_feed_dict" : Dict
                    Upper-level feed dictionary.
                - "auxiliary_model" : _MonkeyPatchBase
                    Patched lower-level model.
                - "auxiliary_opt" : DifferentiableOptimizer
                    Patched lower-level optimizer.
                - "current_iter" : int
                    Current iteration number.
                - "gda_loss" : callable
                    Gradient Descent Aggregation (GDA) loss function.
                - "alpha" : float
                    Coefficient used in the GDA operation, typically in (0, 1).
                - "alpha_decay" : float
                    Decay factor for the coefficient `alpha`.

        Raises
        ------
        AssertionError
            If `next_operation` is not defined.
            If `alpha` is not in the range (0, 1).
            If `gda_loss` is not properly defined.

        Notes
        -----
        - The method assumes that `gda_loss` is defined and accessible from the instance attributes.
        - The coefficient `alpha` and its decay rate `alpha_decay` must be properly configured.
        """

        assert next_operation is not None, "Next operation should be defined."
        assert (self.alpha > 0) and (
            self.alpha < 1
        ), "Set the coefficient alpha properly in (0,1)."
        assert (
            self.gda_loss is not None
        ), "Define the gda_loss properly in loss_func.py."
        return {
            "ll_feed_dict": ll_feed_dict,
            "ul_feed_dict": ul_feed_dict,
            "auxiliary_model": auxiliary_model,
            "auxiliary_opt": auxiliary_opt,
            "current_iter": current_iter,
            "gda_loss": self.gda_loss,
            "alpha": self.alpha,
            "alpha_decay": self.alpha_decay,
            **kwargs,
        }
