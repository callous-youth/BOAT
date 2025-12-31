import jittor as jit
from jittor import Module
from typing import List, Callable, Dict
from ..higher_jit.patch import _MonkeyPatchBase
from boat_jit.utils.op_utils import update_tensor_grads

from boat_jit.operation_registry import register_class
from boat_jit.na_ol.hyper_gradient import HyperGradient


@register_class
class IGA(HyperGradient):
    """
    Computes the hyper-gradient of the upper-level variables using Implicit Gradient Approximation (IGA) [1].

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
        Dictionary containing solver configurations, including:

        - `alpha_init` (float): Initial learning rate for GDA.
        - `alpha_decay` (float): Decay factor for the GDA learning rate.
        - Optional `gda_loss` (Callable): Custom loss function for GDA, if applicable.
        - `gm_op` (List[str]): Specifies dynamic operations, e.g., "DI" for dynamic initialization.

    Attributes
    ----------
    alpha : float
        Initial learning rate for GDA operations, if applicable.
    alpha_decay : float
        Decay factor applied to the GDA learning rate.
    gda_loss : Callable, optional
        Custom loss function for GDA operations, if specified in `solver_config`.
    dynamic_initialization : bool
        Indicates whether dynamic initialization is enabled, based on `gm_op`.

    References
    ----------
    [1] Liu R, Gao J, Liu X, et al., "Learning with constraint learning: New perspective, solution strategy and
        various applications," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
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
        super(IGA, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.gda_loss = solver_config.get("gda_loss", None)

        self.dynamic_initialization = "DI" in solver_config["gm_op"]

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
        Compute the hyper-gradients of the upper-level variables using the given feed dictionaries and patched models.

        Parameters
        ----------
        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization, including training data,
            targets, and other information required for the LL objective computation.
        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization, including validation data,
            targets, and other information required for the UL objective computation.
        auxiliary_model : _MonkeyPatchBase
            A patched lower-level model wrapped by the `higher` library, enabling differentiable optimization.
        max_loss_iter : int, optional
            The number of iterations used for backpropagation, by default 0.
        hyper_gradient_finished : bool, optional
            A flag indicating whether the hypergradient computation is finished, by default False.
        next_operation : str, optional
            The next operator for hypergradient calculation. Not supported in this implementation, by default None.
        **kwargs : dict
            Additional arguments, such as:

            - `lower_model_params` : List[jt.Var]
                List of parameters for the lower-level model.


        Returns
        -------
        Dict
            A dictionary containing:

            - `upper_loss` : jt.Var
                The upper-level objective value after optimization.
            - `hyper_gradient_finished` : bool
                Indicates whether the hypergradient computation is complete.

        Notes
        -----
        - This implementation calculates the Gauss-Newton (GN) loss to refine the gradients using second-order approximations.
        - If `dynamic_initialization` is enabled, the gradients of the lower-level variables are updated with time-dependent parameters.
        - Updates are performed on both lower-level and upper-level variables using computed gradients.

        Returns
        -------
        Dict
            A dictionary containing the upper-level objective and the status of hypergradient computation.
        """
        assert next_operation is None, "FD does not support next_operation"
        lower_model_params = kwargs.get(
            "lower_model_params", list(auxiliary_model.parameters())
        )
        if self.gda_loss is not None:
            ll_feed_dict["alpha"] = self.alpha * self.alpha_decay**max_loss_iter
            lower_loss = self.gda_loss(
                ll_feed_dict,
                ul_feed_dict,
                self.ul_model,
                auxiliary_model,
                params=lower_model_params,
            )
        else:
            lower_loss = self.ll_objective(
                ll_feed_dict, self.ul_model, auxiliary_model, params=lower_model_params
            )

        upper_loss = self.ul_objective(
            ul_feed_dict, self.ul_model, auxiliary_model, params=lower_model_params
        )
        dFy = jit.grad(upper_loss, lower_model_params, retain_graph=True)
        
        dfy = jit.grad(lower_loss, lower_model_params, retain_graph=True)


        # calculate GN loss
        gFyfy = 0
        gfyfy = 0
        for Fy, fy in zip(dFy, dfy):
            gFyfy = gFyfy + (Fy * fy).sum()
            gfyfy = gfyfy + (fy * fy).sum()
        GN_loss = -gFyfy.detach() / gfyfy.detach() * lower_loss

        if self.dynamic_initialization:
            grads_lower = jit.grad(
                upper_loss, list(auxiliary_model.parameters(time=0)), retain_graph=True
            )
            update_tensor_grads(self.ll_var, grads_lower)

        grads_upper = jit.grad(GN_loss + upper_loss, list(self.ul_var))

        update_tensor_grads(self.ul_var, grads_upper)

        return {"upper_loss": upper_loss.item(), "hyper_gradient_finished": True}
