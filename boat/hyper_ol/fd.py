import torch
from torch.nn import Module
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase
from boat.utils.op_utils import update_tensor_grads,grad_unused_zero

from boat.operation_registry import register_class
from boat.hyper_ol.hyper_gradient import HyperGradient


@register_class
class FD(HyperGradient):
    """
    Computes the hyper-gradient of the upper-level variables using Finite Differentiation (FD) [1].

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
        Dictionary containing solver configurations. Expected keys include:

        - `r` (float): Perturbation radius for finite differences.
        - `lower_level_opt` (torch.optim.Optimizer): Lower-level optimizer configuration.
        - `dynamic_op` (str): Indicates dynamic initialization type (e.g., "DI").
        - GDA-specific parameters if applicable, such as:
            - `alpha_init` (float): Initial learning rate for GDA.
            - `alpha_decay` (float): Decay factor for GDA.

    Attributes
    ----------
    ll_lr : float
        Learning rate for the lower-level optimizer, extracted from `lower_level_opt`.
    dynamic_initialization : bool
        Indicates whether dynamic initialization is enabled (based on `dynamic_op`).
    _r : float
        Perturbation radius for finite differences, used for gradient computation.
    alpha : float
        Initial learning rate for GDA operations.
    alpha_decay : float
        Decay factor applied to the learning rate for GDA.
    gda_loss : Callable, optional
        Custom loss function for GDA operations, if specified in `solver_config`.

    References
    ----------
    [1] H. Liu, K. Simonyan, Y. Yang, "DARTS: Differentiable Architecture Search," in ICLR, 2019.
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
        super(FD, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )
        self.ll_lr = solver_config["lower_level_opt"].defaults["lr"]
        self.dynamic_initialization = "DI" in solver_config["dynamic_op"]
        self._r = solver_config["FD"]["r"]
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.gda_loss = solver_config.get("gda_loss", None)

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
        Compute the hyper-gradients of the upper-level variables with the data from `feed_dict` and patched models.

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
            The number of iterations used for backpropagation. Default is 0.

        hyper_gradient_finished : bool, optional
            A boolean flag indicating whether the hyper-gradient computation is finished. Default is False.

        next_operation : str, optional
            The next operator for the calculation of the hyper-gradient. Default is None.

        Returns
        -------
        dict
            A dictionary containing:
            - "upper_loss": The current upper-level objective value.
            - "hyper_gradient_finished": A boolean indicating whether the hyper-gradient computation is complete.

        Raises
        ------
        AssertionError
            If `next_operation` is not None, as FD does not support `next_operation`.
        """
        assert next_operation is None, "FD does not support next_operation"
        lower_model_params = kwargs.get(
            "lower_model_params", list(auxiliary_model.parameters())
        )
        loss = self.ul_objective(
            ul_feed_dict, self.ul_model, auxiliary_model, params=lower_model_params
        )

        grad_x = grad_unused_zero(loss, self.ul_var, retain_graph=True)
        grad_y = grad_unused_zero(
            loss,
            lower_model_params,
            retain_graph=self.dynamic_initialization
        )
        dalpha = [v.data for v in grad_x]
        vector = [v.data for v in grad_y]

        implicit_grads = self._hessian_vector_product(
            vector, ll_feed_dict, ul_feed_dict
        )

        for g, ig in zip(dalpha, implicit_grads):
            g.sub_(ig.data, alpha=self.ll_lr)

        if self.dynamic_initialization:
            grads_lower = torch.autograd.grad(
                loss, list(auxiliary_model.parameters(time=0))
            )
            update_tensor_grads(self.ll_var, grads_lower)
        update_tensor_grads(self.ul_var, dalpha)
        return {"upper_loss": loss.item(), "hyper_gradient_finished": True}

    def _hessian_vector_product(self, vector, ll_feed_dict, ul_feed_dict):
        """
        Compute the first-order approximation of the second-order derivative of upper-level variables.

        Parameters
        ----------
        vector : List[Tensor]
            A vector used for computing the Hessian-vector product.

        ll_feed_dict : Dict
            Dictionary containing the lower-level data used for optimization.

        ul_feed_dict : Dict
            Dictionary containing the upper-level data used for optimization.

        Returns
        -------
        List[Tensor]
            A list of tensors representing the first-order approximation of the second-order derivative (Hessian-vector product).

        Notes
        -----
        The method computes the Hessian-vector product using finite difference approximation, and the hyper-parameter `_r` is used for scaling the perturbation.
        """

        eta = self._r / torch.cat([x.view(-1) for x in vector]).norm()
        for p, v in zip(self.ll_var, vector):
            p.data.add_(v, alpha=eta)
        if self.gda_loss is not None:
            ll_feed_dict["alpha"] = self.alpha
            loss = self.gda_loss(
                ll_feed_dict, ul_feed_dict, self.ul_model, self.ll_model
            )
        else:
            loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        grads_p = torch.autograd.grad(loss, list(self.ul_var))

        for p, v in zip(self.ll_var, vector):
            p.data.sub_(v, alpha=2 * eta)  # w-
        if self.gda_loss is not None:
            ll_feed_dict["alpha"] = self.alpha
            loss = self.gda_loss(
                ll_feed_dict, ul_feed_dict, self.ul_model, self.ll_model
            )
        else:
            loss = self.ll_objective(ll_feed_dict, self.ul_model, self.ll_model)
        grads_n = torch.autograd.grad(loss, list(self.ul_var))

        for p, v in zip(self.ll_var, vector):
            p.data.add_(v, alpha=eta)  # w

        return [(x - y).div_(2 * eta) for x, y in zip(grads_p, grads_n)]
