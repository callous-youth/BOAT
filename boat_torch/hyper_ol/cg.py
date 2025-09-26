import torch
from torch.nn import Module
from typing import List, Callable, Dict
from higher.patch import _MonkeyPatchBase
from boat_torch.utils.op_utils import update_tensor_grads, conjugate_gradient

from boat_torch.operation_registry import register_class
from boat_torch.hyper_ol.hyper_gradient import HyperGradient


@register_class
class CG(HyperGradient):
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
        super(CG, self).__init__(
            ll_objective,
            ul_objective,
            ul_model,
            ll_model,
            ll_var,
            ul_var,
            solver_config,
        )

        self.dynamic_initialization = "DI" in solver_config["dynamic_op"]
        self.ll_lr = solver_config["lower_level_opt"].defaults["lr"]
        self.tolerance = solver_config["CG"]["tolerance"]
        self.K = solver_config["CG"]["k"]
        self.alpha = solver_config["GDA"]["alpha_init"]
        self.alpha_decay = solver_config["GDA"]["alpha_decay"]
        self.gda_loss = (
            solver_config.get("gda_loss", None)
            if "GDA" in solver_config["dynamic_op"]
            else None
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
            The number of iterations used for backpropagation. Default is 0.

        hyper_gradient_finished : bool, optional
            A flag indicating whether the hyper-gradient computation is finished. Default is False.

        next_operation : str, optional
            The next operator for the calculation of the hypergradient. Default is None.

        **kwargs : dict
            Additional arguments, such as:
            - `lower_model_params` (list): Parameters of the lower-level model (default: `list(auxiliary_model.parameters())`).
            - `hparams` (list): Hyper-parameters of the upper-level model (default: `list(self.ul_var)`).

        Returns
        -------
        dict
            A dictionary containing:
            - "upper_loss": The current upper-level objective value.
            - "hyper_gradient_finished": A boolean indicating that the hyper-gradient computation is complete.

        Raises
        ------
        AssertionError
            If `hyper_gradient_finished` is True, as CG does not support multiple hyper-gradient computations.
        """

        assert (
            not hyper_gradient_finished
        ), "CG does not support multiple hypergradient computation"
        lower_model_params = kwargs.get(
            "lower_model_params", list(auxiliary_model.parameters())
        )
        hparams = kwargs.get("hparams", list(self.ul_var))
        import time

        starttime = time.time()

        def fp_map(params, loss_f):
            lower_grads = torch.autograd.grad(loss_f, params, create_graph=True)
            updated_params = []
            for i in range(len(params)):
                updated_params.append(params[i] - self.ll_lr * lower_grads[i])
            return updated_params

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
        #print("step 1 time:", time.time() - starttime)
        starttime = time.time()
        if self.dynamic_initialization:
            grads_lower = torch.autograd.grad(
                upper_loss, list(auxiliary_model.parameters(time=0)), retain_graph=True
            )
            update_tensor_grads(self.ll_var, grads_lower)
        upper_grads = conjugate_gradient(
            lower_model_params,
            hparams,
            upper_loss,
            lower_loss,
            self.K,
            fp_map,
            self.tolerance,
        )
        #print("step 6 time:", time.time() - starttime)
        update_tensor_grads(self.ul_var, upper_grads)

        return {"upper_loss": upper_loss.item(), "hyper_gradient_finished": True}
