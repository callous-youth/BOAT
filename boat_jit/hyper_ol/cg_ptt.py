import jittor as jit
from .hyper_gradient import HyperGradient
from jittor import Module
from typing import List, Callable, Dict
from ..higher_jit.patch import _MonkeyPatchBase
from boat_jit.utils.op_utils import update_tensor_grads, conjugate_gradient


class CG_PTT(HyperGradient):
    """
    Calculation of the hyper gradient of the upper-level variables with Conjugate Gradient (CG) _`[1]` and
    Pessimistic Trajectory Truncation (PTT) _`[2]`.

    Parameters
    ----------
        :param ll_objective: The lower-level objective of the BLO problem.
        :type ll_objective: callable
        :param ul_objective: The upper-level objective of the BLO problem.
        :type ul_objective: callable
        :param ll_model: The lower-level model of the BLO problem.
        :type ll_model: torch.nn.Module
        :param ul_model: The upper-level model of the BLO problem.
        :type ul_model: torch.nn.Module
        :param ll_var: List of variables optimized with the lower-level objective.
        :type ll_var: List
        :param ul_var:  of variables optimized with the upper-level objective.
        :type ul_var: List
        :param solver_config: Dictionary containing solver configurations.
        :type solver_config: dict

    References
    ----------
    _`[1]` A. Rajeswaran, C. Finn, S. M. Kakade, and S. Levine, Meta-learning
     with implicit gradients[C]. in NeurIPS, 2019.
    _`[2]` Liu R, Liu Y, Zeng S, et al. Towards gradient-based bilevel optimization
     with non-convex followers and beyond[C]. In NeurIPS, 2021.
    """

    def __init__(
            self,
            ll_objective: Callable,
            ul_objective: Callable,
            ll_model: Module,
            ul_model: Module,
            ll_var:List,
            ul_var:List,
            solver_config : Dict
    ):
        super(CG_PTT, self).__init__(ul_objective, ul_model, ll_model,ll_var,ul_var)

        self.truncate_max_loss_iter = "PTT" in solver_config["hyper_op"]
        self.dynamic_initialization = "DI" in solver_config['dynamic_op']
        self.ll_lr = solver_config['ll_opt'].defaults["lr"]
        self.ll_objective = ll_objective
        self.tolerance = solver_config["CG"]["tolerance"]
        self.K = solver_config["CG"]["k"]
        self.alpha = solver_config['GDA']["alpha_init"]
        self.alpha_decay = solver_config['GDA']["alpha_decay"]
        self.gda_loss = solver_config['gda_loss']

    def compute_gradients(
            self,
            ll_feed_dict: Dict,
            ul_feed_dict: Dict,
            auxiliary_model: _MonkeyPatchBase,
            max_loss_iter: int = 0
    ):
        """
        Compute the hyper-gradients of the upper-level variables with the data from feed_dict and patched models.

        :param ll_feed_dict: Dictionary containing the lower-level data used for optimization.
            It typically includes training data, targets, and other information required to compute the LL objective.
        :type ll_feed_dict: Dict

        :param ul_feed_dict: Dictionary containing the upper-level data used for optimization.
            It typically includes validation data, targets, and other information required to compute the UL objective.
        :type ul_feed_dict: Dict

        :param auxiliary_model: A patched lower model wrapped by the `higher` library.
            It serves as the lower-level model for optimization.
        :type auxiliary_model: _MonkeyPatchBase

        :param max_loss_iter: The number of iteration used for backpropagation.
        :type max_loss_iter: int

        :returns: the current upper-level objective
        """

        hparams = list(self.ul_var)

        def fp_map(params, loss_f):
            lower_grads = list(jit.grad(loss_f, params))
            updated_params = []
            for i in range(len(params)):
                updated_params.append(params[i] - self.ll_lr * lower_grads[i])
            return updated_params

        assert self.truncate_max_loss_iter and (
                    max_loss_iter >= 0), "With PTT operation, 'max_loss_iter' should be greater than 0"
        lower_model_params = list(
            auxiliary_model.parameters(time=max_loss_iter))

        if self.gda_loss is not None:
            ll_feed_dict['alpha'] = self.alpha * self.alpha_decay ** max_loss_iter
            lower_loss = self.gda_loss(ll_feed_dict, ul_feed_dict, self.ul_model, auxiliary_model,
                                       params=lower_model_params)
        else:
            lower_loss = self.ll_objective(ll_feed_dict, self.ul_model, auxiliary_model,
                                           params=lower_model_params)
        upper_loss = self.ul_objective(ul_feed_dict, self.ul_model, auxiliary_model,
                                       params=lower_model_params)

        if self.dynamic_initialization:
            grads_lower = jit.grad(upper_loss, list(auxiliary_model.parameters(time=0)),retain_graph=True)
            update_tensor_grads(self.ll_var, grads_lower)

        upper_grads = conjugate_gradient(lower_model_params, hparams, upper_loss, lower_loss, self.K, fp_map, self.tolerance)

        update_tensor_grads(self.ul_var,upper_grads)

        return upper_loss


