import torch
from torch import Tensor
from typing import List, Callable, Dict


class ResultStore:
    """
    A simple class to store and manage intermediate results of hyper-gradient computation.
    """

    def __init__(self):
        self.results = []

    def add(self, name: str, result: Dict):
        """
        Add a result to the store.

        :param name: The name of the result (e.g., 'gradient_operator_results_0').
        :type name: str
        :param result: The result dictionary to store.
        :type result: Dict
        """
        self.results.append({name: result})

    def clear(self):
        """Clear all stored results."""
        self.results = []

    def get_results(self) -> List[Dict]:
        """Retrieve all stored results."""
        return self.results


class DynamicalSystemRules:
    """
    A class to store and manage gradient operator rules.
    """

    # Default static gradient operator order
    _gradient_order = [
        ["GDA", "DI"],
        ["NGD", "DM"],
    ]

    @staticmethod
    def get_gradient_order() -> List[List[str]]:
        """
        Get the current gradient operator order.

        Returns
        -------
        List[List[str]]
            The current gradient operator order.
        """
        return DynamicalSystemRules._gradient_order

    @staticmethod
    def set_gradient_order(new_order: List[List[str]]):
        """
        Set a new gradient operator order.

        Parameters
        ----------
        new_order : List[List[str]]
            The new gradient operator order to set.

        Raises
        ------
        ValueError
            If the new order is invalid.
        """
        if not isinstance(new_order, list) or not all(
            isinstance(group, list) for group in new_order
        ):
            raise ValueError("Gradient order must be a list of lists.")
        DynamicalSystemRules._gradient_order = new_order


class HyperGradientRules:
    """
    A class to store and manage gradient operator rules.
    """

    # Default static gradient operator order
    _gradient_order = [
        ["PTT", "FOA", "RGT"],
        ["IAD", "RAD", "FD", "IGA"],
        ["CG", "NS"],
    ]

    @staticmethod
    def get_gradient_order() -> List[List[str]]:
        """
        Get the current gradient operator order.

        Returns
        -------
        List[List[str]]
            The current gradient operator order.
        """
        return HyperGradientRules._gradient_order

    @staticmethod
    def set_gradient_order(new_order: List[List[str]]):
        """
        Set a new gradient operator order.

        Parameters
        ----------
        new_order : List[List[str]]
            The new gradient operator order to set.

        Raises
        ------
        ValueError
            If the new order is invalid.
        """
        if not isinstance(new_order, list) or not all(
            isinstance(group, list) for group in new_order
        ):
            raise ValueError("Gradient order must be a list of lists.")
        HyperGradientRules._gradient_order = new_order


def l2_reg(parameters):
    """
    Compute the L2 regularization term for a list of parameters.

    Parameters
    ----------
    parameters : List[torch.Tensor]
        List of tensors for which the L2 regularization term is computed.

    Returns
    -------
    torch.Tensor
        The L2 regularization loss value.
    """
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def grad_unused_zero(
    output, inputs, grad_outputs=None, retain_graph=False, create_graph=False
):
    """
    Compute gradients for the given inputs, substituting zeros for unused gradients.

    Parameters
    ----------
    output : torch.Tensor
        The output tensor for which gradients are computed.

    inputs : List[torch.Tensor]
        List of input tensors with respect to which gradients are computed.

    grad_outputs : torch.Tensor, optional
        Gradient outputs to compute the gradients of the inputs, by default None.

    retain_graph : bool, optional
        If True, the computation graph is retained after the gradient computation,
        by default False.

    create_graph : bool, optional
        If True, constructs the graph for higher-order gradient computations,
        by default False.

    Returns
    -------
    Tuple[torch.Tensor]
        Gradients for the inputs, with unused gradients replaced by zeros.
    """
    grads = torch.autograd.grad(
        output,
        inputs,
        grad_outputs=grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph,
    )

    def grad_or_zeros(grad, var):
        return (
            torch.zeros_like(var) if grad is None or (torch.isnan(grad).any()) else grad
        )

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def list_tensor_matmul(list1, list2):
    """
    Perform element-wise multiplication and summation for two lists of tensors.

    Parameters
    ----------
    list1 : List[torch.Tensor]
        First list of tensors.

    list2 : List[torch.Tensor]
        Second list of tensors.

    Returns
    -------
    torch.Tensor
        Result of the element-wise multiplication and summation.
    """
    out = 0
    for t1, t2 in zip(list1, list2):
        out = out + torch.sum(t1 * t2)
    return out


def list_tensor_norm(list_tensor, p=2):
    """
    Compute the p-norm of a list of tensors.

    Parameters
    ----------
    list_tensor : List[torch.Tensor]
        List of tensors for which the norm is computed.

    p : int, optional
        Order of the norm, by default 2.

    Returns
    -------
    torch.Tensor
        The computed p-norm of the list of tensors.
    """
    norm = 0
    for t in list_tensor:
        norm = norm + torch.norm(t, p)
    return norm


def require_model_grad(model=None):
    """
    Enable gradient computation for all parameters in the given model.

    Parameters
    ----------
    model : torch.nn.Module, optional
        The model whose parameters require gradient computation.

    Raises
    ------
    AssertionError
        If the model is None.
    """
    assert model is not None, "The module is not defined!"
    for param in model.parameters():
        param.requires_grad_(True)


def update_grads(grads, model):
    """
    Update gradients of a model with the given gradients.

    Parameters
    ----------
    grads : List[torch.Tensor]
        Gradients to be applied to the model's parameters.

    model : torch.nn.Module
        Model whose gradients are updated.
    """
    for p, x in zip(grads, model.parameters()):
        if x.grad is None:
            x.grad = p
        else:
            x.grad += p


def update_tensor_grads(hparams, grads):
    """
    Update gradients of hyperparameters with the given gradients.

    Parameters
    ----------
    hparams : List[torch.Tensor]
        Hyperparameters whose gradients are updated.

    grads : List[torch.Tensor]
        Gradients to be applied to the hyperparameters.
    """
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
            #print('first_time')
        # else:
        if g is not None:
            #print("accumulate")
            l.grad += g


def stop_grads(grads):
    """
    Stop gradient computation for the given gradients.

    Parameters
    ----------
    grads : List[torch.Tensor]
        Gradients to be detached from the computation graph.

    Returns
    -------
    List[torch.Tensor]
        Gradients detached from the computation graph.
    """
    return [
        (grad.detach().requires_grad_(False) if grad is not None else grad)
        for grad in grads
    ]


def average_grad(model, batch_size):
    """
    Average gradients over a batch.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose gradients are averaged.

    batch_size : int
        The batch size used for averaging.
    """
    for param in model.parameters():
        param.grad.data = param.grad.data / batch_size


def stop_model_grad(model=None):
    """
    Disable gradient computation for all parameters in the given model.

    Parameters
    ----------
    model : torch.nn.Module, optional
        The model whose parameters no longer require gradient computation.

    Raises
    ------
    AssertionError
        If the model is None.
    """
    assert model is not None, "The module is not defined!"
    for param in model.parameters():
        param.requires_grad_(False)


def copy_parameter_from_list(y, z):
    """
    Copy parameters from a list to a model.

    Parameters
    ----------
    y : torch.nn.Module
        Target model to which parameters are copied.

    z : List[torch.Tensor]
        List of source parameters.

    Returns
    -------
    torch.nn.Module
        The updated model with copied parameters.
    """
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    return y


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    """
    Compute the gradients of the outer-level loss with respect to parameters and hyperparameters.

    Parameters
    ----------
    outer_loss : Tensor
        The outer-level loss.
    params : List[Tensor]
        List of tensors representing parameters.
    hparams : List[Tensor]
        List of tensors representing hyperparameters.
    retain_graph : bool, optional
        Whether to retain the computation graph, by default True.

    Returns
    -------
    Tuple[List[Tensor], List[Tensor]]
        Gradients with respect to parameters and hyperparameters.
    """
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(
        outer_loss, hparams, retain_graph=retain_graph
    )

    return grad_outer_w, grad_outer_hparams

def cat_list_to_tensor(list_tx):
    """
    Concatenate a list of tensors into a single tensor.

    Parameters
    ----------
    list_tx : List[Tensor]
        List of tensors to concatenate.

    Returns
    -------
    Tensor
        The concatenated tensor.
    """
    return torch.cat([xx.view([-1]) for xx in list_tx])

def neumann(
    params: List[Tensor],
    hparams: List[Tensor],
    upper_loss,
    lower_loss,
    k: int,
    fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
    tol=1e-10,
) -> List[Tensor]:
    """
    Compute hypergradients using Neumann series approximation.

    Parameters
    ----------
    params : List[Tensor]
        List of lower-level parameters.
    hparams : List[Tensor]
        List of upper-level hyperparameters.
    upper_loss : Tensor
        The upper-level loss.
    lower_loss : Tensor
        The lower-level loss.
    k : int
        Number of iterations for Neumann approximation.
    fp_map : Callable
        Fixed-point mapping function.
    tol : float, optional
        Tolerance for early stopping, by default 1e-10.

    Returns
    -------
    List[Tensor]
        Hypergradients for the upper-level hyperparameters.
    """
    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)

    w_mapped = fp_map(params, lower_loss)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for i in range(k):
        gs_prev_vec = gs_vec
        vs = torch.autograd.grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(torch.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = torch.autograd.grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    return grads

def conjugate_gradient(
    params: List[Tensor],
    hparams: List[Tensor],
    upper_loss,
    lower_loss,
    K: int,
    fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
    tol=1e-10,
) -> List[Tensor]:
    """
    Compute hypergradients using the conjugate gradient method.

    Parameters
    ----------
    params : List[Tensor]
        List of lower-level parameters.
    hparams : List[Tensor]
        List of upper-level hyperparameters.
    upper_loss : Tensor
        The upper-level loss.
    lower_loss : Tensor
        The lower-level loss.
    K : int
        Maximum number of iterations for the conjugate gradient method.
    fp_map : Callable
        Fixed-point mapping function.
    tol : float, optional
        Tolerance for early stopping, by default 1e-10.

    Returns
    -------
    List[Tensor]
        Hypergradients for the upper-level hyperparameters.
    """
    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)

    w_mapped = fp_map(params, lower_loss)
    def dfp_map_dw(xs):
        Jfp_mapTv = torch.autograd.grad(
            w_mapped, params, grad_outputs=xs, retain_graph=True
        )
        return [v - j for v, j in zip(xs, Jfp_mapTv)]
    vs = cg_step(dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol)
    grads = torch.autograd.grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]

    return grads

def cg_step(Ax, b, max_iter=100, epsilon=1.0e-5):
    """
    Perform the conjugate gradient optimization step.

    Parameters
    ----------
    Ax : Callable
        Function to compute the matrix-vector product.
    b : List[Tensor]
        Right-hand side of the linear system.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    epsilon : float, optional
        Tolerance for early stopping, by default 1.0e-5.

    Returns
    -------
    List[Tensor]
        Solution vector for the linear system.
    """
    x_last = [torch.zeros_like(bb) for bb in b]
    r_last = [torch.zeros_like(bb).copy_(bb) for bb in b]
    p_last = [torch.zeros_like(rr).copy_(rr) for rr in r_last]

    for ii in range(max_iter):
        Ap = Ax(p_last)
        Ap_vec = cat_list_to_tensor(Ap)
        p_last_vec = cat_list_to_tensor(p_last)
        r_last_vec = cat_list_to_tensor(r_last)
        rTr = torch.sum(r_last_vec * r_last_vec)
        pAp = torch.sum(p_last_vec * Ap_vec)
        alpha = rTr / pAp

        x = [xx + alpha * pp for xx, pp in zip(x_last, p_last)]
        r = [rr - alpha * pp for rr, pp in zip(r_last, Ap)]
        r_vec = cat_list_to_tensor(r)

        if float(torch.norm(r_vec)) < epsilon:
            break

        beta = torch.sum(r_vec * r_vec) / rTr
        p = [rr + beta * pp for rr, pp in zip(r, p_last)]

        x_last = x
        p_last = p
        r_last = r

    return x_last
