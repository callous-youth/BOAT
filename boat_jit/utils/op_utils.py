import jittor as jit
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

        Parameters
        ----------
        name : str
            The name of the result (e.g., 'gradient_operator_results_0').
        result : Dict
            The result dictionary to store.
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
    Compute L2 regularization term (Jittor version).

    Parameters
    ----------
    parameters : List[jt.Var]
        need to compute L2 regularization parameter list

    Returns
    -------
    jt.Var
        L2 regularization loss
    """
    loss = jit.zeros(1)
    for w in parameters:
        loss += (w * w).sum()
    return loss


def grad_unused_zero(output, inputs, retain_graph=False):
    """
    Compute gradients for inputs with respect to the output, filling missing gradients with zeros.

    Parameters
    ----------
    output : jittor.Var
        The output tensor to compute gradients for.
    inputs : List[jittor.Var]
        The input tensors to compute gradients with respect to.
    retain_graph : bool, optional
        Whether to retain the computation graph, by default False.

    Returns
    -------
    Tuple[jittor.Var]
        Gradients with respect to the inputs, with zeros for unused gradients.
    """
    grads = jit.grad(output, inputs, retain_graph=retain_graph)

    def grad_or_zeros(grad, var):
        return jit.zeros_like(var) if grad is None or (jit.isnan(grad).any()) else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def list_tensor_matmul(list1, list2):
    """
    Compute the element-wise multiplication and sum of two lists of tensors.

    Parameters
    ----------
    list1 : List[jittor.Var]
        The first list of tensors.
    list2 : List[jittor.Var]
        The second list of tensors.

    Returns
    -------
    jittor.Var
        The resulting scalar from element-wise multiplication and summation.
    """
    out = 0
    for t1, t2 in zip(list1, list2):
        out += (t1 * t2).sum()
    return out


def list_tensor_norm(list_tensor, p=2):
    """
    Compute the p-norm of a list of tensors.

    Parameters
    ----------
    list_tensor : List[jittor.Var]
        The list of tensors to compute the norm for.
    p : float, optional
        The order of the norm, by default 2 (Euclidean norm).

    Returns
    -------
    jittor.Var
        The computed p-norm of the list of tensors.

    Raises
    ------
    ValueError
        If the list of tensors is empty.
    """
    norm = 0
    for t in list_tensor:
        norm += (t.abs() ** p).sum()
    return norm ** (1 / p)


def require_model_grad(model=None):
    """
    Ensure all model parameters require gradients.

    Parameters
    ----------
    model : jittor.Module
        The model to check and update parameters.

    Raises
    ------
    AssertionError
        If the model is not defined.
    """
    assert model is not None, "The module is not defined!"
    for param in model.parameters():
        if param.is_stop_grad():
            param = param.clone()


def update_grads(grads, model):
    """
    Update the custom_grad attribute of the model's parameters.

    Parameters
    ----------
    grads : List[jittor.Var]
        Gradients to be applied to the parameters.
    model : jittor.Module
        Model whose parameters will be updated.
    """
    for p, x in zip(grads, model.parameters()):
        if not hasattr(x, "_custom_grad"):
            x._custom_grad = p.clone()
        else:
            x._custom_grad += p



def manual_update(optimizer, variables):
    """
    Manually update variables using gradients stored in _custom_grad.

    Parameters
    ----------
    optimizer : jittor.optim.Optimizer
        The Jittor optimizer instance.
    variables : List[jittor.Var]
        A list of Jittor variables to be updated.
    """
    variable_ids = {id(var) for var in variables} 
    for group in optimizer.param_groups:
        lr = group.get("lr", optimizer.lr)

        for param in group["params"]:
            if id(param) not in variable_ids:
                continue

            # if param is not in variables, skip
            if not hasattr(param, "_custom_grad"):
                continue  

            grad = param._custom_grad

            # if shapes do not match, skip
            if grad.shape != param.shape:
                continue  

            param -= lr * grad

            param._custom_grad *= 0



def update_tensor_grads(hparams, grads):
    """
    Update gradients for Jittor variables manually.

    Parameters
    ----------
    hparams : List[jittor.Var]
        List of Jittor variables representing the hyperparameters.
    grads : List[jittor.Var]
        List of gradients corresponding to the hyperparameters.

    Raises
    ------
    ValueError
        If a variable is stop_grad and cannot be updated.
    """
    for l, g in zip(hparams, grads):
        if l.is_stop_grad():
            continue
        if not hasattr(l, "_custom_grad"):
            l._custom_grad = g.clone().detach()
        else:
            l._custom_grad += g


def stop_grads(grads):
    """
    Detach and stop gradient computation for a list of gradients.

    Parameters
    ----------
    grads : List[jittor.Var]
        The gradients to process.

    Returns
    -------
    List[jittor.Var]
        Detached gradients with requires_grad set to False.
    """
    return [(grad.detach().stop_grad() if grad is not None else grad) for grad in grads]


def average_grad(model, batch_size):
    """
    Divide the gradients of all model parameters by the batch size.

    Parameters
    ----------
    model : jittor.Module
        The model whose gradients need to be averaged.
    batch_size : int
        The batch size to divide gradients by.
    """
    for param in model.parameters():
        if param.opt_grad() is not None:
            param.opt_grad().update(param.opt_grad() / batch_size)


def stop_model_grad(model=None):
    """
    Stop gradient computation for all parameters in a model.

    Parameters
    ----------
    model : jittor.Module
        The model to stop gradients for.

    Raises
    ------
    AssertionError
        If the model is not defined.
    """
    assert model is not None, "The module is not defined!"
    for param in model.parameters():
        param.stop_grad()


def cat_list_to_tensor(list_tx):
    """
    Concatenate a list of tensors into a single flattened tensor.

    Parameters
    ----------
    list_tx : List[jittor.Var]
        The list of tensors to concatenate.

    Returns
    -------
    jittor.Var
        A single flattened tensor.
    """
    return jit.concat([xx.flatten() for xx in list_tx])


def copy_parameter_from_list(y, z):
    """
    Copy parameters from a list to the parameters of a Jittor model.

    Parameters
    ----------
    y : jittor.Module
        Jittor model with parameters to be updated.
    z : List[jittor.Var]
        List of variables to copy from.

    Returns
    -------
    jittor.Module
        Updated model.
    """
    for p, q in zip(y.parameters(), z):
        p.update(q.clone().detach())
        p.requires_grad = True
    return y


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    """
    Compute gradients of the outer-level loss with respect to parameters and hyperparameters.

    Parameters
    ----------
    outer_loss : jittor.Var
        The scalar loss from the outer-level optimization problem.
    params : List[jittor.Var]
        The list of parameters for which gradients with respect to the outer loss are computed.
    hparams : List[jittor.Var]
        The list of hyperparameters for which gradients with respect to the outer loss are computed.
    retain_graph : bool, optional
        Whether to retain the computation graph after computing the gradients, by default True.

    Returns
    -------
    Tuple[List[jittor.Var], List[jittor.Var]]
        Gradients with respect to parameters and hyperparameters.
    """
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(
        outer_loss, hparams, retain_graph=retain_graph
    )
    return grad_outer_w, grad_outer_hparams


def custom_grad(outputs, inputs, grad_outputs=None, retain_graph=False):
    """
    Compute the vector-Jacobian product for Jittor, mimicking PyTorch's autograd.grad.

    Parameters
    ----------
    outputs : Sequence[jittor.Var]
        Outputs of the differentiated function.
    inputs : Sequence[jittor.Var]
        Inputs with respect to which the gradient will be computed.
    grad_outputs : Sequence[jittor.Var], optional
        Gradients with respect to the outputs, by default None.
    retain_graph : bool, optional
        Whether to retain the computation graph after computing the gradients, by default False.

    Returns
    -------
    List[jittor.Var]
        Gradients with respect to the inputs.
    """
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    if grad_outputs is None:
        grad_outputs = [jit.ones_like(output) for output in outputs]
    elif not isinstance(grad_outputs, (tuple, list)):
        grad_outputs = (grad_outputs,)

    assert len(outputs) == len(
        grad_outputs
    ), "outputs and grad_outputs must have the same length."

    total_output = sum(
        (output * grad_output).sum()
        for output, grad_output in zip(outputs, grad_outputs)
    )
    grads = jit.grad(total_output, inputs, retain_graph=retain_graph)
    return grads


def neumann(params, hparams, upper_loss, lower_loss, k, fp_map, tol=1e-10):
    """
    Compute hyperparameter gradients using the Neumann series approximation.

    Parameters
    ----------
    params : List[jittor.Var]
        List of parameters for the lower-level optimization problem.
    hparams : List[jittor.Var]
        List of hyperparameters for the upper-level optimization problem.
    upper_loss : jittor.Var
        Loss function for the upper-level problem.
    lower_loss : jittor.Var
        Loss function for the lower-level problem.
    k : int
        Number of iterations for the Neumann series approximation.
    fp_map : Callable
        Fixed-point map function that computes updates to lower-level parameters.
    tol : float, optional
        Tolerance for early stopping based on convergence, by default 1e-10.

    Returns
    -------
    List[jittor.Var]
        Hyperparameter gradients computed using the Neumann series approximation.
    """
    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)
    w_mapped = fp_map(params, lower_loss)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for i in range(k):
        gs_prev_vec = gs_vec
        vs = custom_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(jit.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = custom_grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    return grads


def conjugate_gradient(
    params, hparams, upper_loss, lower_loss, K, fp_map, tol=1e-10, stochastic=False
):
    """
    Compute hyperparameter gradients using the Conjugate Gradient method.

    Parameters
    ----------
    params : List[jittor.Var]
        List of parameters for the lower-level optimization problem.
    hparams : List[jittor.Var]
        List of hyperparameters for the upper-level optimization problem.
    upper_loss : jittor.Var
        Loss function for the upper-level problem.
    lower_loss : jittor.Var
        Loss function for the lower-level problem.
    K : int
        Maximum number of iterations for the Conjugate Gradient method.
    fp_map : Callable
        Fixed-point map function that computes updates to lower-level parameters.
    tol : float, optional
        Tolerance for early stopping based on convergence, by default 1e-10.
    stochastic : bool, optional
        If True, recompute the fixed-point map during each iteration, by default False.

    Returns
    -------
    List[jittor.Var]
        Hyperparameter gradients computed using the Conjugate Gradient method.
    """
    grad_outer_w, grad_outer_hparams = get_outer_gradients(upper_loss, params, hparams)
    if not stochastic:
        w_mapped = fp_map(params, lower_loss)

    def dfp_map_dw(xs):
        if stochastic:
            w_mapped_in = fp_map(params, lower_loss)
            Jfp_mapTv = custom_grad(
                w_mapped_in, params, grad_outputs=xs, retain_graph=False
            )
        else:
            Jfp_mapTv = custom_grad(
                w_mapped, params, grad_outputs=xs, retain_graph=True
            )
        return [v - j for v, j in zip(xs, Jfp_mapTv)]

    vs = cg_step(dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol)
    if stochastic:
        w_mapped = fp_map(params, lower_loss)

    grads = custom_grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    return grads


def cg_step(Ax, b, max_iter=100, epsilon=1.0e-5):
    """
    Perform Conjugate Gradient (CG) optimization to solve Ax = b.

    Parameters
    ----------
    Ax : Callable
        Function that computes the matrix-vector product Ax for a given x.
    b : List[jittor.Var]
        Right-hand side of the equation Ax = b.
    max_iter : int, optional
        Maximum number of iterations for the CG method, by default 100.
    epsilon : float, optional
        Convergence threshold for the residual norm, by default 1e-5.

    Returns
    -------
    List[jittor.Var]
        Solution vector x that approximately solves Ax = b.
    """
    x_last = [jit.zeros_like(bb) for bb in b]
    r_last = [jit.zeros_like(bb) + bb for bb in b]
    p_last = [jit.zeros_like(rr) + rr for rr in r_last]

    for _ in range(max_iter):
        Ap = Ax(p_last)
        Ap_vec = cat_list_to_tensor(Ap)
        p_last_vec = cat_list_to_tensor(p_last)
        r_last_vec = cat_list_to_tensor(r_last)
        rTr = jit.sum(r_last_vec * r_last_vec)
        pAp = jit.sum(p_last_vec * Ap_vec)
        alpha = rTr / pAp

        x = [xx + alpha * pp for xx, pp in zip(x_last, p_last)]
        r = [rr - alpha * pp for rr, pp in zip(r_last, Ap)]
        r_vec = cat_list_to_tensor(r)

        if float(jit.norm(r_vec)) < epsilon:
            break

        beta = jit.sum(r_vec * r_vec) / rTr
        p = [rr + beta * pp for rr, pp in zip(r, p_last)]

        x_last = x
        p_last = p
        r_last = r

    return x_last
