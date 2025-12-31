from typing import List, Callable
from mindspore import ops
import mindspore as ms


def l2_reg(params):
    """
    Compute the L2 regularization loss.

    Parameters
    ----------
    params : list
        List of model parameters (trainable).

    Returns
    -------
    mindspore.Tensor
        The computed L2 regularization loss.
    """
    loss = 0.0
    for param in params:
        loss += ops.ReduceSum()(ops.Pow()(param, 2))
    return loss


def require_model_grad(model=None):
    """
    Ensure all parameters of a MindSpore model require gradients.

    Parameters
    ----------
    model : mindspore.nn.Cell, optional
        MindSpore model instance. Must not be None.

    Raises
    ------
    AssertionError
        If the model is None.
    """
    assert model is not None, "The module is not defined!"
    for param in model.trainable_params():
        param.requires_grad = True


def update_grads(grads, model):
    """
    Update gradients for a model's parameters.

    Parameters
    ----------
    grads : list
        List of gradients to apply.
    model : mindspore.nn.Cell
        The model whose gradients will be updated.
    """
    for grad, param in zip(grads, model.trainable_params()):
        if param.grad is None:
            param.set_grad(grad)
        else:
            param.grad += grad


def update_tensor_grads(hparams, grads):
    """
    Update gradients for hyperparameters.

    Parameters
    ----------
    hparams : list of mindspore.Tensor
        Hyperparameters to update.
    grads : list of mindspore.Tensor
        Gradients to apply to the hyperparameters.
    """
    for param, grad in zip(hparams, grads):
        if param.grad is None:
            param.set_grad(grad)
        else:
            param.grad += grad


def stop_grads(grads):
    """
    Detach and stop gradient computation for a list of gradients.

    Parameters
    ----------
    grads : list of mindspore.Tensor
        Gradients to process.

    Returns
    -------
    list of mindspore.Tensor
        Detached gradients with requires_grad set to False.
    """
    return [
        (grad.detach().requires_grad_(False) if grad is not None else grad)
        for grad in grads
    ]


def average_grad(model, batch_size):
    """
    Average the gradients of all model parameters by the batch size.

    Parameters
    ----------
    model : mindspore.nn.Cell
        The model whose gradients need to be averaged.
    batch_size : int
        The batch size to divide gradients by.
    """
    for param in model.trainable_params():
        if param.grad is not None:
            param.grad /= batch_size


def stop_model_grad(model=None):
    """
    Stop gradient computation for all parameters in a model.

    Parameters
    ----------
    model : mindspore.nn.Cell, optional
        The model to stop gradients for. Must not be None.

    Raises
    ------
    AssertionError
        If the model is None.
    """
    assert model is not None, "The module is not defined!"
    for param in model.trainable_params():
        param.requires_grad = False


def copy_parameter_from_list(model, param_list):
    """
    Copy parameters from a list to a model's trainable parameters.

    Parameters
    ----------
    model : mindspore.nn.Cell
        The model whose parameters need to be updated.
    param_list : list of mindspore.Tensor
        The list of parameters to copy.
    """
    for param, new_param in zip(model.trainable_params(), param_list):
        param.set_data(new_param)

def grad_unused_zero(output, inputs, grad_outputs=None):
    """
    Compute gradients for the given inputs, substituting zeros for unused gradients.
    (MindSpore version)

    Parameters
    ----------
    output : mindspore.Tensor
        The scalar output tensor for which gradients are computed.

    inputs : List[mindspore.Tensor]
        List of input tensors with respect to which gradients are computed.

    grad_outputs : mindspore.Tensor, optional
        Sensitivity tensor (same shape as output). Default = ones_like(output).

    Returns
    -------
    List[mindspore.Tensor]
        Gradients for the inputs, with unused gradients replaced by zeros.
    """
    if grad_outputs is None:
        grad_outputs = ops.ones_like(output)

    grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)
    grads = grad_fn(lambda *x: output, inputs)(*inputs, grad_outputs)

    safe_grads = []
    for g, v in zip(grads, inputs):
        if g is None:
            safe_grads.append(ms.numpy.zeros_like(v))
        else:
            safe_grads.append(g)
    return safe_grads