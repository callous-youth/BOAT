# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable optimizer wrappers around ``jittor.optim`` instances."""

import abc as _abc
import collections as _collections
import copy as _copy
import math as _math
import typing as _typing
import warnings as _warnings

import jittor as jit

from . import patch as _patch
from . import utils as _utils

_GroupedGradsType = _typing.List[_typing.List[jit.Var]]
_StateType = _typing.List[_typing.DefaultDict[int, _typing.Any]]
_GradClosureType = _typing.Callable[[jit.Var], jit.Var]
_OverrideType = _typing.Dict[str, _typing.List[_typing.Any]]
_GradCallbackType = _typing.Callable[[_typing.List[jit.Var]], _typing.List[jit.Var]]


def _get_mask_closure(mask: jit.Var) -> _GradClosureType:
    def closure(grad: jit.Var) -> jit.Var:
        grad = jit.where(mask, jit.zeros_like(grad), grad)
        if grad.requires_grad:
            grad.register_hook(_get_mask_closure(mask))
        return grad

    return closure


def _maybe_mask(tensor: jit.Var, mask: jit.Var) -> None:
    if tensor.requires_grad:
        tensor.register_hook(_get_mask_closure(mask))


class DifferentiableOptimizer(_abc.ABC):
    def __init__(
        self,
        other: jit.optim.Optimizer,
        reference_params: _typing.Iterable[jit.Var],
        fmodel: _typing.Optional[_patch._MonkeyPatchBase] = None,
        device: _typing.Optional[str] = None,
        override: _typing.Optional[_OverrideType] = None,
        grad_callback: _typing.Optional[_GradCallbackType] = None,
        track_higher_grads: bool = True,
        **kwargs,
    ) -> None:
        r"""Initialize the optimizer with the state of an existing optimizer.

        Args:
            other: an existing optimizer instance.
            reference_params: an iterable over the parameters of the original
                model.
            fmodel (optional): a patched stateless module with a view on
                weights.
            device (optional): the device to cast state tensors to.
            override (optional): a dictionary mapping optimizer settings (i.e.
                those which would be passed to the optimizer constructor or
                provided within parameter groups) to either singleton lists of
                override values, or to a list of override values of length equal
                to the number of parameter groups. If a single override is
                provided for a keyword, it is used for all parameter groups. If
                a list is provided, the ``i``\ th element of the list overrides the
                corresponding setting in the ``i``\ th parameter group. This permits
                the passing of tensors requiring gradient to differentiable
                optimizers for use as optimizer settings.
            grad_callback: (optional) a single argument function which will be
                applied to a list of gradients of parameters, which respects the
                order specified by ``reference_params``. This can be used to
                apply a function, such as gradient clipping, to all (or a
                subset) of these gradients every time the step function is
                called. If this keyword argument is provided when calling the
                step method, its value will override the default specified here.
            track_higher_grads: if True, during unrolled optimization the graph
                be retained, and the fast weights will bear grad funcs, so as to
                permit backpropagation through the optimization process. Setting
                this to False allows the differentiable optimizer to be used in
                "test mode", without potentially tracking higher order
                gradients. This can be useful when running the training loop at
                test time, e.g. in k-shot learning experiments, without
                incurring a significant memory overhead.
        """
        reference_params = list(reference_params)

        # Copy param groups and set up structures for copy state
        self.lr = other.lr
        self.param_groups = _copy.deepcopy(other.param_groups)
        self._group_to_param_list: _typing.List[_typing.List[int]] = []

        self.state: _typing.List[_typing.Dict[int, _typing.Dict[str, _typing.Any]]] = [
            _collections.defaultdict(dict) for _ in range(len(self.param_groups))
        ]

        # Deal with override
        if override is not None:
            self._apply_override(override)

        self._grad_callback = grad_callback

        # Initialize state
        zipped = zip(self.param_groups, other.param_groups)
        for group_idx, (group, orig_group) in enumerate(zipped):
            local_list = []
            for param_idx, param in enumerate(orig_group["params"]):
                param_id = id(param)  # Use the unique ID of the parameter
                if param_id in other._grad_map:
                    # Initialize state for the parameter
                    self.state[group_idx][param_idx] = {
                        k: _utils._recursive_copy_and_cast(v, device)
                        for k, v in other._grad_map[param_id].items()
                    }
                index = _utils._find_param_in_list(param, reference_params)
                if index is None:
                    raise ValueError(
                        f"Could not find parameter {param} in reference parameters."
                    )
                local_list.append(index)
            self._group_to_param_list.append(local_list)

        self._fmodel = fmodel
        self._track_higher_grads = track_higher_grads

    def _apply_override(self, override: _OverrideType) -> None:
        for k, v in override.items():
            # Sanity check
            if (len(v) != 1) and (len(v) != len(self.param_groups)):
                raise ValueError(
                    "Mismatch between the number of override tensors for "
                    "optimizer parameter {} and the number of "
                    "parameter groups.".format(k)
                )
            for group_idx, group in enumerate(self.param_groups):
                group[k] = v[0] if len(v) == 1 else v[group_idx]

    def step(
        self,
        loss: jit.Var,
        params: _typing.Iterable[jit.Var] = None,
        override: _typing.Optional[_OverrideType] = None,
        grad_callback: _typing.Optional[_GradCallbackType] = None,
        **kwargs,
    ) -> _typing.Iterable[jit.Var]:
        r"""Perform a model update.

        This would be used by replacing the normal sequence::

            opt.zero_grad()
            loss.backward()
            opt.step()

        with::

            diffopt.step(loss)


        Args:
            loss: the loss tensor.
            params (optional): the parameters with regard to which we measure
                the loss. These must be provided if the differentiable optimizer
                did not receive a patched model with a view over its own fast
                weights at initialisation. If there is such a model, and params
                are provided, they will overwrite the params of the encapsulated
                model.
            override (optional): a dictionary mapping optimizer settings (i.e.
                those which would be passed to the optimizer constructor or
                provided within parameter groups) to either singleton lists of
                override values, or to a list of override values of length equal
                to the number of parameter groups. If a single override is
                provided for a keyword, it is used for all parameter groups. If
                a list is provided, the ``i``\ th element of the list overrides
                the corresponding setting in the ``i``\ th parameter group. This
                permits the passing of tensors requiring gradient to
                differentiable optimizers for use as optimizer settings. Setting
                override here has highest precedence, i.e. it will override any
                tensors provided as override during the creation of the
                differentiable optimizer, where there is name clash.
            grad_callback: (optional) a single argument function which will be
                applied to a list of gradients of parameters, which respects the
                order specified by ``reference_params``. This can be used to
                apply a function, such as gradient clipping, to all (or a
                subset) of these gradients every time the step function is
                called. This callback overrides the default provided when
                constructing the differentiable optimizer.


        Returns:
            The updated parameters, which will individually have ``grad_fn``\ s
            of their own. If the optimizer has an encapsulated patched model,
            its view over its own fast weights will be updated with these
            params.
        """

        # Deal with override
        if override is not None:
            self._apply_override(override)

        if self._fmodel is None or self._fmodel.fast_params is None:
            if params is None:
                raise ValueError(
                    "params kwarg must be passed to step if the differentiable "
                    "optimizer doesn't have a view on a patched model with "
                    "params."
                )
        else:
            params = self._fmodel.fast_params if params is None else params

        params = list(params)

        grad_targets = [
            p if not p.is_stop_grad() else jit.zeros_like(p).stop_grad() for p in params
        ]

        all_grads = jit.grad(
            loss,
            grad_targets,
            # retain_graph=True,  # Jittor does not have allow_unused, retain_graph used here
        )

        if grad_callback is not None:
            all_grads = grad_callback(all_grads)
        elif self._grad_callback is not None:
            all_grads = self._grad_callback(all_grads)

        grouped_grads = []
        for group, mapping in zip(self.param_groups, self._group_to_param_list):
            grads = []
            for i, index in enumerate(mapping):
                group["params"][i] = params[index]
                grads.append(all_grads[index])
            grouped_grads.append(grads)

        self._update(grouped_grads)

        new_params = params[:]
        for group, mapping in zip(self.param_groups, self._group_to_param_list):
            for p, index in zip(group["params"], mapping):
                if self._track_higher_grads:
                    new_params[index] = p
                else:
                    # new_params[index] = p.detach().requires_grad_()
                    new_params[index] = p.detach()
                    new_params[index].start_grad()
        if self._fmodel is not None:
            self._fmodel.update_params(new_params)

        return new_params

    @_abc.abstractmethod
    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        pass


class DifferentiableSGD(DifferentiableOptimizer):
    r"""A differentiable version of the SGD optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            momentum = group.get("momentum", 0.0) 
            weight_decay = group.get("weight_decay", 0.0)
            dampening = group.get("dampening", 0.0)
            nesterov = group.get("nesterov", False)

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                # if weight decay is specified, add it to the gradient
                if weight_decay != 0:
                    g += weight_decay * p

                # use momentum if specified
                param_state = self.state[group_idx].get(p_idx, {})
                if momentum != 0:
                    # initialize momentum buffer if it does not exist
                    if "momentum_buffer" not in param_state:
                        param_state["momentum_buffer"] = jit.zeros_like(p).stop_grad()

                    buf = param_state["momentum_buffer"]
                    buf *= momentum
                    buf += g * (1 - dampening)

                    if nesterov:
                        g += momentum * buf
                    else:
                        g = buf

                    self.state[group_idx][p_idx] = param_state

                p -= self.lr * g


class DifferentiableAdam(DifferentiableOptimizer):
    r"""A differentiable version of the Adam optimizer for Jittor.

    This optimizer creates a gradient tape as it updates parameters.
    """

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            amsgrad = group.get("amsgrad", False)
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            weight_decay = group.get("weight_decay", 0.0)
            eps = group.get("eps", 1e-8)

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                # State initialization
                param_state = self.state[group_idx].get(p_idx, {})
                if not param_state:
                    param_state["step"] = 0
                    param_state["exp_avg"] = jit.zeros_like(p).stop_grad()
                    param_state["exp_avg_sq"] = jit.zeros_like(p).stop_grad()
                    if amsgrad:
                        param_state["max_exp_avg_sq"] = jit.zeros_like(p).stop_grad()

                exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = param_state["max_exp_avg_sq"]

                # Update step count
                param_state["step"] += 1
                step = param_state["step"]

                # Apply weight decay
                if weight_decay != 0:
                    g += weight_decay * p

                # Decay the first and second moment running average coefficient
                exp_avg *= beta1
                exp_avg += (1 - beta1) * g

                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (g * g)

                # Bias correction terms
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                if amsgrad:
                    # Maintain the max of all 2nd moment running avg till now
                    max_exp_avg_sq = jit.maximum(max_exp_avg_sq, exp_avg_sq)
                    param_state["max_exp_avg_sq"] = max_exp_avg_sq
                    denom = (max_exp_avg_sq.sqrt() / jit.sqrt(bias_correction2)) + eps
                else:
                    denom = (exp_avg_sq.sqrt() / jit.sqrt(bias_correction2)) + eps

                step_size = group["lr"] / bias_correction1

                # Update parameters
                p -= step_size * (exp_avg / denom)

                # Save updated state
                self.state[group_idx][p_idx] = param_state


class DifferentiableAdamW(DifferentiableOptimizer):
    r"""A differentiable version of the AdamW optimizer for Jittor.

    This optimizer creates a gradient tape as it updates parameters.
    """

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            amsgrad = group.get("amsgrad", False)
            beta1, beta2 = group.get("betas", (0.9, 0.999))
            weight_decay = group.get("weight_decay", 0.01)  # Typical default for AdamW
            eps = group.get("eps", 1e-8)

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                # State initialization
                param_state = self.state[group_idx].get(p_idx, {})
                if not param_state:
                    param_state["step"] = 0
                    param_state["exp_avg"] = jit.zeros_like(p).stop_grad()
                    param_state["exp_avg_sq"] = jit.zeros_like(p).stop_grad()
                    if amsgrad:
                        param_state["max_exp_avg_sq"] = jit.zeros_like(p).stop_grad()

                exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = param_state["max_exp_avg_sq"]

                # Apply weight decay directly on the parameter (AdamW specific)
                p *= 1 - group["lr"] * weight_decay

                # Update step count
                param_state["step"] += 1
                step = param_state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg *= beta1
                exp_avg += (1 - beta1) * g

                exp_avg_sq *= beta2
                exp_avg_sq += (1 - beta2) * (g * g)

                # Bias correction terms
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                if amsgrad:
                    # Maintain the max of all 2nd moment running avg till now
                    max_exp_avg_sq = jit.maximum(max_exp_avg_sq, exp_avg_sq)
                    param_state["max_exp_avg_sq"] = max_exp_avg_sq
                    denom = (max_exp_avg_sq.sqrt() / jit.sqrt(bias_correction2)) + eps
                else:
                    denom = (exp_avg_sq.sqrt() / jit.sqrt(bias_correction2)) + eps

                step_size = group["lr"] / bias_correction1

                # Update parameters
                p -= step_size * (exp_avg / denom)

                # Save updated state
                self.state[group_idx][p_idx] = param_state


class DifferentiableAdadelta(DifferentiableOptimizer):
    r"""A differentiable version of the Adadelta optimizer for Jittor.

    This optimizer creates a gradient tape as it updates parameters.
    """

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            rho = group.get("rho", 0.9)
            eps = group.get("eps", 1e-6)
            weight_decay = group.get("weight_decay", 0.0)

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                # State initialization
                param_state = self.state[group_idx].get(p_idx, {})
                if not param_state:
                    param_state["step"] = 0
                    param_state["square_avg"] = jit.zeros_like(p).stop_grad()
                    param_state["acc_delta"] = jit.zeros_like(p).stop_grad()

                square_avg = param_state["square_avg"]
                acc_delta = param_state["acc_delta"]

                # Update step count
                param_state["step"] += 1

                # Apply weight decay if needed
                if weight_decay != 0:
                    g += weight_decay * p

                # Update square_avg
                square_avg *= rho
                square_avg += (1 - rho) * (g * g)
                param_state["square_avg"] = square_avg

                # Compute update step
                std = (square_avg + eps).sqrt()
                delta = (acc_delta + eps).sqrt() / std * g

                # Update acc_delta
                acc_delta *= rho
                acc_delta += (1 - rho) * (delta * delta)
                param_state["acc_delta"] = acc_delta

                # Update parameter
                p -= group["lr"] * delta

                # Save updated state
                self.state[group_idx][p_idx] = param_state


class DifferentiableAdagrad(DifferentiableOptimizer):
    r"""A differentiable version of the Adagrad optimizer for Jittor.

    This optimizer creates a gradient tape as it updates parameters.
    """

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            lr = group.get("lr", 1e-2)
            lr_decay = group.get("lr_decay", 0.0)
            weight_decay = group.get("weight_decay", 0.0)
            eps = group.get("eps", 1e-10)

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                # State initialization
                param_state = self.state[group_idx].get(p_idx, {})
                if not param_state:
                    param_state["step"] = 0
                    param_state["sum"] = jit.zeros_like(p).stop_grad()

                sum_ = param_state["sum"]

                # Update step count
                param_state["step"] += 1
                step = param_state["step"]

                # Apply weight decay if needed
                if weight_decay != 0:
                    g += weight_decay * p

                # Compute adjusted learning rate
                clr = lr / (1 + (step - 1) * lr_decay)

                # Update sum of squared gradients
                sum_ += g * g
                param_state["sum"] = sum_

                # Compute parameter update
                std = sum_.sqrt() + eps
                p -= clr * g / std

                # Save updated state
                self.state[group_idx][p_idx] = param_state


class DifferentiableAdamax(DifferentiableOptimizer):
    r"""A differentiable version of the Adamax optimizer for Jittor.

    This optimizer creates a gradient tape as it updates parameters.
    """

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            lr = group.get("lr", 2e-3)
            betas = group.get("betas", (0.9, 0.999))
            eps = group.get("eps", 1e-8)
            weight_decay = group.get("weight_decay", 0.0)

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                # State initialization
                param_state = self.state[group_idx].get(p_idx, {})
                if not param_state:
                    param_state["step"] = 0
                    param_state["exp_avg"] = jit.zeros_like(p).stop_grad()
                    param_state["exp_inf"] = jit.zeros_like(p).stop_grad()

                exp_avg = param_state["exp_avg"]
                exp_inf = param_state["exp_inf"]
                beta1, beta2 = betas

                # Update step count
                param_state["step"] += 1
                step = param_state["step"]

                # Apply weight decay
                if weight_decay != 0:
                    g += weight_decay * p

                # Update biased first moment estimate
                exp_avg = exp_avg * beta1 + (1 - beta1) * g
                param_state["exp_avg"] = exp_avg

                # Update the exponentially weighted infinity norm
                exp_inf = jit.maximum(exp_inf * beta2, g.abs())
                param_state["exp_inf"] = exp_inf

                # Bias correction
                bias_correction = 1 - beta1**step
                clr = lr / bias_correction

                # Parameter update
                p -= clr * exp_avg / (exp_inf + eps)

                # Save updated state
                self.state[group_idx][p_idx] = param_state


class DifferentiableASGD(DifferentiableOptimizer):
    r"""A differentiable version of the ASGD optimizer for Jittor.

    This optimizer creates a gradient tape as it updates parameters.
    """

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            lr = group.get("lr", 1e-2)
            lambd = group.get("lambd", 1e-4)
            alpha = group.get("alpha", 0.75)
            t0 = group.get("t0", 1e6)
            weight_decay = group.get("weight_decay", 0.0)

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                # State initialization
                param_state = self.state[group_idx].get(p_idx, {})
                if not param_state:
                    param_state["step"] = 0
                    param_state["eta"] = lr
                    param_state["mu"] = 1
                    param_state["ax"] = jit.zeros_like(p).stop_grad()

                eta = param_state["eta"]
                mu = param_state["mu"]
                ax = param_state["ax"]

                # Update step count
                param_state["step"] += 1
                step = param_state["step"]

                # Apply weight decay
                if weight_decay != 0:
                    g += weight_decay * p

                # Decay term
                p *= 1 - lambd * eta

                # Update parameter
                p -= eta * g

                # Averaging
                if mu != 1:
                    ax += (p - ax) * mu
                else:
                    ax = p

                # Update eta and mu
                param_state["eta"] = lr / ((1 + lambd * lr * step) ** alpha)
                param_state["mu"] = 1 / max(1, step - t0)

                # Save updated parameter and state
                group["params"][p_idx] = p
                param_state["ax"] = ax
                self.state[group_idx][p_idx] = param_state


class DifferentiableRMSprop(DifferentiableOptimizer):
    r"""A differentiable version of the RMSprop optimizer for Jittor.

    This optimizer creates a gradient tape as it updates parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warnings.warn(
            "Differentiable RMSprop may suffer from gradient correctness issues. "
            "Consider verifying behavior for specific use cases."
        )

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            lr = group.get("lr", 1e-2)
            alpha = group.get("alpha", 0.99)
            eps = group.get("eps", 1e-8)
            weight_decay = group.get("weight_decay", 0.0)
            momentum = group.get("momentum", 0.0)
            centered = group.get("centered", False)

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                # State initialization
                param_state = self.state[group_idx].get(p_idx, {})
                if not param_state:
                    param_state["step"] = 0
                    param_state["square_avg"] = jit.zeros_like(p).stop_grad()
                    if momentum > 0:
                        param_state["momentum_buffer"] = jit.zeros_like(p).stop_grad()
                    if centered:
                        param_state["grad_avg"] = jit.zeros_like(p).stop_grad()

                square_avg = param_state["square_avg"]
                param_state["step"] += 1

                # Apply weight decay
                if weight_decay != 0:
                    g += weight_decay * p

                # Update running average of squared gradients
                square_avg = alpha * square_avg + (1 - alpha) * g * g
                param_state["square_avg"] = square_avg

                # Prevent NaNs for zero values
                if jit.any(square_avg == 0):
                    square_avg += eps

                # Calculate centered RMSprop if applicable
                if centered:
                    grad_avg = param_state["grad_avg"]
                    grad_avg = alpha * grad_avg + (1 - alpha) * g
                    param_state["grad_avg"] = grad_avg
                    avg = (square_avg - grad_avg * grad_avg).sqrt() + eps
                else:
                    avg = square_avg.sqrt() + eps

                # Momentum updates if enabled
                if momentum > 0:
                    buf = param_state.get(
                        "momentum_buffer", jit.zeros_like(p).stop_grad()
                    )
                    buf = momentum * buf + g / avg
                    param_state["momentum_buffer"] = buf
                    p -= lr * buf
                else:
                    p -= lr * g / avg

                # Save updated parameter and state
                group["params"][p_idx] = p
                self.state[group_idx][p_idx] = param_state


class DifferentiableRprop(DifferentiableOptimizer):
    r"""A differentiable version of the Rprop optimizer for Jittor.

    This optimizer creates a gradient tape as it updates parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warnings.warn(
            "Differentiable Rprop correctly yields zero second-order gradients, "
            "as only the sign of the gradient is used in updates. Future versions "
            "may include higher-order gradient approximations."
        )

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)

        for group_idx, (group, grads) in enumerate(zipped):
            etaminus, etaplus = group.get("etas", (0.5, 1.2))
            step_size_min, step_size_max = group.get("step_sizes", (1e-6, 50))

            for p_idx, (p, g) in enumerate(zip(group["params"], grads)):
                if g is None or p.is_stop_grad():
                    continue

                if g.is_sparse:
                    raise RuntimeError("Rprop does not support sparse gradients")

                # State initialization
                param_state = self.state[group_idx].get(p_idx, {})
                if not param_state:
                    param_state["step"] = 0
                    param_state["prev"] = jit.zeros_like(p).stop_grad()
                    param_state["step_size"] = jit.full_like(
                        p, group.get("lr", 1e-2)
                    ).stop_grad()

                prev_grad = param_state["prev"]
                step_size = param_state["step_size"]

                param_state["step"] += 1

                # Determine gradient sign
                sign = (g * prev_grad).sign()
                sign_positive = sign > 0
                sign_negative = sign < 0

                # Update step sizes
                step_size = jit.where(sign_positive, step_size * etaplus, step_size)
                step_size = jit.where(sign_negative, step_size * etaminus, step_size)
                step_size = step_size.clamp(step_size_min, step_size_max)
                param_state["step_size"] = step_size

                # Zero out gradient where sign is negative
                g = jit.where(sign_negative, jit.zeros_like(g), g)

                # Update parameters
                p -= g.sign() * step_size

                # Save state
                param_state["prev"] = g.clone().stop_grad()
                self.state[group_idx][p_idx] = param_state


_OptMappingType = _typing.Dict[
    jit.optim.Optimizer, _typing.Type[DifferentiableOptimizer]
]

# get available optimizers in jittor.optim
available_optimizers = {attr for attr in dir(jit.optim) if not attr.startswith("_")}

# define optimizer mapping
_opt_mapping: _OptMappingType = {
    "Adadelta": DifferentiableAdadelta,
    "Adagrad": DifferentiableAdagrad,
    "Adam": DifferentiableAdam,
    "AdamW": DifferentiableAdamW,
    "Adamax": DifferentiableAdamax,
    "ASGD": DifferentiableASGD,
    "RMSprop": DifferentiableRMSprop,
    "Rprop": DifferentiableRprop,
    "SGD": DifferentiableSGD,
}

# remove entries for optimizers not available in jittor.optim
_opt_mapping = {
    getattr(jit.optim, k): v
    for k, v in _opt_mapping.items()
    if k in available_optimizers
}


def get_diff_optim(
    opt: jit.optim.Optimizer,
    reference_params: _typing.Iterable[jit.Var],
    fmodel: _typing.Optional[_patch._MonkeyPatchBase] = None,
    device: _typing.Optional[str] = None,
    override: _typing.Optional[_OverrideType] = None,
    track_higher_grads: bool = True,
    **kwargs,
) -> DifferentiableOptimizer:
    r"""Construct/initialize a differentiable version of an existing optimizer.

    Args:
        opt: an existing optimizer, assumed to be an instance of
            ``jittor.optim.Optimizer``, of a supported type which is either defined
            in ``jittor.optim``, or a custom implementation which has been added to
            higher at runtime by using ``higher.register_optim``. We assume this
            optimizer tracks the parameters (or some subset thereof) of a single
            ``jittor.Module`` instance, with support for parameter groups.
        reference_params: the parameters of the module tracked by ``opt``, as
            returned by ``module.parameters()``.
        fmodel (optional): a patched version of the ``module`` tracked by ``opt``.
            It is assumed this patched instance has a view on its latest fast
            weights through ``fmodel.parameters()``. If provided, it is not
            necessary to pass the fast weights explicitly to the differentiable
            optimizer's ``step`` function via the keyword arg ``params``. If not
            provided, the fast weights to update must be provided to ``step``.
        device (optional): the device to cast the optimizer state to when
            creating the differentiable optimizer. If not provided, the same
            device as used for the parameters tracked by ``opt`` will be used.
        override (optional): a dictionary mapping optimizer settings (i.e.
            those which would be passed to the optimizer constructor or
            provided within parameter groups) to either singleton lists of
            override values, or to a list of override values of length equal to
            the number of parameter groups. If a single override is provided for
            a keyword, it is used for all parameter groups. If a list is
            provided, the ``i``\ th element of the list overrides the corresponding
            setting in the ``i``\ th parameter group. This permits the passing of
            tensors requiring gradient to differentiable optimizers for use as
            optimizer settings.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows the returned differentiable optimizer to be used in
            "test mode", without potentially tracking higher order gradients.
            This can be useful when running the training loop at test time,
            e.g. in k-shot learning experiments, without incurring a significant
            memory overhead.

    Returns:
        An initialized ``DifferentiableOptimizer`` instance of the right subtype.
    """
    if type(opt) in _opt_mapping:
        return _opt_mapping[type(opt)](
            opt,
            reference_params,
            fmodel=fmodel,
            device=device,
            override=override,
            track_higher_grads=track_higher_grads,
            **kwargs,
        )
    else:
        raise ValueError(
            "Optimizer type {} not supported by higher yet.".format(type(opt))
        )


def create_diff_optim(
    opt_type: _typing.Type[jit.optim.Optimizer],
    opt_kwargs: _typing.Optional[_typing.Dict[str, _typing.Any]] = None,
    params: _typing.Optional[_typing.List[jit.Var]] = None,
    fmodel: _typing.Optional[_patch._MonkeyPatchBase] = None,
    device: _typing.Optional[str] = None,
    override: _typing.Optional[_OverrideType] = None,
    track_higher_grads: bool = True,
    **kwargs,
) -> DifferentiableOptimizer:
    r"""Construct a differentiable version of an new optimizer.

    Args:
        opt_type: the type (constructor) for a jittor.optim.Optimizer subtype
            from amongst the types supported by the library, or registered with
            it a runtime.
        opt_kwargs: a dictionary of keywords to be passed to the optimizer
            constructor.
        params (optional): a list of (fast) weights which the differentiable
            optimizer will update. These must be provided if fmodel is not
            provided. If both, these will be used in lieu. These will only
            be used for shape inference when initializing the optimizer.
            This argument can also take the same format as parameter groups,
            i.e. an iterable over dictionaries which contain the 'params' key
            with fast weights as value, and group-specific hyperparameters.
        fmodel (optional): a patched version of the ``module`` tracked by ``opt``.
            It is assumed this patched instance has a view on its latest fast
            weights through ``fmodel.parameters()``. If provided, it is not
            necessary to pass the fast weights explicitly to the differentiable
            optimizer's ``step`` function via the keyword arg ``params``. If not
            provided, the fast weights to update must be provided to ``step``.
        device (optional): the device to cast the optimizer state to when
            creating the differentiable optimizer. If not provided, the same
            device as used for the parameters tracked by ``opt`` will be used.
        override (optional): a dictionary mapping optimizer settings (i.e.
            those which would be passed to the optimizer constructor or
            provided within parameter groups) to either singleton lists of
            override values, or to a list of override values of length equal to
            the number of parameter groups. If a single override is provided for
            a keyword, it is used for all parameter groups. If a list is
            provided, the ``i``\ th element of the list overrides the corresponding
            setting in the ``i``\ th parameter group. This permits the passing of
            tensors requiring gradient to differentiable optimizers for use as
            optimizer settings.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows the returned differentiable optimizer to be used in
            "test mode", without potentially tracking higher order gradients.
            This can be useful when running the training loop at test time,
            e.g. in k-shot learning experiments, without incurring a significant
            memory overhead.

    Returns:
        An initialized ``DifferentiableOptimizer`` instance of the right subtype.
    """

    if opt_type in _opt_mapping:
        if params is not None:
            params = list(params)
            if isinstance(params[0], dict):
                dummy = [
                    {
                        k: jit.zeros_like(v, requires_grad=True) if k == "params" else v
                        for k, v in group.items()
                    }
                    for group in params
                ]
            else:
                dummy = [jit.zeros_like(p, requires_grad=True) for p in params]
        elif fmodel is not None:
            dummy = [jit.zeros_like(p, requires_grad=True) for p in fmodel.parameters()]
        else:
            raise ValueError("Must specify one of fmodel or params in kwargs.")

        opt_kwargs = {} if opt_kwargs is None else opt_kwargs
        opt = opt_type(dummy, **opt_kwargs)

        return _opt_mapping[opt_type](
            opt,
            dummy,
            fmodel=fmodel,
            device=device,
            override=override,
            track_higher_grads=track_higher_grads,
            **kwargs,
        )
    else:
        raise ValueError(
            "Optimizer type {} not supported by higher yet.".format(opt_type)
        )


def register_optim(
    optim_type: jit.optim.Optimizer,
    diff_optim_type: _typing.Type[DifferentiableOptimizer],
) -> None:
    r"""Registers a new optimizer type for use with higher functions.

    Args:
        optim_type: the type of a new optimizer, assumed to be an instance of
            ``jittor.optim.Optimizer``.
        diff_optim_type: the type of a new differentiable optimizer, assumed to
            be an instance of ``higher.optim.DifferentiableOptimizer`` with
            functionally equivalent logic to ``optim_type``.
    """
    _opt_mapping[optim_type] = diff_optim_type


def get_trainable_opt_params(
    opt: jit.optim.Optimizer, device: _typing.Optional[str] = None
) -> _OverrideType:
    r"""Get an override dictionary from an optimizer instance.

    Args:
        opt: the optimizer to obtain an override dictionary from.
        device (optional): the device to cast the learnable tensors to.

    Returns:
        A dictionary of the format expected for the override kwarg of
        differentiable optimizers. It is initialized with trainable tensors
        with as values those float and int hyperparameters found in the
        optimizer's parameter groups (or stuctures containing these).
        Heuristically, hyperparameters containing mixtures of differentiable
        and non-differentiable types will be ignored (and must be manually
        specified when constructing an override dict).
    """
    override: _OverrideType = _collections.defaultdict(list)

    def map_fn(x: _typing.Union[jit.Var, int, float]) -> jit.Var:
        if isinstance(x, jit.Var):
            return x.clone().detach()
        else:
            return jit.array(float(x)).stop_grad(False)

    for group in opt.param_groups:
        for k, v in group.items():
            if k == "params":
                # Ignore actual model parameters tracked by optim
                continue

            # Ignore hyperparameters that aren't structures containing ints
            # or floats
            if all(
                isinstance(x, int) or isinstance(x, float) for x in _utils.flatten(v)
            ):
                override[k].append(_utils._recursive_map(v, map_fn))

    return override


def apply_trainable_opt_params(
    opt: jit.optim.Optimizer, override: _OverrideType
) -> None:
    r"""Apply learned hyperparameters back to original optimizer.

    Args:
        opt: the original optimizer. The hyperparameters in its parameter groups
            will be modified in place.
        override: dictionary of the format used for the override kwarg of
            differentiable optimizers.
    """
    for k, v in override.items():
        # Sanity check
        if (len(v) != 1) and (len(v) != len(opt.param_groups)):
            raise ValueError(
                "Mismatch between the number of override tensors for "
                "optimizer parameter {} and the number of "
                "parameter groups.".format(k)
            )
        for group_idx, group in enumerate(opt.param_groups):
            replacement = v[0] if len(v) == 1 else v[group_idx]
            group[k] = _recursive_apply(replacement, group[k])


## Local utility functions
# TODO(egrefen): use funcs below instead of x._add, in diffopt
def _add(
    tensor: jit.Var,
    a1: _typing.Union[float, int, jit.Var],
    a2: _typing.Optional[jit.Var] = None,
) -> jit.Var:
    if a2 is None:
        value: _typing.Union[jit.Var, float] = 1.0
        other = a1
    else:
        value = a1
        other = a2
    return tensor + (value * other)


def _addcdiv(
    tensor: jit.Var,
    a1: _typing.Union[float, int, jit.Var],
    a2: jit.Var,
    a3: _typing.Optional[jit.Var] = None,
) -> jit.Var:
    if a3 is None:
        value: _typing.Union[jit.Var, float] = 1.0
        tensor1 = a1
        tensor2 = a2
    else:
        value = a1
        tensor1 = a2
        tensor2 = a3
    return tensor + value * (tensor1 / tensor2)


def _addcmul(
    tensor: jit.Var,
    a1: _typing.Union[float, int, jit.Var],
    a2: jit.Var,
    a3: _typing.Optional[jit.Var] = None,
) -> jit.Var:
    if a3 is None:
        value: _typing.Union[jit.Var, float] = 1.0
        tensor1 = a1
        tensor2 = a2
    else:
        value = a1
        tensor1 = a2
        tensor2 = a3
    return tensor + (value * tensor1 * tensor2)


# TODO(egrefen): this probably could be refactored into utils
def _recursive_apply(
    replacement: _typing.Union[list, tuple, dict, set, jit.Var],
    target: _typing.Union[jit.Var, int, float],
) -> _typing.Union[jit.Var, int, float]:
    if not isinstance(replacement, type(target)):
        if isinstance(replacement, jit.Var) and not _utils._is_container(target):
            return type(target)(replacement.item())
        raise ValueError(
            "Expected an non-container type for target, but got {} with value "
            "{}".format(type(target), target)
        )
    elif isinstance(replacement, jit.Var) and isinstance(target, jit.Var):
        replacement = replacement.to(target.device)
        target.data = replacement.data
        return target
    if isinstance(target, list):
        return type(target)(
            [_recursive_apply(r, t) for r, t in zip(replacement, target)]
        )
    elif isinstance(target, tuple):
        return type(target)(
            [_recursive_apply(r, t) for r, t in zip(replacement, target)]
        )
    elif isinstance(replacement, dict) and isinstance(target, dict):
        return type(target)(
            {
                k: _recursive_apply(r, t)
                for (_, r), (k, t) in zip(replacement.items(), target.items())
            }
        )
    elif isinstance(target, set):
        return type(target)(
            {_recursive_apply(r, t) for r, t in zip(replacement, target)}
        )
    else:
        raise ValueError(
            "Couldn't apply replacement of type {} to target of type "
            "{}".format(type(replacement), type(target))
        )
