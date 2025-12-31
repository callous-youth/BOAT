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

"""

Utility functions for components of ``higher_jit``.

"""

import typing as _typing
import jittor as jit

_T = _typing.TypeVar("_T")
_U = _typing.TypeVar("_U")


def _copy_tensor(
    t: jit.Var, safe_copy: bool, device: _typing.Optional[str] = None
) -> jit.Var:
    if safe_copy:
        t = t.clone().detach()
        t.stop_grad = not t.is_stop_grad()
    else:
        t = t.detach()
        t.stop_grad = not t.is_stop_grad()
    t = t if device is None else t.to(device)
    return t


def _recursive_copy_and_cast(
    target: _typing.Union[list, tuple, dict, set, jit.Var],
    device: _typing.Optional[str],
) -> jit.Var:
    def map_fn(x):
        if isinstance(
            x, jit.Var
        ):  # Replacing _torch.is_tensor(x) with isinstance(x, jit.Var)
            return _copy_tensor(x, True, device=device)
        else:
            return x

    return _recursive_map(target, map_fn)


def _recursive_map(
    target: _typing.Union[list, tuple, dict, set, _T],
    map_fn: _typing.Callable[[_T], _U],
) -> _typing.Union[list, tuple, dict, set, _U]:
    if isinstance(target, list):
        return type(target)([_recursive_map(x, map_fn) for x in target])
    elif isinstance(target, tuple):
        return type(target)([_recursive_map(x, map_fn) for x in target])
    elif isinstance(target, dict):
        return type(target)({k: _recursive_map(v, map_fn) for k, v in target.items()})
    elif isinstance(target, set):
        return type(target)({_recursive_map(x, map_fn) for x in target})
    else:
        return map_fn(target)


def _is_container(target: _typing.Any) -> bool:
    flag = (
        isinstance(target, list)
        or isinstance(target, tuple)
        or isinstance(target, dict)
        or isinstance(target, set)
    )
    return flag


def _find_param_in_list(
    param: jit.Var, l: _typing.Iterable[jit.Var]
) -> _typing.Optional[int]:
    for i, p in enumerate(l):
        if p is param:
            return i
    else:
        return None


def _get_param_mapping(
    module: jit.Module, seen: _typing.List[jit.Var], mapping: _typing.List[int]
) -> _typing.List[int]:

    for _, param in module.named_parameters(False):
        if param is None:
            continue

        found = _find_param_in_list(param, seen)

        if found is None:
            mapping.append(len(seen))
            seen.append(param)
        else:
            mapping.append(found)

    for name, child in module._modules.items():
        _ = _get_param_mapping(child, seen, mapping)

    return mapping


def flatten(x: _typing.Any) -> _typing.List[_typing.Any]:
    r"""Returns a flattened list of objects from a nested structure."""
    l: _typing.List[_typing.Any] = []
    if isinstance(x, dict):
        for y in x.values():
            l.extend(flatten(y))
    elif isinstance(x, list) or isinstance(x, set) or isinstance(x, tuple):
        for y in x:
            l.extend(flatten(y))
    else:
        l.append(x)
    return l


def get_func_params(
    module: jit.Module, device: _typing.Optional[str] = None, safe_copy: bool = True
) -> _typing.List[jit.Var]:
    r"""Returns a detached copy of module parameters which requires gradient."""
    params = [_copy_tensor(p, safe_copy, device) for p in module.parameters()]
    return params
