"""Compare RAD with an operator that clips the upper hypergradient."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn.utils import clip_grad_norm_

from boat_torch import Problem
from boat_torch.na_ol.rad import RAD
from boat_torch.operation_registry import register_class
from boat_torch.utils import HyperGradientRules


@register_class
class CLIPPED_RAD(RAD):
    def compute_gradients(self, *args, **kwargs):
        result = super().compute_gradients(*args, **kwargs)
        clip_grad_norm_(self.ul_var, max_norm=0.5)
        return result


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self):
        return self.value


def lower_loss(feed, upper, lower, params=None):
    y = lower() if params is None else params[0]
    return 0.5 * (y - upper()) ** 2


def upper_loss(feed, upper, lower, params=None):
    y = lower() if params is None else params[0]
    return 0.5 * (y - 2) ** 2


def _hypergradient(method, log_path):
    upper, lower = Scalar(), Scalar()
    config = {
        "gm_op": ["NGD"],
        "na_op": [method],
        "fo_op": None,
        "lower_iters": 1,
        "device": "cpu",
        "accumulate_grad": False,
        "copy_last_param": False,
        "return_grad": True,
        "loss_log_path": str(log_path),
        "upper_level_model": upper,
        "lower_level_model": lower,
        "upper_level_var": list(upper.parameters()),
        "lower_level_var": list(lower.parameters()),
        "upper_level_opt": torch.optim.SGD(upper.parameters(), lr=0.1),
        "lower_level_opt": torch.optim.SGD(lower.parameters(), lr=1.0),
        "GDA": {"alpha_init": 0.0},
        "RGT": {"truncate_iter": 0},
    }
    losses = {
        "lower_level_loss": {"function": f"{__name__}.lower_loss"},
        "upper_level_loss": {"function": f"{__name__}.upper_loss"},
    }
    problem = Problem(config, losses).build_ll_solver().build_ul_solver()
    problem.set_track_trajectory(True)
    gradients, _ = problem.run_iter({}, {}, current_iter=0)
    return gradients[0].item()


def run_demo():
    order = HyperGradientRules.get_gradient_order()
    HyperGradientRules.set_gradient_order(order + [["CLIPPED_RAD"]])
    try:
        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "losses.json"
            return _hypergradient("RAD", log_path), _hypergradient("CLIPPED_RAD", log_path)
    finally:
        HyperGradientRules.set_gradient_order(order)


if __name__ == "__main__":
    rad, clipped = run_demo()
    print(f"RAD:         {rad:.6f}")
    print(f"CLIPPED_RAD: {clipped:.6f}")
