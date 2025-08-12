import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import boat_torch as boat
import torch
import torch.nn.functional as F
from util_file import data_splitting, initialize, accuary, Binarization
from boat_torch.utils import HyperGradientRules, DynamicalSystemRules
from torchvision.datasets import MNIST

base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)
dataset = MNIST(root=os.path.join(parent_folder, "./data"), train=True, download=True)
tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(0.5)
tr.data_flatten()
val.data_flatten()
test.data_flatten()

print(torch.cuda.is_available())
device = torch.device("cpu")


class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(
            torch.zeros(tr.data.shape[0]).to(device).requires_grad_(True)
        )

    def forward(self, y):
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y


x = Net_x(tr)
y = torch.nn.Sequential(torch.nn.Linear(28**2, 10)).to(device)
x_opt = torch.optim.Adam(x.parameters(), lr=0.01)
y_opt = torch.optim.SGD(y.parameters(), lr=0.01)
initialize(x)
initialize(y)

with open(os.path.join(base_folder, "configs/boat_config_dhl.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(base_folder, "configs/loss_config_dhl.json"), "r") as f:
    loss_config = json.load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data HyperCleaner")

    parser.add_argument(
        "--dynamic_method",
        type=str,
        default="",
        help="omniglot or miniimagenet or tieredImagenet",
    )
    parser.add_argument(
        "--hyper_method",
        type=str,
        default="",
        help="convnet for 4 convs or resnet for Residual blocks",
    )
    parser.add_argument(
        "--fo_gm",
        type=str,
        default=None,
        help="convnet for 4 convs or resnet for Residual blocks",
    )

    args = parser.parse_args()
    dynamic_method = args.dynamic_method.split(",") if args.dynamic_method else None
    hyper_method = args.hyper_method.split(",") if args.hyper_method else None
    print(args.dynamic_method)
    print(args.hyper_method)
    if hyper_method is not None and ("RGT" in hyper_method):
        boat_config["RGT"]["truncate_iter"] = 1
    boat_config["dynamic_op"] = dynamic_method
    boat_config["hyper_op"] = hyper_method
    boat_config["fo_gm"] = args.fo_gm
    boat_config["lower_level_model"] = y
    boat_config["upper_level_model"] = x
    boat_config["lower_level_opt"] = y_opt
    boat_config["upper_level_opt"] = x_opt
    boat_config["lower_level_var"] = list(y.parameters())
    boat_config["upper_level_var"] = list(x.parameters())
    b_optimizer = boat.Problem(boat_config, loss_config)
    if boat_config["fo_gm"] is not None and ("PGDM" in boat_config["fo_gm"]):
        boat_config["PGDM"]["gamma_init"] = boat_config["PGDM"]["gamma_max"] + 0.1

    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()
    ul_feed_dict = {"data": val.data.to(device), "target": val.clean_target.to(device)}
    ll_feed_dict = {"data": tr.data.to(device), "target": tr.dirty_target.to(device)}
    HyperGradientRules.set_gradient_order(
        [
            ["PTT", "RGT", "FOA"],
            ["IAD", "RAD", "FD", "IGA"],
            ["CG", "NS"],
        ]
    )
    DynamicalSystemRules.set_gradient_order(
        [
            ["GDA", "DI"],
            ["DM", "NGD"],
        ]
    )
    if boat_config["dynamic_op"] is not None:
        if "DM" in boat_config["dynamic_op"] and ("GDA" in boat_config["dynamic_op"]):
            iterations = 3
        else:
            iterations = 2
            b_optimizer.boat_configs["return_grad"] = True
    else:
        iterations = 3
    for x_itr in range(iterations):
        if boat_config["dynamic_op"] is not None:
            if "DM" in boat_config["dynamic_op"] and (
                "GDA" in boat_config["dynamic_op"]
            ):
                b_optimizer._ll_solver.gradient_instances[-1].strategy = "s" + str(
                    x_itr + 1
                )
        loss, run_time = b_optimizer.run_iter(
            ll_feed_dict, ul_feed_dict, current_iter=x_itr
        )

        if x_itr % 1 == 0:
            with torch.no_grad():
                out = y(test.data.to(device))
                acc = accuary(out, test.clean_target.to(device))
                x_bi = Binarization(x.x.cpu().numpy())
                clean = x_bi * tr.clean
                p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                r = clean.mean() / (1.0 - tr.rho)
                F1_score = 2 * p * r / (p + r + 1e-8)
                dc = 0
                if x_itr == 0:
                    F1_score_last = 0
                if F1_score_last > F1_score:
                    dc = 1
                F1_score_last = F1_score
                valLoss = F.cross_entropy(out, test.clean_target.to(device))
                print(
                    "x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val_loss={:.3f}".format(
                        x_itr,
                        100 * accuary(out, test.clean_target.to(device)),
                        100 * p,
                        100 * r,
                        100 * F1_score,
                        valLoss,
                    )
                )
    b_optimizer.plot_losses()


if __name__ == "__main__":
    main()
