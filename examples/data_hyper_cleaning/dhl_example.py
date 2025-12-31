import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import boat_torch as boat
import torch
from util_file import data_splitting, initialize
from torchvision.datasets import MNIST

base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)
dataset = MNIST(root=os.path.join(parent_folder, "data/"), train=True, download=True)
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

with open(os.path.join(parent_folder, "data_hyper_cleaning/configs/boat_config_dhl.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(parent_folder, "data_hyper_cleaning/configs/loss_config_dhl.json"), "r") as f:
    loss_config = json.load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data HyperCleaner")

    parser.add_argument(
        "--gm_op",
        type=str,
        default="NGD",
        help="omniglot or miniimagenet or tieredImagenet",
    )
    parser.add_argument(
        "--na_op",
        type=str,
        default="RAD",
        help="convnet for 4 convs or resnet for Residual blocks",
    )
    parser.add_argument(
        "--fo_op",
        type=str,
        default=None,
        help="convnet for 4 convs or resnet for Residual blocks",
    )

    args = parser.parse_args()
    gm_op = args.gm_op.split(",") if args.gm_op else None
    na_op = args.na_op.split(",") if args.na_op else None
    boat_config["gm_op"] = gm_op
    boat_config["na_op"] = na_op
    boat_config["fo_op"] = args.fo_op
    boat_config["lower_level_model"] = y
    boat_config["upper_level_model"] = x
    boat_config["lower_level_opt"] = y_opt
    boat_config["upper_level_opt"] = x_opt
    boat_config["lower_level_var"] = list(y.parameters())
    boat_config["upper_level_var"] = list(x.parameters())
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()
    ul_feed_dict = {"data": val.data.to(device), "target": val.clean_target.to(device)}
    ll_feed_dict = {"data": tr.data.to(device), "target": tr.dirty_target.to(device)}
    iterations = 3
    for x_itr in range(iterations):
        b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)


if __name__ == "__main__":
    main()
