import argparse
import numpy as np
import torch
import boat_torch as boat
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from examples.L2_Reg.utils_l2 import get_data, UpperModel, LowerModel


base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)

with open(os.path.join(parent_folder, "L2_Reg/configs/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(parent_folder, "L2_Reg/configs/loss_config_l2.json"), "r") as f:
    loss_config = json.load(f)


def main():
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--generate_data",
            action="store_true",
            default=False,
            help="whether to create data",
        )
        parser.add_argument(
            "--pretrain",
            action="store_true",
            default=False,
            help="whether to create data",
        )
        parser.add_argument("--epochs", type=int, default=1000)
        parser.add_argument("--seed", type=int, default=1234)
        parser.add_argument("--iterations", type=int, default=10, help="T")
        parser.add_argument("--data_path", default="./data", help="where to save data")
        parser.add_argument(
            "--model_path", default="./save_l2reg", help="where to save model"
        )
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

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        return args

    args = parse_args()
    trainset, valset, testset, tevalset = get_data(args)
    torch.save(
        (trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg.pt")
    )
    device = torch.device("cpu")
    n_feats = trainset[0].shape[-1]
    upper_model = UpperModel(n_feats, device)
    lower_model = LowerModel(
        n_feats, device, num_classes=trainset[1].unique().shape[-1]
    )
    upper_opt = torch.optim.Adam(upper_model.parameters(), lr=0.01)
    lower_opt = torch.optim.SGD(lower_model.parameters(), lr=0.01)
    gm_op = args.gm_op.split(",") if args.gm_op else []
    na_op = args.na_op.split(",") if args.na_op else []
    boat_config["gm_op"] = gm_op
    boat_config["na_op"] = na_op
    boat_config["fo_op"] = args.fo_op
    boat_config["lower_level_model"] = lower_model
    boat_config["upper_level_model"] = upper_model
    boat_config["lower_level_opt"] = lower_opt
    boat_config["upper_level_opt"] = upper_opt
    boat_config["lower_level_var"] = list(lower_model.parameters())
    boat_config["upper_level_var"] = list(upper_model.parameters())
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()

    ul_feed_dict = {"data": trainset[0].to(device), "target": trainset[1].to(device)}
    ll_feed_dict = {"data": valset[0].to(device), "target": valset[1].to(device)}
    iterations = 30
    for x_itr in range(iterations):
        b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)


if __name__ == "__main__":
    main()
