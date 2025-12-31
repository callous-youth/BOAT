import argparse
import numpy as np
import os
import torch
import boat_torch as boat
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups_vectorized
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def get_data(args):
    def from_sparse(x):
        x = x.tocoo()
        values = x.data
        indices = np.vstack((x.row, x.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = x.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape))

    val_size = 0.5
    train_x, train_y = fetch_20newsgroups_vectorized(
        subset="train",
        return_X_y=True,
        data_home=args.data_path,
        download_if_missing=True,
    )

    test_x, test_y = fetch_20newsgroups_vectorized(
        subset="test",
        return_X_y=True,
        data_home=args.data_path,
        download_if_missing=True,
    )

    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, stratify=train_y, test_size=val_size
    )
    test_x, teval_x, test_y, teval_y = train_test_split(
        test_x, test_y, stratify=test_y, test_size=0.5
    )

    train_x, val_x, test_x, teval_x = map(
        from_sparse, [train_x, val_x, test_x, teval_x]
    )
    train_y, val_y, test_y, teval_y = map(
        torch.LongTensor, [train_y, val_y, test_y, teval_y]
    )

    print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)


import os
import json

# base_folder = os.path.dirname(os.path.abspath(__file__))


def evaluate(x, w, testset):
    with torch.no_grad():
        test_x, test_y = testset
        y = test_x.mm(x)
        loss = F.cross_entropy(y, test_y).detach().item()
        acc = (y.argmax(-1).eq(test_y).sum() / test_y.shape[0]).detach().cpu().item()
    return loss, acc


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
        parser.add_argument("--iterations", type=int, default=10, help="T")
        parser.add_argument("--data_path", default="./data", help="where to save data")
        parser.add_argument(
            "--model_path", default="./save_l2reg", help="where to save model"
        )
        parser.add_argument("--x_lr", type=float, default=100)
        parser.add_argument("--xhat_lr", type=float, default=100)
        parser.add_argument("--w_lr", type=float, default=1000)

        parser.add_argument("--w_momentum", type=float, default=0.9)
        parser.add_argument("--x_momentum", type=float, default=0.9)

        parser.add_argument("--K", type=int, default=10, help="k")

        parser.add_argument("--u1", type=float, default=1.0)
        parser.add_argument(
            "--BVFSM_decay", type=str, default="log", choices=["log", "power2"]
        )
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument(
            "--alg",
            type=str,
            default="BOME",
            choices=[
                "BOME",
                "BSG_1",
                "penalty",
                "AID_CG",
                "AID_FP",
                "ITD",
                "BVFSM",
                "baseline",
                "VRBO",
                "reverse",
                "stocBiO",
                "MRBO",
            ],
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
    print(f"[info] successfully generated data to {args.data_path}/l2reg.pt")
    device = torch.device("cpu")
    n_feats = trainset[0].shape[-1]
    num_classes = trainset[1].unique().shape[-1]

    class upper_model(torch.nn.Module):

        def __init__(self, n_feats, device):
            super(upper_model, self).__init__()
            self.x = torch.nn.Parameter(
                torch.zeros(n_feats, requires_grad=True, device=device).requires_grad_(
                    True
                )
            )

        def forward(self):
            return self.x

    class lower_model(torch.nn.Module):

        def __init__(self, n_feats, device):
            super(lower_model, self).__init__()
            self.y = torch.nn.Parameter(
                torch.zeros((n_feats, num_classes), requires_grad=True, device=device)
            )
            self.y.data = nn.init.kaiming_normal_(self.y.data.t(), mode="fan_out").t()

        def forward(self):
            return self.y

    upper_model = upper_model(n_feats, device)
    lower_model = lower_model(n_feats, device)
    upper_opt = torch.optim.Adam(upper_model.parameters(), lr=0.01)
    lower_opt = torch.optim.SGD(lower_model.parameters(), lr=0.01)
    print(args.gm_op)
    print(args.na_op)
    gm_op = args.gm_op.split(",") if args.gm_op else None
    na_op = args.na_op.split(",") if args.na_op else None
    if na_op is not None:
        if "RGT" in na_op:
            boat_config["RGT"]["truncate_iter"] = 1
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
    iterations = 10
    if boat_config["gm_op"] is not None:
        if "DM" in boat_config["gm_op"] and ("GDA" in boat_config["gm_op"]):
            iterations = 30
        else:
            iterations = 10
    for x_itr in range(iterations):
        if boat_config["gm_op"] is not None:
            if "DM" in boat_config["gm_op"] and ("GDA" in boat_config["gm_op"]):
                b_optimizer._ll_solver.strategy = "s" + str(x_itr % 3 + 1)
            elif "DM" in boat_config["gm_op"] and (
                not ("GDA" in boat_config["gm_op"])
            ):
                b_optimizer._ll_solver.strategy = "s" + str(1)


        loss, run_time = b_optimizer.run_iter(
            ll_feed_dict, ul_feed_dict, current_iter=x_itr
        )

        if x_itr % 1 == 0:
            test_loss, test_acc = evaluate(lower_model(), upper_model(), testset)
            teval_loss, teval_acc = evaluate(lower_model(), upper_model(), tevalset)
            print(
                f"[info] epoch {x_itr:5d} te loss {test_loss:10.4f} te acc {test_acc:10.4f} teval loss {teval_loss:10.4f} teval acc {teval_acc:10.4f} time {run_time:8.2f}"
            )


if __name__ == "__main__":
    main()
