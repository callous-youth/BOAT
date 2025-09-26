import sys
import os
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import boat_torch as boat
import torch
import torch.nn.functional as F
from util_file import data_splitting, initialize, accuary, Binarization
from boat_torch.utils import HyperGradientRules, DynamicalSystemRules
from torchvision.datasets import MNIST, FashionMNIST
import logging
import random
seed = 2424
random.seed(seed)
torch.manual_seed(seed)


base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)
dataset = MNIST(root=os.path.join(parent_folder, "./data"), train=True, download=True)
#dataset = FashionMNIST(root=os.path.join(parent_folder, "./data"), train=True, download=True)
tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(0.5)
tr.data_flatten()
val.data_flatten()
test.data_flatten()




print(torch.cuda.is_available())
device = torch.device("cpu")



METHOD_MAP = {
    # 二层方法：fo_gm=None
    "RHG":    ("NGD",        "RAD",       None),
    "BDA":    ("NGD,GDA",    "RAD",       None),
    "CG":     ("NGD",        "CG",        None),
    "NS":     ("NGD",        "NS",        None),
    "TRHG":   ("NGD",        "RGT,RAD",       None),
    "BAMM":   ("DM,GDA,NGD",     "CG",       None),
    "IAPTT":  ("NGD,DI",     "PTT,RAD",   None),

    # fo-gm 方法：dynamic_method=None, hyper_method=None
    "BVFSM":  (None, None, "VSM"),
    "BOME":   (None, None, "VFM"),
    "VPBGD":  (None, None, "PGDM"),
    "MEHA":   (None, None, "MESM"),
}


def main():
   
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
    y = torch.nn.Sequential(torch.nn.Linear(784, 10)).to(device)

    # initialize(x)
    # initialize(y)




    parser = argparse.ArgumentParser(description="Data HyperCleaner")

    parser.add_argument(
        "--dynamic_method",
        type=str,
        default="NGD,DI",
        help="omniglot or miniimagenet or tieredImagenet",
    )
    parser.add_argument(
        "--hyper_method",
        type=str,
        default="PTT,RAD",
        help="convnet for 4 convs or resnet for Residual blocks",
    )
    parser.add_argument(
        "--fo_gm",
        type=str,
        default="BAMM",
        help="convnet for 4 convs or resnet for Residual blocks",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="BAMM",
        help="",
    )
    parser.add_argument(
        "--x_lr",
        type=float,
        default=0.01,
        help="",
    )

    parser.add_argument(
        "--y_lr",
        type=float,
        default=0.005,
        help="",
    )

    args = parser.parse_args()
    if args.method in METHOD_MAP:
        args.dynamic_method, args.hyper_method, args.fo_gm = METHOD_MAP[args.method]
        dynamic_method, hyper_method, fo_gm = METHOD_MAP[args.method]

    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    with open(os.path.join(base_folder, "configs/boat_config_dhl.json"), "r") as f:
        boat_config = json.load(f)

    loss_path = "configs/loss_config_dhl.json"

    if args.method == "TRHG":
        # args.x_lr = 0.001
        # args.y_lr = 0.1
        # boat_config["lower_iters"] = 100
        # loss_path = "configs/loss_config_dhl.json"
        # boat_config["RGT"]["truncate_iter"] = 1
        args.x_lr = 0.1
        args.y_lr = 0.01
        boat_config["lower_iters"] = 100
        loss_path = "configs/loss_config_dhl.json"
        boat_config["RGT"]["truncate_iter"] = 5

    elif args.method == "BAMM":
        boat_config["lower_iters"] = 1
        args.x_lr = 0.01
        args.y_lr = 0.1


    elif args.method == "IAPTT":
        args.x_lr = 0.01
        args.y_lr = 0.01
        boat_config["lower_iters"] = 50
    elif args.method == "VPBGD":
        args.x_lr = 0.1
        args.y_lr = 0.1
        boat_config["lower_iters"] = 1
    elif args.method == "CG":
        args.x_lr = 0.01
        args.y_lr = 0.01
        boat_config["lower_iters"] = 100
    else:
        loss_path = "configs/loss_config_dhl.json"
        args.x_lr = 0.01
        args.y_lr = 0.005

    with open(os.path.join(base_folder, loss_path), "r") as f:
        loss_config = json.load(f)


    dynamic_method = args.dynamic_method.split(",") if args.dynamic_method else None
    hyper_method = args.hyper_method.split(",") if args.hyper_method else None
    fo_gm = args.fo_gm if args.fo_gm else None

    print(args.dynamic_method)
    print(args.hyper_method)
    print(args.fo_gm)

    x_opt = torch.optim.Adam(x.parameters(), lr=args.x_lr)
    if args.method == "TRHG" or args.method == "VPBGD":
        x_opt = torch.optim.SGD(x.parameters(), lr=args.x_lr)
    if args.method == "TRHG":
        x_opt = torch.optim.Adam(x.parameters(), lr=args.x_lr)
    y_opt = torch.optim.SGD(y.parameters(), lr=args.y_lr)


    boat_config["dynamic_op"] = dynamic_method
    boat_config["hyper_op"] = hyper_method
    boat_config["fo_gm"] = fo_gm
    boat_config["lower_level_model"] = y
    boat_config["upper_level_model"] = x
    boat_config["lower_level_opt"] = y_opt
    boat_config["upper_level_opt"] = x_opt
    boat_config["lower_level_var"] = list(y.parameters())
    boat_config["upper_level_var"] = list(x.parameters())
    b_optimizer = boat.Problem(boat_config, loss_config)



    # if boat_config["fo_gm"] is not None and ("PGDM" in boat_config["fo_gm"]):
    #     boat_config["PGDM"]["gamma_init"] = boat_config["PGDM"]["gamma_max"] + 0.1

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
            ["NGD", "DM"],
        ]
    )
    if boat_config["dynamic_op"] is not None:
        if "DM" in boat_config["dynamic_op"] and ("GDA" in boat_config["dynamic_op"]):
            iterations = 3
        else:
            iterations = 2
            b_optimizer.boat_configs["return_grad"] = True
    else:
        iterations = 40000

    b_optimizer.boat_configs["return_grad"] = False

    iterations = 3000

    # 初始化 logger
    log_path = os.path.join(
        base_folder, f"{args.method}.txt"
    )

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    for x_itr in range(iterations):
        if args.method == "BAMM":
            alpha = 0.99 * 1 / (x_itr + 1) ** (1 / 75)
            eta = (x_itr + 1) ** (-0.5 * 0.001) * alpha * args.y_lr
            args.x_lr = (x_itr + 1) ** (-1.5 * 0.001) * alpha ** 5 * args.y_lr
            for params in x_opt.param_groups:
                params['lr'] = args.x_lr


        loss, run_time = b_optimizer.run_iter(
            ll_feed_dict, ul_feed_dict, current_iter=x_itr
        )

        if x_itr % 10 == 0:
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

                log_msg = (
                    "x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val_loss={:.3f}".format(
                        x_itr,
                        100 * accuary(out, test.clean_target.to(device)),
                        100 * p,
                        100 * r,
                        100 * F1_score,
                        valLoss,
                    )
                )
                print(log_msg)  # 控制台
                logger.info(log_msg)  # 文件

    #b_optimizer.plot_losses()


if __name__ == "__main__":
    main()
