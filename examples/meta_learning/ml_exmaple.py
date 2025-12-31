import os
import json
import math
import torch
import boat_torch as boat
import torch.nn.functional as F
from torch import nn
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm


# Model definitions
def get_cnn_omniglot(hidden_size, n_classes):
    def conv_block(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1.0, affine=True, track_running_stats=True),
        )

    feature_extractor = nn.Sequential(
        conv_block(1, hidden_size),
        conv_block(hidden_size, hidden_size),
        conv_block(hidden_size, hidden_size),
        conv_block(hidden_size, hidden_size),
        nn.Flatten(),
    )

    classifier = nn.Linear(hidden_size, n_classes)
    return feature_extractor, classifier


def initialize(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="BOAT Omniglot Meta-Training ")
    parser.add_argument("--dynamic_method", type=str, default="NGD")
    parser.add_argument("--hyper_method", type=str, default="CG")
    parser.add_argument("--fo_gm", type=str, default=None)
    parser.add_argument("--ways", type=int, default=20)
    parser.add_argument("--shot", type=int, default=1)
    args = parser.parse_args()


    # ===== device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== dataset  =====
    dataset = omniglot(
        "./data/",
        ways=args.ways,
        shots=args.shot,
        test_shots=15,
        meta_train=True,
        download=True,
    )

    # ===== model =====
    meta_model_x, meta_model_y = get_cnn_omniglot(64, args.ways)
    meta_model_x = meta_model_x.to(device)
    meta_model_y = meta_model_y.to(device)
    initialize(meta_model_x)
    initialize(meta_model_y)

    # ===== dataloader =====
    batch_size = 8
    dataloader = BatchMetaDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=False,
    )

    # ===== optimizers =====
    inner_opt = torch.optim.SGD(meta_model_y.parameters(), lr=0.4)
    outer_opt = torch.optim.Adam(meta_model_x.parameters(), lr=0.05)

    # ===== BOAT config =====
    with open("./configs/boat_config_CG.json", "r") as f:
        boat_config = json.load(f)
    with open("./configs/loss_config_CG.json", "r") as f:
        loss_config = json.load(f)

    boat_config["gm_op"] = args.dynamic_method.split(",") if args.dynamic_method else None
    boat_config["na_op"] = args.hyper_method.split(",") if args.hyper_method else None
    boat_config["fo_op"] = args.fo_gm

    boat_config["lower_level_model"] = meta_model_y
    boat_config["upper_level_model"] = meta_model_x
    boat_config["lower_level_var"] = list(meta_model_y.parameters())
    boat_config["upper_level_var"] = list(meta_model_x.parameters())
    boat_config["lower_level_opt"] = inner_opt
    boat_config["upper_level_opt"] = outer_opt

    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()

    # ===== training loop =====
    max_iters = 20
    print("Start meta-training ")

    with tqdm(dataloader, total=max_iters, desc="Meta Training") as pbar:
        for meta_iter, batch in enumerate(pbar):
            initialize(meta_model_y)

            ll_feed = [
                {
                    "data": batch["train"][0][k].float().to(device),
                    "target": batch["train"][1][k].to(device),
                }
                for k in range(batch_size)
            ]

            ul_feed = [
                {
                    "data": batch["test"][0][k].float().to(device),
                    "target": batch["test"][1][k].to(device),
                }
                for k in range(batch_size)
            ]

            log_results, _ = b_optimizer.run_iter(
                ll_feed, ul_feed, current_iter=meta_iter
            )

            if meta_iter >= max_iters:
                print(f"Reached {max_iters} iterations. Stop.")
                break



if __name__ == "__main__":
    main()
