import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import boat_torch as boat
from torch import nn
from torchmeta.toy.helpers import sinusoid
from torchmeta.utils.data import BatchMetaDataLoader
import math
from tqdm import tqdm


def get_cnn_omniglot(hidden_size, n_classes):
    def conv_layer(
        ic,
        oc,
    ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(
                oc,
                momentum=1.0,
                affine=True,
                track_running_stats=True,  # When this is true is called the "transductive setting"
            ),
        )

    net = nn.Sequential(
        conv_layer(1, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size, n_classes),
    )

    initialize(net)
    return net


def get_sinuoid():
    fc_net = nn.Sequential(
        nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(inplace=True), nn.LayerNorm(normalized_shape=64)
        ),
        nn.Linear(64, 64),
        nn.ReLU(inplace=True),
        nn.LayerNorm(normalized_shape=64),
        nn.Sequential(nn.Linear(64, 1)),
    )
    initialize(fc_net)
    return fc_net


def initialize(net):
    # initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            # m.bias.data = torch.ones(m.bias.data.size())
            # m.weight.data.zero_()
            m.bias.data.zero_()
    return net


# device = torch.device("cpu")
# dataset = omniglot("C:/Users/ASUS/Documents/GitHub/BOAT/data/", ways=5, shots=1, test_shots=15, meta_train=True,download=True)
# test_dataset = omniglot("C:/Users/ASUS/Documents/GitHub/BOAT/data/", ways=5, shots=1, test_shots=15, meta_test=True,download=True)
#
# meta_model = get_cnn_omniglot(64, 5)
# initialize(meta_model)

batch_size = 4
kwargs = {"num_workers": 1, "pin_memory": True}
device = torch.device("cpu")
dataset = sinusoid(shots=10, test_shots=100, seed=0)
meta_model = get_sinuoid()
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
test_dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)

inner_opt = torch.optim.SGD(lr=0.1, params=meta_model.parameters())
outer_opt = torch.optim.Adam(meta_model.parameters(), lr=0.01)
y_lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=outer_opt, T_max=80000, eta_min=0.001
)
import os
import json

base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)
with open(os.path.join(base_folder, "configs/boat_config_ml.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(base_folder, "configs/loss_config_ml.json"), "r") as f:
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
        default="",
        help="convnet for 4 convs or resnet for Residual blocks",
    )
    args = parser.parse_args()

    dynamic_method = args.dynamic_method.split(",") if args.dynamic_method else None
    hyper_method = args.hyper_method.split(",") if args.hyper_method else None
    print(args.dynamic_method)
    print(args.hyper_method)
    boat_config["dynamic_op"] = dynamic_method
    boat_config["hyper_op"] = hyper_method
    boat_config["lower_level_model"] = meta_model
    boat_config["upper_level_model"] = meta_model
    boat_config["lower_level_var"] = list(meta_model.parameters())
    boat_config["upper_level_var"] = list(meta_model.parameters())
    boat_config["lower_level_opt"] = inner_opt
    boat_config["upper_level_opt"] = outer_opt
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()

    with tqdm(dataloader, total=1, desc="Meta Training Phase") as pbar:
        for meta_iter, batch in enumerate(pbar):
            ul_feed_dict = [
                {
                    "data": batch["test"][0][k].float().to(device),
                    "target": batch["test"][1][k].float().to(device),
                }
                for k in range(batch_size)
            ]
            ll_feed_dict = [
                {
                    "data": batch["train"][0][k].float().to(device),
                    "target": batch["train"][1][k].float().to(device),
                }
                for k in range(batch_size)
            ]
            # print(ll_feed_dict[0]['data'].shape,ll_feed_dict[0]['target'].shape)
            loss, run_time = b_optimizer.run_iter(
                ll_feed_dict, ul_feed_dict, current_iter=meta_iter
            )
            y_lr_schedular.step()
            print("validation loss:", loss[-1][-1])
            if meta_iter >= 1:
                break
    b_optimizer.plot_losses()


if __name__ == "__main__":
    main()
