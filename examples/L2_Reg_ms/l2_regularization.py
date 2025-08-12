import argparse
import os
import json
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import boat_ms as boat
from mindspore.common import COOTensor
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import mindspore as ms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups_vectorized


def get_data(args):
    def from_sparse(x):
        x = x.tocoo()
        indices = np.vstack((x.row, x.col)).astype(np.int32).T
        values = x.data.astype(np.float32)
        shape = x.shape
        return COOTensor(
            indices=Tensor(indices, ms.int32),
            values=Tensor(values, ms.float32),
            shape=shape,
        )

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
        lambda y: Tensor(y, ms.int32), [train_y, val_y, test_y, teval_y]
    )

    print(train_y.shape[0], val_y.shape[0], test_y.shape[0], teval_y.shape[0])

    return (train_x, train_y), (val_x, val_y), (test_x, test_y), (teval_x, teval_y)


def evaluate(x, w, testset):
    """
    Evaluate the model on the test set.

    Args:
        x (ms.Tensor): Lower-level model weights.
        w (ms.Tensor): Upper-level model weights.
        testset (tuple): Test set containing test_x and test_y.

    Returns:
        tuple: Test loss and accuracy.
    """
    test_x, test_y = testset

    if isinstance(test_x, ms.COOTensor):
        y = ops.SparseTensorDenseMatmul()(
            test_x.indices, test_x.values, test_x.shape, x
        )
    else:
        y = ops.MatMul()(test_x, x)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loss = loss_fn(y, test_y).mean()

    loss = loss.asnumpy().item()

    predictions = y.argmax(axis=-1)
    acc = (predictions == test_y).sum().asnumpy() / test_y.shape[0]

    return loss, acc


base_folder = os.path.dirname(os.path.abspath(__file__))
# parent_folder = os.path.dirname(base_folder)

with open(os.path.join(base_folder, "configs_ms/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(base_folder, "configs_ms/loss_config_l2.json"), "r") as f:
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
            "--dynamic_method",
            type=str,
            default=None,
            help="omniglot or miniimagenet or tieredImagenet",
        )
        parser.add_argument(
            "--hyper_method",
            type=str,
            default=None,
            help="convnet for 4 convs or resnet for Residual blocks",
        )
        parser.add_argument(
            "--fo_gm",
            type=str,
            default="VFM",
            help="convnet for 4 convs or resnet for Residual blocks",
        )
        args = parser.parse_args()

        np.random.seed(args.seed)
        ms.set_seed(args.seed)
        return args

    import pickle

    def save_data(data, path):
        """
        Save custom data, including COOTensor, by decomposing into indices, values, and shape.

        Args:
            data: Data to be saved (e.g., datasets with COOTensor).
            path: Path to save the data.
        """
        processed_data = []
        for dataset in data:
            tensors = []
            for item in dataset:
                if isinstance(item, ms.COOTensor):
                    # Convert COOTensor to a savable format
                    tensors.append(
                        {
                            "indices": item.indices.asnumpy(),
                            "values": item.values.asnumpy(),
                            "shape": item.shape,
                        }
                    )
                else:
                    tensors.append(item.asnumpy())
            processed_data.append(tensors)

        with open(path, "wb") as f:
            pickle.dump(processed_data, f)
        print(f"[info] Successfully saved data to {path}")

    def load_data(path):
        """
        Load custom data, reconstructing COOTensor from saved sparse data.

        Args:
            path: Path to load the data from.

        Returns:
            Loaded data with COOTensor reconstructed.
        """
        with open(path, "rb") as f:
            raw_data = pickle.load(f)

        reconstructed_data = []
        for dataset in raw_data:
            tensors = []
            for item in dataset:
                if (
                    isinstance(item, dict)
                    and "indices" in item
                    and "values" in item
                    and "shape" in item
                ):
                    # Reconstruct COOTensor
                    tensors.append(
                        ms.COOTensor(
                            indices=ms.Tensor(item["indices"], ms.int32),
                            values=ms.Tensor(item["values"], ms.float32),
                            shape=item["shape"],
                        )
                    )
                else:
                    tensors.append(ms.Tensor(item, ms.float32))
            reconstructed_data.append(tensors)

        return reconstructed_data

    args = parse_args()
    trainset, valset, testset, tevalset = get_data(args)

    # Save the datasets using pickle
    save_path = os.path.join(args.data_path, "l2reg.pkl")
    save_data((trainset, valset, testset, tevalset), save_path)
    # ms.save((trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg.ms"))
    print(f"[info] successfully generated data to {args.data_path}/l2reg.ms")
    device = ms.context.set_context(device_target="CPU")
    n_feats = trainset[0].shape[-1]
    num_classes = int(trainset[1].max().asnumpy()) + 1

    from mindspore.common.initializer import HeNormal

    class UpperModel(nn.Cell):
        def __init__(self, n_feats):
            super(UpperModel, self).__init__()
            self.x = ms.Parameter(ms.Tensor(np.zeros(n_feats), ms.float32))

        def construct(self):
            return self.x

    class LowerModel(nn.Cell):
        def __init__(self, n_feats, num_classes):
            super(LowerModel, self).__init__()
            he_normal = HeNormal()
            self.y = ms.Parameter(
                ms.Tensor(
                    shape=(n_feats, num_classes), dtype=ms.float32, init=he_normal
                )
            )

        def construct(self):
            return self.y

    upper_model = UpperModel(n_feats)
    lower_model = LowerModel(n_feats, num_classes)
    upper_opt = nn.Adam(upper_model.trainable_params(), learning_rate=0.1)
    lower_opt = nn.SGD(lower_model.trainable_params(), learning_rate=0.1)
    print(args.dynamic_method)
    print(args.hyper_method)
    dynamic_method = args.dynamic_method.split(",") if args.dynamic_method else []
    hyper_method = args.hyper_method.split(",") if args.hyper_method else []
    if "RGT" in hyper_method:
        boat_config["RGT"]["truncate_iter"] = 1
    boat_config["dynamic_op"] = dynamic_method
    boat_config["hyper_op"] = hyper_method
    boat_config["fo_gm"] = args.fo_gm
    boat_config["lower_level_model"] = lower_model
    boat_config["upper_level_model"] = upper_model
    boat_config["lower_level_opt"] = lower_opt
    boat_config["upper_level_opt"] = upper_opt
    boat_config["lower_level_var"] = lower_model.trainable_params()
    boat_config["upper_level_var"] = upper_model.trainable_params()
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()

    ul_feed_dict = {"data": trainset[0], "target": trainset[1]}
    ll_feed_dict = {"data": valset[0], "target": valset[1]}

    if "DM" in boat_config["dynamic_op"] and ("GDA" in boat_config["dynamic_op"]):
        iterations = 30
    else:
        iterations = 10
    for x_itr in range(iterations):
        if "DM" in boat_config["dynamic_op"] and ("GDA" in boat_config["dynamic_op"]):
            b_optimizer._ll_solver.strategy = "s" + str(x_itr % 3 + 1)
        elif "DM" in boat_config["dynamic_op"] and (
            not ("GDA" in boat_config["dynamic_op"])
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
    b_optimizer.plot_losses()


if __name__ == "__main__":
    main()
