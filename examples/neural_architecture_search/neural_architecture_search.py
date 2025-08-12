import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import glob
import numpy as np
import jittor as jit
jit.flags.log_silent = True 
from jittor import nn
import logging
import utils  # 假设 utils 模块已经转换为 Jittor 兼容
import argparse
import boat_jit as boat  # 假设 boat_torch 模块已经转换为 Jittor 兼容
from model_search import Network  # 假设 model_search 模块已经转换为 Jittor 兼容
from jittor.dataset import Dataset
parser = argparse.ArgumentParser("cifar")
parser.add_argument("--data", type=str, default="data/", help="location of the data corpus")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
parser.add_argument("--learning_rate_min", type=float, default=0.001, help="min learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=5, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=1, help="num of training epochs")
parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
parser.add_argument("--layers", type=int, default=3, help="total number of layers")
parser.add_argument("--model_path", type=str, default="saved_models", help="path to save the model")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--load", action="store_true", default=False, help="Whether to load model")
parser.add_argument("--load_iter", type=int, default=0, help="the itrations to load the model")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument("--drop_path_prob", type=float, default=0.3, help="drop path probability")
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument("--train_portion", type=float, default=0.5, help="portion of training data")
parser.add_argument("--inner_loop", type=int, default=1, help="number of inner loops for IAPTT")
parser.add_argument("--unrolled", action="store_true", default=False, help="use one-step unrolled validation loss")
parser.add_argument("--arch_learning_rate", type=float, default=3e-4, help="learning rate for arch encoding")
parser.add_argument("--arch_weight_decay", type=float, default=1e-3, help="weight decay for arch encoding")
args = parser.parse_args()

args.save = "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
args.save = os.path.join("NAS", args.save)
if not os.path.exists("NAS"):
    os.makedirs("NAS")

utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10

def main():
    if not jit.has_cuda:
        logging.info("no gpu device available")
        sys.exit(1)
    jit.flags.use_cuda = jit.has_cuda
    jit.cudnn.set_max_workspace_ratio(0.0)
    jit.flags.lazy_execution=0
    print(jit.has_cuda)
    print(jit.flags.use_cuda)
    # print(jit.current_device())
    np.random.seed(args.seed)
    # jit.set_device(args.gpu)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    if args.load:
        utils.load(model, args.model_path)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = jit.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    outer_opt = jit.optim.Adam(
        model.arch_parameters(),
        lr=args.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay,
    )

    import json

    base_folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_folder, "configs/boat_config_nas.json"), "r") as f:
        boat_config = json.load(f)

    with open(os.path.join(base_folder, "configs/loss_config_nas.json"), "r") as f:
        loss_config = json.load(f)

    boat_config["lower_level_model"] = model
    boat_config["upper_level_model"] = model
    boat_config["lower_level_var"] = list(model.parameters())
    boat_config["upper_level_var"] = list(model.arch_parameters())
    boat_config["lower_level_opt"] = optimizer
    boat_config["upper_level_opt"] = outer_opt
    b_optimizer = boat.Problem(boat_config, loss_config)
    b_optimizer.build_ll_solver()
    b_optimizer.build_ul_solver()

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = jit.dataset.CIFAR10(
        train=True,
        transform=train_transform,
        root=args.data,
    )

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_indices = indices[:split]
    valid_indices = indices[split:]

    # 自定义数据集类用于索引切片
    class SubsetDataset(Dataset):
        def __init__(self, dataset, indices, transform=None):
            super().__init__()
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            original_idx = self.indices[idx]
            img, target = self.dataset[original_idx]
            if self.transform:
                img = self.transform(img)
            return img, target
                
    class CustomBatchSampler:
        def __init__(self, dataset, batch_size, shuffle=True):
            """
            Custom BatchSampler to create batches of data.

            :param dataset: The dataset to sample from.
            :param batch_size: Number of samples per batch.
            :param shuffle: Whether to shuffle the dataset indices.
            """
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = list(range(len(dataset)))

        def __iter__(self):
            if self.shuffle:
                np.random.shuffle(self.indices)

            batch = []
            for idx in self.indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield self._collate_fn(batch)
                    batch = []

            # Drop the last incomplete batch if needed
            if batch:
                yield self._collate_fn(batch)

        def _collate_fn(self, batch_indices):
            """
            Collate function to create a batch of data.

            :param batch_indices: List of indices for the current batch.
            :return: Tuple (inputs, targets)
            """
            inputs, targets = [], []
            for idx in batch_indices:
                img, target = self.dataset[idx]
                inputs.append(img)
                targets.append(target)

            # Convert to numpy arrays
            inputs = np.stack(inputs)  # Shape: [batch_size, 3, 32, 32]
            targets = np.array(targets)  # Shape: [batch_size]
            return inputs, targets

        def __len__(self):
            return len(self.indices) // self.batch_size

    # 创建子数据集
    train_subset = SubsetDataset(train_data, train_indices, transform=train_transform)
    valid_subset = SubsetDataset(train_data, valid_indices, transform=valid_transform)
    train_queue = CustomBatchSampler(train_subset, batch_size=args.batch_size)
    valid_queue = CustomBatchSampler(valid_subset, batch_size=args.batch_size)
    scheduler = jit.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )
    average_runtime = 0
    for epoch in range(args.load_iter, args.epochs):
        if args.load_iter > 0:
            for j in range(args.load_iter - 1):
                scheduler.step()

        scheduler.step()
        # lr = scheduler.get_last_lr()
        # logging.info("epoch %d lr %e", epoch, lr[0])

        genotype = model.genotype()
        logging.info("genotype = %s", genotype)

        train_acc, train_obj, epoch_time = train(
            train_queue, valid_queue, model, criterion, optimizer, b_optimizer
        )

        logging.info("train_acc %f , train_obj %f ", train_acc, train_obj)
        utils.save(model, os.path.join(args.save, "weights.pt"))
        print("epoch_step_time:", epoch_time)
        utils.save(model, os.path.join(args.save, "weights.pt"))
    print("average_step_time", average_runtime / 3)


def train(train_queue, valid_queue, model, criterion, optimizer, boat_optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    time = utils.AvgrageMeter()
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.shape[0]
        input = jit.array(input, dtype='float32')
        target = jit.array(target,dtype='float32')
        input_search, target_search = next(iter(valid_queue))
        input_search = jit.array(input_search, dtype='float32')
        target_search = jit.array(target_search,dtype='float32')
        ul_feed_dict = {"data": input_search, "target": target_search}
        ll_feed_dict = {"data": input, "target": target}
        loss, run_time = boat_optimizer.run_iter(
            ll_feed_dict, ul_feed_dict, current_iter=step
        )
        jit.sync_all()
        jit.display_memory_info()
        logits = model(input)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        time.update(run_time, 1)
        print(step, " step_time: ", run_time)
        if step % args.report_freq == 0:
            logging.info(
                "train %03d loss: %e top1: %f top5: %f runtime: %f",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
                time.avg,
            )

        if step > 20:
            break
    boat_optimizer.plot_losses()
    return top1.avg, objs.avg, time.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = jit.array(input)
        target = jit.array(target)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.shape[0]
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info(
                "valid %03d loss: %e top1: %f top5: %f",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
            )

        if step > 200:
            break

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()