import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import boat_torch as boat
from torch.autograd import Variable
from model_search import Network

parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="/data", help="location of the data corpus"
)
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.025, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.001, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument(
    "--epochs", type=int, default=1, help="num of training epochs"
)  # default=50
parser.add_argument(
    "--init_channels", type=int, default=16, help="num of init channels"
)
parser.add_argument("--layers", type=int, default=3, help="total number of layers")
parser.add_argument(
    "--model_path", type=str, default="saved_models", help="path to save the model"
)
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument(
    "--load", action="store_true", default=False, help="Whether to load model"
)
parser.add_argument(
    "--load_iter", type=int, default=0, help="the itrations to load the model"
)
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training data"
)
parser.add_argument(
    "--inner_loop", type=int, default=1, help="number of inner loops for IAPTT"
)
parser.add_argument(
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=3e-4,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay",
    type=float,
    default=1e-3,
    help="weight decay for arch encoding",
)
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
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    if args.load:
        utils.load(model, args.model_path)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    outer_opt = torch.optim.Adam(
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
    train_data = dset.CIFAR10(
        root=args.data, train=True, download=True, transform=train_transform
    )

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=0,
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )
    average_runtime = 0
    for epoch in range(args.load_iter, args.epochs):
        if args.load_iter > 0:
            for j in range(args.load_iter - 1):
                scheduler.step()

        scheduler.step()
        lr = scheduler.get_last_lr()
        logging.info("epoch %d lr %e", epoch, lr[0])

        genotype = model.genotype()
        logging.info("genotype = %s", genotype)

        train_acc, train_obj, epoch_time = train(
            train_queue, valid_queue, model, criterion, optimizer, b_optimizer
        )

        logging.info("train_acc %f , train_obj %f ", train_acc, train_obj)
        # validation
        # valid_acc, valid_obj = infer(valid_queue, model, criterion)
        # logging.info('valid_acc %f, valid_obj %f', valid_acc, valid_obj)
        average_runtime += epoch_time
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
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(
            target_search, requires_grad=False
        ).cuda()  # cuda(async=True)
        ul_feed_dict = {"data": input_search, "target": target_search}
        ll_feed_dict = {"data": input, "target": target}
        loss, run_time = boat_optimizer.run_iter(
            ll_feed_dict, ul_feed_dict, current_iter=step
        )
        print(step, ' step_run_time', run_time)
        logits = model(input)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        time.update(run_time, n)
        if step % args.report_freq == 0:
            logging.info(
                "train %03d loss: %e top1: %f top5: %f runtime: %f",
                step,
                objs.avg,
                top1.avg,
                top5.avg,
                time.avg,
            )

        if step > 200:
            break
    return top1.avg, objs.avg, time.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
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
