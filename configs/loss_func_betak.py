from torch.nn import functional as F
import torch
from torch import nn

criterion = nn.CrossEntropyLoss().cuda()


def ul_loss(ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    loss = 0
    for model_i in ul_feed_dict["model"]:
        loss += nn.CrossEntropyLoss()(
            model_i(lower_model(**kwargs)), ul_feed_dict["label"]
        )
    return loss


def ll_loss(ll_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    out = -nn.CrossEntropyLoss()(
        ll_feed_dict["model"](lower_model(**kwargs)), ll_feed_dict["label"]
    )
    return out
