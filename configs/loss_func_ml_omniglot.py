from torch.nn import functional as F
import torch
from torch.nn.functional import mse_loss


def ul_loss(ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    loss = F.cross_entropy(
        lower_model(ul_feed_dict["data"], **kwargs), ul_feed_dict["target"]
    )
    return loss


def ll_loss(ll_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    out = F.cross_entropy(
        lower_model(ll_feed_dict["data"], **kwargs), ll_feed_dict["target"]
    )
    return out
