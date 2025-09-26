from torch.nn import functional as F
import torch
from torch.nn.functional import mse_loss


def ul_loss(ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    loss = mse_loss(lower_model(ul_feed_dict["data"], **kwargs), ul_feed_dict["target"])
    return loss


def ll_loss(ll_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    out = mse_loss(lower_model(ll_feed_dict["data"], **kwargs), ll_feed_dict["target"])
    return out


def gda_loss(
    ll_feed_dict, ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs
):
    out = ll_feed_dict["alpha"] * mse_loss(
        lower_model(ul_feed_dict["data"], **kwargs), ul_feed_dict["target"]
    ) + (1 - ll_feed_dict["alpha"]) * mse_loss(
        lower_model(ll_feed_dict["data"], **kwargs), ll_feed_dict["target"]
    )
    return out
