from torch.nn import functional as F
import torch
from torch.nn.functional import mse_loss


def ul_loss(ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    y = ul_feed_dict["data"].mm(lower_model(**kwargs))
    loss = F.cross_entropy(y, ul_feed_dict["target"], reduction="mean")
    return loss


def ll_loss(ll_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    y = ll_feed_dict["data"].mm(lower_model(**kwargs))
    loss = F.cross_entropy(y, ll_feed_dict["target"], reduction="mean")
    reg_loss = (
        0.5 * (lower_model(**kwargs).pow(2) * upper_model().view(-1, 1).exp()).mean()
    )  # l2 reg loss
    return loss + reg_loss


def gda_loss(
    ll_feed_dict, ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs
):
    y_val = ul_feed_dict["data"].mm(lower_model(**kwargs))
    loss_val = F.cross_entropy(y_val, ul_feed_dict["target"], reduction="mean")
    y_tr = ll_feed_dict["data"].mm(lower_model(**kwargs))
    loss = F.cross_entropy(y_tr, ll_feed_dict["target"], reduction="mean")
    reg_loss = (
        0.5 * (lower_model(**kwargs).pow(2) * upper_model().view(-1, 1).exp()).mean()
    )  # l2 reg loss
    out = (
        ll_feed_dict["alpha"] * (loss + reg_loss)
        + (1 - ll_feed_dict["alpha"]) * loss_val
    )
    return out
