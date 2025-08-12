from torch.nn import functional as F


def ul_loss(ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    loss = F.cross_entropy(
        lower_model(ul_feed_dict["data"], **kwargs), ul_feed_dict["target"]
    )
    return loss


def ll_loss(ll_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    out = upper_model(
        F.cross_entropy(
            lower_model(ll_feed_dict["data"], **kwargs),
            ll_feed_dict["target"],
            reduction="none",
        )
    )
    return out


def gda_loss(
    ll_feed_dict, ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs
):
    out = ll_feed_dict["alpha"] * F.cross_entropy(
        lower_model(ul_feed_dict["data"], **kwargs), ul_feed_dict["target"]
    ) + (1 - ll_feed_dict["alpha"]) * upper_model(
        F.cross_entropy(
            lower_model(ll_feed_dict["data"], **kwargs),
            ll_feed_dict["target"],
            reduction="none",
        )
    )
    return out
