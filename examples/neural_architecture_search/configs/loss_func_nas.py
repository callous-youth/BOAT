

def ul_loss(ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    loss = lower_model._criterion(
        lower_model(ul_feed_dict["data"], **kwargs), ul_feed_dict["target"]
    )
    return loss


def ll_loss(ll_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    out = lower_model._criterion(
        lower_model(ll_feed_dict["data"], **kwargs), ll_feed_dict["target"]
    )
    return out


def gda_loss(ll_feed_dict, ul_feed_dict, upper_model, lower_model, weights=0.0, **kwargs):
    loss = lower_model._criterion(
        lower_model(ll_feed_dict["data"], **kwargs), ll_feed_dict["target"]
    )
    loss_val=lower_model._criterion(
        lower_model(ul_feed_dict["data"], **kwargs), ul_feed_dict["target"]
    )
    out = ll_feed_dict["alpha"] * loss + (1 - ll_feed_dict["alpha"]) * loss_val
    return out
