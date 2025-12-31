from torch import nn
import math


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


def initialize(net):
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
            m.bias.data.zero_()
    return net
