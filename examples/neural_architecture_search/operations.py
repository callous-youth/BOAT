import jittor as jit
from jittor import nn
import collections

class MySequential(nn.Module):
    def __init__(self, name='sequential',*args):
        super().__init__()
        self.layers = collections.OrderedDict()

        
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(name+'_'+str(idx), module)

    def execute(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def append(self, module):
        name = str(len(self.layers))
        self.add_module(name, module)



    def add_module(self, name, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(name)]=mod
        super().add_module(str(name), mod)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.layers):
                raise IndexError(f"Index {idx} is out of range")
            return list(self.layers.values())[idx]
        elif isinstance(idx, str):
            if idx not in self.layers:
                raise KeyError(f"Module {idx} not found")
            return self.layers[idx]
        else:
            raise TypeError("Index must be an integer or string")

    def __iter__(self):
        return iter(self.layers.values())

    def __len__(self):
        return len(self.layers)

    def named_children(self):
        return list(self.layers.items())

    def keys(self):
        return self.layers.keys()

    def values(self):
        return self.layers.values()

    def items(self):
        return self.layers.items()

    def __getattr__(self, key):
        if key in self.layers:
            return self.layers[key]
        return super().__getattr__(key)




OPS = {
    "none": lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3": lambda C, stride, affine: nn.Pool(
        3, stride=stride, padding=1, count_include_pad=False, op="mean"
    ),
    "max_pool_3x3": lambda C, stride, affine: nn.Pool(3, stride=stride, padding=1, op="maximum"),
    "skip_connect": lambda C, stride, affine: (
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine)
    ),
    "sep_conv_3x3": lambda C, stride, affine: SepConv(
        C, C, 3, stride, 1, affine=affine
    ),
    "sep_conv_5x5": lambda C, stride, affine: SepConv(
        C, C, 5, stride, 2, affine=affine
    ),
    "sep_conv_7x7": lambda C, stride, affine: SepConv(
        C, C, 7, stride, 3, affine=affine
    ),
    "dil_conv_3x3": lambda C, stride, affine: DilConv(
        C, C, 3, stride, 2, 2, affine=affine
    ),
    "dil_conv_5x5": lambda C, stride, affine: DilConv(
        C, C, 5, stride, 4, 2, affine=affine
    ),
    "conv_7x1_1x7": lambda C, stride, affine: nn.Sequential(
    nn.ReLU(), nn.Conv(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm(C, affine=affine),),
}


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(),nn.Conv(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm(C_out, affine=affine),)

    def execute(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()

        self.op = nn.Sequential(nn.ReLU(), 
                       nn.Conv( C_in,C_in,kernel_size=kernel_size,stride=stride, padding=padding,dilation=dilation,groups=C_in,bias=False,),
                       nn.Conv(C_in, C_out, kernel_size=1, padding=0, bias=False),
                       nn.BatchNorm(C_out, affine=affine),)


    def execute(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()

        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm(C_in, affine=affine),
            nn.ReLU(),
            nn.Conv(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=C_in,
                bias=False,
            ),
            nn.Conv(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm(C_out, affine=affine),
        )

    def execute(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def execute(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def execute(self, x):
        if self.stride == 1:
            return x * 0.0
        return x[:, :, :: self.stride, :: self.stride] * 0.0


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm(C_out, affine=affine)

    def execute(self, x):
        x = self.relu(x)
        out = jit.contrib.concat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out