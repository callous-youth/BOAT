from operations import *
import jittor as jit
from jittor import nn
from operations import *  
from genotypes import PRIMITIVES, Genotype 
import collections


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = []
        k=0
        for primitive in PRIMITIVES:
            k+=1
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                op = nn.Sequential(op, nn.BatchNorm(C, affine=False))
            self._ops.append(op)

    def execute(self, x, weights):
        # Initialize a variable to accumulate the sum
        result = 0

        # Loop over the weights and operations in self._ops
        for w, op in zip(weights, self._ops):
            # Compute the product of weight and the output of op(x)
            product = w * op(x)

            # Add the product to the result
            result += product

        # Return the final accumulated result
        return result


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)

                self._ops.append(op)

    def execute(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return jit.concat(states[-self._multiplier:], dim=1)


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
        """依次执行子模块"""
        for layer in self.layers.values():
            x = layer(x)
        return x

    def add_module(self, name, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(name)]=mod
        super().add_module(str(name), mod)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers_num, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers_num = layers_num
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C

        self.stem = MySequential('stem', nn.Conv(3, C_curr, 3, padding=1, bias=False),nn.BatchNorm(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = []
        reduction_prev = False
        for i in range(layers_num):
            if i in [layers_num // 3, 2 * layers_num // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers_num, self._criterion)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def execute(self, input):
        s0 = s1 = self.stem(input)


        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = jit.nn.softmax(self.hidden_params['alphas_reduce'], dim=-1)
            else:
                weights = jit.nn.softmax(self.hidden_params['alphas_normal'], dim=-1)

            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
       
        mid_out = out.view(out.size(0), -1)

        logits = self.classifier(mid_out)

        
        return logits


    def _loss(self, input, target):
        logits = self.execute(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.hidden_params = {
            "alphas_normal": jit.array(1e-3 * jit.randn(k, num_ops)).start_grad(),
            'alphas_reduce':jit.array(1e-3 * jit.randn(k, num_ops)).start_grad()
        }

        self._arch_parameters = list(self.hidden_params.values())
        
    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k]
                        for k in range(len(W[x]))
                        if k != PRIMITIVES.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index("none"):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(jit.nn.softmax(self.hidden_params['alphas_normal'], dim=-1).data)
        gene_reduce = _parse(jit.nn.softmax(self.hidden_params['alphas_reduce'], dim=-1).data)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype