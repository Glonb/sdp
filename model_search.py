import  torch
from    torch import nn
import  torch.nn.functional as F
from    operations import OPS, FactorizedReduce, ReLUConvBN
from    genotypes import PRIMITIVES, Genotype


class MixedLayer(nn.Module):
    def __init__(self, c, stride):
        
        super(MixedLayer, self).__init__()

        self.layers = nn.ModuleList()
    
        for primitive in PRIMITIVES:
            # create corresponding layer
            layer = OPS[primitive](c, stride, False)
            # append batchnorm after pool layer
            if 'pool' in primitive:
                # disable affine w/b for batchnorm
                layer = nn.Sequential(layer, nn.BatchNorm1d(c, affine=False))

            self.layers.append(layer)

    def forward(self, x, weights):
        for i,layer in enumerate(self.layers):
            print("after: " , i)
            print(layer(x).shape)
        res = [w * layer(x) for w, layer in zip(weights, self.layers)]
        # element-wise add by torch.add
        res = sum(res)
        return res
        

class Cell(nn.Module):

    def __init__(self, steps, multiplier, cpp, cp, c, reduction, reduction_prev):
       
        super(Cell, self).__init__()

        # indicating current cell is reduction or not
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        # preprocess0 deal with output from prev_prev cell
        if reduction_prev:
            # if prev cell has reduced channel/double width,
            # it will reduce width by half
            self.preprocess0 = FactorizedReduce(cpp, c, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(cpp, c, 1, 1, 0, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ReLUConvBN(cp, c, 1, 1, 0, affine=False)

        self.steps = steps # 4
        self.multiplier = multiplier # 4

        self.layers = nn.ModuleList()

        for i in range(self.steps):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            for j in range(2 + i):
                # for reduction cell, it will reduce the heading 2 inputs only
                stride = 2 if reduction and j < 2 else 1
                layer = MixedLayer(c, stride)
                self.layers.append(layer)

    def forward(self, s0, s1, weights):
       
        print('s0:', s0.shape,end='=>')
        s0 = self.preprocess0(s0) 
        print(s0.shape, self.reduction_prev)
        print('s1:', s1.shape,end='=>')
        s1 = self.preprocess1(s1) 
        print(s1.shape)

        states = [s0, s1]
        offset = 0
        # for each node, receive input from all previous intermediate nodes and s0, s1
        for i in range(self.steps): # 4
            
            s = sum(self.layers[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            # append one state since s is the elem-wise addition of all output
            states.append(s)
            print('node:',i, s.shape, self.reduction)

        # concat along dim=channel
        return torch.cat(states[-self.multiplier:], dim=1) # 6 of [40, 16, 32, 32]


class Network(nn.Module):
    """
    stack number:layer of cells and then flatten to fed a linear layer
    """
    def __init__(self, c, num_classes, layers, criterion, steps=4, multiplier=4):
        
        super(Network, self).__init__()

        self.c = c
        self.num_classes = num_classes
        self.layers = layers
        self.criterion = criterion
        self.steps = steps
        self.multiplier = multiplier

        cpp, cp, c_curr = c * multiplier, c * multiplier, c
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):

            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(steps, multiplier, cpp, cp, c_curr, reduction, reduction_prev)
            print('cell:',i, cpp, cp, c_curr, cell.reduction, cell.reduction_prev)
            print('\n')
            # update reduction_prev
            reduction_prev = reduction

            self.cells += [cell]

            cpp, cp = cp, multiplier * c_curr

        # adaptive pooling output size to 1x1
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        # since cp records last cell's output channels
        # it indicates the input channel number
        self.classifier = nn.Linear(cp, num_classes)

        # k is the total number of edges inside single cell, 14
        k = sum(1 for i in range(self.steps) for j in range(2 + i))
        num_ops = len(PRIMITIVES) # 8

        self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))
        self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops))
        with torch.no_grad():
            # initialize to smaller value
            self.alpha_normal.mul_(1e-3)
            self.alpha_reduce.mul_(1e-3)
        self._arch_parameters = [
            self.alpha_normal,
            self.alpha_reduce,
        ]

    def new(self):
        
        model_new = Network(self.c, self.num_classes, self.layers, self.criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        s0 = s1 = x

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alpha_reduce, dim=-1)
            else:
                weights = F.softmax(self.alpha_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
            print('cell:',i, s1.shape, cell.reduction, cell.reduction_prev)
            print('\n')

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits

    def loss(self, x, target):
        logits = self(x)
        return self.criterion(logits, target)


    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.steps): # for each node
                end = start + n
                W = weights[start:end].copy() # [2, 8], [3, 8], ...
                edges = sorted(range(i + 2), # i+2 is the number of connection for node i
                            key=lambda x: -max(W[x][k] # by descending order
                                               for k in range(len(W[x])) # get strongest ops
                                               if k != PRIMITIVES.index('none'))
                               )[:2] # only has two inputs
                for j in edges: # for every input nodes j of current node i
                    k_best = None
                    for k in range(len(W[j])): # get strongest ops for current input j->i
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j)) # save ops and input node
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self.steps - self.multiplier, self.steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

        return genotype
