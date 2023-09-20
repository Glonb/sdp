import  torch
from    torch import nn
import  torch.nn.functional as F
from    operations import OPS, Flatten
from    genotypes import PRIMITIVES, Genotype


class MixedLayer(nn.Module):
    def __init__(self, c, stride):
        
        super(MixedLayer, self).__init__()

        self.layers = nn.ModuleList()
    
        for primitive in PRIMITIVES:
            
            # create corresponding layer
            layer = OPS[primitive](c, stride, False)
            
            # append batchnorm after pool layer
            # if 'pool' in primitive:
            #     # disable affine w/b for batchnorm
            #     layer = nn.Sequential(layer, nn.BatchNorm1d(c, affine=False))

            self.layers.append(layer)

    def forward(self, x, weights):
        # for i,layer in enumerate(self.layers):
        #     print("after: " , i)
        #     print(layer(x).shape)
        res = [w * layer(x) for w, layer in zip(weights, self.layers)]
        
        # element-wise add by torch.add
        res = sum(res)
        
        return res
        

class Network(nn.Module):
    
    def __init__(self, c, steps, criterion):
        
        super(Network, self).__init__()

        self.c = c
        self.steps = steps 
        self.criterion = criterion
        
        self.layers = nn.ModuleList()

        for i in range(self.steps):
            
            # for each i, it connects with all previous output
            for j in range(1 + i):
                stride = 1
                layer = MixedLayer(c, stride)
                self.layers.append(layer)

        # adaptive pooling output
        self.global_pooling = nn.AdaptiveMaxPool1d(1)

        self.flatten = Flatten()
        
        self.classifier = nn.Linear(c * steps, 1)

        # k is the total number of edges
        k = sum(1 for i in range(self.steps) for j in range(1 + i))
        num_ops = len(PRIMITIVES) 

        self.alpha = nn.Parameter(torch.randn(k, num_ops))
        with torch.no_grad():
            # initialize to smaller value
            self.alpha.mul_(1e-3)
        self._arch_parameters = [self.alpha]

    def new(self):
        
        model_new = Network(self.c, self.steps, self.criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):

        states = [x]
        offset = 0
        
        # for each node, receive input from all previous intermediate nodes and x
        for i in range(self.steps):
            
            weights = F.softmax(self.alpha, dim=-1)
            s = sum(self.layers[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            
            # append one state since s is the elem-wise addition of all output
            states.append(s)

            # print('node:',i, s.shape)

        # concat along dim=channel
        res = torch.cat(states[1:], dim=1)

        out = self.global_pooling(res)
        
        logits = self.classifier(self.flatten(out))
        
        return logits

    def loss(self, x, target):
        logits = self(x)
        return self.criterion(logits, target.float())


    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self.steps): # for each node
                end = start + n
                W = weights[start:end].copy() # [1, 8], [2, 8], ...
                edges = sorted(range(i + 1), # i+1 is the number of connection for node i
                            key=lambda x: -max(W[x][k] # by descending order
                                               for k in range(len(W[x])) # get strongest ops
                                               if k != PRIMITIVES.index('none'))
                               )[:1] # only has one inputs
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

        gene = _parse(F.softmax(self.alpha, dim=-1).data.cpu().numpy())

        concat = range(1, self.steps + 1)
        genotype = Genotype(
            geno=gene, geno_concat=concat
        )

        return genotype
