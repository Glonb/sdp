import  torch
from    torch import nn
import  torch.nn.functional as F
from    operations import *
from    genotypes import PRIMITIVES, Genotype


class MixedLayer(nn.Module):
    def __init__(self, c):
        
        super(MixedLayer, self).__init__()

        self.layers = nn.ModuleList()
    
        for primitive in PRIMITIVES:
            
            layer = OPS[primitive](c)
            self.layers.append(layer)

    def forward(self, x, weights):
        max_length = x.size(-1)
        out = [w * layer(x) for w, layer in zip(weights, self.layers)]

        padded_tensors = [F.pad(tensor, (0, max_length - tensor.size(-1))) for tensor in out]
        output = sum(padded_tensors)
        
        return output
        

class Network(nn.Module):
    
    def __init__(self, c, steps, hidden_size, dropout_prob, criterion):
        
        super(Network, self).__init__()

        self.c = c
        self.steps = steps 
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.criterion = criterion
        
        out_dim = self.c + 2 * self.hidden_size
        
        self.layers = nn.ModuleList()

        for i in range(self.steps):
            
            # for each i, it connects with all previous output
            for j in range(1 + i):
                layer = MixedLayer(self.c)
                self.layers.append(layer)

            self.c //= 2

        self.tr_gru = nn.GRU(input_size=20, hidden_size=2*self.hidden_size, batch_first=True)
        self.tr_dropout = nn.Dropout(self.dropout_prob)
        
        # adaptive pooling output
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        
        self.gate = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(out_dim, 1)

        # k is the total number of edges
        k = sum(1 for i in range(self.steps) for j in range(1 + i))
        num_ops = len(PRIMITIVES) 

        self.alpha = nn.Parameter(torch.randn(k, num_ops))
        with torch.no_grad():
            # initialize to smaller value
            self.alpha.mul_(1e-3)
        self._arch_parameters = [self.alpha]

    def new(self):
        
        model_new = Network(self.c, self.steps, self.hidden_size, self.dropout_prob, self.criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x, trf):
        input = x.permute(0, 2, 1)
        trf = trf.unsqueeze(1)
        states = [x]
        offset = 0
        
        # for each node, receive input from all previous intermediate nodes and x
        for i in range(self.steps):
            
            weights = F.softmax(self.alpha, dim=-1)
            s = sum(self.layers[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            
            # append one state since s is the elem-wise addition of all output
            states.append(s)
        
        cnn_out = self.global_pooling(states[-1])
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        trf_out, _ = self.tr_gru(self.tr_dropout(trf))
        trf_out = trf_out[:, -1, :]
        
        out = torch.cat([cnn_out, trf_out], dim=-1)
        cat_gate_out = self.sigmoid(self.gate(out))
        out = cat_gate_out * out
        
        logits = self.fc(out)
        
        return logits

    def loss(self, x, trf, target):
        logits = self(x, trf)
        return self.criterion(logits, target.float())


    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self.steps): # for each node
                idx = 1
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 1), # i+1 is the number of connection for node i
                            key=lambda x: -max(W[x][k] # by descending order
                                               for k in range(len(W[x])) )# get strongest ops
                               )[:idx] # select inputs
                for j in edges: # for every input nodes j of current node i
                    k_best = None
                    for k in range(len(W[j])): # get strongest ops for current input j->i
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
