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
            
            # create corresponding layer
            layer = OPS[primitive](c)
            self.layers.append(layer)

    def forward(self, x, weights):
        max_length = x.size(-1)
        # for i,layer in enumerate(self.layers):
        #     print(layer(x).shape)
        out = [w * layer(x) for w, layer in zip(weights, self.layers)]

        # max_length = max(tensor.size(-1) for tensor in out)
        padded_tensors = [F.pad(tensor, (0, max_length - tensor.size(-1))) for tensor in out]
        output = sum(padded_tensors)
        
        return output
        

class Network(nn.Module):
    
    def __init__(self, c, steps, hidden_size, criterion):
        
        super(Network, self).__init__()

        self.c = c
        self.steps = steps 
        self.hidden_size = hidden_size
        self.criterion = criterion
        
        # out_dim = c * 2 + 2 * hidden_size + 48
        out_dim = 128
        
        self.layers = nn.ModuleList()

        for i in range(self.steps):
            
            # for each i, it connects with all previous output
            for j in range(1 + i):
                layer = MixedLayer(c)
                self.layers.append(layer)

        # self.bilstm = nn.LSTM(input_size=self.c, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(input_size=self.c, hidden_size=64, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.tr_gru = nn.GRU(input_size=20, hidden_size=24, batch_first=True)
        self.tr_dropout = nn.Dropout(0.2)
        
        # adaptive pooling output
        self.global_pooling = nn.AdaptiveMaxPool1d(1)

        # self.gate = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(out_dim, 1)

        # k is the total number of edges
        k = sum(1 for i in range(self.steps) for j in range(1 + i))
        num_ops = len(PRIMITIVES) 

        self.alpha = nn.Parameter(torch.randn(k, num_ops))
        with torch.no_grad():
            # initialize to smaller value
            self.alpha.mul_(1e-3)
        self._arch_parameters = [self.alpha,]

    def new(self):
        
        model_new = Network(self.c, self.steps, self.hidden_size, self.criterion).cuda()
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

        # pooled_states = [self.global_pooling(h) for h in states[-2:]]
        # cnn_out = torch.cat(pooled_states, dim=-1)
        
        cnn_out = self.global_pooling(states[-1])
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        # print(cnn_out.shape)
        
        # bl_out, (h_n, c_n) = self.bilstm(input)
        # bilstm_out = torch.cat((h_n[0], h_n[1]), dim=-1) 
        # print(bilstm_out.shape)
        
        gru_out, _ = self.gru(self.dropout(input))
        gru_out = gru_out[:, -1, :]
        # gru_out = torch.cat((h_n[0], h_n[1]), dim=-1)

        trf_out, _ = self.tr_gru(self.tr_dropout(trf))
        trf_out = trf_out[:, -1, :]
        # trf_gate_out = self.sigmoid(self.tr_gate(trf_out))
        # trf_out = trf_out * trf_gate_out
        
        out = torch.cat([cnn_out, gru_out, trf_out], dim=-1)
        # cat_gate_out = self.sigmoid(self.gate(out))
        # cat_out = cat_gate_out * out
        # print(out.shape)
        
        logits = self.fc(out)
        output = self.sigmoid(logits)
        
        return output

    def loss(self, x, trf, target):
        logits = self(x, trf)
        # print(logits.shape)
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
