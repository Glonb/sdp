import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from    operations import *


class Network(nn.Module):

    def __init__(self, C, hidden_size, dropout_prob, genotype):
        super(Network, self).__init__()
        
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        out_dim = C + 2 * self.hidden_size

        op_names, indices = zip(*genotype.geno)
        concat = genotype.geno_concat
        self._compile(C, op_names, indices, concat)

        self.tr_gru = nn.GRU(input_size=20, hidden_size=2*self.hidden_size, batch_first=True)
        self.tr_dropout = nn.Dropout(self.dropout_prob)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        
        self.gate = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(out_dim, 1)

    def _compile(self, C, op_names, indices, concat):
        
        assert len(op_names) == len(indices)

        self._steps = len(op_names)
        self._concat = concat

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C)
            self._ops += [op]
            C //= 2
        self._indices = indices

    def forward(self, x, trf):
        input = x.permute(0, 2, 1)
        trf = trf.unsqueeze(1)
        states = [x]
        
        for i in range(self._steps):
            h = states[self._indices[i]]
            op = self._ops[i]
            h = op(h)
            states += [h]

        cnn_out = self.global_pooling(states[-1])
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        trf_out, _ = self.tr_gru(self.tr_dropout(trf))
        trf_out = trf_out[:, -1, :]
        
        out = torch.cat([cnn_out, trf_out], dim=-1)
        cat_gate_out = self.sigmoid(self.gate(out))
        out = cat_gate_out * out
        logits = self.fc(out)
        
        return logits
