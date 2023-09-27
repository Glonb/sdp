import  torch
import  torch.nn as nn
from    operations import *
from    utils import drop_path


class Network(nn.Module):

    def __init__(self, C, genotype):
        super(Network, self).__init__()
        hidden_size = 64

        op_names, indices = zip(*genotype.geno)
        concat = genotype.geno_concat
        self._compile(C, op_names, indices, concat)

        self.bilstm = nn.LSTM(input_size=C, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(C * len(op_names) + 2 * hidden_size, 1)

    def _compile(self, C, op_names, indices, concat):
        
        assert len(op_names) == len(indices)

        self._steps = len(op_names)
        self._concat = concat

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, input):
        
        states = [input]
        
        for i in range(self._steps):
            h = states[self._indices[i]]
            op = self._ops[i]
            h = op(h)
            states += [h]
            
        cnn_out = torch.cat([states[i] for i in self._concat], dim=1)
        cnn_out = self.global_pooling(cnn_out)

        bilstm_out, (h_n, c_n) = self.bilstm(input)
        bilstm_out = self.global_pooling(bilstm_out)

        out = torch.cat(cnn_out, bilstm_out, dim=1)
        
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
