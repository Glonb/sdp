import  torch
import  torch.nn as nn
from    operations import *


class Network(nn.Module):

    def __init__(self, C, hidden_size, genotype):
        super(Network, self).__init__()
        
        self.hidden_size = hidden_size
        out_dim = C + 2 * hidden_size

        op_names, indices = zip(*genotype.geno)
        concat = genotype.geno_concat
        self._compile(C, op_names, indices, concat)

        self.bilstm = nn.LSTM(input_size=C, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(out_dim, 1)

    def _compile(self, C, op_names, indices, concat):
        
        assert len(op_names) == len(indices)

        self._steps = len(op_names)
        self._concat = concat

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C)
            self._ops += [op]
        self._indices = indices

    def forward(self, x):
        
        input = x.permute(0, 2, 1)
        states = [x]
        
        for i in range(self._steps):
            h = states[self._indices[i]]
            op = self._ops[i]
            h = op(h)
            states += [h]
            
        # cnn_out = torch.cat(states[-2:], dim=1)
        
        cnn_out = states[-1]
        cnn_out = self.global_pooling(cnn_out)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)

        bilstm_out, (h_n, c_n) = self.bilstm(input)
        forward_state, backward_state = h_n[0], h_n[1]

        combined_state = torch.cat((forward_state, backward_state), dim=-1)
        print(combined_state.shape)
        bilstm_out = combined_state.permute(1, 0, 2)
        bilstm_out = bilstm_out.view(bilstm_out.size(0), -1)

        out = torch.cat([cnn_out, bilstm_out], dim=-1)
        logits = self.fc(out)
        
        return torch.sigmoid(logits)
