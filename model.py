import  torch
import  torch.nn as nn
from    operations import *


class Network(nn.Module):

    def __init__(self, C, dropout_prob, vocab_size, genotype):
        super(Network, self).__init__()
        self.dropout_prob = dropout_prob
        self.vocab_size = vocab_size
        hidden_size = 64

        op_names, indices = zip(*genotype.geno)
        concat = genotype.geno_concat
        self._compile(C, op_names, indices, concat)

        self.embed = nn.Embedding(vocab_size, C)
        self.bilstm = nn.LSTM(input_size=C, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        
        # out_dim = C * len(op_names) + 2 * hidden_size
        out_dim = C + 2 * hidden_size
        hidden_dim = out_dim // 2
        self.fc1 = nn.Linear(out_dim, hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def _compile(self, C, op_names, indices, concat):
        
        assert len(op_names) == len(indices)

        self._steps = len(op_names)
        self._concat = concat

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride)
            self._ops += [op]
        self._indices = indices

    def forward(self, x):

        x = self.embed(x)
        input = x.permute(0, 2, 1)
        states = [input]
        
        for i in range(self._steps):
            h = states[self._indices[i]]
            op = self._ops[i]
            h = op(h)
            states += [h]
            
        # cnn_out = torch.cat([states[i] for i in self._concat], dim=1)
        cnn_out = states[-1]
        cnn_out = self.global_pooling(cnn_out)

        bilstm_out, (h_n, c_n) = self.bilstm(x)
        bilstm_out = bilstm_out.transpose(1, 2)
        bilstm_out = self.global_pooling(bilstm_out)

        out = torch.cat([cnn_out, bilstm_out], dim=1)
        out = out.view(out.size(0), -1)
        fc1_out = self.fc1(out)    
        out_drop = self.dropout(fc1_out)
        logits = self.fc2(out_drop)
        return torch.sigmoid(logits).view(-1)
