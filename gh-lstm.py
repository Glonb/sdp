import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from my_dataset import MyDataset


def metrics(output, target, threshold = 0.5):

    pred = (output > threshold).long()
    
    # Compute True Positives, False Positives, False Negatives, True Negatives
    tp = (pred == target) & (target == 1)
    fp = (pred == 1) & (target == 0)
    fn = (pred == 0) & (target == 1)
    tn = (pred == target) & (target == 0)
    
    # accuracy = (tp.sum() + tn.sum()) / (tp.sum() + fp.sum() + fn.sum() + tn.sum() + 1e-10)
    precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
    recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
    fpr = fp.sum() / (fp.sum() + tn.sum() + 1e-10)
    fnr = fn.sum() / (fn.sum() + tn.sum() + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    g1 = 2 * recall * (1 - fpr) / (recall - fpr + 1)
    mcc = (tp.sum() * tn.sum() - fp.sum() * fn.sum()) / (torch.sqrt((tp.sum() + fp.sum()) * (tp.sum() + fn.sum()) * (tn.sum() + fp.sum()) * (tn.sum() + fn.sum()))+ 1e-10)
    
    return  precision, recall, fpr, fnr, f1, g1, mcc


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyModel, self).__init__()

        # LSTM层
        self.sce_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, dropout=0.2, batch_first=True)
        self.promise_lstm = nn.LSTM(input_size=18, hidden_size=hidden_dim, dropout=0.2, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(input_dim * 2, 1)

        # 门控层
        self.sce_gate = nn.Linear(hidden_dim, 128)
        self.promise_gate = nn.Linear(hidden_dim, 128)

        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, sce_input, promise_input):
        # 处理sce_input序列数据
        lstm_out, _ = self.sce_lstm(sce_input)
        lstm_out_last = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 应用门控机制
        gate_output = self.sigmoid(self.sce_gate(lstm_out_last))
        print(gate_output.shape)
        # gated_lstm_out = lstm_out * gate_output.unsqueeze(1)

        # 处理promise_input数据
        promise_lstm_out, _ = self.promise_lstm(promise_input)
        promise_lstm_out_last = promise_lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 应用门控机制
        promise_gate_output = self.sigmoid(self.promise_gate(promise_lstm_out_last))
        print(promise_gate_output.shape)
        # gated_promise_lstm_out = promise_lstm_out * promise_gate_output.unsqueeze(1)

        # 合并两个部分
        merged = torch.cat((gate_output, promise_gate_output), dim=1)
        print(merged.shape)

        # 全连接层
        fc_output = self.fc(merged)

        return fc_output


# 创建模型实例
model = MyModel(input_dim=40, hidden_dim=128)

# 定义损失函数
pos_weight = torch.tensor(2.0)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = MyDataset('/kaggle/input/sdp-own/ant16_train.pt', '/kaggle/input/sdp-own/ant16.csv')

dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

# 训练模型和预测的过程需要根据你的数据和训练流程进行调整
for epoch in range(200):
    model.train()
    lr = optimizer.param_groups[0]['lr']
    print('Epoch: %d' % epoch)
    total_loss = 0.0

    for i, (emb_data, tr_data, label) in enumerate(dataloader):

        optimizer.zero_grad()
        sce = emb_data.permute(0, 2, 1)
        trf = tr_data.unsqueeze(1)
        output = model(sce, trf)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/ {200}, Loss: {total_loss / len(dataloader)}')



