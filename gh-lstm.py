import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torch.utils.data import DataLoader
from my_dataset import MyDataset


# 检查 GPU 可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyModel, self).__init__()

        # LSTM层
        self.sce_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, dropout=0.2, batch_first=True)
        self.promise_lstm = nn.LSTM(input_size=18, hidden_size=hidden_dim, dropout=0.2, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, 1)

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
        # print(gate_output.shape)
        # gated_lstm_out = lstm_out * gate_output.unsqueeze(1)

        # 处理promise_input数据
        promise_lstm_out, _ = self.promise_lstm(promise_input)
        promise_lstm_out_last = promise_lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 应用门控机制
        promise_gate_output = self.sigmoid(self.promise_gate(promise_lstm_out_last))
        # print(promise_gate_output.shape)
        # gated_promise_lstm_out = promise_lstm_out * promise_gate_output.unsqueeze(1)

        # 合并两个部分
        merged = torch.cat((gate_output, promise_gate_output), dim=1)
        # print(merged.shape)

        # 全连接层
        fc_output = self.fc(merged)

        return fc_output


# 创建模型实例
model = MyModel(input_dim=40, hidden_dim=128).to(device)

# 定义损失函数
pos_weight = torch.tensor(2.0)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_data = MyDataset('/kaggle/input/sdp-own/poi25_train.pt', '/kaggle/input/sdp-own/poi25.csv')
test_data = MyDataset('/kaggle/input/sdp-own/poi30_test.pt', '/kaggle/input/sdp-own/poi30.csv')

batch_size = 512
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 训练模型和预测的过程需要根据你的数据和训练流程进行调整
for epoch in range(200):
    model.train()
    # lr = optimizer.param_groups[0]['lr']
    losses = utils.AverageMeter()
    precision = utils.AverageMeter()
    recall = utils.AverageMeter()
    f_measure = utils.AverageMeter()

    for i, (emb_data, tr_data, label) in enumerate(train_dataloader):
        emb_data = emb_data.to(device)
        tr_data = tr_data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        sce = emb_data.permute(0, 2, 1)
        trf = tr_data.unsqueeze(1)
        output = model(sce, trf)
        loss = criterion(output, label.float())

        loss.backward()
        optimizer.step()

        prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(output, label)
        losses.update(loss.item(), batch_size)
        precision.update(prec, batch_size)
        recall.update(rec, batch_size)
        f_measure.update(f1, batch_size)

    print(f'Epoch {epoch + 1}/{200}, Loss: {losses.avg}, Precision: {precision.avg}, Recall: {recall.avg}, F1 Score: {f_measure.avg}')

model.eval
with torch.no_grad():
    losses = utils.AverageMeter()
    precision = utils.AverageMeter()
    recall = utils.AverageMeter()
    f_measure = utils.AverageMeter()

    for i, (emb_data, tr_data, label) in enumerate(test_dataloader):
        emb_data = emb_data.to(device)
        tr_data = tr_data.to(device)
        label = label.to(device)

        sce = emb_data.permute(0, 2, 1)
        trf = tr_data.unsqueeze(1)
        output = model(sce, trf)
        loss = criterion(output, label.float())

        prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(output, label)
        losses.update(loss.item(), batch_size)
        precision.update(prec, batch_size)
        recall.update(rec, batch_size)
        f_measure.update(f1, batch_size)

    print(f'Test Loss: {losses.avg}, Precision: {precision.avg}, Recall: {recall.avg}, F1 Score: {f_measure.avg}')
