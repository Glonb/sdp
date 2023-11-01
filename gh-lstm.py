import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import utils
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from sklearn.utils.class_weight import compute_class_weight


# 检查 GPU 可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def my_loss(y_pred, y_true):
    # print(y_pred)
    margin = 0.6

    # Define theta function
    def theta(t):
        return (torch.sign(t) + 1) / 2

    # Compute the loss
    loss = -(
        (1 - theta(y_true - margin) * theta(y_pred - margin) 
        - theta(1 - margin - y_true) * theta(1 - margin - y_pred)) * 
        (y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8))
    )
    
    return loss.mean()  # You can use .mean() to compute the average loss
    

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyModel, self).__init__()

        # LSTM层
        self.sce_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.sce_dropout = nn.Dropout(p=0.2)
        self.promise_lstm = nn.LSTM(input_size=18, hidden_size=hidden_dim, batch_first=True)
        self.promise_dropout = nn.Dropout(p=0.2)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, 1)

        # 门控层
        self.sce_gate = nn.Linear(hidden_dim, 128)
        self.promise_gate = nn.Linear(hidden_dim, 128)

        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, sce_input, promise_input):
        # 处理sce_input序列数据
        sce_lstm_out, _ = self.sce_lstm(self.sce_dropout(sce_input))
        sce_lstm_out_last = sce_lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 应用门控机制
        sce_gate_output = self.sigmoid(self.sce_gate(sce_lstm_out_last))
        # print(sce_gate_output.shape)
        gated_sce_lstm_out = sce_lstm_out_last * sce_gate_output

        # 处理promise_input数据
        promise_lstm_out, _ = self.promise_lstm(self.promise_dropout(promise_input))
        promise_lstm_out_last = promise_lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 应用门控机制
        promise_gate_output = self.sigmoid(self.promise_gate(promise_lstm_out_last))
        # print(promise_gate_output.shape)
        gated_promise_lstm_out = promise_lstm_out_last * promise_gate_output

        # 合并两个部分
        merged = torch.cat((gated_sce_lstm_out, gated_promise_lstm_out), dim=-1)
        # print(merged.shape)

        # 全连接层
        fc_output = self.fc(merged)

        return self.sigmoid(fc_output)


train_data = MyDataset('/kaggle/input/sdp-own/poi25_train.pt', '/kaggle/input/sdp-own/poi25.csv')
test_data = MyDataset('/kaggle/input/sdp-own/poi30_test.pt', '/kaggle/input/sdp-own/poi30.csv')
df = pd.read_csv('/kaggle/input/sdp-own/poi25.csv')
labels = df["bug"]

class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
print(class_weights)

pos_weight = torch.tensor(class_weights[1] / class_weights[0])
print(pos_weight)

# 创建模型实例
model = MyModel(input_dim=40, hidden_dim=128).to(device)

# 定义损失函数
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = my_loss

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
epoch_count = 100
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# 训练模型和预测的过程需要根据你的数据和训练流程进行调整
for epoch in range(epoch_count):
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

    print(f'Epoch {epoch + 1}/{epoch_count}, Loss: {losses.avg:.3f}, Precision: {precision.avg:.3f}, Recall: {recall.avg:.3f}, F1 Score: {f_measure.avg:.3f}')

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

    print(f'Test Loss: {losses.avg:.3f}, Precision: {precision.avg:.3f}, Recall: {recall.avg:.3f}, F1 Score: {f_measure.avg:.3f}')
