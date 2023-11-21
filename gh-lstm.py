import    torch
import    torch.nn as nn
import    torch.optim as optim
import    argparse
import    pandas as pd
import    logging
import    time,os,sys
import    utils
from      torch.utils.data import DataLoader
from      my_dataset import MyDataset


parser = argparse.ArgumentParser("GH-LSTM")
parser.add_argument('--train_data', type=str, default='ant-1.5', help='train dataset')
parser.add_argument('--test_data', type=str, default='ant-1.6', help='test dataset')
parser.add_argument('--input_dim', type=int, default=40, help='input dim')
parser.add_argument('--batchsz', type=int, default=2048, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--log_path', type=str, default='gh-lstm', help='log path')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
args = parser.parse_args()

utils.create_exp_dir(args.log_path)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format)
fh = logging.FileHandler(os.path.join(args.log_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# 检查 GPU 可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

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
        self.sce_gate = nn.Linear(hidden_dim, hidden_dim)
        self.promise_gate = nn.Linear(hidden_dim, hidden_dim)

        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, sce_input, promise_input):
        # 处理sce_input序列数据
        sce_lstm_out, _ = self.sce_lstm(self.sce_dropout(sce_input))
        sce_lstm_out_last = sce_lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 应用门控机制
        sce_gate_output = self.sigmoid(self.sce_gate(sce_lstm_out_last))
        gated_sce_lstm_out = sce_lstm_out_last * sce_gate_output

        # 处理promise_input数据
        promise_lstm_out, _ = self.promise_lstm(self.promise_dropout(promise_input))
        promise_lstm_out_last = promise_lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 应用门控机制
        promise_gate_output = self.sigmoid(self.promise_gate(promise_lstm_out_last))
        gated_promise_lstm_out = promise_lstm_out_last * promise_gate_output

        # 合并两个部分
        merged_out = torch.cat((gated_sce_lstm_out, gated_promise_lstm_out), dim=-1)

        # 全连接层
        fc_output = self.fc(merged_out)
        # output = self.sigmoid(fc_output)

        return fc_output


# 加载训练集和测试集
data_loc = '/kaggle/input/new-sdp/'
train_data = MyDataset(data_loc + 'data/' + args.train_data + '_train.pt', data_loc + args.train_data + '.csv')
test_data = MyDataset(data_loc + 'data/' + args.test_data + '_test.pt', data_loc + args.test_data + '.csv')

train_dataloader = DataLoader(train_data, batch_size=args.batchsz, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=args.batchsz, shuffle=True)
logging.info('________________________________Project %s -> %s________________________________', args.train_data, args.test_data)

# 创建模型实例
model = MyModel(input_dim=args.input_dim, hidden_dim=128).to(device)
# print(f'Total param size: {utils.count_parameters_in_MB(model)} MB')

# 定义损失函数
df = pd.read_csv(data_loc + args.train_data + '.csv')
labels = df['bug'].values.reshape(-1, 1)

# 计算正类别和负类别的样本数量
num_positive = (labels == 1).sum()
num_negative = (labels == 0).sum()

# 计算 pos_weight，避免除零错误
pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float)
# print(pos_weight)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
# criterion = utils.GH_Loss().to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

start_training_time = time.time()

# 训练模型和预测
for epoch in range(args.epochs):
    model.train()
    # lr = optimizer.param_groups[0]['lr']
    losses = utils.AverageMeter()
    precision = utils.AverageMeter()
    recall = utils.AverageMeter()
    fpr = utils.AverageMeter()
    fnr = utils.AverageMeter()
    f_measure = utils.AverageMeter()
    g_measure = utils.AverageMeter()
    mcc = utils.AverageMeter()

    for i, (emb_data, tr_data, label) in enumerate(train_dataloader):
        emb_data = emb_data.to(device)
        tr_data = tr_data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        sce = emb_data.permute(0, 2, 1)
        trf = tr_data.unsqueeze(1)
        output = model(sce, trf)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        m = nn.Sigmoid()
        
        prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(m(output), label)
        losses.update(loss.item(), args.batchsz)
        precision.update(prec, args.batchsz)
        recall.update(rec, args.batchsz)
        fpr.update(FPR, args.batchsz)
        fnr.update(FNR, args.batchsz)
        f_measure.update(f1, args.batchsz)
        g_measure.update(g1, args.batchsz)
        mcc.update(MCC, args.batchsz)

    logging.info('Epoch:%03d loss:%.3f prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f', 
                         epoch+1, losses.avg, precision.avg, recall.avg, fpr.avg, 
                         fnr.avg, f_measure.avg, g_measure.avg, mcc.avg)

end_training_time = time.time()

training_time = end_training_time - start_training_time
logging.info('Train time:%.3fs', training_time)
# print(f"模型训练时间：{training_time}秒")

start_testing_time = time.time()

model.eval
with torch.no_grad():
    losses = utils.AverageMeter()
    precision = utils.AverageMeter()
    recall = utils.AverageMeter()
    fpr = utils.AverageMeter()
    fnr = utils.AverageMeter()
    f_measure = utils.AverageMeter()
    g_measure = utils.AverageMeter()
    mcc = utils.AverageMeter()

    for i, (emb_data, tr_data, label) in enumerate(test_dataloader):
        emb_data = emb_data.to(device)
        tr_data = tr_data.to(device)
        label = label.to(device)

        sce = emb_data.permute(0, 2, 1)
        trf = tr_data.unsqueeze(1)
        output = model(sce, trf)
        loss = criterion(output, label)

        m = nn.Sigmoid()
        
        prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(m(output), label)
        losses.update(loss.item(), args.batchsz)
        precision.update(prec, args.batchsz)
        recall.update(rec, args.batchsz)
        fpr.update(FPR, args.batchsz)
        fnr.update(FNR, args.batchsz)
        f_measure.update(f1, args.batchsz)
        g_measure.update(g1, args.batchsz)
        mcc.update(MCC, args.batchsz)

    logging.info('Test loss:%.3f prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f', 
                         losses.avg, precision.avg, recall.avg, fpr.avg, 
                         fnr.avg, f_measure.avg, g_measure.avg, mcc.avg)
    print(f'Test Loss: {losses.avg:.3f}, Precision: {precision.avg:.3f}, Recall: {recall.avg:.3f}, F1: {f_measure.avg:.3f}')

end_testing_time = time.time()

testing_time = end_testing_time - start_testing_time
logging.info('Test time:%.3fs', testing_time)
# print(f"模型测试时间：{testing_time}秒")
