import    torch
import    torch.nn as nn
import    torch.optim as optim
import    pandas as pd
import    utils
import    time,os,sys,logging
import    argparse
from      torch.utils.data import DataLoader
from      my_dataset import MyDataset


parser = argparse.ArgumentParser("DP-CNN")
parser.add_argument('--train_data', type=str, default='ant-1.5', help='train dataset')
parser.add_argument('--test_data', type=str, default='ant-1.6', help='test dataset')
parser.add_argument('--input_dim', type=int, default=40, help='input dim')
parser.add_argument('--batchsz', type=int, default=32, help='batch size')
parser.add_argument('--log_path', type=str, default='dp-cnn', help='log path')
parser.add_argument('--epochs', type=int, default=15, help='num of training epochs')
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


class CNNModel(nn.Module):
    def __init__(self, embed_dim):
        super(CNNModel, self).__init__()

        # 卷积层
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=10, kernel_size=5)

        # 最大池化层
        self.pool = nn.AdaptiveMaxPool1d(1)

        # ReLU激活函数
        self.relu = nn.ReLU()

        # 全连接层
        self.fc = nn.Linear(10, 100)

        # 输出层
        self.output_layer = nn.Linear(120, 1)

        # Sigmoid激活函数
        # self.sigmoid = nn.Sigmoid()

    def forward(self, emb_data, tr_data):

        # 卷积和池化
        conv_out = self.relu(self.pool(self.conv(emb_data)))
        flat_out = conv_out.view(conv_out.size(0), -1)

        # 全连接层
        fc_output = self.relu(self.fc(flat_out))

        # 连接第二个输入
        cat_out = torch.cat((fc_output, tr_data), dim=-1)

        # 输出层
        output = self.output_layer(cat_out)

        # 使用Sigmoid激活函数输出
        # output = self.sigmoid(output)

        return output


# 模型实例化
model = CNNModel(args.input_dim).to(device)
# print(f'Total param size: {utils.count_parameters_in_MB(model)} MB')
logging.info("Param Size = %f MB", utils.count_parameters_in_MB(model))

# 加载训练集和测试集
data_loc = '/kaggle/input/new-sdp/'
train_data = MyDataset(data_loc + 'data/' + args.train_data + '_train.pt', data_loc + args.train_data + '.csv')
test_data = MyDataset(data_loc + 'data/' +args.test_data + '_test.pt', data_loc + args.test_data + '.csv')

train_dataloader = DataLoader(train_data, batch_size=args.batchsz, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=args.batchsz, shuffle=True)
logging.info('_________________________________Project %s -> %s________________________________', args.train_data, args.test_data)

df = pd.read_csv(data_loc + args.train_data + '.csv')
labels = df['bug'].values.reshape(-1, 1)

# 计算正类别和负类别的样本数量
num_positive = (labels == 1).sum()
num_negative = (labels == 0).sum()

# 计算 pos_weight，避免除零错误
pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float)
# print(pos_weight)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
        output = model(emb_data, tr_data)
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
    # print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {losses.avg:.3f}, Precision: {precision.avg:.3f}, Recall: {recall.avg:.3f}, F1 Score: {f_measure.avg:.3f}')

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

        output = model(emb_data, tr_data)
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
