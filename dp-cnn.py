import   torch
import   torch.nn as nn
import   torch.optim as optim
import   utils
from     torch.utils.data import DataLoader
from     my_dataset import MyDataset


parser = argparse.ArgumentParser("DP-CNN")
parser.add_argument('--train_data', type=str, default='ant15', help='train dataset')
parser.add_argument('--test_data', type=str, default='ant16', help='test dataset')
parser.add_argument('--input_dim', type=int, default=40, help='input dim')
parser.add_argument('--batchsz', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=15, help='num of training epochs')
args = parser.parse_args()

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
        self.sigmoid = nn.Sigmoid()

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
        output = self.sigmoid(output)

        return output


# 模型实例化
embedding_dim = 40
model = CNNModel(embedding_dim).to(device)
print(f'Total param size: {utils.count_parameters_in_MB(model)} MB')

# 定义损失函数和优化器
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
