import  os, sys, time, glob, re
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  genotypes
import  torch.utils
import  pandas as pd
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn
from    model import Network
from    my_dataset import MyDataset
from    genotypes import get_Genotype

parser = argparse.ArgumentParser("SDP")
parser.add_argument('--data', type=str, default='ant-1.5', help='dataset')
parser.add_argument('--batchsz', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--channels', type=int, default=40, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--hiddensz', type=int, default=64, help='number of hidden_size in bilstm')
parser.add_argument('--dropout_prob', type=float, default=0.2, help='dropout probability')
parser.add_argument('--exp_path', type=str, default='log/train', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--arch', type=str, default='SDP', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

utils.create_exp_dir(args.exp_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.exp_path, 'train_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    data_path = '/kaggle/input/new-sdp/'
    train_data = MyDataset(data_path + 'data/' + args.data + '_train.pt', data_path + args.data + '.csv')
  
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz, shuffle=True, pin_memory=False, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz, shuffle=True, pin_memory=False, num_workers=2)
  
    # genotype = eval("genotypes.%s" % args.arch)
    genotype = get_Genotype()
    model = Network(args.channels, args.hiddensz, args.dropout_prob, genotype).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    df = pd.read_csv(data_path + args.data + '.csv')
    labels = df['bug'].values.reshape(-1, 1)
    
    # 计算正类别和负类别的样本数量
    num_positive = (labels == 1).sum()
    num_negative = (labels == 0).sum()
    
    # 计算 pos_weight，避免除零错误
    pos_weight = torch.tensor([num_negative / max(num_positive, 1)], dtype=torch.float)
    # print(pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    start_training_time = time.time()
  
    for epoch in range(args.epochs):

        lr = optimizer.param_groups[0]['lr']
        logging.info('epoch %d lr %e', epoch, lr)
      
        train_f1 = train(train_queue, model, criterion, optimizer)
        print('train f1_score: %.5f' %train_f1.item())

        valid_f1 = infer(valid_queue, model, criterion)
        print('valid f1_score: %.5f' %valid_f1.item())
      
        utils.save(model, os.path.join(args.save, 'trained.pt'))

    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    logging.info('Train time:%.3fs', training_time)


def train(train_queue, model, criterion, optimizer):

    losses = utils.AverageMeter()
    precision = utils.AverageMeter()
    recall = utils.AverageMeter()
    fpr = utils.AverageMeter()
    fnr = utils.AverageMeter()
    f_measure = utils.AverageMeter()
    g_measure = utils.AverageMeter()
    mcc = utils.AverageMeter()
    model.train()

    for step, (x, trf, target) in enumerate(train_queue):
        x = x.cuda()
        trf = trf.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(x, trf)
  
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        m = nn.Sigmoid()
      
        prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(m(logits), target)
        batchsz = x.size(0)
        losses.update(loss.item(), batchsz)
        precision.update(prec, batchsz)
        recall.update(rec, batchsz)
        fpr.update(FPR, batchsz)
        fnr.update(FNR, batchsz)
        f_measure.update(f1, batchsz)
        g_measure.update(g1, batchsz)
        mcc.update(MCC, batchsz)

        if step % args.report_freq == 0:
            logging.info('Step:%03d loss:%.3f prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f', 
                         step, losses.avg, precision.avg, recall.avg, fpr.avg, 
                         fnr.avg, f_measure.avg, g_measure.avg, mcc.avg)
          
    logging.info('Train %03d loss:%.3f prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f', 
                         step, losses.avg, precision.avg, recall.avg, fpr.avg, 
                         fnr.avg, f_measure.avg, g_measure.avg, mcc.avg)

    return f_measure.avg


def infer(valid_queue, model, criterion):

    losses = utils.AverageMeter()
    precision = utils.AverageMeter()
    recall = utils.AverageMeter()
    fpr = utils.AverageMeter()
    fnr = utils.AverageMeter()
    f_measure = utils.AverageMeter()
    g_measure = utils.AverageMeter()
    mcc = utils.AverageMeter()
    model.eval()

    for step, (x, trf, target) in enumerate(valid_queue):
        x = x.cuda()
        trf = trf.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits = model(x, trf)
            
            loss = criterion(logits, target)

            m = nn.Sigmoid()
          
            prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(m(logits), target)
            batchsz = x.size(0)
            losses.update(loss.item(), batchsz)
            precision.update(prec, batchsz)
            recall.update(rec, batchsz)
            fpr.update(FPR, batchsz)
            fnr.update(FNR, batchsz)
            f_measure.update(f1, batchsz)
            g_measure.update(g1, batchsz)
            mcc.update(MCC, batchsz)

        if step % args.report_freq == 0:
            logging.info('>> Validation: %3d %e prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f', 
                             step, losses.avg, precision.avg, recall.avg, fpr.avg,
                             fnr.avg, f_measure.avg, g_measure.avg, mcc.avg)
          
    logging.info('>> Validation: %3d %e prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f', 
                             step, losses.avg, precision.avg, recall.avg, fpr.avg,
                             fnr.avg, f_measure.avg, g_measure.avg, mcc.avg)

    return f_measure.avg


if __name__ == '__main__':
    main()
