import  os,sys,time, glob, re
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
from    torch import optim
import  torch.backends.cudnn as cudnn

from    model_search import Network
from    arch import Arch
from    my_dataset import MyDataset


parser = argparse.ArgumentParser("SDP")
parser.add_argument('--data', type=str, default='xalan25', help='dataset')
parser.add_argument('--batchsz', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--channels', type=int, default=40, help='num of channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--hiddensz', type=int, default=64, help='hidden size of bilstm')
parser.add_argument('--dropout_prob', type=float, default=0.5, help='dropout probability')
parser.add_argument('--exp_path', type=str, default='search', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training/val splitting')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_wd', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.exp_path += str(args.gpu)
utils.create_exp_dir(args.exp_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %H:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.exp_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device('cuda:0')


def main():
    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)

    total, used = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')[args.gpu].split(',')
    total = int(total)
    used = int(used)

    logging.info('Total GPU memory: %d used: %d', total, used)
    print('Total GPU mem:', total, 'used:', used)

    logging.info('GPU device = %d' % args.gpu)
    logging.info("args = %s", args)

    data_path = '/kaggle/input/new-sdp/'
    train_data = MyDataset(data_path + args.data + '_train.pt', data_path + args.data + '.csv')

    # num_train = len(train_data) 
    # indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))

    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batchsz,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    #     pin_memory=True, num_workers=2)

    # valid_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batchsz,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    #     pin_memory=True, num_workers=2)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=2)

    criterion = nn.BCELoss().to(device)
    model = Network(args.channels, args.layers, args.hiddensz, criterion).to(device)

    logging.info("Total param size = %f MB", utils.count_parameters_in_MB(model))

    # this is the optimizer to optimize
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.lr_min)

    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    arch = Arch(model, args)

    for epoch in range(args.epochs):

        lr = scheduler.get_last_lr()[0]
        # lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch: %d lr: %e', epoch, lr)
        print('Epoch: %d' %epoch)

        genotype = model.genotype()
        logging.info('Genotype: %s', genotype)

        # training
        train_prec, train_rec, train_f1 = train(train_queue, valid_queue, model, arch, criterion, optimizer, lr)
        print('train precision: %.5f' %train_prec.item())

        # update lr
        scheduler.step()

        # validation
        valid_prec, valid_rec, valid_f1 = infer(valid_queue, model, criterion)
        print('valid precision: %.5f' %valid_prec.item())


def train(train_queue, valid_queue, model, arch, criterion, optimizer, lr):
    
    losses = utils.AverageMeter()
    precision = utils.AverageMeter()
    recall = utils.AverageMeter()
    fpr = utils.AverageMeter()
    fnr = utils.AverageMeter()
    f_measure = utils.AverageMeter()
    g_measure = utils.AverageMeter()
    mcc = utils.AverageMeter()

    valid_iter = iter(valid_queue)

    for step, (x, trf, target) in enumerate(train_queue):

        batchsz = x.size(0)
        model.train()

        x, trf, target = x.to(device), trf.to(device), target.cuda()
        x_search, trf_search, target_search = next(valid_iter) 
        x_search, trf_search, target_search = x_search.to(device), trf_search.to(device), target_search.cuda()

        # 1. update alpha
        arch.step(x, trf, target, x_search, trf_search, target_search, lr, optimizer, unrolled=True)

        logits = model(x, trf)
        loss = criterion(logits, target.float())

        # 2. update weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(logits, target)
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

    logging.info('Step:%03d loss:%.3f prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f', 
                         step, losses.avg, precision.avg, recall.avg, fpr.avg, 
                         fnr.avg, f_measure.avg, g_measure.avg, mcc.avg)

    return precision.avg, recall.avg, f_measure.avg


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

    with torch.no_grad():
        for step, (x, trf, target) in enumerate(valid_queue):

            x, trf, target = x.to(device), trf.to(device), target.cuda()
            batchsz = x.size(0)

            logits = model(x, trf)
            loss = criterion(logits, target.float())

            prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(logits, target)
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
  
    return precision.avg, recall.avg, f_measure.avg


if __name__ == '__main__':
    main()
