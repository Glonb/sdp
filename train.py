import  os
import  sys
import  time
import  glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  genotypes
import  torch.utils
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn

from    model import Network
from    my_dataset import MyDataset

parser = argparse.ArgumentParser("SDP")
parser.add_argument('--data', type=str, default='xalan25', help='dataset')
parser.add_argument('--batchsz', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--channels', type=int, default=40, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--dropout_prob', type=float, default=0.4, help='dropout probability')
parser.add_argument('--exp_path', type=str, default='exp/sdp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--arch', type=str, default='SDP', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = args.exp_path + '-' + time.strftime("%Y%m%d-%H%M%S")
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
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
 
    train_data = MyDataset('/kaggle/input/new-sdp/' + args.data + '_train.pt')
    # valid_data = MyDataset('/kaggle/input/new-sdp/' + args.data + '.pt')

    num_data = len(train_data) 
    indices = list(range(num_data))
    split = int(np.floor(0.8 * num_data))
    
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=2)
  
    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=2)

    mapping_file_path = '/kaggle/input/new-sdp/xalan_mapping.txt'
    with open(mapping_file_path, 'r') as mf:
        lines = mf.readlines()

    vocab_size = len(lines)
    print(vocab_size)
  
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.channels, args.dropout_prob, vocab_size, genotype).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  
    criterion = nn.BCELoss().cuda()
  
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(args.epochs):
    
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_prec, train_rec, train_f1 = train(train_queue, model, criterion, optimizer)
        print('train precision: %.5f' %train_prec.item())

        valid_prec, valid_rec, valid_f1 = infer(valid_queue, model, criterion)
        print('valid precision: %.5f' %valid_prec.item())

        scheduler.step()
      
        utils.save(model, os.path.join(args.save, 'trained.pt'))


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

    for step, (x, target) in enumerate(train_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        # print(logits)
        loss = criterion(logits, target.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(logits, target)
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

    for step, (x, target) in enumerate(valid_queue):
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits = model(x)
            # print(logits)
            loss = criterion(logits, target.float())

            prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(logits, target)
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

    return precision.avg, recall.avg, f_measure.avg


if __name__ == '__main__':
    main()
