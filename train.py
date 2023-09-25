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
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=40, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='exp/sdp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
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

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_ch, genotype).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.wd
    )

    train_data = MyDataset('/kaggle/input/sdp-data/xalan25_embed.npy', '/kaggle/input/sdp-data/xalan25_label.csv')
    # valid_data = MyDataset('/kaggle/input/sdp-data/xalan26_embed.npy', '/kaggle/input/sdp-data/xalan26_label.csv')

    num_valid = len(train_data) 
    indices = list(range(num_valid))
    split = int(np.floor(0.2 * num_valid))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    for epoch in range(args.epochs):
    
        logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_prec, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('train_prec: %f', train_prec)

        valid_prec, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_prec: %f', valid_prec)

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

    return precision.avg, losses.avg


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

    return precision.avg, losses.avg


if __name__ == '__main__':
    main()
