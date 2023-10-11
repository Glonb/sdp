import  os,sys,glob,time,re
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  genotypes
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn

from    model import Network
from    my_dataset import MyDataset

parser = argparse.ArgumentParser("SDP")
parser.add_argument('--data', type=str, default='xalan25', help='dataset')
parser.add_argument('--batchsz', type=int, default=16, help='batch size')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--exp_path', type=str, default='exp/trained.pt', help='path of pretrained model')
parser.add_argument('--dropout_prob', type=float, default=0.5, help='drop probability')
parser.add_argument('--seed', type=int, default=3, help='random seed')
parser.add_argument('--arch', type=str, default='SDP', help='which architecture to use')
args = parser.parse_args()

utils.create_exp_dir('sdp-test')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    data_path = '/kaggle/input/new-sdp/'
    test_data = MyDataset(data_path + args.data + '_test.pt')

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batchsz, 
        shuffle=True, pin_memory=True, num_workers=2)

    letters = "".join(re.findall(r'[a-zA-Z]+', args.data))
    mapping_file_path = data_path + letters + '_mapping.txt'
    with open(mapping_file_path, 'r') as mf:
        lines = mf.readlines()

    vocab_size = len(lines)
    print('vocabulary size: %d'%vocab_size)

    genotype = eval("genotypes.%s" % args.arch)
    print('Load genotype:', genotype)
    model = Network(args.channels, args.dropout_prob, vocab_size, genotype).cuda()
    utils.load(model, args.exp_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.BCEWithLogitsLoss().cuda()
  
    test_prec, test_rec, test_f1 = infer(test_queue, model, criterion)
    
    print('test_precision: ', test_prec.item())


def infer(test_queue, model, criterion):

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

        for step, (x, target) in enumerate(test_queue):

            x, target = x.cuda(), target.cuda(non_blocking=True)

            logits = model(x)
            loss = criterion(logits, target.float())

            batchsz = x.size(0)
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
                logging.info('test %03d loss:%e prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f',
                             step, losses.avg, precision.avg, recall.avg, fpr.avg, fnr.avg,
                             f_measure.avg, g_measure.avg, mcc.avg)

    logging.info('test %03d loss:%e prec:%.3f recall:%.3f fpr:%.3f fnr:%.3f f1:%.3f g1:%.3f mcc:%.3f',
                             step, losses.avg, precision.avg, recall.avg, fpr.avg, fnr.avg,
                             f_measure.avg, g_measure.avg, mcc.avg)
  
    return precision.avg, recall.avg, f_measure.avg


if __name__ == '__main__':
    main()
