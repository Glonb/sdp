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
from    genotypes import get_Genotype

parser = argparse.ArgumentParser("SDP")
parser.add_argument('--data', type=str, default='ant-1.5', help='dataset')
parser.add_argument('--batchsz', type=int, default=16, help='batch size')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--channels', type=int, default=40, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--dropout_prob', type=float, default=0.5, help='dropout probability')
parser.add_argument('--model_path', type=str, default='log/train/trained.pt', help='path of pretrained model')
parser.add_argument('--exp_path', type=str, default='log/test', help='path of log file')
parser.add_argument('--hiddensz', type=int, default=64, help='number of hidden_size in bilstm')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--arch', type=str, default='SDP', help='which architecture to use')
args = parser.parse_args()

utils.create_exp_dir(args.exp_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.exp_path, 'test_log.txt'))
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
    test_data = MyDataset(data_path + 'data/' + args.data + '_test.pt', data_path + args.data + '.csv')

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batchsz, 
        shuffle=True, pin_memory=True, num_workers=2)

    # genotype = eval("genotypes.%s" % args.arch)
    genotype = get_Genotype()
    logging.info('Load genotype: %s', genotype)
    print('Load genotype:', genotype)
  
    model = Network(args.channels, args.hiddensz, args.dropout_prob, genotype).cuda()
    utils.load(model, args.model_path)

    print('param size:% .6f' % utils.count_parameters_in_MB(model))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.BCEWithLogitsLoss().cuda()

    start_test_time = time.time()
    
    test_prec, test_recall, test_f1 = infer(test_queue, model, criterion)

    end_test_time = time.time()
    test_time = end_test_time - start_test_time
    logging.info('Test time:%.3fs', test_time)
    
    print(f'Precision: {test_prec:.3f}, Recall: {test_recall:.3f}, F1: {test_f1:.3f}')


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

        for step, (x, trf, target) in enumerate(test_queue):

            x, trf, target = x.cuda(), trf.cuda(), target.cuda(non_blocking=True)

            logits = model(x, trf)
            loss = criterion(logits, target)

            batchsz = x.size(0)

            m = nn.Sigmoid()
          
            prec, rec, FPR, FNR, f1, g1, MCC = utils.metrics(m(logits), target)
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
