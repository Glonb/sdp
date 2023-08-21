import  os,sys,glob
import  numpy as np
import  torch
import  utils
import  logging
import  argparse
import  torch.nn as nn
import  genotypes
import  torchvision.datasets as dset
import  torch.backends.cudnn as cudnn

from    model import NetworkCIFAR as Network
from    my_dataset import MyDataset

parser = argparse.ArgumentParser("SDP")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_ch', type=int, default=10, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--exp_path', type=str, default='exp/trained.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def main():

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # equal to: genotype = genotypes.DARTS_v2
    genotype = eval("genotypes.%s" % args.arch)
    print('Load genotype:', genotype)
    model = Network(args.init_ch, args.layers, args.auxiliary, genotype).cuda()
    utils.load(model, args.exp_path)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.BCEWithLogitsLoss().cuda()

    test_data = MyDataset('/kaggle/input/sdp-data/xalan6_embed.npy', '/kaggle/input/sdp-data/xalan6_label.csv')
  
    num_test = len(test_data) 
    indices = list(range(num_test))
    split = int(np.floor(0.2 * num_test))

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batchsz, 
        shuffle=False, 
        pin_memory=True, num_workers=2)

    model.drop_path_prob = args.drop_path_prob
    test_prec, test_obj = infer(test_queue, model, criterion)
    logging.info('test_prec %f', test_prec)
    print('test_prec: ', test_prec.item())


def infer(test_queue, model, criterion):

    losses = utils.AverageMeter()
    precision = utils.AverageMeter()
    recall = utils.AverageMeter()
    f_measure = utils.AverageMeter()
    model.eval()

    with torch.no_grad():

        for step, (x, target) in enumerate(test_queue):

            x, target = x.cuda(), target.cuda(non_blocking=True)

            logits, _ = model(x)
            loss = criterion(logits, target.float())

            prec, rec, f1 = utils.metrics(logits, target)
            batchsz = x.size(0)
            losses.update(loss.item(), batchsz)
            precision.update(prec, batchsz)
            recall.update(rec, batchsz)
            f_measure.update(f1, batchsz)

            if step % args.report_freq == 0:
                logging.info('test %03d loss:%e prec:%f recall:%f f1:%f',
                             step, losses.avg, precision.avg, recall.avg, f_measure.avg)

    return precision.avg, losses.avg


if __name__ == '__main__':
    main()
