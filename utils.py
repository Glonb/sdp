import  os
import  numpy as np
import  torch
import  shutil
import  torchvision.transforms as transforms


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def metrics(output, target, threshold = 0.5):

    pred = (output > threshold).long()
    
    # Compute True Positives, False Positives, False Negatives, True Negatives
    tp = (pred == target) & (target == 1)
    fp = (pred == 1) & (target == 0)
    fn = (pred == 0) & (target == 1)
    tn = (pred == target) & (target == 0)
    
    # accuracy = (tp.sum() + tn.sum()) / (tp.sum() + fp.sum() + fn.sum() + tn.sum() + 1e-10)
    precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
    recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
    fpr = fp.sum() / (fp.sum() + tn.sum() + 1e-10)
    fnr = fn.sum() / (fn.sum() + tn.sum() + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    g1 = 2 * recall * (1 - fpr) / (recall - fpr + 1)
    mcc = (tp.sum() * tn.sum() - fp.sum() * fn.sum()) / (torch.sqrt((tp.sum() + fp.sum()) * (tp.sum() + fn.sum()) * (tn.sum() + fp.sum()) * (tn.sum() + fn.sum()))+ 1e-10)
    
    return  precision, recall, fpr, fnr, f1, g1, mcc


class GH_Loss(torch.nn.Module):
    def __init__(self):
        super(GH_Loss, self).__init__()

    def forward(self, y_pred, y_true):
        margin = 0.6

        # Define theta function
        def theta(t):
            return (torch.sign(t) + 1) / 2

        # Move tensors to the same device (GPU or CPU)
        # y_true = y_true.to(y_pred.device)
        # y_pred = y_pred.to(y_true.device)

        # Compute the loss
        loss = -(
            (1 - theta(y_true - margin) * theta(y_pred - margin) 
            - theta(1 - margin - y_true) * theta(1 - margin - y_pred)) * 
            (y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8))
        )
        
        return loss.mean()


def count_parameters_in_MB(model):
    
    return np.sum(v.numel() for name, v in model.named_parameters()) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    print('saved to model:', model_path)
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    print('load from model:', model_path)
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
