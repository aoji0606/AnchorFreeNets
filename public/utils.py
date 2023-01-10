import os
import math
import torch
import logging
import warnings
from logging.handlers import TimedRotatingFileHandler


def get_logger(name, log_dir='log'):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = TimedRotatingFileHandler(info_name,
                                            when='D',
                                            encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    error_name = os.path.join(log_dir, '{}.error.log'.format(name))
    error_handler = TimedRotatingFileHandler(error_name,
                                             when='D',
                                             encoding='utf-8')
    error_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)

    return logger


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            sample = next(self.loader)
            self.next_input, self.next_target = sample
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class StepLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs=2, warmup_ratio=0.2, milestones=[18, 22], gamma=0.1, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio

        self.milestones = milestones
        self.gamma = gamma

        super(StepLRWithWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch < self.warmup_epochs:
            k = (1 - self.last_epoch / self.warmup_epochs) * (1 - self.warmup_ratio)
            lrs = [base_lr * (1 - k) for base_lr in self.base_lrs]
        else:
            miles = 0
            for milestone in self.milestones:
                if self.last_epoch > milestone:
                    miles += 1
            lrs = [base_lr * self.gamma ** miles for base_lr in self.base_lrs]

        return lrs


class CosineAnnealingLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epochs, warmup_epochs=2, warmup_ratio=0.2, min_lr=1e-6, last_epoch=-1, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.warmup_ratio = warmup_ratio

        self.min_lr = min_lr
        self.total_epochs = total_epochs

        super(CosineAnnealingLRWithWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )

        """Decay the learning rate with half-cycle cosine after warmup"""
        if self.last_epoch < self.warmup_epochs:
            k = (1 - self.last_epoch / self.warmup_epochs) * (1 - self.warmup_ratio)
            lrs = [base_lr * (1 - k) for base_lr in self.base_lrs]
        else:
            lrs = [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))) for base_lr in self.base_lrs
            ]

        return lrs


def param_groups_lrd(model, lr=0, weight_decay=0, no_weight_decay_list=[]):
    param_groups_names = dict()
    param_groups = dict()

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if n in no_weight_decay_list:
            g_name = 'no_decay'
            this_decay = 0.
            this_lr = lr
        elif p.ndim == 1 or n in no_weight_decay_list:
            g_name = 'no_decay_double_lr' if 'bias' in n else 'no_decay'
            this_decay = 0.
            this_lr = lr * 2 if 'bias' in n else lr
        else:
            g_name = 'decay'
            this_decay = weight_decay
            this_lr = lr
        
        group_name = 'layer_{}'.format(g_name)
        if group_name not in param_groups_names:
            param_groups_names[group_name] = {
                'weight_decay': this_decay,
                'lr': this_lr,
                'params': [],
            }
            param_groups[group_name] = {
                'weight_decay': this_decay,
                'lr': this_lr,
                'params': [],
            }
    
        param_groups_names[group_name]['params'].append(n)
        param_groups[group_name]['params'].append(p)

    return list(param_groups.values()), param_groups_names
