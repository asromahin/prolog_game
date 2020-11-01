import os
import random
import sys
import typing as tp
from shutil import copyfile

import cv2
import numpy as np
import torch


def make_os_settings(cuda_num: str) -> None:
    """Set some os settings."""
    os.environ['TORCH_HOME'] = '/home/ds'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['NO_PROXY'] = 'koala03.ftc.ru'
    os.environ['no_proxy'] = 'koala03.ftc.ru'
    # torch.set_num_threads(4)
    cv2.setNumThreads(0)
    #cv2.ocl.setUseOpenCL(False)


def seed_everything(seed: tp.Optional[int] = 999) -> None:
    """Add reproduce."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def copyfile2dir(filepath: str, dir_: str) -> None:
    """Copy file to dir."""
    filename = os.path.basename(filepath)
    copyfile(filepath, os.path.join(dir_, filename))


def make_logdirs_if_needit(logdir: str, experiment_dir: str, model_checkpoints: str) -> None:
    """Creates all the necessary folders for the experiment and saves the experiment config."""
    #os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, experiment_dir), exist_ok=True)
    os.makedirs(os.path.join(logdir, experiment_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(logdir, experiment_dir, 'qat_models'), exist_ok=True)
    os.makedirs(os.path.join(model_checkpoints, experiment_dir), exist_ok=True)
    copyfile2dir(sys.modules[__name__].__file__, os.path.join(logdir, experiment_dir))
    for fn in os.listdir():
        if fn.endswith('.py'):
            copyfile2dir(fn, os.path.join(logdir, experiment_dir))


def write2log(logpath: str, epoch: int, accuracy_1) -> None:
    add2log = ''
    if epoch == 0:
        add2log = '\t'.join(('accuracy_doc'))
    add2log += '\t'.join((f'\n{epoch}', '\t'.join(map(lambda x: str(round(x, 4)), (accuracy_1)))))
    with open(logpath, 'a') as log:
        log.write(add2log)

def push_logs(name, logger, epoch, metrics, config):
    list_log = ['iou', 'iou_dots', 'acc_document', 'clf_quality', 'lr','Fscore','acc_titul', config.loss.__name__]
    for log_name in list_log:
        try:
            logger.report_scalar(log_name, name, iteration=epoch, value=metrics[log_name])
        except KeyError:
            if log_name not in list_log:
                print('Error key logs in push logs.')