import argparse
import json
import os
import time
import shutil
import logging
import numpy as np

import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import lib.core.utils as utils
from lib.core.config import config
from lib.core.config import update_config
from lib.core.functions import train
from lib.core.lfw_eval import eval as lfw_eval
from lib.datasets.dataset import WebFace_Folder
from lib.datasets.dataset import WebFace_LMDB
from lib.datasets.dataset import LFW_Image
from lib.models.fpn import LResNet50E_IR_Occ as LResNet50E_IR_FPN
from lib.models.fpn import LResNet50E_IR_Occ_2D
from lib.models.fpn import LResNet50E_IR_Occ_FC
from lib.models.metrics import ArcMarginProduct
from lib.models.metrics import CosMarginProduct
from lib.models.metrics import SphereMarginProduct

# computing flops
# from ptflops import get_model_complexity_info

# setup random seed
torch.manual_seed(0)
np.random.seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch End2End Occluded Face')
    parser.add_argument('--cfg', help='experiment configure file name', type=str, default="experiments/CASIA-112x96-LMDB-Mask.yaml")
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--gpus', help='gpus', type=str, default='0')
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--binary_thres', help='thres for binary mask', type=float)
    parser.add_argument('--soft_binary', help='whether use soft binary mask', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('--pattern', help='num of pattern', type=int, default=5)
    parser.add_argument('--pretrained', help='whether use pretrained model', type=str)
    parser.add_argument('--debug', help='whether debug', default=1, type=int)
    parser.add_argument('--model', help=' model name', type=str, default='LResNet50E_IR_FPN')
    parser.add_argument('--factor', help='factor of mask',  type=float)
    parser.add_argument('--ratio', help='ratio of masked img for training', default=3, type=int)
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus:
        config.TRAIN.GPUS = args.gpus
    if args.workers:
        config.TRAIN.WORKERS = args.workers
    if args.model:
        print('update model type')
        config.TRAIN.MODEL = args.model
    if args.pattern:
        print('update pattern and num_mask')
        config.TRAIN.PATTERN = args.pattern
        config.TRAIN.NUM_MASK = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, args.pattern))
    if args.batch_size:
        print('update batch_size')
        config.TRAIN.BATCH_SIZE = args.batch_size
        config.TEST.BATCH_SIZE = args.batch_size
    if args.pretrained =='No':
        print('update pretrained')
        config.NETWORK.PRETRAINED = ''
    if args.factor: 
        print('update factor')
        config.NETWORK.FACTOR = args.factor
    if args.binary_thres:
        print('update binary_thres')
        config.TRAIN.BINARY_THRES = args.binary_thres
    if args.soft_binary:
        print('update soft_binary')
        config.TRAIN.SOFT_BINARY = args.soft_binary

def main():
    # --------------------------------------model----------------------------------------
    args = parse_args()
    reset_config(config, args)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS
    gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
    gpus = range(len(gpus))

    logger, final_output_dir, tb_log_dir = utils.create_temp_logger()

    logger.info('Configs: \n' + json.dumps(config, indent=4, sort_keys=True))
    logger.info('Args: \n' + json.dumps(vars(args), indent=4, sort_keys=True))

    test_transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    test_loader = torch.utils.data.DataLoader(
        LFW_Image(config, test_transform),
        batch_size=config.TEST.BATCH_SIZE*len(gpus), 
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.TEST.WORKERS, 
        pin_memory=True)

    # model_root = 'pretrained/'
    model_root ='/home/leishu/pycode/FROM/output/WebFace/LResNet50E_IR_FPN_CosMargin-Mask/CASIA-112x96-LMDB-Mask-pattern_5-weight_1-lr_0.01-optim_sgd-pretrained_1_factor_1'
    # model_list = ['model_best_p5_2023-12-21-03-36_0.9915_0.9480_0.5587_作者给的预训练权重_改动opp模块.pth.tar',
    #               'model_best_p5_2023-12-28-16-35_0.9905_0.9405_0.5527_作者给的预训练权重_改动MD模块.pth.tar',
    #               'model_best_p5_2023-12-22-15-20_0.9878_0.9312_0.6440_作者给的预训练权重_transformer模块.pth.tar',
    #               'model_best_p5_2023-12-14-14-37_0.9915_0.9588_0.7163.pth.tar']
    model_list = ['model_best_p5_39epoch_2024-01-12-18-23_0.9942_0.9575_0.7180_使用basep4权重_BN后面加分类.pth.tar',
                  'model_best_p5_59epoch_2024-01-13-08-01_0.9942_0.9575_0.7180_使用basep4权重_BN后面加分类.pth.tar',
                  'model_best_p5_50epoch_2024-01-09-03-47_0.9887_0.9278_0.4933.pth.tar',
                  'model_best_p5_2023-12-22-15-20_0.9878_0.9312_0.6440_作者给的预训练权重_transformer模块.pth.tar',
                  'model_best_p5_2023-12-14-14-37_0.9915_0.9588_0.7163.pth.tar',]
    # model_root = 'pretrained/'
    # model_list = ['model_best_2023-12-14-14-37_0.9915_0.9588_0.7163.pth.tar',
    #               'checkpoint.pth.tar']
    # model_list = [
        # 'model_best_p5_w0.pth.tar',
        # 'model_best_p5_w1.pth.tar',
        # 'model_best_p5_occ.pth.tar'
    # ]
    for model_name in model_list:
        pattern = int(model_name[model_name.find('p')+1])
        num_mask = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, pattern))
        model = LResNet50E_IR_FPN(num_mask=num_mask)
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

        model_path = os.path.join(model_root, model_name)
        lfw_eval(model, model_path, config, test_loader, 'temp', 0)
        print(' ')

if __name__ == '__main__':
    main()

