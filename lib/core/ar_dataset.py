import os
import json
import time
import argparse
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys

sys.path.append('./')

import lib.core.utils as utils
from lib.core.config import config
from lib.models.fpn import LResNet50E_IR_Occ as LResNet50E_IR_FPN
from lib.models.resnets import LResNet50E_IR_Occ
from lib.datasets.dataset import ARdataset


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--protocol', type=int, help='', default=2)
    parser.add_argument('--gallery-root', type=str, help='',
                        default='data/datasets/ar_face_data/images_112_96/gallery_single/')
    parser.add_argument('--glass-root', type=str, help='',
                        default='data/datasets/ar_face_data/images_112_96/prob_glass_all')
    parser.add_argument('--scarf-root', type=str, help='',
                        default='data/datasets/ar_face_data/images_112_96/prob_scarf_all')
    return parser.parse_args()


def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


def get_models_params_new():
    # models_root = 'pretrained/'
    # # models_names = ['model_p5_w1_9938_9063.pth.tar']
    # models_names = ['model_p5_w1_9938_9470_6503_原来.pth.tar']
    # models_names = ['model_p4_baseline_9938_9563.pth.tar',]
    # models_names = ['baseline_p4_arcface_9917.pth.tar',
    # 'baseline_p4_cosface_9927.pth.tar']

    models_root = 'output/WebFace/LResNet50E_IR_FPN_CosMargin-Mask/CASIA-112x96-LMDB-Mask-pattern_5-weight_1-lr_0.01-optim_sgd-pretrained_1_factor_1/'
    # models_names = ['model_p5_w1_9938_9063.pth.tar']
    # models_names = ['model_best_p5_2023-12-21-03-36_0.9915_0.9480_0.5587_作者给的预训练权重_改动opp模块.pth.tar',
    #                 'model_best_p5_2023-12-28-16-35_0.9905_0.9405_0.5527_作者给的预训练权重_改动MD模块.pth.tar',
    #                 'model_best_p5_2023-12-22-15-20_0.9878_0.9312_0.6440_作者给的预训练权重_transformer模块.pth.tar',
    #                 'model_best_p5_2023-12-14-14-37_0.9915_0.9588_0.7163.pth.tar']
    # models_names = ['model_best_p5_40epoch_2024-01-08-20-59_0.9887_0.9278_0.4933.pth.tar',
    #                 'model_best_p5_50epoch_2024-01-09-03-47_0.9887_0.9278_0.4933.pth.tar',
    #                 'model_best_p5_2023-12-22-15-20_0.9878_0.9312_0.6440_作者给的预训练权重_transformer模块.pth.tar',
    #                 'model_best_p5_2023-12-14-14-37_0.9915_0.9588_0.7163.pth.tar',
    #                 'model_best_p5.pth.tar']
    models_names = ['model_best_p5_39epoch_2024-01-12-18-23_0.9942_0.9575_0.7180_使用basep4权重_BN后面加分类.pth.tar',
                    'model_best_p5_59epoch_2024-01-13-08-01_0.9942_0.9575_0.7180_使用basep4权重_BN后面加分类.pth.tar',
                    'model_best_p5_50epoch_2024-01-09-03-47_0.9887_0.9278_0.4933.pth.tar',
                    'model_best_p5_2023-12-22-15-20_0.9878_0.9312_0.6440_作者给的预训练权重_transformer模块.pth.tar',
                    'model_best_p5_2023-12-14-14-37_0.9915_0.9588_0.7163.pth.tar', ]
    models_params = []
    for name in models_names:
        model_path = os.path.join(models_root, name)
        assert os.path.exists(model_path), 'invalid model name!'

        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']

        model_name = name.split('.')[0]
        models_params.append((model_name, state_dict))
    return models_params


def get_feature(img, model, model_name, mask=None):
    if isinstance(mask, torch.Tensor):
        # expand mask to match the batch size of img
        batch_size = img.shape[0]
        mask = mask.unsqueeze(0)
        mask = mask.repeat(batch_size, 1, 1, 1)

    img = img.to('cuda')
    fc_mask, mask, _, fc = model(img, mask)
    fc, fc_mask = fc.to('cpu').squeeze(), fc_mask.to('cpu').squeeze()
    mask = mask.to('cpu')

    if ('baseline' not in model_name):
        fc = fc_mask
    fc1 = fc[0::2, :]
    fc2 = fc[1::2, :]
    fc = fc1 + fc2
    fc = F.normalize(fc1)

    m1 = mask[0::2]
    m2 = mask[1::2]
    mask = (m1 + m2) / 2.0
    return fc.detach(), mask.detach()


def cosine_similarity(f1, f2):
    # compute cosine_similarity for 1-D tensor
    f1 = F.normalize(f1, dim=0)
    f2 = F.normalize(f2, dim=0)

    similarity = torch.sum(f1 * f2).item()

    return similarity


def main():
    args = parse_arguments()
    if args.protocol == 1:
        args.gallery_root = 'data/datasets/ar_face_data/images_112_96/gallery/'

    print('Args: \n' + json.dumps(vars(args), indent=4))

    os.environ['CUDA_VISIBLE_DEVICES'] = config.TRAIN.GPUS
    gpus = [int(i) for i in config.TRAIN.GPUS.split(',')]
    gpus = range(len(gpus))

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    gallery_loader = torch.utils.data.DataLoader(
        ARdataset(args.gallery_root, transform),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.TEST.WORKERS,
        pin_memory=True)
    gallery_length = len(gallery_loader.dataset)

    glass_loader = torch.utils.data.DataLoader(
        ARdataset(args.glass_root, transform),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.TEST.WORKERS,
        pin_memory=True)
    glass_length = len(glass_loader.dataset)
    batch_len = len(glass_loader)

    scarf_loader = torch.utils.data.DataLoader(
        ARdataset(args.scarf_root, transform),
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.TEST.WORKERS,
        pin_memory=True)
    scarf_length = len(scarf_loader.dataset)
    assert glass_length == scarf_length

    for models_params in get_models_params_new():
        model_name, state_dict = models_params
        pattern = int(model_name[model_name.find('p') + 1])
        num_mask = len(utils.get_grids(*config.NETWORK.IMAGE_SIZE, pattern))
        model = LResNet50E_IR_FPN(num_mask=num_mask)
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

        model.module.load_state_dict(state_dict, strict=False)
        model.eval()

        features, masks, labels = preprocessing(glass_length, glass_loader, model, model_name)
        identification(model, model_name, features, masks, labels, gallery_length, gallery_loader, 'glass')

        features, masks, labels = preprocessing(scarf_length, scarf_loader, model, model_name)
        identification(model, model_name, features, masks, labels, gallery_length, gallery_loader, 'scarf')


def preprocessing(length, loader, model, model_name):
    features = torch.zeros(length // 2, 512)
    masks = torch.zeros(length // 2, 512, 7, 6)
    labels = []
    begin = end = 0
    for batch_idx, (img, label) in enumerate(loader):
        fc, mask = get_feature(img, model, model_name)

        begin = end
        end = begin + len(label[::2])
        features[begin:end] = fc
        masks[begin:end] = mask
        labels += label[::2]
    return features, masks, labels


def match(fc, label, features_gallery, labels_gallery):
    similarity_label = []
    for fc_gallery, label_gallery in zip(features_gallery, labels_gallery):
        similarity = cosine_similarity(fc, fc_gallery)
        similarity_label.append((similarity, label_gallery))
    similarity_label = sorted(similarity_label, key=lambda x: x[0], reverse=True)
    largest_similarity = similarity_label[0][0]
    label_match = similarity_label[0][1]

    return int(label == label_match)


def identification(model, model_name, features, masks, labels, gallery_length, gallery_loader, probe_name):
    mask_avg = torch.mean(masks, dim=0)
    features_gallery = torch.zeros(gallery_length // 2, 512)
    labels_gallery = []
    start = end = 0
    for batch_idx, (img, label_gallery) in enumerate(gallery_loader):
        fc_gallery, _ = get_feature(img, model, model_name)

        begin = end
        end = begin + len(label_gallery[::2])
        features_gallery[begin:end] = fc_gallery
        labels_gallery += label_gallery[::2]

    length = len(labels)
    correct = 0
    for i, fc, mask, label in zip(range(length), features, masks, labels):
        correct += match(fc, label, features_gallery, labels_gallery)
    acc = float(correct) / length
    time_cur = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())
    print('{}, model:{}, probe set:{}, rank 1 accuracy:{}'.format(time_cur, model_name, probe_name, acc))


if __name__ == '__main__':
    main()
