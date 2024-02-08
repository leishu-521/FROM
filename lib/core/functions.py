import os
import time
import logging
import numpy as np
import math

import torch
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def train(train_loader, model, classifier, criterion, optimizer, epoch, tb_log_dir, config, centers, counts, pattern):
    model.train()
    time_curr = time.time()
    loss_display = 0.0
    loss_cls_dis = 0.0
    loss_pred_dis = 0.0

    for batch_idx, data in enumerate(train_loader):
        img, label, mask_label, imgPaths = data

        img, label = img.cuda(), label.cuda()
        mask_label = mask_label.cuda()

        features = model(img)

        # compute output
        if config.TRAIN.MODE == 'Clean' or config.TRAIN.MODE == 'Occ':
            feature = features[-1]
            output = classifier(feature, label)
            loss = criterion(output, label)

        elif config.TRAIN.MODE == 'Mask':
            output, loss, loss_cls, loss_pred, mask, preds = occ_train(features, label, mask_label, config, classifier,
                                                                       criterion, centers, counts, pattern)
        else:
            raise ValueError('Unknown training mode!')

        loss_display += loss.item()
        if config.TRAIN.MODE == 'Mask':
            loss_cls_dis += loss_cls.item()
            loss_pred_dis += loss_pred.item()
        else:
            loss_cls_dis = 0
            loss_pred_dis = 0
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iters = epoch * len(train_loader) + batch_idx

        if iters % config.TRAIN.PRINT_FREQ == 0 and iters != 0:

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))

            time_used = time.time() - time_curr
            if batch_idx < config.TRAIN.PRINT_FREQ:
                num_freq = batch_idx + 1
            else:
                num_freq = config.TRAIN.PRINT_FREQ
            speed = num_freq / time_used
            loss_display /= num_freq
            loss_cls_dis /= num_freq
            loss_pred_dis /= num_freq

            INFO = ' Margin: {:.2f}, Scale: {:.2f}'.format(classifier.module.m, classifier.module.s)
            logger.info(
                'Train Epoch: {} [{:03}/{} ({:.0f}%)]{:05}, Loss: {:.6f}, Acc: {:.4f}, Elapsed time: {:.4f}s, Batches/s {:.4f}'.format(
                    epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader),
                    iters, loss_display, acc, time_used, speed) + INFO)
            if config.TRAIN.MODE == 'Mask':
                logger.info('Cls Loss: {:.4f}; Pred Loss: {:.4f}*{}'.format(loss_cls_dis, loss_pred_dis,
                                                                            config.LOSS.WEIGHT_PRED))
            with SummaryWriter(tb_log_dir) as sw:
                sw.add_scalar('TRAIN_LOSS', loss_display, iters)
                sw.add_scalar('TRAIN_ACC', acc, iters)
                if config.TRAIN.MODE == 'Mask':
                    sw.add_scalar('CLS_LOSS', loss_cls_dis, iters)
                    sw.add_scalar('PRED_LOSS', loss_pred_dis, iters)
            time_curr = time.time()
            loss_display = 0.0
            loss_cls_dis = 0.0
            loss_pred_dis = 0.0


def occ_train(features, label, mask_label, config, classifier, criterion, centers, counts, pattern):
    fc_mask, mask, vec, fc = features

    output = classifier(fc_mask, label)
    loss_cls = criterion(output, label)

    # 自己加的mse
    vec_preds = np.argmax(vec, axis=1)
    mse = mask_Mse(centers, counts, vec_preds, mask_label, pattern)

    loss_pred = criterion(vec, mask_label)
    print(type(loss_pred))
    print(mask_label)
    preds = vec.cpu().detach().numpy()
    print(preds)
    preds = np.argmax(preds, axis=1)
    print(preds)
    loss = loss_cls + config.LOSS.WEIGHT_PRED * (loss_pred + mse)


    return output, loss, loss_cls, loss_pred, mask, preds


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def mask_Mse(centers, counts, vec, mask_label, N):
    grids_counts = (N * (N + 1) / 2) ** 2 + 1
    print(grids_counts)
    mask_centers = [list(centers[i]) for i in vec]
    mask_centers_label = [list(centers[i]) for i in mask_label]

    mask_counts = [counts[i] for i in vec]
    mask_counts_label = [counts[i] for i in mask_label]
    print("预测图片的掩码位置中心坐标:{}".format(mask_centers))
    print("图片的真实标签掩码位置中心坐标:{}".format(mask_centers_label))
    print("预测图片的掩码块数量:{}".format(mask_counts))
    print("图片的真实块数量:{}".format(mask_counts_label))

    if len(mask_centers) != len(mask_centers_label) or len(mask_counts) != len(mask_counts_label):
        raise ValueError("Input lists must have the same length.")
    n = len(mask_counts)
    for a, b in zip(mask_centers, mask_centers_label):
        print(a, b)
    center_squared_diff_sum = sum(calculate_distance(a, b) for a, b in zip(mask_centers, mask_centers_label))
    print("center_squared_diff_sum:{}".format(center_squared_diff_sum))
    centers_mse = center_squared_diff_sum / n / 147  # 112*96 的图像两点距离最大为147
    print("centers_mse:{}".format(centers_mse))
    counts_squared_diff_sum = sum(abs(a - b) for a, b in zip(mask_counts, mask_counts_label))
    print("counts_squared_diff_sum:{}".format(counts_squared_diff_sum))
    counts_mse = counts_squared_diff_sum / n / (N * N)
    print("counts_mse:{}".format(counts_mse))

    mse = centers_mse + counts_mse
    return mse


def get_grids(H, W, N):
    grid_ori = np.zeros((H, W))
    centers = []
    counts = []

    x_axis = np.linspace(0, W, N + 1, True, dtype=int)
    y_axis = np.linspace(0, H, N + 1, True, dtype=int)

    vertex_set = []
    for y in y_axis:
        for x in x_axis:
            vertex_set.append((y, x))

    grids = [grid_ori]
    grid_ori_centers = (0, 0)
    grid_ori_counts = 0
    centers.append(grid_ori_centers)
    counts.append(grid_ori_counts)

    # i = 0
    for start in vertex_set:
        for end in vertex_set:
            if end[0] > start[0] and end[1] > start[1]:
                grid = grid_ori.copy()
                grid[start[0]:end[0], start[1]:end[1]] = 1.0
                # if int(((end[0] - start[0]) / (H / N)) * ((end[1] - start[1]) / (W / N)) + 0.5) == 9:
                #     print(grid)
                # print("左上角坐标：({},{}),右下角坐标({},{})".format(start[1], start[0], end[1], end[0]))
                # print('({},{})'.format(start[1] + (end[1] - start[1]) / 2, start[0] + (end[0] - start[0]) / 2))
                # print("方块数量：{}".format(int(((end[0] - start[0]) / (H / N)) * ((end[1] - start[1]) / (W / N)) + 0.5)))
                grids.append(grid)
                centers.append((start[1] + (end[1] - start[1]) / 2, start[0] + (end[0] - start[0]) / 2))
                counts.append(int(((end[0] - start[0]) / (H / N)) * (
                            (end[1] - start[1]) / (W / N)) + 0.5))  # int会直接截断取整数，如果想要使用四舍五入，则在int里面的数加0.5即可
                # i += 1
    # print(i)
    return grids, centers, counts
