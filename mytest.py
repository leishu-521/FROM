# # import netron
# #
# # modelData = "./pretrained/model_p4_baseline_9938_8205_3610.pth.tar"
# # netron.start(modelData)
# #
# # import torch
# # import torchvision.models as models
# #
# # checkpoint = \
# #     torch.load("./pretrained/model_p4_baseline_9938_8205_3610.pth.tar") # 加载模型
# # print(checkpoint.keys()) # 查看模型元素
# # state_dict = checkpoint['state_dict']
# #
# # # print(checkpoint['epoch'])
# # # print(checkpoint['arch'])
# # # print(checkpoint['best_prec1'])
# # print(checkpoint)
#
#
# #
# # import torch
# #
# # # a = torch.randn([64,128])
# # # print(a.shape)
# # #
# # # batch = torch.nn.BatchNorm1d(128)
# # # b = batch(a)
# # # print(b.shape)
# # # c = sum(b[0:-1][0])
# # # print(c)
# # mask = None
# # if not isinstance(mask, torch.Tensor):
# #     print("为none，执行")
# #
# # # print("为none，不执行")
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # def get_grids(H, W, N):
# #     grid_ori = np.zeros((H, W))
# #
# #     x_axis = np.linspace(0, W, N+1, True, dtype=int)
# #     y_axis = np.linspace(0, H, N+1, True, dtype=int)
# #
# #     vertex_set = []
# #     for y in y_axis:
# #         for x in x_axis:
# #             vertex_set.append((y, x))
# #
# #     grids = [grid_ori]
# #     for start in vertex_set:
# #         for end in vertex_set:
# #             if end[0] > start[0] and end[1] > start[1]:
# #                 grid = grid_ori.copy()
# #                 grid[start[0]:end[0], start[1]:end[1]] = 1.0
# #                 grids.append(grid)
# #     return grids
# #
# # def plot_grid(grid):
# #     plt.imshow(grid, cmap='gray', interpolation='none')
# #     plt.show()
# #
# # # 示例用法
# # # 假设 grids 是你从 get_grids 函数获得的列表
# # grids = get_grids(H=112, W=96, N=4)
# #
# # # 绘制第一个网格
# # for i in range(10):
# #     plot_grid(grids[i])
#
# # def main():
# #     import lib.core.utils as utils
# #     import random
# #
# #     img_occ, mask, _ = utils.occluded_image_ratio(img.copy(), occ, factor)
# #
# #     # cal mask label
# #     mask_label = utils.cal_similarity_label(grids, mask)
# #
# #
# #     def occluded_image_ratio(img, occ, factor=1.0):
# #         W, H = img.size
# #         occ_w, occ_h = occ.size
# #
# #         new_w, new_h = min(W - 1, int(factor * occ_w)), min(H - 1, int(factor * occ_h))
# #         occ = occ.resize((new_w, new_h))
# #
# #         center_x = random.choice(range(0, W))
# #         center_y = random.choice(range(0, H))
# #
# #         start_x = center_x - new_w // 2
# #         start_y = center_y - new_h // 2
# #
# #         end_x = center_x + (new_w + 1) // 2
# #         end_y = center_y + (new_h + 1) // 2
# #         # occlude the img
# #         # box = (x, y, x+new_w, y+new_h)
# #         box = (start_x, start_y, end_x, end_y)
# #         img.paste(occ, box)
# #
# #         # cal the corresponding mask
# #         start_x = max(start_x, 0)
# #         start_y = max(start_y, 0)
# #         end_x = min(W - 1, end_x)
# #         end_y = min(H - 1, end_y)
# #         mask = np.zeros((H, W))
# #         mask[start_y:end_y, start_x:end_x] = 1.0
# #
# #         ratio = ((end_y - start_y) * (end_x - start_x)) / float(H * W)
# #
# #         return img, mask, ratio
# #
# #     def cal_similarity_label(grids, mask):
# #         scores = []
# #         for i, grid in enumerate(grids):
# #             score = cal_IoU(grid, mask)
# #             scores.append(score)
# #         occ_label = np.argmax(scores)
# #         return occ_label
# #
# #     def cal_IoU(mask1, mask2):
# #         inter = np.sum(mask1 * mask2)
# #         union = np.sum(np.clip(mask1 + mask2, 0, 1)) + 1e-10
# #         return inter / union
#
#
# import numpy as np
#
# # def get_grid_info(H, W, N):
# #     grid_centers = []
# #     grid_counts = []
# #
# #     x_axis = np.linspace(0, W, N + 1, True, dtype=int)
# #     y_axis = np.linspace(0, H, N + 1, True, dtype=int)
# #
# #     for start_y in y_axis:
# #         for start_x in x_axis:
# #             end_y = min(H, start_y + H // N)
# #             end_x = min(W, start_x + W // N)
# #
# #             center_y = (start_y + end_y) // 2
# #             center_x = (start_x + end_x) // 2
# #
# #             grid_centers.append((center_y, center_x))
# #             grid_counts.append(N * N)
# #
# #     return grid_centers, grid_counts
# #
# # # Example Usage:
# # H, W, N = 112, 96, 4
# # centers, counts = get_grid_info(H, W, N)
# # print("Grid Centers:", centers)
# # print("Grid Counts:", counts)
#
# import numpy as np
#
# import numpy as np
# import math
#
#
# # def get_grids_info(H, W, N):
# #     grid_ori = np.zeros((H, W))
# #
# #     x_axis = np.linspace(0, W, N+1, True, dtype=int)
# #     y_axis = np.linspace(0, H, N+1, True, dtype=int)
# #
# #     vertex_set = []
# #     for y in y_axis:
# #         for x in x_axis:
# #             vertex_set.append((y, x))
# #
# #     grids = [grid_ori]
# #     centers = []
# #     occupied_blocks = []
# #
# #     for start_y in y_axis:
# #         for start_x in x_axis:
# #             end_y = min(H, start_y + H // N)
# #             end_x = min(W, start_x + W // N)
# #
# #             center_y = (start_y + end_y) // 2
# #             center_x = (start_x + end_x) // 2
# #
# #             grid = grid_ori.copy()
# #             grid[start_y:end_y, start_x:end_x] = 1.0
# #             grids.append(grid)
# #
# #             centers.append((center_y, center_x))
# #
# #             # Calculate the number of blocks occupied by the rectangle
# #             occupied_block_count = (end_y - start_y) // (H // N) * (end_x - start_x) // (W // N)
# #             occupied_blocks.append(occupied_block_count)
# #
# #     return grids, centers, occupied_blocks
# #
# # # Example Usage:
# # H, W, N = 112, 96, 4
# # grids, centers, counts = get_grids_info(H, W, N)
# # print("Grids:", grids)
# # print("Centers:", centers)
# # print("Occupied Blocks:", counts)

import numpy as np
import math
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
                counts.append(int(((end[0] - start[0]) / (H / N)) * ((end[1] - start[1]) / (W / N)) + 0.5)) # int会直接截断取整数，如果想要使用四舍五入，则在int里面的数加0.5即可
                # i += 1
    # print(i)
    return grids, centers, counts

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def mask_Mse(centers, counts, vec, mask_label, N):
    grids_counts = (N * (N + 1) / 2 ) ** 2 + 1
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
        print(a,b)
    center_squared_diff_sum = sum(calculate_distance(a, b) for a, b in zip(mask_centers, mask_centers_label))
    print("center_squared_diff_sum:{}".format(center_squared_diff_sum))
    centers_mse = center_squared_diff_sum / n / 147 # 112*96 的图像两点距离最大为147
    print("centers_mse:{}".format(centers_mse))
    counts_squared_diff_sum = sum(abs(a - b) for a, b in zip(mask_counts, mask_counts_label))
    print("counts_squared_diff_sum:{}".format(counts_squared_diff_sum))
    counts_mse = counts_squared_diff_sum / n / (N * N)
    print("counts_mse:{}".format(counts_mse))

    mse = centers_mse + counts_mse
    return mse

# for a, b in zip(mask_centers, mask_centers_label):
#     print(a,b)


import random
import matplotlib.pyplot as plt
import time

# print(dataset[0])
H, W, N = 112, 96, 5
grids, centers, counts = get_grids(H, W, N)
vec = random.sample(range(101),8)
mask_label = random.sample(range(101), 8)
print("centers:{}".format(centers))
print("counts:{}".format(counts))
mask_Mse(centers, counts, vec, mask_label, 5)


# print("Grids:", grids)
# print("Centers:", centers)
# print("Occupied Blocks:", counts)
# for i in range(101):
#     plt.imshow(grids[i])
#     plt.show()
# print(len(get_grids(H, W, N)[0]))
# print(len(centers), centers)
# print(len(counts), counts)
# for i in range(37):
#     time.sleep(0.5)
#     plt.imshow(grids[i], cmap ='gray', vmin=0, vmax=1)
#     plt.show()
# a = np.ones((H, W))
# a[0][0] = 0

# print(a)
# plt.imshow(a, cmap='gray')
# plt.show()
#
#
# # def get_grids(H, W, N):
# #     grid_ori = np.zeros((H, W))
# #     centers = []
# #     counts = []
# #
# #     x_axis = np.linspace(0, W, N + 1, True, dtype=int)
# #     y_axis = np.linspace(0, H, N + 1, True, dtype=int)
# #
# #     vertex_set = []
# #     for y in y_axis:
# #         for x in x_axis:
# #             vertex_set.append((y, x))
# #
# #     grids = [grid_ori]
# #     grid_ori_centers = (0, 0)
# #     grid_ori_counts = 0
# #     centers.append(grid_ori_centers)
# #     counts.append(grid_ori_counts)
# #
# #     i = 0
# #     for start in vertex_set:
# #         for end in vertex_set:
# #             if end[0] > start[0] and end[1] > start[1]:
# #                 grid = grid_ori.copy()
# #                 grid[start[0]:end[0], start[1]:end[1]] = 1.0
# #                 # if int(((end[0] - start[0]) / (H / N)) * ((end[1] - start[1]) / (W / N)) + 0.5) == 9:
# #                 #     print(grid)
# #                 # print("左上角坐标：({},{}),右下角坐标({},{})".format(start[1], start[0], end[1], end[0]))
# #                 # print('({},{})'.format(start[1] + (end[1] - start[1]) / 2, start[0] + (end[0] - start[0]) / 2))
# #                 # print("方块数量：{}".format(int(((end[0] - start[0]) / (H / N)) * ((end[1] - start[1]) / (W / N)) + 0.5)))
# #                 grids.append(grid)
# #                 centers.append((start[1] + (end[1] - start[1]) / 2, start[0] + (end[0] - start[0]) / 2))
# #                 counts.append(int(((end[0] - start[0]) / (H / N)) * ((end[1] - start[1]) / (W / N)) + 0.5)) # int会直接截断取整数，如果想要使用四舍五入，则在int里面的数加0.5即可
# #                 i += 1
# #     print(i)
# #     return grids, centers, counts
#
