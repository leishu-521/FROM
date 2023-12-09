# import netron
#
# modelData = "./pretrained/model_p4_baseline_9938_8205_3610.pth.tar"
# netron.start(modelData)

import torch
import torchvision.models as models

checkpoint = \
    torch.load("./pretrained/model_p4_baseline_9938_8205_3610.pth.tar") # 加载模型
print(checkpoint.keys()) # 查看模型元素
state_dict = checkpoint['state_dict']

# print(checkpoint['epoch'])
# print(checkpoint['arch'])
# print(checkpoint['best_prec1'])
print(checkpoint)




# 
# import torch
# 
# # a = torch.randn([64,128])
# # print(a.shape)
# #
# # batch = torch.nn.BatchNorm1d(128)
# # b = batch(a)
# # print(b.shape)
# # c = sum(b[0:-1][0])
# # print(c)
# mask = None
# if not isinstance(mask, torch.Tensor):
#     print("为none，执行")
# 
# # print("为none，不执行")
