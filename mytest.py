import netron
#
# # modelData = "./pretrained/model_p4_baseline_9938_8205_3610.pth.tar"
modelData = '/home/leishu/pycode/FROM/output/WebFace/LResNet50E_IR_FPN_CosMargin-Mask/CASIA-112x96-LMDB-Mask-pattern_5-weight_1-lr_0.01-optim_sgd-pretrained_1_factor_1/'

# modelData = '/home/leishu/pycode/FROM/output/WebFace/LResNet50E_IR_FPN_CosMargin-Mask/CASIA-112x96-LMDB-Mask-pattern_5-weight_1-lr_0.01-optim_sgd-pretrained_1_factor_1/model_best_p5_2023-12-28-16-35_0.9905_0.9405_0.5527_作者给的预训练权重_改动MD模块.pth.tar'
netron.start(modelData+"model_best_p5_2023-12-14-14-37_0.9915_0.9588_0.7163.pth.tar")

# import torch
# import torchvision.models as models
#
# # checkpoint = \
# #     torch.load("./pretrained/model_p4_baseline_9938_8205_3610.pth.tar") # 加载模型
# model_root = '/home/leishu/pycode/FROM/output/WebFace/LResNet50E_IR_FPN_CosMargin-Mask/CASIA-112x96-LMDB-Mask-pattern_5-weight_1-lr_0.01-optim_sgd-pretrained_1_factor_1/'
# checkpoint = torch.load(model_root + "model_best_p5_2023-12-14-14-37_0.9915_0.9588_0.7163.pth.tar")
# print(checkpoint.keys()) # 查看模型元素
# state_dict = checkpoint['state_dict']
# #
# # # print(checkpoint['epoch'])
# # # print(checkpoint['arch'])
# # # print(checkpoint['best_prec1'])
# print(state_dict)
#
#
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



# 以下为各个模块调试
#
# import torch
# import torch.nn as nn
#
#
# class CustomAttention(nn.Module):
#     def __init__(self,
#                  dim,  # 输入token的dim
#                  num_heads=8,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(CustomAttention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim * 3, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#
#     def forward(self, q, k, v):
#         # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
#         # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
#         # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         B, N, C = q.shape
#
#         qkv = self.qkv(torch.cat([q, k, v], dim=-1)) \
#             .reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#
#         # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class PatchEmbed(nn.Module):
#     """
#     2D Image to Patch Embedding
#     """
#
#     def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
#         super().__init__()
#         img_size = (img_size, img_size)
#         patch_size = (patch_size, patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x
#
# class Regress(nn.Module):
#     def __init__(self, input_channels, num_mask):
#         super().__init__()
#         self.input_channels = input_channels
#         self.num_mask = num_mask
#         self.softmax_a = nn.Softmax(dim=1)
#         self.conv1x1 = nn.Conv2d(self.input_channels, 2, 1, 1, 0)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(self.input_channels, self.num_mask)
#
#     def forward(self, x):
#         out = self.conv1x1(x)
#         out = self.softmax_a(out)
#         out, _ = torch.max(out, dim=1, keepdim=True)
#         x1 = x * out.repeat(1, 512, 1, 1)
#         # 全局平均池化
#         x1 = self.avg_pool(x1)
#         if x1.shape[0] != 1:
#             x1 = x1.squeeze()
#         else:
#             x1 = x1.squeeze(2).squeeze(2)
#         x1 = self.fc(x1)
#         return x1
#
#
# class MaskDecoder(nn.Module):
#     def __init__(self, input_channels, out_channels):
#         super().__init__()
#         self.input_channels = input_channels
#         self.out_channels = out_channels
#         self.softmax_a = nn.Softmax(dim=1)
#         self.conv1x1 = nn.Conv2d(self.input_channels, out_channels, 1, 1, 0)
#         # self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # self.fc = nn.Linear(self.input_channels, self.num_mask)
#
#     def forward(self, x):
#         out = self.conv1x1(x)
#         out = self.softmax_a(out)
#         out, _ = torch.max(out, dim=1, keepdim=True)
#         x1 = x * out.repeat(1, self.input_channels, 1, 1)
#         # # 全局平均池化
#         # x1 = self.avg_pool(x1)
#         # if x1.shape[0] != 1:
#         #     x1 = x1.squeeze()
#         # else:
#         #     x1 = x1.squeeze(2).squeeze(2)
#         # x1 = self.fc(x1)
#         return x1
#
#
#
#
# if __name__ == "__main__":
#     a = torch.randn([8, 512, 7, 6])
#     maskdecoder = MaskDecoder(512, 2)
#     out = maskdecoder(a)
#     print(out.shape)
#     # a = torch.randn([8, 512, 7, 6])
#     # regress = Regress(512, 101)
#     # out = regress(a)
#     # print(out.shape)
#     # fmap = torch.randn([32, 42, 512])
#     # mask = torch.randn([32, 42, 512])
#     # attention = CustomAttention(512)
#     # a = attention(fmap, mask, fmap)
#     # print(a.shape)
#     # a = torch.randn(8, 512, 7, 6)
#     # softmax_a = nn.Softmax(dim=1)
#     # conv1x1 = nn.Conv2d(512, 2, 1, 1, 0)
#     # a_mid = conv1x1(a)
#     #
#     # b = softmax_a(a_mid)
#     # c, _ = torch.max(b, dim=1, keepdim=True)
#     # a = a * c.repeat(1, 512, 1, 1)
#     #
#     # # 全局平均池化
#     # avg_pool = nn.AdaptiveAvgPool2d(1)
#     # a = avg_pool(a).squeeze(2).squeeze(2)
#     #
#     # # 全连接层
#     # fc = nn.Linear(512, 101)
#     # a = fc(a)
#     #
#     # # b = torch.squeeze(a，N)
#     # print(a.shape)

