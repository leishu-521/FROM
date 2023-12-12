# import netron
#
# modelData = "./pretrained/model_p4_baseline_9938_8205_3610.pth.tar"
# netron.start(modelData)
#
# import torch
# import torchvision.models as models
#
# checkpoint = \
#     torch.load("./pretrained/model_p4_baseline_9938_8205_3610.pth.tar") # 加载模型
# print(checkpoint.keys()) # 查看模型元素
# state_dict = checkpoint['state_dict']
#
# # print(checkpoint['epoch'])
# # print(checkpoint['arch'])
# # print(checkpoint['best_prec1'])
# print(checkpoint)
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


import torch
import torch.nn as nn


class CustomAttention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(CustomAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim * 3, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, q, k, v):
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        B, N, C = q.shape

        qkv = self.qkv(torch.cat([q, k, v], dim=-1))\
            .reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x



if __name__ == "__main__":
    fmap = torch.randn([32, 42, 512])
    mask = torch.randn([32, 42, 512])
    attention = CustomAttention(512)
    a = attention(fmap, mask, fmap)
    print(a.shape)

