import torch
import torch.nn as nn


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class LResNet_Occ_FC(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_Occ_FC, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        # --Begin--Triplet branch
        # 把掩码作为向量的形式
        self.mask = nn.Sequential(
            nn.BatchNorm1d(64 * 7 * 6),
            nn.Linear(64 * 7 * 6, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
            nn.Sigmoid(),
        )
        self.fpn = PyramidFeatures(filter_list[2], filter_list[3], filter_list[4])

        self.reduces = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(64),
            nn.BatchNorm2d(64)
        )

        # 此处是OPP预测结构
        self.regress = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(512, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )
        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 7 * 6, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        fmap = self.layer4(x3)

        # generate mask
        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            fmap_reduce = self.reduces(features[0])
            mask = self.mask(fmap_reduce.reshape(fmap_reduce.size(0), -1))

        # regress
        vec = self.regress(mask)

        fc = self.fc(fmap.reshape(fmap.size(0), -1))

        fc_mask = fc * mask

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ_FC(is_gray=False, num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    model = LResNet_Occ_FC(BlockIR, layers, filter_list, is_gray)
    return model


class LResNet_Occ_2D(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        self.filter_list = filter_list
        super(LResNet_Occ_2D, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        # --Begin--Triplet branch 
        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(filter_list[4]),
            nn.BatchNorm2d(filter_list[4]),
            nn.Conv2d(filter_list[4], 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.fpn = PyramidFeatures(filter_list[2], filter_list[3], filter_list[4])

        self.regress = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 7 * 6, filter_list[5], bias=False),
            nn.BatchNorm1d(filter_list[5]),
        )
        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 7 * 6, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        fmap = self.layer4(x3)

        # generate mask
        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            mask = self.mask(features[0])
            mask = mask.repeat(1, self.filter_list[4], 1, 1)

        # regress
        vec = self.regress(mask.reshape(mask.size(0), -1))

        fmap_mask = fmap * mask

        fc_mask = self.fc(fmap_mask.reshape(fmap_mask.size(0), -1))

        fc = self.fc(fmap.reshape(fmap.size(0), -1))

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ_2D(is_gray=False, num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    model = LResNet_Occ_2D(BlockIR, layers, filter_list, is_gray)
    return model

class LResNet_Occ(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_Occ, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        # --Begin--Triplet branch 
        self.mask = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.PReLU(256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
            # nn.PReLU(filter_list[4]),
            # nn.BatchNorm2d(filter_list[4]),
            nn.Sigmoid(),
        )
        # 下面是MD自研模块
        # self.mask = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.PReLU(256),
        #     nn.BatchNorm2d(256),
        #     nn.Conv2d(256, filter_list[4], kernel_size=3, stride=2, padding=1, bias=False),
        #     MaskDecoder(filter_list[4], 2),
        #     # nn.PReLU(filter_list[4]),
        #     # nn.BatchNorm2d(filter_list[4]),
        #     nn.Sigmoid(),
        # )

        self.fpn = PyramidFeatures(filter_list[2], filter_list[3], filter_list[4])
        self.attention = CustomAttention(512)

        # self.regress = nn.Sequential(
        #     nn.BatchNorm1d(filter_list[4]*7*6),
        #     nn.Dropout(p=0.5),  # No drop for triplet dic
        #     nn.Linear(filter_list[4]*7*6, filter_list[5], bias=False),
        #     nn.BatchNorm1d(filter_list[5]),
        # )
        # self.regress1 = nn.Sequential(
        #     Regress(filter_list[4], filter_list[5]),
        #     nn.BatchNorm1d(filter_list[4] * 7 * 6),
        #     nn.Dropout(p=0.5),  # No drop for triplet dic
        #     nn.Linear(filter_list[4] * 7 * 6, filter_list[5], bias=False),
        #     nn.BatchNorm1d(filter_list[5]),
        # )
        self.regress2 = Regress(filter_list[4], filter_list[5])

        # --End--Triplet branch
        self.fc = nn.Sequential(
            nn.BatchNorm1d(filter_list[4] * 7 * 6),
            nn.Dropout(p=0.5),  # No drop for triplet dic
            nn.Linear(filter_list[4] * 7 * 6, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        fmap = self.layer4(x3)
        # print("fmap:".format(fmap.shape))

        # generate mask
        if not isinstance(mask, torch.Tensor):
            features = self.fpn([x2, x3, fmap])
            mask = self.mask(features[0])
        # print("mask:".format(mask.shape))

        # regress
        # vec = self.regress(mask.reshape(mask.size(0), -1))

        # leishu regress1
        vec = self.regress2(mask)

        ## 下面是自定义的transformer注意力机制
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        fmap_patch = fmap.flatten(2).transpose(1, 2)
        mask_patch = mask.flatten(2).transpose(1, 2)
        fmap_mask = self.attention(fmap_patch, mask_patch, fmap_patch)

        # fmap_mask = fmap * mask   #舍弃原来的哈达玛积的注意力机制的方式

        fc_mask = self.fc(fmap_mask.reshape(fmap_mask.size(0), -1))

        fc = self.fc(fmap.reshape(fmap.size(0), -1))

        return fc_mask, mask, vec, fc

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_Occ(is_gray=False, num_mask=101):
    filter_list = [64, 64, 128, 256, 512, num_mask]
    layers = [3, 4, 14, 3]
    model = LResNet_Occ(BlockIR, layers, filter_list, is_gray)
    return model


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

        qkv = self.qkv(torch.cat([q, k, v], dim=-1)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

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


# class Regress(nn.Module):
#     def __init__(self, input_channels, num_mask):
#         super().__init__()
#         self.input_channels = input_channels
#         self.num_mask = num_mask
#         self.BN = nn.BatchNorm2d(self.input_channels)
#         self.softmax_a = nn.Softmax(dim=1)
#         self.conv1x1 = nn.Conv2d(self.input_channels, 2, 1, 1, 0)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(self.input_channels, self.num_mask)
#
#     def forward(self, x):
#         out = self.BN(x)
#         out = self.conv1x1(x)
#         out = self.softmax_a(out)
#         out, _ = torch.max(out, dim=1, keepdim=True)
#         x1 = x * out.repeat(1, 512, 1, 1)
#
#         # 为了接上论文的后半部分，进行reshape处理
#         x1 = x1.reshape(x1.size(0), -1)
#         # # 全局平均池化
#         # x1 = self.avg_pool(x1)
#         # if x1.shape[0] != 1:
#         #     x1 = x1.squeeze()
#         # else:
#         #     x1 = x1.squeeze(2).squeeze(2)
#         # x1 = self.fc(x1)
#         return x1


class MaskDecoder(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.softmax_a = nn.Softmax(dim=1)
        self.conv1x1 = nn.Conv2d(self.input_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = self.softmax_a(out)
        out, _ = torch.max(out, dim=1, keepdim=True)
        x1 = x * out.repeat(1, self.input_channels, 1, 1)
        return x1


class Regress(nn.Module):
    def __init__(self, input_channels, num_mask):
        super().__init__()
        self.input_channels = input_channels
        self.num_mask = num_mask
        self.BN = nn.BatchNorm2d(self.input_channels)
        # self.softmax_a = nn.Softmax(dim=1)
        # self.conv1x1 = nn.Conv2d(self.input_channels, 2, 1, 1, 0)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(self.input_channels, self.num_mask)
        self.conv1x1 = nn.Conv2d(self.input_channels, 25, 1, 1, 0)
        self.softmax_a = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(12800, self.num_mask)


    def forward(self, x):
        # out = self.BN(x)
        out = self.conv1x1(x)
        out = self.softmax_a(out)
        y = []
        for i in range(25):
            p_i = out[:, i, :, :]
            p_i = torch.unsqueeze(p_i, 1)
            y_i = torch.mul(x, p_i)
            y_i = self.BN(y_i)
            y_i = self.relu(y_i)
            y_i = self.avgpool(y_i)
            y_i = torch.squeeze(y_i)
            # print(y_i.shape)
            y.append(y_i)

        m = torch.cat(y, 1)
        # print("m:{}".format(m.shape))
        m = self.fc(m)
        return m






if __name__ == "__main__":
    model = LResNet50E_IR_Occ()
    a = torch.randn([8, 3, 112, 96])
    b = model(a)
    print(b[2].shape)

    # regress = Regress(512, 101)
    # a = torch.randn([8, 512, 7, 6])
    # b = regress(a)
    # print(b.shape)


    # regress1 = nn.Sequential(
    #     Regress(512, 226),
    #     nn.BatchNorm1d(512 * 7 * 6),
    #     nn.Dropout(p=0.5),  # No drop for triplet dic
    #     nn.Linear(filter_list[4] * 7 * 6, filter_list[5], bias=False),
    #     nn.BatchNorm1d(filter_list[5]),
    # )
