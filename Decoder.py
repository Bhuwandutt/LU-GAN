import torch
from torch import nn
import torch.nn.functional as f


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1,bias=False)


# Upscale the spatial size by a factor of 2
def upBlock(in_planes,
            out_planes,
            use_batchnorm =True):
    block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv3x3(in_planes, out_planes),
            nn.LeakyReLU(0.2, True))

    return block


class ResBlockv2(nn.Module):
    def __init__(self, channels):
        super(ResBlockv2, self).__init__()
        self.conv1 = conv3x3(channels, channels)
        self.prelu = nn.LeakyReLU(0.2, True)
        self.conv2 = conv3x3(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)

        return x + residual


class Decoder(nn.Module):
    def __init__(self,
                 input_dim,
                 feature_base_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.feature_base_dim = feature_base_dim
        self.num_res = 3

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_base_dim//2),
            nn.LeakyReLU(0.2, True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.feature_base_dim//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True))

        res_layers = [ResBlockv2(self.feature_base_dim) for i in range(self.num_res)]
        self.Res_block = nn.Sequential(*res_layers)
        self.upblock = upBlock(self.feature_base_dim, self.feature_base_dim // 2)

        self.to_img = nn.Sequential(
            nn.Conv2d(self.feature_base_dim // 2, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, image, c_code):
        x = self.conv1(image)
        s_size = x.size(2)

        c_code = self.fc(c_code)
        c_code = c_code.view(-1, self.feature_base_dim//2, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)

        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, x), 1)

        # state size ngf x in_size x in_size
        out_code = self.Res_block(h_c_code)
        out_code = self.upblock(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        pre_img = self.to_img(out_code)

        return pre_img*0.5 + f.interpolate(image,scale_factor=2,mode="bilinear")*0.5