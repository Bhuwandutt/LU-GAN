import torch
import torch.nn as nn
import numpy as np


def Convblock(in_planes,
              out_planes,
              kernel = 3,
              stride=1,
              padding = 1):
    # convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                     padding=padding,bias = False)


class Discriminator(nn.Module):
    # Patch discriminator, success in 256

    def __init__(self,
                 base_feature,
                 txt_input_dim,
                 down_rate,
                 norm="InstanceNorm"):
        super(Discriminator, self).__init__()
        self.feature_base_dim = base_feature
        self.txt_input_dim = txt_input_dim
        self.down_rate = down_rate
        self.txt_dim = 128

        if norm == "InstanceNorm":
            self.norm2d = nn.InstanceNorm2d
        elif norm == "BatchNorm":
            self.norm2d = nn.BatchNorm2d

        kw = 3
        padw = int(np.floor((kw - 1.0) / 2))

        # 32 x 32 x 1 -> 32 x 32 x bf
        nf = self.feature_base_dim
        self.conv1 = nn.Sequential(
                        Convblock(1,
                                  nf,
                                  kernel=kw,
                                  stride=1,
                                  padding=padw),
                        nn.LeakyReLU(0.2, True))

        # 32 x 32 x bf -> 32 x 32 x bf
        self.downs = nn.Sequential()
        for i in range(self.down_rate):
            nf_pre = nf
            nf = min(nf*2,512)
            block = nn.Sequential(
                Convblock(nf_pre, nf, kernel=kw, stride=2, padding=padw),
                self.norm2d(nf),
                nn.LeakyReLU(0.2, True)
            )
            self.downs.add_module('down_{}'.format(i),block)

        # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
        self.textfc = nn.Linear(self.txt_input_dim, self.txt_dim)
        self.textBN = self.norm2d(self.txt_dim)
        self.textAcc = nn.LeakyReLU(0.2, True)

        nf = nf + self.txt_dim
        self.output = nn.Sequential(
            nn.Conv2d(nf,
                      1,
                      kernel_size=4,
                      padding =1)
        )

        def forward(self, x, txt_embedding):
            # 64 x 64 x 1 -> 64 x 64 x bf
            x = self.conv1(x)

            x = self.downs(x)
            s_size = x.size(2)
            # 1 x 1 x input_dim -> 1 x 1 x (8 x bf)
            embedding = self.textfc(txt_embedding)

            # 1 x 1 x (8 x bf) -> 4 x 4 x (8 x bf)
            embedding = embedding.view(-1, self.txt_dim, 1, 1)
            embedding = embedding.repeat(1, 1, s_size, s_size)
            embedding = self.textBN(embedding)
            embedding = self.textAcc(embedding)

            # 4 x 4 x (8 x bf)+ 4 x 4 x (8 x bf) -> 4 x 4 x (16 x bf)
            x = torch.cat((x, embedding), dim=1)

            # 4 x 4 x (16 x bf) -> 1
            return self.output(x)
