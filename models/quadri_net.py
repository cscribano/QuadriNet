import torch
from torch import nn
from torchvision.models import vgg16

from models import BaseModel

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        # type: (int, int, int, int) -> BasicBlock
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(True),
        )

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return self.main(x)


class QuadriFcn(BaseModel):

    def __init__(self, n_class=1):
        super().__init__()

        features = vgg16(pretrained=True).features
        features[0] = nn.Conv2d(3,64,kernel_size=3, stride=1, padding=100)
        #print(features)

        self.pool3 = nn.Sequential(features[:17])#1/8
        self.pool4 = nn.Sequential(features[17:24])#1/16
        self.pool5 = nn.Sequential(features[24:])#1/32

        # conv6
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        # conv7
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        self.score_conv7 = nn.Conv2d(4096, n_class, 1)

        self.upsample_conv7 = nn.ConvTranspose2d(n_class, n_class, 4, 2, padding=1) #deconv-1 = (conv7*2): 1/32 -> 1/16
        self.upsample_dc1 = nn.ConvTranspose2d(n_class, n_class, 4, 2, padding=1) #deconv-2 = (deconv-1*2): 1/16->1/8
        self.upsample_out = nn.ConvTranspose2d(n_class, n_class, 16, 8, padding=4) #(out*8): 1:8 -> 1


    def forward(self, x):
        h = x

        #input -> pool3
        h = self.pool3(h)
        pool3 = h #1/8

        #pool4
        h = self.pool4(h)
        pool4 = h #1/16
        #pool5
        h = self.pool5(h)#1/32
        #conv6
        h = self.conv6(h)
        #conv7
        h = self.conv7(h)

        #deconv-1
        h = self.score_conv7(h)
        h = self.upsample_conv7(h) #1/32 -> 1/16
        upsample_conv7 = h  # 1/16

        h = self.score_pool4(pool4) #512 -> 1
        h = h[:, :, 1:1 + upsample_conv7.size()[2], 1:1 + upsample_conv7.size()[3]]
        score_pool4 = h  # 1/16

        #print(upsample_conv7.shape)
        #print(score_pool4.shape)
        h = torch.add(score_pool4, upsample_conv7) #1/16

        #deconv-2
        h = self.upsample_dc1(h) #1/16 -> 1/8
        upsample_dc1 = h  # 1/8

        h = self.score_pool3(pool3) #256 -> 1
        h = h[:, :,
            2:2 + upsample_dc1.size()[2],
            2:2 + upsample_dc1.size()[3]]
        score_pool3 = h  # 1/8

        #print(upsample_dc1.shape)
        #print(score_pool3.shape)
        h = torch.add(upsample_dc1, score_pool3) #1/8

        #output
        h = self.upsample_out(h)
        h = h[:, :, 115:115 + x.size()[2], 115:115 + x.size()[3]].contiguous()

        return h