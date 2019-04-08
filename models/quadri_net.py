import torch
from torch import nn
import torch.nn.functional as F
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

        vgg = vgg16(pretrained=True)
        features = vgg.features

        self.pool3 = nn.Sequential(features[:17])#1/8
        self.pool4 = nn.Sequential(features[17:24])#1/16
        self.pool5 = nn.Sequential(features[24:])#1/32

        # conv6
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        #Load weights from vgg-fc1
        self.conv6[0].load_state_dict({
            "weight": vgg.classifier[0].state_dict()["weight"].view(4096,512, 7, 7),
            "bias": vgg.classifier[0].state_dict()["bias"]#4096
        })

        # conv7
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        #Load weights from vgg-fc2
        self.conv7[0].load_state_dict({
            "weight": vgg.classifier[3].state_dict()["weight"].view(4096,4096, 1, 1),
            "bias": vgg.classifier[3].state_dict()["bias"]
        })

        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        self.score_conv7 = nn.Conv2d(4096, n_class, 1, padding=1)

        self.upsample_conv7 = nn.ConvTranspose2d(n_class, n_class, 4, 2, padding=0, output_padding=1) #deconv-1 = (conv7*2): 1/32 -> 1/16
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
        h = self.pool5(h)
        #conv6
        h = self.conv6(h)
        #conv7
        h = self.conv7(h)

        #deconv-1
        h = self.score_conv7(h)

        #L,R,T,B
        h = F.pad(h, [1,2,1,2])
        h = self.upsample_conv7(h)
        upsample_conv7 = h  # 1/16

        h = self.score_pool4(pool4)

        score_pool4 = h  # 1/16

        h = torch.add(score_pool4, upsample_conv7) #1/16

        #deconv-2
        h = self.upsample_dc1(h) #1/16 -> 1/8
        upsample_dc1 = h  # 1/8

        h = self.score_pool3(pool3) #256 -> 1

        score_pool3 = h  # 1/8

        h = torch.add(upsample_dc1, score_pool3) #1/8

        #output
        #output
        h = F.pad(h, [0,0,0,1])
        h = self.upsample_out(h)
        h = h[:, :, 3:3 + x.size()[2]].contiguous()

        return h

if __name__ == '__main__':
    nn = QuadriFcn()
    nn.eval()

    #crop 405,720 -> 384, 704
    t = torch.randn(1,3,405,720) #-> 135*240

    x = nn(t)
    print(x.shape)