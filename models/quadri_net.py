import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50
from torchvision.models.resnet import Bottleneck

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


class Upsample2Bloc(nn.Module):

    def __init__(self, insize, outsize, padding=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(insize, outsize, 4, 2, padding=padding),
            nn.BatchNorm2d(outsize),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.main(x)


class QuadriNetFancy(BaseModel):

    def __init__(self, n_class=1):
        super(QuadriNetFancy, self).__init__()
        features = resnet50(pretrained=True)

        self.conv1 = features.conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = features.layer1
        self.layer2 = features.layer2
        self.layer3 = features.layer3
        self.layer4 = features.layer4

        #x2 upsamples + residual block evry 2 upsamples
        self.upsample1 = Upsample2Bloc(2048, 1024) #1/32->1/16
        self.upsample2 = Upsample2Bloc(1024, 512) #1/16->1/8
        self.residual1 = Bottleneck(512, 128)

        self.upsample3 = Upsample2Bloc(512, 256) #1/8 -> 1/4
        self.upsample4 = Upsample2Bloc(256, 64) #1/4 -> 1/2
        self.residual2 = Bottleneck(64, 16)

        #single x2 upsample + residual block
        self.upsample5 = Upsample2Bloc(64, 32) #1/2 -> 1
        self.residual3 = Bottleneck(32, 8)

        self.out = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        #Downsample phase
        x = self.conv1(x)#64x203x360 (1/2)
        conv1 = F.pad(x,[4,4,3,2]) #208x368

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#64x102x180 (1/4)

        x = self.layer1(x)#256x102x180 (1/4)
        layer1 = F.pad(x,[2,2,1,1]) #104x184

        x = self.layer2(x)#512x51x90 (1/8)
        layer2 = F.pad(x,[1,1,0,1])#52x92

        x = self.layer3(x)  # 1024x26x45 (1/16)
        layer3 = F.pad(x, [0,1,0,0]) #26x46

        x = self.layer4(x)  #2048x13x23 (1/32)

        #Upsample phase
        x = self.upsample1(x) #1/16 (26x46)
        upsample1 = x
        torch.add(upsample1, layer3)

        x = self.upsample2(x) #1/8 (52x90)
        upsample2 = x
        torch.add(upsample2, layer2)

        x = self.residual1(x)

        x = self.upsample3(x) #1/4 (104x184)
        upsample3 = x
        torch.add(upsample3, layer1)

        x = self.upsample4(x) #1/2 (208x368)
        upsample4 = x
        x = torch.add(upsample4, conv1)

        x = self.residual2(x)

        x = self.upsample5(x) #1 (832x
        upsample5 = x
        torch.add(upsample5, x)

        x = self.residual3(x)

        x = x[:,:,11:11+x.size()[2],16:16 + x.size()[3]].contiguous()
        x = self.out(x)

        return x


#-------------
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
    nn = QuadriNetFancy()
    nn.eval()

    #crop 405,720 -> 384, 704
    t = torch.randn(1,3,405,720) #-> 135*240

    x = nn(t)
    print(x.shape)