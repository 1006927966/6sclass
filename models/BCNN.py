import torch
import torchvision
from config.value_config import *

def make_model(key):
    return BCNN(key)


class BCNN(torch.nn.Module):
    def __init__(self, key):
        super(BCNN, self).__init__()
        backbone = torchvision.models.__dict__['resnext101_32x8d'](pretrained=True)
        self.layer0 = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.conv = torch.nn.Conv2d(self.layer3[-1].conv1.in_channels, 512, 1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=512*512, out_features=NUMCLASS),
        )


    def forward(self, x, weights=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv(x)
        x = self.relu(x)
        N = x.size()[0]
        h = x.size()[-1]
        x = torch.reshape(x, (N, 512, h*h))
        x = torch.bmm(x, torch.transpose(x, 1, 2))/(h*h)
        x = torch.reshape(x, (N, 512*512))
        x = torch.sqrt(x+1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    c = np.ones((224, 224, 3))
    img = Image.fromarray(np.uint8(c))
    img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    net = make_model('resnext101_32x8d')
    x = net(img)
    print(x)