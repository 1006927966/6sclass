import torch
import torchvision
from config.value_config import *

def make_model(key):

    return ResNextmulti(key)


class ResNextmulti(torch.nn.Module):
    def __init__(self, key):
        super(ResNextmulti, self).__init__()
        backbone = torchvision.models.__dict__['resnext101_32x8d'](pretrained=True)
        self.layer0 = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=self.layer4[-1].conv1.in_channels, out_features=NUMCLASS),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=self.layer4[-1].conv1.in_channels, out_features=2),
        )

        pass

    def forward(self, x, weights=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2
