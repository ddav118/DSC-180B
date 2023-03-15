import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import resnet152
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from typing import Type
import torchvision.models as models
from torchvision.models import ResNet152_Weights


class VanillaResNet(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(VanillaResNet, self).__init__()
        self.resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)
        return x


class ResNet152(nn.Module):
    def __init__(self, num_classes, input_channels=3, data_features=4):
        super(ResNet152, self).__init__()
        self.resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features + data_features, num_classes
        )

    def forward(self, images: Tensor, data: Tensor) -> Tensor:
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)  # feature vector
        # append the clinical data features to the feature vector (convolutional output features)
        comb = torch.cat((x, data), dim=1)
        x = self.resnet.fc(comb)
        return x
