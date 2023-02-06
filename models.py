import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import resnet152
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

        # self.features_conv = self.features[:40]

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32768, out_features=500),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=500, out_features=500),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=500, out_features=1),
        )

    def forward(self, x):
        out = self.features(x)  # conv layers
        out = self.linear_layers(out)  # fully-connected layers
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(x),
                    nn.LeakyReLU(),
                ]
                in_channels = x
        return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None

        # PRETRAINED MODEL
        self.pretrained = resnet152(pretrained=True)
        self.pretrained.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
        self.layerhook.append(
            self.pretrained.layer4.register_forward_hook(self.forward_hook())
        )

        for p in self.pretrained.parameters():
            p.requires_grad = True

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out
