import torch.nn as nn
import torch

class CustomConvolution(nn.Module):
    def __init__(self):
        super(CustomConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 8, 64
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, 2)

        # self.dense1 = nn.Linear(512, 1000)
        # self.dense2 = nn.Linear(1000, 100)
        # self.dense3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x