import torch as th
from torch import nn


class Residual_Block(nn.Module):
    def __init__(self, input_dim : int = 256, feature_dim : int = 32, activation : nn.Module = nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        
        self.cnn1 = nn.Conv2d(input_dim, feature_dim, kernel_size=(3,3), stride=1, padding="same")
        self.cnn2 = nn.Conv2d(feature_dim, input_dim, kernel_size=(3,3), stride=1, padding="same")
        self.activation1 = activation()
        self.activation2 = activation()
        self.batchnorm1 = nn.BatchNorm2d(feature_dim)

    def forward(self, x):
        x1 = self.cnn1(x)
        x1 = self.activation1(x1)
        x1 = self.batchnorm1(x1)
        x2 = self.cnn2(x1)
        return self.activation2(x2 + x)

class ResNet(nn.Module):
    def __init__(self, input_dim : int = 256, feature_dim : int = 32, num_blocks : int = 4, activation : nn.Module = nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_blocks = num_blocks
        self.residual_blocks = nn.ModuleList([Residual_Block(input_dim, feature_dim, activation) for _ in range(num_blocks)])
        self.cnn1 = nn.Conv2d(input_dim, feature_dim, kernel_size=3, stride=1, padding="same")
        self.cnn2 = nn.Conv2d(feature_dim, input_dim, kernel_size=3, stride=1, padding="same")
        self.activation1 = activation()
        self.activation2 = activation()
        self.batchnorm1 = nn.BatchNorm2d(feature_dim)
        
    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)
        x1 = self.cnn1(x)
        x1 = self.activation1(x1)
        x1 = self.batchnorm1(x1)
        x2 = self.cnn2(x1)
        return self.activation2(x2 + x)