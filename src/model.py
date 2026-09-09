import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class ConvBlock(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size : int,
                stride : int,
                padding : int,
                activation: nn.Module,
                dilation=1
                ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation()
        
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.activation(x)
        return x




class RezNetBlock(nn.Module):
    def __init__(self, 
                in_channels: int,
                out_channels: int, 
                kernel_size: int,
                stride: int,
                padding: int,
                activation: nn.Module,
                dilation=1):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, activation)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 1, padding, nn.Identity)
        self.act   = activation()
        
        self.id_match = nn.ModuleList()
        if stride > 1:
            self.id_match.append(nn.AvgPool2d(kernel_size, stride , padding))
        if in_channels != out_channels:
            self.id_match.append(nn.Conv2d(in_channels, out_channels, 1))
        self.id_match = nn.Sequential(*self.id_match)
        
        
    def forward(self, x: torch.Tensor):
        identity_x = self.id_match(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity_x
        self.act(x)
        return x



class RezNetStage(nn.Module):
        def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int,
                padding: int,
                activation: nn.Module,
                num_layers: int):
            super().__init__()
            
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(RezNetBlock(in_channels, out_channels, kernel_size, stride, padding, activation))
                else:
                    self.layers.append(RezNetBlock(out_channels, out_channels, kernel_size, 1, padding , activation))
            
        
        def forward(self, x: torch.Tensor):
            for layer in self.layers:
                x = layer(x)
            return x



class Stem(nn.Module):
    def __init__(self,                 
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int,
                padding: int,
                activation: nn.Module):
        super().__init__()
        
        self.zero_pad = nn.ZeroPad2d(3)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding , activation)
        self.pool = nn.MaxPool2d(3,2,1)
        
    def forward(self, x: torch.Tensor):
        x = self.zero_pad(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
    


class RezNet(nn.Module):
    def __init__(self,
                num_features : int,
                num_classes: int,
                stride: int = 4,
                kernel_size: int = 5,
                activation: nn.Module = nn.ReLU,
                embedding_dims: Tuple[int] = (112, 56 , 28, 14, 7),
                layers_per_stage: Tuple[int] = (3,4,6,3)):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.stem = Stem(num_features, embedding_dims[0], 7, 2, 4, activation)
        
        self.layers = nn.ModuleList()
        for i in range(len(layers_per_stage)):
            if i == 0:
                self.layers.append(RezNetStage(embedding_dims[i], embedding_dims[i + 1], kernel_size, stride, padding, activation, layers_per_stage[i]))
            else:
                self.layers.append(RezNetStage(embedding_dims[i], embedding_dims[i + 1], kernel_size, 2, padding, activation, layers_per_stage[i]))
                
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.cls_head = nn.Conv2d(embedding_dims[-1], num_classes, 1)
        
        
    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = self.cls_head(x)
        
        x = x.flatten(1)
        
        return x