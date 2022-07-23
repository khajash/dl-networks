# from turtle import forward
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn

## Identity Shortcut

## Projection Shortcut
def proj_shortcut(in_channels, out_channels):
    proj = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=2, padding=0, bias=False),
        nn.BatchNorm2d(out_channels)
    )
    return proj

class BasicBlock(nn.Module):
    """ 
    Basic building block as described in 'Deep residual learning for image recognition'
    https://arxiv.org/abs/1512.03385

    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            to_downsample: bool = True,
            shortcut: str = "identity",
        ) -> None:
        
        super().__init__()
        
        # first stride
        fstride = 2 if to_downsample else 1
        self.use_projection = False if shortcut == "identity" else True
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=fstride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # if shortcut == "proj":
        # self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=2, padding=0, bias=False)
        self.proj = proj_shortcut(in_channels, out_channels)
        # elif shortcut == "pad":
            # pass

    def forward(self, x):
        
        identity = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_projection:
            identity = self.proj(x)
        
        out += identity
        out = self.relu(out)
        return out
        
        
class ResNet(nn.Module):
    def __init__(
        self, 
        layers: List[int],
        in_channels: int = 3,
        num_classes: int = 10,
        block_type: nn.Module = BasicBlock,
    ) -> None:
        
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels) # check parameters
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    