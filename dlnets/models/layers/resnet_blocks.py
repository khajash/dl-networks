import torch
import torch.nn as nn


## Shorcuts 
# There are two methods mentioned in (He et al., 2015) when number of channels change from input to output of block: projection
# shortcuts and zero-padding shortcuts. In general, the padding method is not recommended (see F. Chollet comment
# https://github.com/keras-team/keras/issues/2608) But if you are working on a lightweight solution where you need to 
# keep model size small, it's ok to use.


class PadShortcut(nn.Module):
    # Option A: zero-padding shortcuts
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1,1), stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pool(x)
        pad = torch.zeros_like(x)
        out = torch.cat((x, pad), 1)
        return out


class ProjShortcut(nn.Module):
    # Option B: projection shortcuts (recommended)
    def __init__(self,
        in_channels: int, 
        out_channels: int, 
        stride: int = 2
    ) -> None:

        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.proj(x)


class BasicBlock(nn.Module):
    """ 
    Basic building block as described in 'Deep residual learning for image recognition'
    Contains two 3x3 Conv layers with residual connection
    https://arxiv.org/abs/1512.03385

    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        to_downsample: bool = True,
        shortcut: str = "proj",
    ) -> None:
        
        super().__init__()
        
        # first stride
        fstride = 2 if to_downsample else 1
        self.to_downsample = to_downsample
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=fstride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if (in_channels != out_channels):
            if shortcut == "proj":
                self.proj = ProjShortcut(in_channels, out_channels, stride=2)
            else:
                self.proj = PadShortcut()
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # use projection when used stride 2 and size mismatches
        if self.to_downsample:
            identity = self.proj(x)

        out += identity
        out = self.relu(out)
        return out
        

class BottleneckBlock(nn.Module):
    """ 
    Bottleneck building block as described in 'Deep residual learning for image recognition'
    Contains two 3x3 Conv layers with residual connection
    https://arxiv.org/abs/1512.03385

    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            to_downsample: bool = True,
            shortcut: str = "proj",
        ) -> None:
        
        super().__init__()
        
        # first stride
        fstride = 2 if to_downsample else 1
    
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=(1,1), stride=fstride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)

        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # use projection when size mismatches
        if (in_channels != out_channels):
            if shortcut == "proj":
                self.proj = ProjShortcut(in_channels, out_channels, fstride)
            else:
                self.proj = PadShortcut()
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.proj:
            identity = self.proj(x)
    
        out += identity
        out = self.relu(out)
        return out