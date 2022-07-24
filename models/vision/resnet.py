from turtle import down, forward
from typing import Type, Any, Callable, Union, List, Optional
from numpy import block
import torch
import torch.nn as nn
from collections import OrderedDict


## Identity Mapping
# In general, the padding method is not recommended (see F. Chollet comment https://github.com/keras-team/keras/issues/2608) 
# But if you are working on a lightweight solution where you need to keep model size small, it's ok to use.


def pad_shortcut():
    # perform pooling
    # zero padding
    # pad = nn.Sequential(
    #     nn.MaxPool2d(kernel_size=(1,1), stride=2, padding="same"),
    # )
    pass

## Projection Shortcut
def proj_shortcut(in_channels, out_channels, stride=2):
    proj = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(out_channels)
    )
    return proj

# class PadShortcut(nn.Module):



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

        if shortcut == "proj":
            self.proj = proj_shortcut(in_channels, out_channels, stride=2)
        else:
            self.proj = pad_shortcut()

    def forward(self, x):
        
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
        # self.to_project = in_channels != out_channels
    
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=(1,1), stride=fstride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)

        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # TODO: change this so it's not creating extra weights
        if (in_channels != out_channels):
            if shortcut == "proj":
                self.proj = proj_shortcut(in_channels, out_channels, fstride)
            else:
                self.proj = pad_shortcut()
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

        # use projection when used stride 2 and size mismatches
        print("project: ", self.proj)
        if self.proj:
            identity = self.proj(x)
        print(out.shape)
        print(identity.shape)
        
        out += identity
        out = self.relu(out)
        return out

# Number of layers within each block
CONFIG_LAYERS = {
    18:  [2, 2, 2, 2], 
    34:  [3, 4, 6, 3], 
    50:  [3, 4, 6, 3],
    101: [3, 4, 23, 3], 
    152: [3, 4, 36, 3] 
}
        
class ResNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3,
        num_classes: int = 10,
        config_key: int = 18,
        conv_layers: List[int] = [64, 128, 256, 512],
    ) -> None:
        
        super().__init__()
        
        if config_key not in CONFIG_LAYERS.keys():
            raise ValueError(f"{config_key} is not a valid setting. Choose from {CONFIG_LAYERS.keys()}")

        block_config = CONFIG_LAYERS[config_key]
        channels = [64] + conv_layers
        cnn_layer_dict = OrderedDict()
        self.block_type = BasicBlock if config_key < 50 else BottleneckBlock

        cnn_layer_dict["conv1"] = nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        cnn_layer_dict["bn1"] = nn.BatchNorm2d(channels[0])
        cnn_layer_dict["relu1"] = nn.ReLU(inplace=True)
        cnn_layer_dict["maxpool1"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for idx, n_layers in enumerate(block_config):
            cnn_layer_dict[f"conv_block{idx + 2}"] = self._add_conv_block(channels[idx], channels[idx+1], 
                                                                          n_layers, ds_first=(idx > 0))

        self.cnn = nn.Sequential(cnn_layer_dict)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(conv_layers[-1], num_classes)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv layers
        x = self.cnn(x)

        # classifier
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    
    def _add_conv_block(self, in_channels, out_channels, n_layers, ds_first):
        
        block = []

        for j in range(n_layers):
            if j == 0:
                block.append(self.block_type(in_channels, out_channels, to_downsample=ds_first, shortcut="proj"))
            else:
                block.append(self.block_type(out_channels, out_channels, to_downsample=False, shortcut="proj"))

        return nn.Sequential(*block)


if __name__ == "__main__": 
    import torch
    from torchsummary import summary

    config_key = 18 
    if config_key >= 50:
        conv_layers = [256, 512, 1024, 2048]
    else:
        conv_layers = [64, 128, 256, 512]
    
    model = ResNet(3, 10, config_key=50, conv_layers=conv_layers).cuda()
    print(model)

    b, h, w, d = 64, 224, 224, 3
    summary(model, (d, h, w), batch_size=-1)
    input = torch.randn(b, d, h, w).cuda()
    print(input.shape)

    out = model(input)
    print(out.shape)