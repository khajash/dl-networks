# from turtle import down, forward
from typing import List
# from numpy import block
import torch
import torch.nn as nn
from collections import OrderedDict
from models.layers.resnet_blocks import BasicBlock, BottleneckBlock


# Number of layers within each block
CONFIG_LAYERS = {
    18:  [2, 2, 2, 2], 
    34:  [3, 4, 6, 3], 
    50:  [3, 4, 6, 3],
    101: [3, 4, 23, 3], 
    152: [3, 4, 36, 3] 
}

# TODO: implement base class for full customization of layers and first layer
# class ResNetBase(nn.Module):


class ResNet(nn.Module):
    """
    ResNet model and configurations as described in 'Deep residual learning for image recognition'
    https://arxiv.org/abs/1512.03385
    """
    def __init__(
        self, 
        in_channels: int = 3,
        num_classes: int = 10,
        config_key: int = 18,
        conv_layers: List[int] = [64, 128, 256, 512],
        shortcut: str = "proj"
    ) -> None:
        
        super().__init__()
        
        if config_key not in CONFIG_LAYERS.keys():
            raise ValueError(f"{config_key} is not a valid setting. Choose from {CONFIG_LAYERS.keys()}")

        block_config = CONFIG_LAYERS[config_key]
        channels = [64] + conv_layers
        cnn_layer_dict = OrderedDict()
        self.block_type = BasicBlock if config_key < 50 else BottleneckBlock

        # Conv1 Block
        cnn_layer_dict["conv1"] = nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        cnn_layer_dict["bn1"] = nn.BatchNorm2d(channels[0])
        cnn_layer_dict["relu1"] = nn.ReLU(inplace=True)
        cnn_layer_dict["maxpool1"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Conv2+ Blocks
        for idx, n_layers in enumerate(block_config):
            cnn_layer_dict[f"conv_block{idx + 2}"] = self._add_conv_block(channels[idx], channels[idx+1], 
                                                                          n_layers, ds_first=(idx > 0), shortcut=shortcut)

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

    
    def _add_conv_block(self, in_channels, out_channels, n_layers, ds_first, shortcut="proj"):
        
        block = []

        for j in range(n_layers):
            if j == 0:
                block.append(self.block_type(in_channels, out_channels, to_downsample=ds_first, shortcut=shortcut))
            else:
                block.append(self.block_type(out_channels, out_channels, to_downsample=False, shortcut=shortcut))

        return nn.Sequential(*block)


if __name__ == "__main__": 
    import torch
    from torchsummary import summary

    config_key = 18 
    if config_key >= 50:
        conv_layers = [256, 512, 1024, 2048]
    else:
        conv_layers = [64, 128, 256, 512]
    
    model = ResNet(3, 10, config_key=50, conv_layers=conv_layers, shortcut="proj").cuda()
    print(model)

    b, h, w, d = 64, 224, 224, 3
    summary(model, (d, h, w), batch_size=-1)
    input = torch.randn(b, d, h, w).cuda()
    print(input.shape)

    out = model(input)
    print(out.shape)