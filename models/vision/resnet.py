from typing import List
import torch
import torch.nn as nn
from collections import OrderedDict
from models.layers.resnet_blocks import BasicBlock, BottleneckBlock


class ResNetBase(nn.Module):
    """
    ResNet base class model. This is a flexible model that can take many configurations including the 
    different ResNet models and the smaller CIFAR10 model variations.
    """
    def __init__(
        self, 
        in_channels: int = 3,
        num_classes: int = 10,
        fkernel: int = 3,
        fstride: int = 1, 
        ffilters: int = 16, 
        to_pool: bool = False,
        block_config: List[int] = [14, 14, 14],
        conv_layers: List[int] = [16, 32, 64],
        shortcut: str = "proj",
        block_type: str = "basic"
    ) -> None:

        super().__init__()

        self.block_type = BasicBlock if (block_type == "basic") else BottleneckBlock
        channels = [ffilters] + conv_layers

        # Conv1 Block
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=fkernel, stride=fstride, padding=fkernel//2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        if to_pool:
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else: 
            self.pool1 = None

        # Conv2+ Blocks
        cnn_layer_dict = OrderedDict()
        for idx, n_layers in enumerate(block_config):
            cnn_layer_dict[f"conv_block{idx + 2}"] = self._add_conv_block(channels[idx], channels[idx+1], 
                                                                          n_layers, ds_first=(idx > 0), shortcut=shortcut)

        self.cnn_blocks = nn.Sequential(cnn_layer_dict)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(conv_layers[-1], num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv layers
        x = self.cnn1(x)
        if self.pool1:
            x = self.pool1(x)
        x = self.cnn_blocks(x)

        # classifier
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    
    def _add_conv_block(self, 
        in_channels: int, 
        out_channels: int, 
        n_layers: int, 
        ds_first: bool, 
        shortcut : str = "proj"
    ) -> nn.Sequential:
        
        block = []

        for j in range(n_layers):
            if j == 0:
                block.append(self.block_type(in_channels, out_channels, to_downsample=ds_first, shortcut=shortcut))
            else:
                block.append(self.block_type(out_channels, out_channels, to_downsample=False, shortcut=shortcut))

        return nn.Sequential(*block)


# Number of layers within each block
CONFIG_LAYERS = {
    18:  [2, 2, 2, 2], 
    34:  [3, 4, 6, 3], 
    50:  [3, 4, 6, 3],
    101: [3, 4, 23, 3], 
    152: [3, 8, 36, 3] 
}

class ResNet(ResNetBase):
    """
    ResNet model and configurations as described in 'Deep residual learning for image recognition'
    Supports ResNet models: 18, 24, 50, 101, 152
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
                
        if config_key not in CONFIG_LAYERS.keys():
            raise ValueError(f"{config_key} is not a valid setting. Choose from {CONFIG_LAYERS.keys()}")

        block_config = CONFIG_LAYERS[config_key]
        block_type = "basic" if config_key < 50 else "bottleneck"

        super().__init__(in_channels, num_classes, fkernel=7, fstride=2, ffilters=64, to_pool=True, 
            block_config=block_config, conv_layers=conv_layers, shortcut=shortcut, block_type=block_type)



if __name__ == "__main__": 
    import torch
    from torchsummary import summary

    # config_key = 152
    # # for config_key in [18, 34, 50, 101, 152]:

    # print("-"*20)
    # print(f"Resnet Model: {config_key}")

    # if config_key >= 50:
    #     conv_layers = [256, 512, 1024, 2048]
    # else:
    #     conv_layers = [64, 128, 256, 512]
    
    # model = ResNet(3, 10, config_key, conv_layers=conv_layers, shortcut="proj").cuda()
    model = ResNetBase(3, 3, fkernel=7, fstride=2, block_config = [14]*3).cuda()
    print(model)

    b, h, w, d = 2, 224, 224, 3
    summary(model, (d, h, w), batch_size=-1)
    input = torch.randn(b, d, h, w).cuda()
    print(input.shape)

    out = model(input)
    print(out.shape)
