import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Number of layers within each block, as specified in https://arxiv.org/abs/1409.1556 (Table 1)
CONFIG_LAYERS = {
    "A": [1, 1, 2, 2, 2], # 11 layers
    "B": [2, 2, 2, 2, 2], # 13 layers
    "C": [2, 2, 3, 3, 3], # 16 layers w/ 1x1 CONV
    "D": [2, 2, 3, 3, 3], # 16 layers
    "E": [2, 2, 4, 4, 4]  # 19 layers
}

class VGG(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
            config_key: str = "D",
            conv_layers: list = [64, 128, 256, 512, 512],
            lin_layers: list = [4096, 4096]
        ) -> None:
        super().__init__()

        if config_key not in CONFIG_LAYERS.keys():
            raise ValueError(f"{config_key} is not a valid setting. Choose from {CONFIG_LAYERS.keys()}")

        block_config = CONFIG_LAYERS[config_key]
        channels = [in_channels] + conv_layers
        cnn_layer_dict = OrderedDict()
        self.i = 1 # used for naming each layer
        
        # variable for adding 1x1 convs for config C
        last_single_conv = False 
        
        # create each conv block
        for idx, n_layers in enumerate(block_config):
            if config_key == "C" and idx > 1:
                last_single_conv = True
            self.add_conv_block(cnn_layer_dict, channels[idx], channels[idx+1], block_idx=idx+1, 
                                n_layers=n_layers, last_single_conv=last_single_conv)
        
        self.cnn = nn.Sequential(cnn_layer_dict)

        # flatten
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(channels[-1]*7*7, lin_layers[0]),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(lin_layers[0], lin_layers[1]),    
            nn.ReLU(),
            nn.Linear(lin_layers[1], num_classes)
        )

    def add_conv_block(self, layer_dict, in_channels, block_channels, block_idx, n_layers, last_single_conv=False):
        for j in range(n_layers):
            if last_single_conv and j == (n_layers - 1):
                kernel = (1,1)
            else:
                kernel = (3,3)
            
            layer_dict[f"block{block_idx}_conv{self.i}"] = nn.Conv2d(
                in_channels, block_channels, kernel_size=kernel, stride=1, padding="same", bias=True)
            layer_dict[f"block{block_idx}_relu{self.i}"] = nn.ReLU()
            
            in_channels = block_channels
            self.i += 1
        layer_dict[f"block{block_idx}_maxpool"] = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    def forward(self, x):
        x = self.cnn(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__": 
    import torch
    from torchsummary import summary

    model = VGG(3, 10, config="A").cuda()
    print(model)

    b, h, w, d = 64, 224, 224, 3
    summary(model, (d, h, w))
    input = torch.randn(b, d, h, w).cuda()
    print(input.shape)

    out = model(input)
    print(out.shape)