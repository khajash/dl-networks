import torch
from torch import ne, nn
# import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
from typing import List


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, to_downsample=True, bias=True) -> None:
        super().__init__()

        if to_downsample:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.maxpool = None

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.maxpool:
            x = self.maxpool(x)
        out = self.cnn(x)
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True) -> None:
        super().__init__()

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2,2), stride=2)

        self.cnn = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=(3,3), stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=0, bias=bias),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, crop) -> torch.Tensor:
        # out, crop = x
        out = x
        out = self.deconv(out)
        out = torch.concat([out, crop], dim=1)

        out = self.cnn(out)

        return out


class UNet(nn.Module):
    """
    UNet model based on paper "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/abs/1505.04597
    """
    def __init__(
        self, 
        in_channels: int = 1,
        num_classes: int = 2,
        img_size: int = 572,
        net_depth: int = 5,
        ffilters: int = 64, 
    ) -> None:
        super().__init__()

        # contracting layers
        self.crop_size = self._get_crop_sizes(img_size, net_depth)
        self.conv_layers = nn.ModuleList()
        prev_channels, out_channels = in_channels, ffilters
        
        for i in range(net_depth):
            # only downsample after first block
            self.conv_layers.append(DownBlock(prev_channels, out_channels, to_downsample=(i > 0), bias=False))
            
            prev_channels = out_channels
            out_channels *= 2
        
        # expanding layers
        self.deconv_layers = nn.ModuleList()
        out_channels //= 4

        for _ in range(net_depth - 1):
            self.deconv_layers.append(UpBlock(prev_channels, out_channels, bias=False))
            prev_channels = out_channels
            out_channels //= 2

        self.classifier = nn.Conv2d(prev_channels, num_classes, kernel_size=(1,1), bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        conv_outputs = []
        
        # contracting layers
        for i, block in enumerate(self.conv_layers):
            x = block(x)
            # store crop for later
            if i < len(self.crop_size):
                conv_outputs.append(center_crop(x, self.crop_size[~i]))
        
        # expanding layers
        for j, block in enumerate(self.deconv_layers):
            x = block(x, conv_outputs[~j])

        logits = self.classifier(x)
        return logits


    def _get_crop_sizes(self, img_size, net_depth):

        size = img_size
        crop_sizes = []
        
        # downsample
        for i in range(net_depth):
            size -= 4
            size //= 2
        
        # account for extra ds
        size *= 2
        
        # upsample
        for i in range(net_depth - 1):
            size *= 2
            crop_sizes.append(size)
            size -= 4
        
        return crop_sizes


if __name__ == "__main__": 
    import torch
    from torchsummary import summary

    b, h, w, d = 1, 572, 572, 1

    model = UNet(d, 2, img_size=h, net_depth=5, ffilters=64).cuda()
    print(model)

    summary(model, (d, h, w), batch_size=b, device="cuda")
    input = torch.randn(b, d, h, w).cuda()
    print(input.shape)

    out = model(input)
    print(out.shape)