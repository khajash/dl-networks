import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
            conv_layers: list = [96, 256, 384, 384, 256],
            lin_layers: list = [4096, 4096]
        ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            # Input: 3x224x224 -- add (2,2) padding to enable this to work
            nn.Conv2d(in_channels, conv_layers[0], kernel_size=(11,11), stride=4, padding=(2,2), bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv_layers[0]),
            # Output: C0x55x55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Output: C0x27x27

            nn.Conv2d(conv_layers[0], conv_layers[1], kernel_size=(5,5), stride=1, padding='same', bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(conv_layers[1]),
            # Output: C1x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Output: C1x13x13

            nn.Conv2d(conv_layers[1], conv_layers[2], kernel_size=(3,3), stride=1, padding='same', bias=True),
            nn.ReLU(inplace=True),
            # Output: C2x13x13
            nn.Conv2d(conv_layers[2], conv_layers[3], kernel_size=(3,3), stride=1, padding='same', bias=True),
            nn.ReLU(inplace=True),
            # Output: C3x13x13
            nn.Conv2d(conv_layers[3], conv_layers[4], kernel_size=(3,3), stride=1, padding='same', bias=True),
            nn.ReLU(inplace=True),
            # Output: C4x13x13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Output: C5x6x6
        )

        # flatten
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_layers[-1]*6*6, lin_layers[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(lin_layers[0], lin_layers[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(lin_layers[1], num_classes)
        )


    def forward(self, x):  
        x = self.cnn(x)
        logits = self.classifier(x)
        return logits

if __name__ == "__main__": 
    import torch
    from torchsummary import summary

    model = AlexNet(3, 10)
    print(model)

    b, h, w, d = 64, 224, 224, 3
    summary(model, input_size=(d, h, w), batch_size=b, device='cpu')
    input = torch.randn(b, d, h, w)
    print(input.shape)

    out = model(input)
    print(out.shape)