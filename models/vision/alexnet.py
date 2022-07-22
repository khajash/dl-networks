import torch.nn as nn
import torch.nn.functional as F


def local_response_norm():
    # TODO - implement to compare to bn
    pass

class AlexNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
            conv_layers: list = [96, 256, 384, 384, 256],
            lin_layers: list = [4096, 4096]
        ) -> None:
        super().__init__()

        # Input: 3x224x224 -- add (2,2) padding to enable this to work
        self.conv1 = nn.Conv2d(in_channels, conv_layers[0], kernel_size=(11,11), stride=4, padding=(2,2), bias=True)
        self.bn1 = nn.BatchNorm2d(conv_layers[0])
        # Output: C0x55x55
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Output: C0x27x27

        self.conv2 = nn.Conv2d(conv_layers[0], conv_layers[1], kernel_size=(5,5), stride=1, padding='same', bias=True) 
        self.bn2 = nn.BatchNorm2d(conv_layers[1])
        # Output: C1x27x27
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Output: C1x13x13

        self.conv3 = nn.Conv2d(conv_layers[1], conv_layers[2], kernel_size=(3,3), stride=1, padding='same', bias=True)
        # Output: C2x13x13
        self.conv4 = nn.Conv2d(conv_layers[2], conv_layers[3], kernel_size=(3,3), stride=1, padding='same', bias=True)
        # Output: C3x13x13
        self.conv5 = nn.Conv2d(conv_layers[3], conv_layers[4], kernel_size=(3,3), stride=1, padding='same', bias=True)
        # Output: C4x13x13
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Output: C5x6x6

        # flatten
        self.fc1 = nn.Linear(conv_layers[-1]*6*6, lin_layers[0])
        self.fc2 = nn.Linear(lin_layers[0], lin_layers[1])
        self.fc3 = nn.Linear(lin_layers[1], num_classes)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        # x = local_response_norm(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        # x = local_response_norm(x)a
        x = self.bn2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))

        x = x.view(-1, 9216)
        # nn.Flatten()

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        logits = self.fc3(x)
        return logits

if __name__ == "__main__": 
    import torch
    from torchsummary import summary


    model = AlexNet(3, 10)
    print(model)

    b, h, w, d = 64, 224, 224, 3
    summary(model, input_size=(d, h, w), batch_size=b, device='cpu')
    # input = torch.randn(b, h, w, d)
    # print(input.shape)

    # out = model(input)
    # print(out.shape)