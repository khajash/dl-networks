import torch.nn as nn
import torch.nn.functional as F


def local_response_norm():
    # TODO - implement to compare to bn
    pass

class AlexNet(nn.module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 10,
        ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size=(11,11), stride=4, padding='valid', bias=True)
        self.bn1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5,5), stride=1, padding='same', bias=True) 
        self.bn1 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3,3), stride=1, padding='same', bias=True)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3,3), stride=1, padding='same', bias=True)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3,3), stride=1, padding='same', bias=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # flatten
        self.fc1 = nn.Linear(43264, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        # x = local_response_norm(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        # x = local_response_norm(x)
        x = self.bn1(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool5(F.relu(self.conv5(x)))

        x = x.view(-1, 43264)
        # nn.Flatten()

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        logits = self.fc3(x)
        return logits
        