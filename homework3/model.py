# David Fried
# 4/1/2021
# 4 layer convolutional neural network
from torch import nn
from torch.nn import functional as F


class LandmarkCNN(nn.Module):
    def __init__(self, landmark_num: int, feature=False):
        super().__init__()
        self.landmark_num = landmark_num
        self.feature = feature

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        self.fc5 = nn.Linear(512 * 13 ** 2, landmark_num * 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # batch_size (0) * out_channels (1) * height (2) * width (3)
        x = x.view(x.size(0), -1)  # batch_size (0) * (out_channels * height * width)
        x = self.fc5(x)

        return x.reshape(-1, 98, 2)
