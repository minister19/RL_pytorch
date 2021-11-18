import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, cnn_image_width, cnn_image_height, action_dim, device):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(cnn_image_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(cnn_image_height)))

        self.fc1 = nn.Linear(32*convw*convh, 20)
        self.fc2 = nn.Linear(20, action_dim)
        self.to(device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_image_width = 100
    cnn_image_height = 100
    action_dim = 3
    cnn = CNN(cnn_image_width, cnn_image_height, action_dim, device)
    x = torch.rand((1, 3, cnn_image_width, cnn_image_height), device=device)
    y = cnn(x)
    print(y)
