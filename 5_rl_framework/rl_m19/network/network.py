import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork(nn.Module):
    def __init__(self, label):
        super().__init__()
        self.label = label


class PureLinear(BaseNetwork):
    def __init__(self, config):
        super().__init__('PureLinear')
        self.fc1 = self.__layer_init(nn.Linear(config.states_dim, 16))
        self.fc2 = self.__layer_init(nn.Linear(16, 32))
        self.fc3 = self.__layer_init(nn.Linear(32, 16))
        self.fc4 = self.__layer_init(nn.Linear(16, config.actions_dim))
        self.to(config.device)

    def forward(self, x: Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    @staticmethod
    def __layer_init(layer: nn.Linear, w_scale=1.0):
        # nn.init.normal_(layer.weight.data)
        nn.init.orthogonal_(layer.weight.data)
        layer.weight.data.mul_(w_scale)
        nn.init.constant_(layer.bias.data, 0)
        return layer


class DQN(BaseNetwork):
    def __init__(self, h: int, w: int, outputs: int):
        super().__init__('DQN')
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
