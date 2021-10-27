import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from torchviz import make_dot
from rl_m19.config import Config


class BaseNetwork(nn.Module):
    def __init__(self, label):
        super().__init__()
        self.label = label


class PureLinear(BaseNetwork):
    def __init__(self, config: Config):
        super().__init__('PureLinear')
        self.fc1 = nn.Linear(config.states_dim, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, config.actions_dim)
        self.to(config.device)

    def forward(self, x: Tensor):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CNN(BaseNetwork):
    def __init__(self, config: Config):
        super().__init__('CNN')
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(config.cnn_image_width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(config.cnn_image_height)))

        self.fc1 = nn.Linear(32*convw*convh, 20)
        self.fc2 = nn.Linear(20, config.actions_dim)
        self.to(config.device)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Nematode(BaseNetwork):
    def __init__(self, config: Config):
        super().__init__('Nematode')
        self.hidden = config.states_dim // 2
        self.fc1 = nn.Linear(config.states_dim, self.hidden, bias=False)
        self.fc2 = nn.Linear(self.hidden, config.actions_dim, bias=False)
        self.to(config.device)

    def forward(self, x: Tensor):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    config = Config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.states_dim = None
    config.actions_dim = 5
    config.cnn_image_width = 100
    config.cnn_image_height = 100
    cnn = CNN(config)
    x = torch.rand(1, 3, config.cnn_image_width, config.cnn_image_height, device=config.device)
    y = cnn(x)
    print(y)

    config.states_dim = 21
    config.actions_dim = 1
    nematode = Nematode(config)
    x = torch.rand(1, config.states_dim, device=config.device)
    y = nematode(x)
    print(y)

    dot = make_dot(y, params=dict(nematode.named_parameters()), show_attrs=True, show_saved=True)
    dot.render('data/round-table.gv', view=True)
