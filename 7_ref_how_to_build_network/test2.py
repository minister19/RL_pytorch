import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# define the network class


class MyNetwork(nn.Module):
    def __init__(self):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.fc1 = nn.Linear(16, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # define forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# instantiate the model
model = MyNetwork()

# print model architecture
print(model)

# print graphic model architecture
x = torch.zeros(1, 16).requires_grad_(False)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.render('test-output/round-table.gv', view=True)
