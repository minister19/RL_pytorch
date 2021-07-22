import torch
import torch.nn as nn
from torchviz import make_dot


class MyNetwork2(nn.Module):
    def __init__(self):
        super().__init__()

        # define the layers
        self.layers = nn.Sequential(
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        # forward pass
        x = torch.sigmoid(self.layers(x))
        return x


# instantiate the model
model = MyNetwork2()

# print model architecture
print(model)

# print graphic model architecture
x = torch.randn(1, 16)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.render('test-output/round-table.gv', view=True)
