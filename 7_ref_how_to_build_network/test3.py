import torch
import torch.nn as nn
from torchviz import make_dot

# define model architecture
model = nn.Sequential(
    nn.Linear(16, 12),
    nn.ReLU(),
    nn.Linear(12, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# print model architecture
print(model)

# print graphic model architecture
x = torch.randn(1, 16)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.render('data/round-table.gv', view=True)
