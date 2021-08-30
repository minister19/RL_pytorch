import torch
import torch.nn as nn
from collections import OrderedDict
from torchviz import make_dot

# define model architecture
model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(16, 12)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(12, 10)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(10, 1)),
    ('sigmoid', nn.Sigmoid())
]))

# print model architecture
print(model)

# print graphic model architecture
x = torch.randn(1, 16)
y = model(x)

dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.render('data/round-table.gv', view=True)
