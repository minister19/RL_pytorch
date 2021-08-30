import torch
import torch.nn as nn
from torchviz import make_dot

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

# print graphic model architecture
x = torch.randn(1, 8)
y = model(x)

dot = make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
dot.render('data/round-table.gv', view=True)
