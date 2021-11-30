import torch
import torch.nn.functional as F
from torchviz import make_dot

# generating some random features
features = torch.randn(1, 16)

# define the weights
W1 = torch.randn((16, 12), requires_grad=True)
W2 = torch.randn((12, 10), requires_grad=True)
W3 = torch.randn((10, 1), requires_grad=True)

# define the bias terms
B1 = torch.randn((12), requires_grad=True)
B2 = torch.randn((10), requires_grad=True)
B3 = torch.randn((1), requires_grad=True)

# calculate hidden and output layers
h1 = F.relu((features @ W1) + B1)
h2 = F.relu((h1 @ W2) + B2)
output = torch.sigmoid((h2 @ W3) + B3)

# print graphic model architecture
dot = make_dot(output.mean(), show_attrs=True, show_saved=True)
dot.render('data/round-table.gv', view=True)
