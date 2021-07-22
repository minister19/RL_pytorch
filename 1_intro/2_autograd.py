# %%
# 1. Tensor
import torch

# If you set its attribute .requires_grad as True, it starts to track all operations on it.
# torch.Tensor.requires_grad = True

# Computes the gradient of current tensor w.r.t. graph leaves.
# torch.Tensor.backward()

# To stop a tensor from tracking history, you can call .detach() to detach it from the computation history, and to prevent future computation from being tracked.
# torch.Tensor.detach()

# To prevent tracking history (and using memory), you can also wrap the code block in with torch.no_grad():
# with torch.no_grad():
#     pass

# Each tensor has a .grad_fn attribute that references a Function that has created the Tensor (except for Tensors created by the user - their grad_fn is None).

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# %%
# 2. Gradients
out.backward()
print(x.grad)

# Generally speaking, torch.autograd is an engine for computing vector-Jacobian product.
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

# if we just want the vector-Jacobian product, simply pass the vector to backward as argument:
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# You can also stop autograd from tracking history on Tensors with .requires_grad=True either by wrapping the code block in with torch.no_grad():
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# Or by using .detach() to get a new Tensor with the same content but that does not require gradients:
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
