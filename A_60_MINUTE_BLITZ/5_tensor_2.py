# %%
import numpy as np
import pandas as pd
import torch

# %%
a = torch.randn(4, 4)
b = a.argmax() % 4
c = a.max(1)
d = a.max(1).values
e = a.max(1)[0]
f = a.max(1)[1]
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)

# %%
a = torch.randn(4, 4)
b = a[:, :2]
c = a[:, -2:]
print(a)
print(b)
print(c)

# %%
a = torch.randn(4, 4)
b = map(lambda s: s is not None, a)
c = tuple(b)
print(a)
print(b)
print(c)

# %%
a = torch.tensor([[1, 2], [3, 4]])
b = a.gather(0, torch.tensor([[0, 0], [1, 0]]))
c = a.gather(1, torch.tensor([[0, 0], [1, 0]]))
print(a)
print(b)
print(c)

# %%
a = [-4, -3, -2, -1]
print(a[-3:-1])

# %%
a = np.array([1, 2, 3, 4])
b = np.array([np.mean(a[-3:])])
c = np.zeros(10)
d = np.concatenate([c, b])
e = np.append(a, 1)
print(a)
print(b)
print(c)
print(d)
print(e)

# %%
a = np.array([1, 2, 3, 4])
b = torch.tensor(a)
print(a)
print(type(a))
print(b)

# %%
a = {"symbol": "DCE.P", "frequency": "180s", "open": 5794.0, "high": 5796.0, "low": 5788.0, "close": 5790.0, "volume": 1613,
     "amount": 93421700.0, "position": 152533, "bob": "2020-08-12 22:45:00+08:00", "eob": "2020-08-12 22:48:00+08:00", "pre_close": 0.0}
b = pd.DataFrame(a)
print(b.eob - b.bob)

# %%
a = torch.tensor([[1, 1]])
b = torch.tensor([[2, 2]])
c = torch.cat((a, b))
print(c)

# %%
columns = range(10)
q_table = pd.DataFrame(columns=columns, dtype=np.float32)
a = pd.Series(
    [0]*len(columns),
    index=columns,
    name='a',
)
b = pd.Series(
    np.zeros(len(columns)),
    index=columns,
    name='b',
)
q_table = q_table.append([a, b])
q_table = q_table.append(b)
print(q_table)

# %%
a = torch.tensor([[1, 1]])
b = torch.squeeze(a).numpy().tolist()
string = ""
for x in b:
    string += str(x)
print(string)

# %%
a = [1, 2]
print(a[-2])
# %%
