import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MySmallModel(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        hidden_nodes = nodes * 2
        self.fc1 = nn.Linear(nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, nodes)
        self.fc3 = nn.Linear(nodes, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class Classifier(nn.Module):
    def __init__(self, input_nodes):
        super().__init__()
        self.networks = nn.ModuleList()
        self.input_nodes = input_nodes

        for i in range(len(input_nodes)):
            self.networks.append(MySmallModel(input_nodes[i]))
        '''
        self.network1 = MySmallModel(i1)
        self.network2 = MySmallModel(i2)
        self.network3 = MySmallModel(i3)
        '''

        nodes = len(input_nodes)
        hidden_nodes = len(input_nodes) * 2
        self.fc1 = nn.Linear(nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, nodes)
        self.fc_out = nn.Linear(nodes, 1)

    def forward(self, input_):
        x_list = []
        for i in range(len(self.input_nodes)):
            x_item = torch.relu(self.networks[i](torch.tensor(input_[i])))
            x_list.append(x_item)

        x = torch.cat((x_list), 0)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc_out(x))
        return x


input_nodes = [1, 2, 3]
model = Classifier(input_nodes)
print(model)
criterion = nn.BCELoss()  # nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.08)

epochs = 10
trainloader = [
    (([1., ], [1., 1., ], [1., 1., 1., ]), torch.tensor([1.])),
]
train_accuracy = []
train_loss = []
for e in range(epochs):
    running_loss = 0
    i = 0
    print('Epochs: ', e)
    for data, label in trainloader:
        y_hat = model(data)

        loss = criterion(y_hat, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_hat_class = np.where(y_hat.detach().numpy() < 0.5, 0, 1)

        accuracy = np.sum(label.numpy() == y_hat_class.flatten()) / len(label)
        print('Train Accuracy: ', accuracy)
        train_accuracy.append(accuracy)
        train_loss.append(loss.item())

        running_loss += loss.item()
    else:

        print(f"Training loss: {running_loss/len(trainloader)}")
