import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import copy
import numpy as np
import syft as sy
from syft.frameworks.torch.federated import utils
from syft.workers.websocket_client import WebsocketClientWorker

with open('./data/boston_housing.pickle','rb') as f:
    ((x, y), (x_test, y_test)) = pickle.load(f)

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

mean = x.mean(0, keepdim=True)
dev = x.std(0, keepdim=True)
mean[:, 3] = 0.
dev[:, 3] = 1.
x = (x - mean) / dev
x_test = (x_test - mean) / dev
train = TensorDataset(x, y)
test = TensorDataset(x_test, y_test)
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = x.view(-1, 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc4(x))
        x = self.fc3(x)
        return x


hook = sy.TorchHook(torch)
bob_worker = sy.VirtualWorker(hook, id="bob")
alice_worker = sy.VirtualWorker(hook, id="alice")
# kwargs_websocket = {"host": "localhost", "hook": hook}
# alice = WebsocketClientWorker(id='alice', port=8779, **kwargs_websocket)
# bob = WebsocketClientWorker(id='bob', port=8778, **kwargs_websocket)
compute_nodes = [bob_worker, alice_worker]


remote_dataset = (list(), list())
train_distributed_dataset = []

for batch_idx, (data,target) in enumerate(train_loader):
    data = data.send(compute_nodes[batch_idx % len(compute_nodes)])
    target = target.send(compute_nodes[batch_idx % len(compute_nodes)])
    remote_dataset[batch_idx % len(compute_nodes)].append((data, target))


bobs_model = Net()
alices_model = Net()
bobs_optimizer = optim.SGD(bobs_model.parameters(), lr=args.lr)
alices_optimizer = optim.SGD(alices_model.parameters(), lr=args.lr)


models = [bobs_model, alices_model]
optimizers = [bobs_optimizer, alices_optimizer]

model = Net()

def update(data, target, model, optimizer):
    model.send(data.location)
    optimizer.zero_grad()
    prediction = model(data)
    loss = F.mse_loss(prediction.view(-1), target)
    loss.backward()
    optimizer.step()
    return model

def train():
    for data_index in range(len(remote_dataset[0])-1):
        for remote_index in range(len(compute_nodes)):
            data, target = remote_dataset[remote_index][data_index]
            models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index])
        for model in models:
            model.get()
        return utils.federated_avg({
            "bob": models[0],
            "alice": models[1]
        })

def test(federated_model):
    federated_model.eval()
    test_loss = 0
    for data, target in test_loader:
        output = federated_model(data)
        test_loss += F.mse_loss(output.view(-1), target, reduction='sum').item()
        predection = output.data.max(1, keepdim=True)[1]
        
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))


for epoch in range(args.epochs):
    start_time = time.time()
    print(f"Epoch Number {epoch + 1}")
    federated_model = train()
    model = federated_model
    test(federated_model)
    total_time = time.time() - start_time
    print('Communication time over the network', round(total_time, 2), 's\n')