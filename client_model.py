import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ClientModel(nn.Module):
	def __init__(self):
		super(ClientModel, self).__init__()

		self.conv1 = nn.Conv2d(1, 16, (5,5), 1)
		self.conv2 = nn.Conv2d(16, 32, (3,3), 1)
		self.fc1 = nn.Linear(5*5*32, 64)
		self.fc2 = nn.Linear(64, 10)
		self.relu = F.relu

	def forward(self, input):
		x = self.conv1(input)
		x = nn.MaxPool2d((2,2), 1)(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = nn.MaxPool2d((2,2), 1)(x)
		x = self.relu(x)
		
		x = x.view(-1, 5*5*32)
		x = self.relu(self.fc1(x))
		
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)

		return x