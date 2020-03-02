import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ClientModel(nn.Module):
	def __init__(self):
		super(ClientModel, self).__init__()

		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4 * 4 * 50, 500)
		self.fc2 = nn.Linear(500, 10)
		self.relu = F.relu

	def forward(self, input):
		x = self.conv1(input)
		x = F.max_pool2d(x, 2, 2)
		x = self.relu(x)

		x = self.conv2(x)
		x = F.max_pool2d(x, 2, 2)
		x = self.relu(x)
		
		x = x.view(-1, 4*4*50)
		x = self.relu(self.fc1(x))
		
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)

		return x