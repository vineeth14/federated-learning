import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ClientModel(nn.Module):
	def __init__(self):
		super(ClientModel, self).__init__()

		self.lin1 = nn.Linear(1, 8)
		self.lin2 = nn.Linear(8, 4)
		self.lin3 = nn.Linear(4, 1)
		self.relu = F.relu

	def forward(self, input):
		x = self.lin1(input)
		x = self.relu(x)
		x = self.lin2(x)
		x = self.relu(x)
		x = self.lin3(x)

		return x