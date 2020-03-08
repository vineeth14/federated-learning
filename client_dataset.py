import torch
from torchvision import datasets, transforms

import syft as sy

'''
Dataset:

data = th.rand((10000, 1))*1000
target = data.sin()

Alices share
alice_data = data[0:2500]
alice_target = target[0:2500]

Bobs share
bob_data = data[2500:5000]
bob_target = target[2500:5000]

Charlies share
charlie_data = data[5000:7500]
charlie_target = target[5000:7500]

Test data
test_data = data[7500:]
test_target = target[7500:]
'''


def GetTrainLoader(workers: tuple, args):
	loader = sy.FederatedDataLoader(
		datasets.MNIST(
			"../data",
			train=True,
			download=True,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		).federate(tuple(workers)),
		batch_size=args.batch_size,
		shuffle=True,
		iter_per_worker=True
	)

	return loader

def GetTestLoader(workers: tuple, args):
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
			"../data",
			train=False,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
			),
		),
		batch_size=args.batch_size,
		shuffle=True
	)

	return test_loader	
