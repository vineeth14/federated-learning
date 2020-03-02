import torch
from torchvision import datasets, transforms

import syft as sy

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