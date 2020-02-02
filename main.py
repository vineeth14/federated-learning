import torch
import torch.nn as nn
import torch.nn.functional as F

import syft as sy
from syft.workers.websocket_client import WebsocketClientWorker
from syft.workers.virtual import VirtualWorker

import mylogger
import sys
import argparse

import utils as Federator
import client_dataset
from client_model import ClientModel

def getArgs(args=sys.argv[1:]):
	parser = argparse.ArgumentParser(
		description="Federated Learning using PySyft Workers"
	)

	parser.add_argument("--batch_size", type=int, default=64, help="batch size of the training")
	parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train")
	parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
	parser.add_argument("--seed", type=int, default=1, help="seed used for random init values")
	parser.add_argument("--save_model", action="store_true", help="model will be saved")
	parser.add_argument("--use_virtual", action="store_true", help="virtual workers will be used instead of websocket workers")
	parser.add_argument("--use_cuda", action="store_true", help="use gpu for training")

	return parser.parse_args(args=args)

def createWorkers(SyftWorker, worker_ports, worker_ids, worker_kwargs):
	workers = []

	for (id, port) in zip(worker_ids, worker_ports):
		workers.append(
			SyftWorker(id=id, port=port, **worker_kwargs)
		)

	return workers

def main():
	args = getArgs()
	hook = sy.TorchHook(torch)
	SyftWorker = WebsocketClientWorker if args.use_virtual else VirtualWorker

	worker_kwargs = {'host': 'localhost', 'hook': hook, 'verbose': args.verbose}
	worker_ports = (8777, 8778, 8779)
	worker_ids = ('alice', 'bob', 'charlie')

	print("Creating Workers...")
	workers = createWorkers(SyftWorker, worker_ports=worker_ports, worker_ids=worker_ids, worker_kwargs=worker_kwargs)

	use_cuda = torch.cuda.is_available() and args.use_cuda
	print("Using Cuda:", use_cuda)
	
	torch.manual_seed(args.seed)

	device = torch.device('cuda' if use_cuda else 'cpu')

	model = ClientModel().to(device)

	federated_train_loader = dataset.federated_train_loader
	test_loader = dataset.federated_train_loader

	for epoch in range(args.epochs):
		mylogger.logger.info("Epoch %s/%s", epoch, args.epochs)
		
		model = Federator.FederatedTrainer(model, device, federated_train_loader, args.lr, fed_after_n_batches=25)
		Federator.test(model, device, test_loader)
	
	if args.save_model:
		torch.save(model.state_dict(), "fed_model.pt")

if __name__ == '__main__':
	
	main()