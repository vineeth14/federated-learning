import torch
import torch.nn.functional as F

import syft as sy
from syft.frameworks.torch.fl import utils as SyftUtils

import mylogger

def GetNextBatch(fed_data_loader: sy.FederatedDataLoader, n_batches):
	batches = {}
	
	for worker_id in fed_data_loader.workers:
		worker = fed_data_loader.federated_dataset.datasets[worker_id].location
		batches[worker] = []

	try:
		for _ in range(n_batches):
			next_batches = next(fed_data_loader)

			for worker in next_batches:
				batches[worker].append(next_batches[worker])
	except StopIteration:
		pass

	return batches

def TrainOnBatches(worker, batches, model_in, device, lr):
	model = model_in.copy()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	model.train()
	model.send(worker)
	loss_local = False

	for batch_idx, (data, target) in enumerate(batches):
		loss_local = False
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()

		if batch_idx % mylogger.LOG_INTERVAL == 0:
			loss = loss.get()
			loss_local = True
			mylogger.logger.debug(
                "Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )
	if not loss_local:
		loss = loss.get()
	model.get()

	return model, loss

def FederatedTrainer(model, device, fed_data_loader, lr, fed_after_n_batches):
	model.train()

	n_batches = fed_after_n_batches
	data_for_all_workers = True
	batch_counter = 0
	models = {}
	losses = {}
	
	iter(fed_data_loader)
	batches = GetNextBatch(fed_data_loader, n_batches)

	while True:
		logger.debug(
            "Starting training round, batches [{}, {}]".format(counter, counter + nr_batches)
        )

		for worker in batches:
			curr_batches = batches[worker]

			if curr_batches:
				models[worker], losses[worker] = TrainOnBatches()
			else:
				data_for_all_workers = False
		
		batch_counter += 1

		if not data_for_all_workers:
			logger.debug("stopping.")

			break

		model = SyftUtils.federated_avg(models)
		batches = GetNextBatch(fed_data_loader, n_batches)

	return model

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  
            pred = output.argmax(1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    mylogger.logger.debug("\n")
    accuracy = 100.0 * correct / len(test_loader.dataset)
    mylogger.logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
