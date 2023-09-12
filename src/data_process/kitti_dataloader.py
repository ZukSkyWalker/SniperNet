import torch
from torch.utils.data import DataLoader

from data_process.kitti_dataset import KittiDataset


def create_train_dataloader(configs):
	"""Create dataloader for training"""
	train_dataset = KittiDataset(configs, mode='train', num_samples=configs.num_samples)
	train_sampler = None
	if configs.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
																pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler)

	return train_dataloader, train_sampler

def create_val_dataloader(configs):
	"""Create dataloader for validation"""
	val_sampler = None
	val_dataset = KittiDataset(configs, mode='val', num_samples=configs.num_samples)
	if configs.distributed:
		val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
	val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
															pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler)

	return val_dataloader

def create_test_dataloader(configs):
	"""Create dataloader for testing phase"""
	test_dataset = KittiDataset(configs, mode='test', num_samples=configs.num_samples)
	test_sampler = None
	if configs.distributed:
		test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
	test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
			      pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

	return test_dataloader
