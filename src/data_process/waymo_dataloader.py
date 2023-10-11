from torch.utils.data import DataLoader
from data_process.waymo_dataset import WaymoDataset


def create_train_dataloader(configs):
	"""Create dataloader for training"""
	train_dataset = WaymoDataset(configs, mode='train')
	# train_sampler = None
	train_dataloader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=False,
																pin_memory=configs["pin_memory"], num_workers=configs["num_workers"],
																sampler=None)

	return train_dataloader

