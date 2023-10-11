from torch.utils.data import Dataset
import torch
import os

import config.waymo_config as cnf


class WaymoDataset(Dataset):
	def __init__(self, cfg, mode='train'):
		assert mode in ['train', 'val', 'test'], f'Invalid mode: {mode}'
		self.mode = mode
		self.device = torch.device(cfg["device"])
		self.dataset_dir = cfg["dataset_dir"] + mode + '/'
		self.file_list = os.listdir(self.dataset_dir)

		if cnf.num_samples > 0:
			self.file_list = self.file_list[:cnf.num_samples]
		self.num_samples = len(self.file_list)

	def __len__(self):
		return self.num_samples

	def __getitem__(self, index):
		return torch.load(self.dataset_dir + self.file_list[index], map_location=self.device)
