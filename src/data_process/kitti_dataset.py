import numpy as np
from torch.utils.data import Dataset
import torch
import os

from data_process.kitti_data_utils import frame, agents


class KittiDataset(Dataset):
	def __init__(self, configs, mode='train', num_samples=None):
		self.dataset_dir = configs.dataset_dir
		self.input_size = configs.input_size

		self.num_classes = configs.num_classes
		self.max_objects = configs.max_objects

		assert mode in ['train', 'val', 'test'], f'Invalid mode: {mode}'
		self.mode = mode
		self.is_test = (self.mode == 'test')
		sub_folder = 'testing' if self.is_test else 'training'

		self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
		self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
		self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
		split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', f'{mode}.txt')
		self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]

		if num_samples is not None:
			self.sample_id_list = self.sample_id_list[:num_samples]
		self.num_samples = len(self.sample_id_list)

	def __len__(self):
		return len(self.sample_id_list)

	def __getitem__(self, index):
		if self.is_test:
			return self.load_features(index)
		else:
			return self.load_feature_with_labels(index)

	def load_features(self, index):
		"""Load only image for the testing phase"""
		sample_id = int(self.sample_id_list[index])

		frm = frame(os.path.join(self.lidar_dir, f"{sample_id:06d}.bin"))
		frm.set_bev_map()
		bev_map = torch.from_numpy(frm.bev)
		metadatas = {'sample id': sample_id}

		return metadatas, bev_map

	def load_feature_with_labels(self, index):
		"""Load bev and labels for the training and validation phase"""
		sample_id = int(self.sample_id_list[index])
		frm = frame(os.path.join(self.lidar_dir, f"{sample_id:06d}.bin"))
		frm.set_bev_map()
		bev_map = torch.from_numpy(frm.bev)

		agt = agents(os.path.join(self.label_dir, f'{sample_id:06d}.txt'),
							 	 os.path.join(self.calib_dir, f"{sample_id:06d}.txt"))

		metadatas = {'sample id': sample_id}
		return metadatas, bev_map, agt.labels
