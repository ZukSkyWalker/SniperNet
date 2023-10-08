from torch.utils.data import Dataset
import torch
import numpy as np
import math
import os

import data_process.kitti_data_utils as kitti_util
import config.kitti_config as cnf


class KittiDataset(Dataset):
	def __init__(self, configs, mode='train'):
		self.cfg = configs
		self.dataset_dir = configs["dataset_dir"]
		self.input_size = (configs["NX"], configs["NY"])

		# The BEV is divided into hm_l * hm_w grids, every grid is supposed to be occupied by only one object
		self.heatmap_size = 152, 152

		self.num_classes = configs["num_classes"]
		self.max_objects = configs["max_objects"]

		assert mode in ['train', 'val', 'test'], f'Invalid mode: {mode}'
		self.mode = mode
		self.is_test = (self.mode == 'test')
		sub_folder = 'testing' if self.is_test else 'training'

		self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
		self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
		self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
		split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', f'{mode}.txt')
		self.sample_id_list = [int(x.strip()) for x in open(split_txt_path).readlines()]

		if configs["num_samples"] > 0:
			self.sample_id_list = self.sample_id_list[:configs["num_samples"]]
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

		frm = kitti_util.frame(os.path.join(self.lidar_dir, f"{sample_id:06d}.bin"))
		frm.set_bev_map()
		bev_map = torch.from_numpy(frm.bev)
		metadatas = {'sample id': sample_id}

		return metadatas, bev_map

	def load_feature_with_labels(self, index):
		"""Load bev and labels for the training and validation phase"""
		sample_id = int(self.sample_id_list[index])
		frm = kitti_util.frame(os.path.join(self.lidar_dir, f"{sample_id:06d}.bin"))
		frm.set_bev_map()
		bev_map = torch.from_numpy(frm.bev)

		agt = kitti_util.agents(os.path.join(self.label_dir, f'{sample_id:06d}.txt'),
													os.path.join(self.calib_dir, f"{sample_id:06d}.txt"))

		metadatas = {'sample id': sample_id}
		targets = self.build_targets(agt.labels, hflipped=False)

		return metadatas, bev_map, targets


	def build_targets(self, labels, hflipped):
		minX = self.cfg['minX']
		maxX = self.cfg['maxX']
		minY = self.cfg['minY']
		maxY = self.cfg['maxY']
		minZ = self.cfg['minZ']
		maxZ = self.cfg['maxZ']

		num_objects = min(len(labels), self.max_objects)

		hm_l, hm_w = self.heatmap_size

		hm_main_center = np.zeros((self.num_classes, hm_l, hm_w), dtype=np.float32)
		cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
		direction = np.zeros((self.max_objects, 2), dtype=np.float32)
		z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)
		dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

		indices_center = np.zeros((self.max_objects), dtype=np.int64)
		obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

		for k in range(num_objects):
			# location (x, y, z) and dimension (w, l) in meters and lidar coordinates 
			cls_id, x, y, z, h, w, l, yaw = labels[k]

			# print("label info", cls_id, x, y, z, h, w, l, yaw)

			cls_id = int(cls_id)
			# Invert yaw angle
			yaw = -yaw
			if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
				continue
			if (h <= 0) or (w <= 0) or (l <= 0):
				continue

			bbox_l = l / cnf.bound_size_x * hm_l
			bbox_w = w / cnf.bound_size_y * hm_w
			radius = kitti_util.compute_radius(math.ceil(bbox_l), math.ceil(bbox_w))
			radius = max(0, int(radius))

			center_x = (x - minX) / cnf.bound_size_x * hm_l
			center_y = (y - minY) / cnf.bound_size_y * hm_w
			center = np.array([center_x, center_y], dtype=np.float32)

			if hflipped:
				center[0] = hm_w - center[0] - 1

			center_int = center.astype(np.int32)
			if cls_id < 0:
				ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == -1 else [-cls_id - 2]
				# Consider to make mask ignore
				for cls_ig in ignore_ids:
					kitti_util.gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
				hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
				continue

			# Generate heatmaps for main center
			kitti_util.gen_hm_radius(hm_main_center[cls_id], center, radius)
			# Index of the center
			indices_center[k] = center_int[1] * hm_w + center_int[0]

			# targets for center offset
			cen_offset[k] = center - center_int

			print(f"label info: cls_id={cls_id}, at ({x:.2f}, {y:.2f}, {z:.2f}), h={h:.2f}, w={w:.2f}, l={l:.2f}, yaw={yaw:.2f}")
			print(f"center_x = {center_x:.2f}; center_y = {center_y:.2f}")

			# targets for dimension
			dimension[k, 0] = h
			dimension[k, 1] = w
			dimension[k, 2] = l

			# targets for direction
			direction[k, 0] = math.sin(float(yaw))  # im
			direction[k, 1] = math.cos(float(yaw))  # re
			# im -->> -im
			if hflipped:
				direction[k, 0] = - direction[k, 0]

			# targets for depth
			z_coor[k] = z - minZ

			# Generate object masks
			obj_mask[k] = 1

		targets = {
			'hm_cen': hm_main_center,
			'cen_offset': cen_offset,
			'direction': direction,
			'z_coor': z_coor,
			'dim': dimension,
			'indices_center': indices_center,
			'obj_mask': obj_mask,
		}

		return targets
