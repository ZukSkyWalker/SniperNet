import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import os


class KittiDataset(Dataset):
	def __init__(self, configs, mode='train', lidar_aug=None, hflip_prob=None, num_samples=None):
		self.dataset_dir = configs.dataset_dir
		self.input_size = configs.input_size
		self.hm_size = configs.hm_size

		self.num_classes = configs.num_classes
		self.max_objects = configs.max_objects

		assert mode in ['train', 'val', 'test'], f'Invalid mode: {mode}'
		self.mode = mode
		self.is_test = (self.mode == 'test')
		sub_folder = 'testing' if self.is_test else 'training'

		self.lidar_aug = lidar_aug
		self.hflip_prob = hflip_prob

		self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
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
			return self.load_img_only(index)
		else:
			return self.load_img_with_targets(index)

	def load_img_only(self, index):
		"""Load only image for the testing phase"""
		sample_id = int(self.sample_id_list[index])
		img_path, img_rgb = self.get_image(sample_id)
		lidarData = self.get_lidar(sample_id)
		lidarData = get_filtered_lidar(lidarData, kitti_cfg.boundary)
		bev_map = makeBEVMap(lidarData, kitti_cfg.boundary)
		bev_map = torch.from_numpy(bev_map)

		metadatas = {'img_path': img_path}

		return metadatas, bev_map, img_rgb

	def load_img_with_targets(self, index):
		"""Load images and targets for the training and validation phase"""
		sample_id = int(self.sample_id_list[index])
		img_path = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
		lidarData = self.get_lidar(sample_id)
		calib = self.get_calib(sample_id)
		labels, has_labels = self.get_label(sample_id)
		if has_labels:
			labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P2)

		if self.lidar_aug:
			lidarData, labels[:, 1:] = self.lidar_aug(lidarData, labels[:, 1:])

		lidarData, labels = get_filtered_lidar(lidarData, kitti_cfg.boundary, labels)

		bev_map = makeBEVMap(lidarData, kitti_cfg.boundary)
		bev_map = torch.from_numpy(bev_map)

		hflipped = False
		if np.random.random() < self.hflip_prob:
			hflipped = True
			# C, H, W
			bev_map = torch.flip(bev_map, [-1])

		targets = self.build_targets(labels, hflipped)

		metadatas = {
			'img_path': img_path,
			'hflipped': hflipped
		}

		return metadatas, bev_map, targets

	def get_image(self, idx):
		img_path = os.path.join(self.image_dir, f'{idx:06d}.png')
		img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

		return img_path, img

	def get_calib(self, idx):
		calib_file = os.path.join(self.calib_dir, f'{idx:06d}.txt')
		# assert os.path.isfile(calib_file)
		return Calibration(calib_file)

	def get_lidar(self, idx):
		lidar_file = os.path.join(self.lidar_dir, f'{idx:06d}.bin')
		# assert os.path.isfile(lidar_file)
		return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

	def get_label(self, idx):
		labels = []
		label_path = os.path.join(self.label_dir, f'{idx:06d}.txt')
		for line in open(label_path, 'r'):
			line = line.rstrip()
			line_parts = line.split(' ')
			obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
			cat_id = int(kitti_cfg.CLASS_NAME_TO_ID[obj_name])
			if cat_id <= -99:  # ignore Tram and Misc
				continue
			'''
			truncated = int(float(line_parts[1]))  # truncated pixel ratio [0..1]
			occluded = int(line_parts[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
			alpha = float(line_parts[3])  # object observation angle [-pi..pi]
			# xmin, ymin, xmax, ymax
			bbox = np.array([float(line_parts[4]), float(line_parts[5]), float(line_parts[6]), float(line_parts[7])])
			'''
			
			# height, width, length (h, w, l)
			h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10])
			# location (x,y,z) in camera coord.
			x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13])
			ry = float(line_parts[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

			object_label = [cat_id, x, y, z, h, w, l, ry]
			labels.append(object_label)

		if len(labels) == 0:
			labels = np.zeros((1, 8), dtype=np.float32)
			has_labels = False
		else:
			labels = np.array(labels, dtype=np.float32)
			has_labels = True

		return labels, has_labels

	def build_targets(self, labels, hflipped):
		minX = kitti_cfg.boundary['minX']
		maxX = kitti_cfg.boundary['maxX']
		minY = kitti_cfg.boundary['minY']
		maxY = kitti_cfg.boundary['maxY']
		minZ = kitti_cfg.boundary['minZ']
		maxZ = kitti_cfg.boundary['maxZ']

		num_objects = min(len(labels), self.max_objects)
		hm_l, hm_w = self.hm_size

		hm_main_center = np.zeros((self.num_classes, hm_l, hm_w), dtype=np.float32)
		cen_offset = np.zeros((self.max_objects, 2), dtype=np.float32)
		direction = np.zeros((self.max_objects, 2), dtype=np.float32)
		z_coor = np.zeros((self.max_objects, 1), dtype=np.float32)
		dimension = np.zeros((self.max_objects, 3), dtype=np.float32)

		indices_center = np.zeros((self.max_objects), dtype=np.int64)
		obj_mask = np.zeros((self.max_objects), dtype=np.uint8)

		for k in range(num_objects):
			cls_id, x, y, z, h, w, l, yaw = labels[k]
			cls_id = int(cls_id)
			# Invert yaw angle
			yaw = -yaw
			if not ((minX <= x <= maxX) and (minY <= y <= maxY) and (minZ <= z <= maxZ)):
				continue
			if (h <= 0) or (w <= 0) or (l <= 0):
				continue

			bbox_l = l / kitti_cfg.bound_size_x * hm_l
			bbox_w = w / kitti_cfg.bound_size_y * hm_w
			radius = compute_radius((math.ceil(bbox_l), math.ceil(bbox_w)))
			radius = max(0, int(radius))

			center_y = (x - minX) / kitti_cfg.bound_size_x * hm_l  # x --> y (invert to 2D image space)
			center_x = (y - minY) / kitti_cfg.bound_size_y * hm_w  # y --> x
			center = np.array([center_x, center_y], dtype=np.float32)

			if hflipped:
				center[0] = hm_w - center[0] - 1

			center_int = center.astype(np.int32)
			if cls_id < 0:
				ignore_ids = [_ for _ in range(self.num_classes)] if cls_id == - 1 else [- cls_id - 2]
				# Consider to make mask ignore
				for cls_ig in ignore_ids:
					gen_hm_radius(hm_main_center[cls_ig], center_int, radius)
				hm_main_center[ignore_ids, center_int[1], center_int[0]] = 0.9999
				continue

			# Generate heatmaps for main center
			gen_hm_radius(hm_main_center[cls_id], center, radius)
			# Index of the center
			indices_center[k] = center_int[1] * hm_w + center_int[0]

			# targets for center offset
			cen_offset[k] = center - center_int

			# targets for dimension
			dimension[k, 0] = h
			dimension[k, 1] = w
			dimension[k, 2] = l

			# targets for direction
			direction[k, 0] = np.sin(float(yaw))  # im
			direction[k, 1] = np.cos(float(yaw))  # re
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

	def draw_img_with_label(self, index):