import tensorflow as tf
import torch
import numpy as np
import os
import plotly.graph_objects as go
from enum import IntFlag
import config.waymo_config as cnf

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

HALF_PI = 0.5 * np.pi
x_grid_size, y_grid_size = 4*cnf.DX, 4*cnf.DY

class ClassEnum(IntFlag):
	TYPE_VEHICLE = 1
	TYPE_PEDESTRIAN = 2
	TYPE_SIGN=3
	TYPE_CYCLIST=4

class_list = ['vehicle', 'pedestrian', 'sign', 'cyclist']
color_list = ['red', 'orange', 'yellow', 'green']

# For training, only use vehicle, pedestrian, cyclist


def convert_range_image_to_pointcloud(*args, **kwargs):
	points, cp_points = frame_utils.convert_range_image_to_point_cloud(*args, **kwargs)
	return points, cp_points

def process_raw_frame(frame):
	# Extract the lidar points
	(range_images, camera_projections, seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
	
	points, cp_points = convert_range_image_to_pointcloud(frame, range_images, camera_projections, range_image_top_pose)
	points_ri2, cp_points_ri2 = convert_range_image_to_pointcloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=1)
	
	return np.vstack(points+points_ri2), [x for x in frame.laser_labels if (0 < x.type < 5) and (x.type != 3)]

def get_vertices2D(box):
	hw = 0.5 * box.width
	hl = 0.5 * box.length
	cos_theta, sin_theta = np.cos(box.heading), np.sin(box.heading)
	return np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]]) @ np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

def get_vertices(box):
	hh = 0.5 * box.height
	vxy = get_vertices2D(box)

	vertices = np.array([[vxy[0, 0], vxy[0, 1], -hh], [vxy[1, 0], vxy[1, 1], -hh],
											 [vxy[2, 0], vxy[2, 1], -hh], [vxy[3, 0], vxy[3, 1], -hh],
											 [vxy[0, 0], vxy[0, 1], hh], [vxy[1, 0], vxy[1, 1], hh], 
											 [vxy[2, 0], vxy[2, 1], hh], [vxy[3, 0], vxy[3, 1], hh]])

	vertices += np.array([box.center_x, box.center_y, box.center_z])

	return vertices

def visualize_frame(frame):
	points, labels = process_raw_frame(frame)
	fig = go.Figure(data=[go.Scatter3d(x=points[:, 0], y= points[:, 1], z=points[:, 2], mode = 'markers',
																			marker=dict(size=1, opacity=0.5, color=points[:, 2], colorscale='jet'))])

	annotations = []
	for lb in labels:
		label_text = f"{class_list[lb.type-1]}"
		annotations.append(dict(x=lb.box.center_x, y=lb.box.center_y, z=lb.box.center_z,
													text=label_text, xanchor='center', showarrow=False,
													font=dict(size=12, color=color_list[lb.type-1])))
		
		vertices = get_vertices(lb.box)

		fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
														i = [0, 5, 1, 6, 2, 3, 3, 4, 4, 6, 0, 2],
														j = [1, 4, 2, 5, 6, 7, 7, 0, 5, 7, 1, 3],
														k = [5, 0, 6, 1, 3, 6, 4, 3, 6, 4, 2, 0],
														opacity=0.2, color=color_list[lb.type-1]))

	fig.update_layout(title=f"{len(points)} points, z range: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}m",
									 template='plotly_dark', scene=dict(annotations=annotations, aspectmode='data'))
	fig.show()


class BEV():
	def __init__(self, frame) -> None:
		points, self.labels = process_raw_frame(frame)
		self.bev_map = np.zeros((3, cnf.BEV_HEIGHT, cnf.BEV_WIDTH))
		
		# boundary select
		sel  = (points[:, 0] > cnf.boundary["minX"]) & (points[:, 0] < cnf.boundary["maxX"])
		sel &= (points[:, 1] > cnf.boundary["minY"]) & (points[:, 1] < cnf.boundary["maxY"])

		points = points[sel]
		idx_x = ((points[:, 0] - cnf.boundary["minX"]) / cnf.DX).astype(np.uint32)
		idx_y = ((points[:, 1] - cnf.boundary["minY"]) / cnf.DY).astype(np.uint32)

		glb_idx = idx_x * cnf.BEV_WIDTH + idx_y

		sorted_idx = np.lexsort((points[:, 2], glb_idx))
		points = points[sorted_idx]
		i0 = 0
		unique_idx, idx_counts = np.unique(glb_idx[sorted_idx], return_counts=True)

		for i in range(len(unique_idx)):
			gx, gy = divmod(unique_idx[i], cnf.BEV_WIDTH)
			# Set the grid height and base
			self.set_grid(points[i0:(i0+idx_counts[i]), 2], gx, gy)
			i0 += idx_counts[i]

	def set_grid(self, z_arr, gx, gy):
		"""
		z_arr is sorted (for all the points in this very grid)
		"""
		b, h = z_arr.min(), z_arr.max()
		cnt = 1
		if len(z_arr) > 1:
			idx_arr = np.where((z_arr[1:] > z_arr[:-1] + cnf.max_z_gap)
													& (z_arr[1:] > cnf.sig_z_thresh))[0]
			if len(idx_arr) > 0:
				h = z_arr[idx_arr[0]]
				cnt = idx_arr[0]

		self.bev_map[0, gx, gy] = b
		self.bev_map[1, gx, gy] = h
		self.bev_map[2, gx, gy] = cnt


	def lable_target(self, label):
		box = label.box
		class_id = max(label.type-1, 2)

		# get the heading, make it within (-np.pi/2, np.pi/2)
		yaw = box.heading
		if yaw > HALF_PI:
			yaw -= np.pi
		elif yaw < -HALF_PI:
			yaw += np.pi
		cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

		# Center position in pixel unit
		box_cx = (box.center_x - cnf.boundary['minX']) / x_grid_size
		box_cy = (box.center_y - cnf.boundary['minY']) / y_grid_size

		# Dimension in pixel unit
		half_l, half_w = 0.5 * box.length / x_grid_size, 0.5 * box.width / y_grid_size

		# get vertices
		vertices_xy = get_vertices2D(box) + np.array([box.center_x, box.center_y])
		vertices_xy[:, 0] = (vertices_xy[:, 0] - cnf.boundary['minX']) / x_grid_size
		vertices_xy[:, 1] = (vertices_xy[:, 1] - cnf.boundary['minY']) / y_grid_size
		
		# Get the range of the pixels
		ix0 = max(0, int(vertices_xy[:, 0].min()))
		ix1 = min(self.hm_l-1, int(vertices_xy[:, 0].max()))
		iy0 = max(0, int(vertices_xy[:, 1].min()))
		iy1 = min(self.hm_w-1, int(vertices_xy[:, 1].max()))

		# Fill the heat map
		for ix in range(ix0, ix1+1):
			d_x = ix + 0.5 - box_cx
			for iy in range(iy0, iy1+1):
				d_y = iy + 0.5 - box_cy
				# rotate:
				rcx = d_x * cos_yaw + d_y * sin_yaw
				rcy = d_y * cos_yaw - d_x * sin_yaw

				# approximate covered area
				cover_x = min(1, max(0, min(half_l - rcx+0.5, rcx+0.5 + half_l)))
				cover_y = min(1, max(0, min(half_w - rcy+0.5, rcy+0.5 + half_w)))

				cover_area = cover_x * cover_y
				if cover_area < 1e-5:
					continue

				self.heat_map[ix, iy] = cover_area
				self.class_onehot[class_id, ix, iy] = 1
				self.cxy_offset[0, ix, iy] = -d_x
				self.cxy_offset[1, ix, iy] = -d_y
				self.z_coor[ix, iy] = box.center_z  # in meters
				self.heading[ix, iy] = yaw
				self.dimension[0, ix, iy] = box.length # in meters
				self.dimension[1, ix, iy] = box.width  # in meters
				self.dimension[2, ix, iy] = box.height # in meters


	def build_targets(self):
		num_classes = 3
		# Down sample factor: 4
		self.hm_l = cnf.BEV_HEIGHT // 4    # X
		self.hm_w = cnf.BEV_WIDTH // 4     # Y

		# Orgnize the label map:
		self.class_onehot = np.zeros((num_classes, self.hm_l, self.hm_w), dtype=np.float32)
		self.heat_map = np.zeros((self.hm_l, self.hm_w), dtype=np.float32) # approximate overlap area
		self.cxy_offset = np.zeros((2, self.hm_l, self.hm_w), dtype=np.float32)
		self.z_coor = np.zeros((self.hm_l, self.hm_w), dtype=np.float32)
		self.heading = np.zeros((self.hm_l, self.hm_w), dtype=np.float32)
		self.dimension = np.zeros((3, self.hm_l, self.hm_w))

		for lb in self.labels:
			self.lable_target(lb)


def get_frames(file_path):
	"""
	Decode the tfrecord file from waymo dataset
	Parameters:
		file_path (string): file path to the tfrecord file
	
	Returns:
		a list of frames
	"""
	frm_list = []

	for data in tf.data.TFRecordDataset(file_path, compression_type=''):
		frm = open_dataset.Frame()
		frm.ParseFromString(bytearray(data.numpy()))
		frm_list.append(frm)

	return frm_list

def save_frame(in_dir, out_path):
	for f in os.listdir(in_dir):
		if not f.endswith('.tfrecord'):
			continue
		
		if os.path.exists(f'{out_path+f[:-9]}0.pt'):
			continue

		print("processing", f)
		idx = 0

		for data in tf.data.TFRecordDataset(in_dir+f, compression_type=''):
			frm = open_dataset.Frame()
			frm.ParseFromString(bytearray(data.numpy()))
			frm_proc, labels = process_raw_frame(frm)
			torch.save({'pos': frm_proc.pos, 'sn': frm_proc.sn_arr, 'cid': frm_proc.cids, 'cls_type': frm_proc.cls_type}, f'{out_path+f[:-9]}{idx}.pt')
			idx += 1
