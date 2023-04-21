"""@package single_frame_reco
This module takes in a full frame with full coverage of the lidar view,
returns the flag of each point for later processing.
Appreximation is made based on the assumption that the frame length is in the order
of 0.1s so the scene is roughly static.

Author: zukai.wang@cepton.com
Date: 9/18/2021
Version: 1.0.0
"""
import numpy as np
import plotly.graph_objects as go
import util

_IS_NOISE  = np.uint8(1)
_IS_GROUND = np.uint8(2)
_IS_BKG    = np.uint8(4)       # points belonging to the static backgrounds
_IS_VEHCLE = np.uint8(8)
_IS_SIGN   = np.uint8(16)
_IS_PED    = np.uint8(32)
_IS_BIKER  = np.uint32(64)


class Frame():
	"""
	Single Frame Reconstruction Result.
	"""
	def __init__(self, pts, cfg):
		self.cfg = cfg
		self.n_points = (~pts['invalid']).sum()
		self.pos = np.c_[pts['x'], pts['y'], pts['z']][~pts['invalid']]
		self.intensities = pts['r'][~pts['invalid']]
		self.h_arr = np.zeros(self.n_points)
		self.distXY = np.sqrt((self.pos[:, :2]**2).sum(axis=1))
		self.flags = np.zeros(self.n_points, dtype=np.uint8)

		self.grd_par = np.array([0, 0, -cfg["lidar_height"]])

	def gridding(self):
		# Partition the points by the angular & distance grids
		self.theta = np.arcsin(self.pos[:, 0] / self.distXY) 
		self.x_idx = ((self.theta + self.cfg['max_theta']) / self.cfg["theta_grid_size"]).astype(np.int16)
		self.y_idx = ((self.distXY - self.cfg['min_dist']) / self.cfg["dist_grid_size"]).astype(np.int16)

	def ground_detection(self):
		self.flags = np.zeros(self.n_points, dtype=np.uint8)
		# The bool array to indicate if the point is a potential ground point
		grd_candidates = np.abs(self.pos[:, 2] + self.cfg["lidar_height"]) < (self.distXY * self.cfg["max_slope"]).clip(self.cfg["dz_local"], self.cfg["dz_global"])
		
		grd_candidates = np.where(grd_candidates)[0]
		# Fit a global ground plane: z = ax + by + c
		a, b, c = util.plane_fit(self.pos[grd_candidates])

		# Set default heights
		self.h_arr = self.pos[:, 2] - a * self.pos[:, 0] - b * self.pos[:, 1] - c
		
		# Mark the selected seed ground points
		seed_grd = self.h_arr[grd_candidates] < self.cfg["dz_local"]
		self.flags[grd_candidates[seed_grd]] |= _IS_GROUND

		# Loop through the grids to get local points marked
		for iy in range(self.cfg["n_dist_grids"]):
			in_y = self.y_idx == iy
			if in_y.sum() < self.cfg["min_grd_pts"]:
				continue
			nearby_y = np.abs(self.y_idx - iy) < 2
			for ix in range(self.cfg["n_angular_grids"]):
				nearby_x = np.abs(self.x_idx - ix) < 2
				nearby_idx = np.where(nearby_x & nearby_y)[0]

				to_fit_idx = nearby_idx[self.flags[nearby_idx] & _IS_GROUND > 0]

				if len(to_fit_idx) < self.cfg["min_grd_pts"]:
					continue

				# Fit the local piece of the ground
				grd_par = util.plane_fit(self.pos[to_fit_idx])
				
				# clip the slope
				grd_par[:2] = grd_par[:2].clip(-self.cfg["max_slope"], self.cfg["max_slope"])

				# Update the heights for the local points
				in_x = self.x_idx == ix
				local_idx = np.where(in_x & in_y)[0]
				self.h_arr[local_idx] = self.pos[local_idx, 2] - self.pos[local_idx, :2] @ grd_par[:2] - grd_par[2]

				# Label the local ground points
				on_local_plane = (self.h_arr[local_idx] > -self.cfg["dz_global"]) & (self.h_arr[local_idx] < self.cfg["dz_local"])
				self.flags[local_idx[~on_local_plane]] = 0
				self.flags[local_idx[on_local_plane]] |= _IS_GROUND


	def visualize(self):
		is_grd = self.flags & _IS_GROUND > 0
		data = [go.Scatter3d(x=self.pos[is_grd, 0], y=self.pos[is_grd, 1], z=self.pos[is_grd, 2],
		       mode='markers', name=f"{is_grd.sum()} points on ground", marker=dict(size=1, opacity=0.3, color='green'))]
		
		abv_grd = self.h_arr >= self.cfg["dz_local"]
		data.append(go.Scatter3d(x=self.pos[abv_grd, 0], y=self.pos[abv_grd, 1], z=self.pos[abv_grd, 2],
		       mode='markers', name=f"{abv_grd.sum()} points above ground", marker=dict(size=1, opacity=0.3, color='red')))

		fig = go.Figure(data=data)

		fig.update_layout(title=f"{self.n_points} points in the frame", scene=dict(aspectmode='data'), template="plotly_dark")
		fig.show()
		

	def static_reco(self):
		# (N, 1)
		t_arr = self.ts - self.anchor_time

		# At anchor time, the front direction is ey
		alpha = self.omega[0] * t_arr
		beta  = self.omega[1] * t_arr
		gamma = self.omega[2] * t_arr

		cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
		cos_beta, sin_beta = np.cos(beta), np.sin(beta)
		cos_gamma, sin_gamma = np.cos(gamma), np.sin(gamma)

		ez0 = -sin_beta
		ez1 = sin_alpha * cos_beta
		ez2 = cos_alpha * cos_beta

		ey0 = cos_beta * sin_gamma
		ey1 = sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma
		ey2 = cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma

		# 3. ex = ey cross ez
		ex0 = ey1 * ez2 - ey2 * ez1
		ex1 = ey2 * ez0 - ey0 * ez2
		ex2 = ey0 * ez1 - ey1 * ez0

		pos_shift = self.velocity*t_arr

		self.positions = (np.vstack((self.xs*ex0+self.ys*ey0+self.zs*ez0 + ey0 * pos_shift,
									 self.xs*ex1+self.ys*ey1+self.zs*ez1 + ey1 * pos_shift,
									 self.xs*ex2+self.ys*ey2+self.zs*ez2 + ey2 * pos_shift))).T

