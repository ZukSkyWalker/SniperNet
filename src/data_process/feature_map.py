import numpy as np
from util.timing import timer_func
import data_process.kitti_config as kitti_cfg


class frame():
  def __init__(self, pos_arr) -> None:
    in_range  = (pos_arr[:, 0] > kitti_cfg.boundary["minX"]) & (pos_arr[:, 0] < kitti_cfg.boundary["maxX"])
    in_range &= (pos_arr[:, 1] > kitti_cfg.boundary["minY"]) & (pos_arr[:, 1] < kitti_cfg.boundary["maxY"])
    in_range &= (pos_arr[:, 2] > kitti_cfg.boundary["minZ"]) & (pos_arr[:, 2] < kitti_cfg.boundary["maxZ"])
    
    self.pos = pos_arr[in_range]

    # Initialize the BEV Maps
    self.base_map = np.zeros((kitti_cfg.W, kitti_cfg.H))
    self.height_map = np.zeros((kitti_cfg.W, kitti_cfg.H))
    self.density_map = np.zeros((kitti_cfg.W, kitti_cfg.H))


  def load_point_cloud(self, pos_arr):
    in_range  = (pos_arr[:, 0] > kitti_cfg.boundary["minX"]) & (pos_arr[:, 0] < kitti_cfg.boundary["maxX"])
    in_range &= (pos_arr[:, 1] > kitti_cfg.boundary["minY"]) & (pos_arr[:, 1] < kitti_cfg.boundary["maxY"])
    in_range &= (pos_arr[:, 2] > kitti_cfg.boundary["minZ"]) & (pos_arr[:, 2] < kitti_cfg.boundary["maxZ"])
    
    self.pos = pos_arr[in_range]


  @timer_func
  def get_base(self):
    cnts, edges = np.histogram(self.pos[:, 2],
                               bins=np.arange(kitti_cfg.boundary["minZ"],
                                              kitti_cfg.boundary["maxZ"],
                                              kitti_cfg.vx))
    pts_thresh = max(len(self.pos) * kitti_cfg.min_ground_ratio, kitti_cfg.min_ground_layer_pts)
    base_idx = np.where(cnts > pts_thresh)[0][0]
    return edges[base_idx]
  
  def set_grid(self, z_arr, gx, gy):
    b, h = z_arr.min(), z_arr.max()
    cnt = 1
    if len(z_arr) > 1:
      idx_arr = np.where((z_arr[1:] > z_arr[:-1] + kitti_cfg.max_z_gap)
                         & (z_arr[1:] > kitti_cfg.sig_z_thresh))[0]
      if len(idx_arr) > 0:
        h = z_arr[idx_arr[0]]
        cnt = idx_arr[0]

    self.base_map[gx, gy] = b
    self.height_map[gx, gy] = h
    # Normalize by the distance^2
    self.density_map[gx, gy] = cnt * (gx*gx + (gy - kitti_cfg.H//2)**2)


  @timer_func
  def set_bev_map(self):
    """
    Return base_map, height_map and the counts map as a supervisor
    """
    # Extract x, y, z coordinates
    x, y = self.pos[:, 0], self.pos[:, 1]

    # Compute the grid indices for every point
    idx_x = (x / kitti_cfg.vx).astype(np.int32)
    idx_y = ((y - kitti_cfg.boundary['minY']) / kitti_cfg.vy).astype(np.int32)

    glb_idx = idx_x * kitti_cfg.H + idx_y
    # The last key in the sequence is used for the primary sort order, 
    # the second-to-last key for the secondary sort order, and so on.
    sorted_indices = np.lexsort((self.pos[:, 2], glb_idx))
    
    i0 = 0
    self.pos = self.pos[sorted_indices]
    unique_idx, idx_counts = np.unique(glb_idx[sorted_indices], return_counts=True)

    for i in range(len(unique_idx)):
      # Decode the grid index
      gx, gy = divmod(unique_idx[i], kitti_cfg.H)

      # Set the grid height and base
      self.set_grid(self.pos[i0:(i0+idx_counts[i]), 2], gx, gy)
      i0 += idx_counts[i]
