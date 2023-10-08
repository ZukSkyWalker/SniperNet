import numpy as np
import os

from utils.timing import timer_func
import config.kitti_config as kitti_cfg

# Default transform matrix from the camera to lidar
c2l_mat = kitti_cfg.Tr_velo_to_cam_inv @ kitti_cfg.R0_inv

# BEV Image coordinates format
def get_corners(x, y, w, l, yaw):
  bev_corners = np.zeros((4, 2), dtype=np.float32)
  cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

  # front left
  bev_corners[0, 0] = x - 0.5 * w * cos_yaw - 0.5 * l * sin_yaw
  bev_corners[0, 1] = y - 0.5 * w * sin_yaw + 0.5 * l * cos_yaw

  # rear left
  bev_corners[1, 0] = x - 0.5 * w * cos_yaw + 0.5 * l * sin_yaw
  bev_corners[1, 1] = y - 0.5 * w * sin_yaw - 0.5 * l * cos_yaw

  # rear right
  bev_corners[2, 0] = x + 0.5 * w * cos_yaw + 0.5 * l * sin_yaw
  bev_corners[2, 1] = y + 0.5 * w * sin_yaw - 0.5 * l * cos_yaw

  # front right
  bev_corners[3, 0] = x + 0.5 * w * cos_yaw - 0.5 * l * sin_yaw
  bev_corners[3, 1] = y + 0.5 * w * sin_yaw + 0.5 * l * cos_yaw

  return bev_corners


class frame():
  def __init__(self, lidar_file) -> None:
    pos_arr = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    in_range  = (pos_arr[:, 0] > kitti_cfg.boundary["minX"]) & (pos_arr[:, 0] < kitti_cfg.boundary["maxX"])
    in_range &= (pos_arr[:, 1] > kitti_cfg.boundary["minY"]) & (pos_arr[:, 1] < kitti_cfg.boundary["maxY"])
    in_range &= (pos_arr[:, 2] > kitti_cfg.boundary["minZ"]) & (pos_arr[:, 2] < kitti_cfg.boundary["maxZ"])
    self.pos = pos_arr[in_range]

    # Initialize the BEV Maps: 0=>base; 1=>height; 2:density
    self.bev = np.zeros((3, kitti_cfg.BEV_HEIGHT, kitti_cfg.BEV_WIDTH))

  def set_grid(self, z_arr, gx, gy):
    b, h = z_arr.min(), z_arr.max()
    cnt = 1
    if len(z_arr) > 1:
      idx_arr = np.where((z_arr[1:] > z_arr[:-1] + kitti_cfg.max_z_gap)
                         & (z_arr[1:] > kitti_cfg.sig_z_thresh))[0]
      if len(idx_arr) > 0:
        h = z_arr[idx_arr[0]]
        cnt = idx_arr[0]

    self.bev[0, gx, gy] = b
    self.bev[1, gx, gy] = h
    # Normalizing the dist square
    self.bev[2, gx, gy] = cnt * (gx*gx + (gy - kitti_cfg.BEV_WIDTH/2)**2) * 2e-4


  # @timer_func
  def set_bev_map(self):
    """
    Return base_map, height_map and the counts map as a supervisor
    """
    # Extract x, y, z coordinates
    x, y = self.pos[:, 0], self.pos[:, 1]

    # Compute the grid indices for every point
    idx_x = (x / kitti_cfg.DISCRETIZATION).astype(np.int32)
    idx_y = ((y - kitti_cfg.boundary['minY']) / kitti_cfg.DISCRETIZATION).astype(np.int32)

    glb_idx = idx_x * kitti_cfg.BEV_WIDTH + idx_y
    # The last key in the sequence is used for the primary sort order, 
    # the second-to-last key for the secondary sort order, and so on.
    sorted_indices = np.lexsort((self.pos[:, 2], glb_idx))
    i0 = 0
    self.pos = self.pos[sorted_indices]
    unique_idx, idx_counts = np.unique(glb_idx[sorted_indices], return_counts=True)

    for i in range(len(unique_idx)):
      # Decode the grid index
      gx, gy = divmod(unique_idx[i], kitti_cfg.BEV_WIDTH)

      # Set the grid height and base
      self.set_grid(self.pos[i0:(i0+idx_counts[i]), 2], gx, gy)
      i0 += idx_counts[i]

    # Reset the base shift
    # base = self.bev[gx, gy, 0].min()
    # self.bev[gx, gy, :2] -= base


class agents():
  """
  For a given frame of point cloud, get the label ready for the frame

  Calibration matrices and utils
      3d XYZ in <label>.txt are in rect camera coord.
      2d box xy are in image2 coord
      Points in <lidar>.bin are in Velodyne coord.

      y_image2 = P^2_rect * x_rect
      y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
      x_ref = Tr_velo_to_cam * x_velo
      x_rect = R0_rect * x_ref

      P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                  0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                  0,      0,      1,      0]
                = K * [1|t]

      image2 coord:
        ----> x-axis (u)
      |
      |
      v y-axis (v)

      velodyne coord:
      front x, left y, up z

      rect/ref camera coord:
      right x, down y, front z

      Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

  """
  def __init__(self, label_path, calib_filepath) -> None:
    self.c2l = None
    self.labels = []

    if os.path.isfile(calib_filepath):
      with open(calib_filepath) as f:
        lines = f.readlines()

      obj = lines[4].strip().split(' ')[1:]
      R0 = np.eye(4)
      R0[:3, :3] = (np.array(obj, dtype=np.float32)).reshape(3, 3)

      obj = lines[5].strip().split(' ')[1:]
      V2C = np.eye(4)
      V2C[:3, :] = (np.array(obj, dtype=np.float32)).reshape(3, 4)
      self.c2l = np.linalg.inv(R0 @ V2C)

    for line in open(label_path, 'r'):
      line = line.rstrip()
      line_parts = line.split(' ')
      obj_name = line_parts[0]  # 'Car', 'Pedestrian', ...
      cat_id = int(kitti_cfg.CLASS_NAME_TO_ID[obj_name])
      if cat_id <= -99:  # ignore Tram and Misc
        continue

      # height, width, length (h, w, l)
      h, w, l = float(line_parts[8]), float(line_parts[9]), float(line_parts[10])
      # location (x,y,z) in camera coord.
      x, y, z = float(line_parts[11]), float(line_parts[12]), float(line_parts[13])

      # Convert this camera coord (x, y, z) into lidar coord
      x, y, z = self.camera_to_lidar(np.array([x,y,z,1]))
      # yaw
      ry = float(line_parts[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
      self.labels.append([cat_id, x, y, z, h, w, l, ry])

  def camera_to_lidar(self, p):
    if self.c2l is None:
      p = c2l_mat @ p
    else:
      p = self.c2l @ p
    return p[:3]

def compute_radius(height, width, min_overlap=0.7):
  a1 = 1
  b1 = (height + width)
  c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1 = (b1 + sq1) / 2

  a2 = 4
  b2 = 2 * (height + width)
  c2 = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2 = (b2 + sq2) / 2

  a3 = 4 * min_overlap
  b3 = -2 * min_overlap * (height + width)
  c3 = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3 = (b3 + sq3) / 2

  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m + 1, -n:n + 1]
  h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0

  return h

def gen_hm_radius(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

  return heatmap
