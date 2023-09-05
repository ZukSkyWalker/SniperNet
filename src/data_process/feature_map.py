import numpy as np


class frame():
  def __init__(self, pos_arr, cfg) -> None:
    in_range  = (pos_arr[:, 0] > -cfg["x_range"]) & (pos_arr[:, 0] < cfg["x_range"])
    in_range &= (pos_arr[:, 1] > -cfg["y_range"]) & (pos_arr[:, 0] < cfg["y_range"])
    in_range &= (pos_arr[:, 2] >  cfg["z_min"]) & (pos_arr[:, 0] < cfg["z_max"])
    
    self.pos = pos_arr[in_range]
    self.base = self.get_base(cfg)

    # General shift
    self.pos[:, 2] -= self.base
    self.base_map = np.zeros((cfg["NX"], cfg["NY"]))
    self.height_map = np.zeros((cfg["NX"], cfg["NY"]))
    self.density_map = np.zeros((cfg["NX"], cfg["NY"]))


  def get_base(self, cfg):
    cnts, edges = np.histogram(self.pos[:, 2], bins=np.arange(cfg["z_min"],cfg["z_max"],cfg["dz"]))
    pts_thresh = max(len(self.pos) * cfg["min_ground_ratio"], cfg["min_ground_layer_pts"])
    base_idx = np.where(cnts > pts_thresh)[0][0]
    return edges[base_idx]


  def get_h_map(self, z_arr, cfg):
    # Notice the z distribution has already been shifted so the ground level is at around 0
    cnts, edges = np.histogram(z_arr, bins=np.arange(0,cfg["z_max"] - cfg["z_min"],cfg["dz"]))
    base = cfg["z_min"] - 1
    height = base
    gap = 0

    for i in range(len(cnts)):
      if cnts[i] > 0:
        if base < cfg["z_min"]:
          base = edges[i]
        height = edges[i]
        gap = 0

      elif edges[i] > cfg["h_range"]:
        gap += 1
        if gap > cfg["max_z_gap"]:
          # having a gap after high enough range
          break

    return base, height

 

  def set_bev_map_orig(self, cfg):
    """
    TODO:
    Fill self.base_map and self.height_map using the get_h_map
    """
    # Extract x, y, z coordinates
    x, y, z = self.pos[:, 0], self.pos[:, 1], 

    # Compute the grid indices for every point
    idx_x = (0.5 * (x + cfg["x_range"]) * cfg["NX"] / cfg["x_range"]).astype(np.uint16)
    idx_y = (0.5 * (y + cfg["y_range"]) * cfg["NY"] / cfg["y_range"]).astype(np.uint16)

    for gx in range(cfg['NX']):
      for gy in range(cfg["NY"]):
        in_grid = (gx == idx_x) & (gy == idx_y)
        b, h = self.get_h_map(self.pos[in_grid, 2])
        self.base_map[gx, gy] = b
        self.height_map[gx, gy] = h

  def set_bev_map(self, cfg):
    # Extract x, y, z coordinates
    x, y, z = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]

    # Compute the grid indices for every point
    idx_x = np.digitize(x, bins=np.linspace(-cfg["x_range"], cfg["x_range"], cfg["NX"] + 1)) - 1
    idx_y = np.digitize(y, bins=np.linspace(-cfg["y_range"], cfg["y_range"], cfg["NY"] + 1)) - 1

    # Create a 2D index array from idx_x and idx_y
    idx_2d = idx_x * cfg['NY'] + idx_y
    
    # Sort idx_2d and z for efficient processing
    sorted_indices = np.argsort(idx_2d)
    sorted_z = z[sorted_indices]
    sorted_idx_2d = idx_2d[sorted_indices]

    # Find the unique elements and their counts in sorted_idx_2d
    unique_elements, element_counts = np.unique(sorted_idx_2d, return_counts=True)

    # Compute the base and height for each unique 2D grid index
    for unique_idx, count in zip(unique_elements, element_counts):
      corresponding_z_values = sorted_z[:count]
      sorted_z = sorted_z[count:]  # Remove processed z values
      b, h = self.get_h_map(corresponding_z_values, cfg)
      gx, gy = divmod(unique_idx, cfg['NY'])
      self.base_map[gx, gy] = b
      self.height_map[gx, gy] = h

