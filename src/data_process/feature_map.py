import numpy as np
from numba import cuda
from util.timing import timer_func


class frame():
  def __init__(self, pos_arr, cfg) -> None:
    self.cfg = cfg
    in_range  = (pos_arr[:, 0] > -cfg["x_range"]) & (pos_arr[:, 0] < cfg["x_range"])
    in_range &= (pos_arr[:, 1] > -cfg["y_range"]) & (pos_arr[:, 1] < cfg["y_range"])
    in_range &= (pos_arr[:, 2] >  cfg["z_min"]) & (pos_arr[:, 2] < cfg["z_max"])
    
    self.pos = pos_arr[in_range]
    self.base = self.get_base()

    # General shift
    self.pos[:, 2] -= self.base
    self.base_map = np.zeros((cfg["NX"], cfg["NY"]))
    self.height_map = np.zeros((cfg["NX"], cfg["NY"]))
    self.density_map = np.zeros((cfg["NX"], cfg["NY"]))

  def load_point_cloud(self, pos_arr):
    in_range  = (pos_arr[:, 0] > -self.cfg["x_range"]) & (pos_arr[:, 0] < self.cfg["x_range"])
    in_range &= (pos_arr[:, 1] > -self.cfg["y_range"]) & (pos_arr[:, 1] < self.cfg["y_range"])
    in_range &= (pos_arr[:, 2] >  self.cfg["z_min"]) & (pos_arr[:, 2] < self.cfg["z_max"])
    
    self.pos = pos_arr[in_range]
    self.base = self.get_base()

    # General shift
    self.pos[:, 2] -= self.base


  def get_base(self):
    cnts, edges = np.histogram(self.pos[:, 2], bins=np.arange(self.cfg["z_min"],self.cfg["z_max"],self.cfg["dz"]))
    pts_thresh = max(len(self.pos) * self.cfg["min_ground_ratio"], self.cfg["min_ground_layer_pts"])
    base_idx = np.where(cnts > pts_thresh)[0][0]
    return edges[base_idx]
  
  def set_map(self, z_arr, gx, gy):
    b, h = z_arr.min(), z_arr.max()
    cnt = 1
    if len(z_arr) > 1:
      idx_arr = np.where((z_arr[1:] > z_arr[:-1] + self.cfg["max_z_gap"]) 
                         & (z_arr[1:] > self.base + self.cfg["sig_height_range"]))[0]
      if len(idx_arr) > 0:
        h = z_arr[idx_arr[0]]
        cnt = idx_arr[0]

    self.base_map[gx, gy] = b
    self.height_map[gx, gy] = h
    self.density_map[gx, gy] = cnt


  @timer_func
  def set_bev_map_cpu(self):
    """
    Return base_map, height_map and the counts map as a supervisor
    """
    # Extract x, y, z coordinates
    x, y = self.pos[:, 0], self.pos[:, 1]

    # Compute the grid indices for every point
    idx_x = (0.5 * (x + self.cfg["x_range"]) * self.cfg["NX"] / self.cfg["x_range"]).astype(np.int32)
    idx_y = (0.5 * (y + self.cfg["y_range"]) * self.cfg["NY"] / self.cfg["y_range"]).astype(np.int32)

    glb_idx = idx_x * self.cfg["NY"] + idx_y
    # The last key in the sequence is used for the primary sort order, 
    # the second-to-last key for the secondary sort order, and so on.
    sorted_indices = np.lexsort((self.pos[:, 2], glb_idx))
    
    i0 = 0
    self.pos = self.pos[sorted_indices]
    unique_idx, idx_counts = np.unique(glb_idx[sorted_indices], return_counts=True)

    for i in range(len(unique_idx)):
      # Decode the grid index
      gx, gy = divmod(unique_idx[i], self.cfg['NY'])

      # Set the grid height and base
      self.set_map(self.pos[i0:(i0+idx_counts[i]), 2], gx, gy)
      i0 += idx_counts[i]


  @timer_func
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


@cuda.jit
def bev_map_kernel(pos, x_range, y_range, NX, NY, max_z_gap, sig_height_range, base_map, height_map, density_map):
    tx = cuda.threadIdx.x  # Thread id in a 1D block
    ty = cuda.blockIdx.x   # Block id in a 1D grid
    bw = cuda.blockDim.x   # Block width, i.e. number of threads per block
    pos_idx = tx + ty * bw

    if pos_idx >= pos.shape[0]:  # Ensure we don't go out of bounds
        return

    x, y, z = pos[pos_idx]

    # Compute the grid indices for the point
    idx_x = int(0.5 * (x + x_range) * NX / x_range)
    idx_y = int(0.5 * (y + y_range) * NY / y_range)

    # Atomic operations ensure that updates from multiple threads are serialized
    cuda.atomic.min(base_map, (idx_x, idx_y), z)
    cuda.atomic.max(height_map, (idx_x, idx_y), z)
    cuda.atomic.add(density_map, (idx_x, idx_y), 1)  # Increment count by 1

def set_bev_map_gpu(self):
    # ... Preparation of data ...

    # Memory allocation and data transfer to device
    pos_device = cuda.to_device(self.pos)

    base_map_device = cuda.device_array((self.cfg["NX"], self.cfg["NY"]), dtype=np.float32)
    height_map_device = cuda.device_array((self.cfg["NX"], self.cfg["NY"]), dtype=np.float32)
    density_map_device = cuda.device_array((self.cfg["NX"], self.cfg["NY"]), dtype=np.int32)

    # Initialize base_map_device to a very large value for the atomic.min operation
    base_map_device[:] = np.finfo(np.float32).max

    # Initialize height_map_device to a very small value for the atomic.max operation
    height_map_device[:] = np.finfo(np.float32).min

    # Launching the kernel
    threadsperblock = 32
    blockspergrid = (self.pos.shape[0] + (threadsperblock - 1)) // threadsperblock
    bev_map_kernel[blockspergrid, threadsperblock](pos_device, self.cfg["x_range"], self.cfg["y_range"], self.cfg["NX"], self.cfg["NY"], self.cfg["max_z_gap"], self.cfg["sig_height_range"], base_map_device, height_map_device, density_map_device)

    # Copy data back from device to host
    base_map = base_map_device.copy_to_host()
    height_map = height_map_device.copy_to_host()
    density_map = density_map_device.copy_to_host()

    return base_map, height_map, density_map
