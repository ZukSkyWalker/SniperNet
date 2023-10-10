
boundary = {
  "minX": -80,
  "maxX": 80,
  "minY": -80,
  "maxY": 80
}

bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']

BEV_WIDTH  = 1600
BEV_HEIGHT = 1600
DX = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT
DY = (boundary["maxY"] - boundary["minY"]) / BEV_WIDTH


# Minimum ratio and number of points, used to determine the ground base
min_ground_ratio = 1e-3
min_ground_layer_pts = 30

# Z gap to cut off the ceiling like points above targets
max_z_gap = 0.4

# If below the height, we simply include the points (any gap may be due to occlusion)
sig_z_thresh = 1.2

# maximum number of points per voxel
T = 35

# max number of samples constraint: (-1 for no constraint, use everything we have)
num_samples = -1