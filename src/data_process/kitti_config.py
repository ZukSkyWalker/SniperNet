import numpy as np

# Car and Van ==> Car class
# Pedestrian and Person_Sitting ==> Pedestrian Class
CLASS_NAME_TO_ID = {
		'Pedestrian': 0,
		'Car': 1,
		'Cyclist': 2,
		'Van': 1,
		'Truck': -3,
		'Person_sitting': 0,
		'Tram': -99,
		'Misc': -99,
		'DontCare': -1
}

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 120, 0],
					[255, 120, 120], [0, 120, 0], [120, 255, 255], [120, 0, 255]]

boundary = {
		"minX": 0,
		"maxX": 64,
		"minY": -32,
		"maxY": 32,
		"minZ": -2.73,
		"maxZ": 6
}

bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']
bound_size_z = boundary['maxZ'] - boundary['minZ']

boundary_back = {
		"minX": -50,
		"maxX": 0,
		"minY": -25,
		"maxY": 25,
		"minZ": -2.73,
		"maxZ": 1.27
}

# Minimum ratio and number of points, used to determine the ground base
min_ground_ratio = 1e-3
min_ground_layer_pts = 30

# Z gap to cut off the ceiling like points above targets
max_z_gap = 0.4

# If below the height, we simply include the points (any gap may be due to occlusion)
sig_z_thresh = 1.2

# maximum number of points per voxel
T = 35

# voxel size
vd = 0.1   # z
vx = 0.05  # x
vy = 0.05  # y

# voxel grid
W = int(np.ceil(bound_size_x / vx))
H = int(np.ceil(bound_size_y / vy))
D = int(np.ceil(bound_size_z / vd))

# Following parameters are calculated as an average from KITTI dataset for simplicity
#####################################################################################
Tr_velo_to_cam = np.array([
		[7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
		[1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
		[9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
		[0, 0, 0, 1]
])

# cal mean from train set
R0 = np.array([
		[0.99992475, 0.00975976, -0.00734152, 0],
		[-0.0097913, 0.99994262, -0.00430371, 0],
		[0.00729911, 0.0043753, 0.99996319, 0],
		[0, 0, 0, 1]
])

P2 = np.array([[719.787081, 0., 608.463003, 44.9538775],
							 [0., 719.787081, 174.545111, 0.1066855],
							 [0., 0., 1., 3.0106472e-03],
							 [0., 0., 0., 0]
							 ])

R0_inv = np.linalg.inv(R0)
Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam)
P2_inv = np.linalg.pinv(P2)
#####################################################################################