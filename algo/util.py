import numpy as np


def plane_fit(pos):
	"""
	Return the local ground parameters
	O(N) complexity
	"""
	# mat = np.vstack([pos[:, :2].T, np.ones(len(pos))]) # (3 * N), 3 -> x, y, 1
	# grd_par = np.linalg.lstsq(mat.T, pos[:, 2], rcond=None)[0]
	# return grd_par

	x0, y0, z0 = pos.mean(axis=0)
	dx = pos[:, 0] - x0
	dy = pos[:, 1] - y0
	dz = pos[:, 2] - z0

	pxx = (dx*dx).sum()
	pyy = (dy*dy).sum()
	pxy = (dx*dy).sum()
	pxz = (dx*dz).sum()
	pyz = (dy*dz).sum()

	# Calculate the slope
	dxy = pxx * pyy - pxy * pxy
	a = 0
	if dxy != 0:
			a = (pxz * pyy - pyz * pxy) / dxy

	b = 0
	if pxy != 0:
			b = (pxz - a*pxx) / pxy
			
	return a, b, z0 - a*x0 - b*y0