#include "frame.h"
#include "util.h"
#include <math.h> 
#include <algorithm>
#include <iomanip>


void Frame::load_cfg(const json & cfg) {
	max_theta = cfg["max_theta"].get<float>();
	min_dist = cfg["min_dist"].get<float>();
	inv_theta_grid_size = 1.0 / cfg["theta_grid_size"].get<float>();
	inv_dist_grid_size = 1.0 / cfg["dist_grid_size"].get<float>();

	lidar_height = cfg["lidar_height"].get<float>();
	max_slope = cfg["max_slope"].get<float>();
	dz_local = cfg["dz_local"].get<float>();
	dz_global = cfg["dz_global"].get<float>();

	n_angular_grids = static_cast<size_t>(2 * max_theta * inv_theta_grid_size);
	n_dist_grids = cfg["n_dist_grids"].get<size_t>();
	min_grd_pts = cfg["min_grd_pts"].get<size_t>();

}

void Frame::load_pts(const std::string& npz_filename) {
	// Load npz file
	cnpy::npz_t npz = cnpy::npz_load(npz_filename);
	size_t firings = npz["x"].shape[0];

	// Create pointers:
	float* xs = npz["x"].data<float>();
	float* ys = npz["y"].data<float>();
	float* zs = npz["z"].data<float>();
	float* rs = npz["r"].data<float>();
	bool * is_invalid = npz["invalid"].data<bool>();

	n_points = 0;
	float dxy, dz_cut, h_est;
	grd_candidates.reserve(max_points);

	for (size_t i = 0; i < firings; ++i) {
		if (n_points >= max_points) break;
		if (is_invalid[i]) continue;

		// Skip points that are below the global cut
		dxy = sqrt(xs[i]*xs[i] + ys[i]*ys[i]);
		dz_cut = std::max(std::min(dxy * max_slope, dz_global), dz_local);
		h_est = zs[i] + lidar_height;
		if (h_est + dz_cut < 0) continue;
		if (h_est < dz_cut)	grd_candidates.emplace_back(n_points);

		pos(n_points, 0) = xs[i];
		pos(n_points, 1) = ys[i];
		pos(n_points, 2) = zs[i];
		intensities[n_points] = rs[i];

		distXY[n_points] = dxy;
		thetas[n_points] = asin(xs[i] / dxy);

		++n_points;
	}
}


void Frame::gridding() {
	size_t grid_idx, ix, iy;
	float dz_cut;

	// Partition the points into the grids
	grid_indices.resize(n_dist_grids * n_angular_grids);

	// Loop through the points to get indices:
	for (size_t i = 0; i < n_points; ++i) {
		// Calculate the index in the for loop
		ix = static_cast<size_t>(std::min(std::max(thetas[i] + max_theta, 0.f) * inv_theta_grid_size, n_angular_grids-1.f));
		iy = static_cast<size_t>(std::min(std::max(distXY[i] - min_dist, 0.f) * inv_dist_grid_size, n_dist_grids-1.f));
		grid_idx = ix + iy * n_angular_grids;

		// Convert to the grid index
		grid_indices[grid_idx].emplace_back(i);
	}
}

void Frame::ground_detection() {
	//Reset the flags array
	flags.setZero();

	std::cout << "Ground candidates: " << grd_candidates.size() << std::endl;

	// Get grd_candidates
	std::array<float, 3> grd_par = plane_fit(pos(grd_candidates, Eigen::all));
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "a=" << grd_par[0] << ", " << "b=" << grd_par[1] << ", " << "c=" << grd_par[2] << std::endl;

	// Get the default h_arr
	h_arr = pos.col(2) - grd_par[0] * pos.col(0) - grd_par[1] * pos.col(1) - grd_par[2];

	std::cout << "Seed points: " << ((h_arr < dz_local).cast<int>()).sum() << std::endl;
	size_t ix, iy, nb_idx;
	int g_ix, g_iy;

	// Loop through the grounds to get the local points marked
	for (size_t i = 0; i< grid_indices.size();  ++i) {
		// Skip this grid if not enough seed ground points
		// if (grid_indices[i].size() < min_grd_pts) continue;

		if ((h_arr(grid_indices[i]) < dz_local).cast<size_t>().sum() < min_grd_pts) continue;

		// Get the ix and iy
		ix = i % n_angular_grids;
		iy = i / n_angular_grids;

		// get Neiboring indices of the grid and perform the fit
		std::vector<size_t> nearby_to_fit_idx;

		for (int d_ix = -1; d_ix < 2; d_ix++) {
			for (int d_iy = -1; d_iy < 2; d_iy++) {
				// if (d_ix == 0 && d_iy==0) continue;
				g_ix = d_ix + ix;
				if (g_ix < 0 || g_ix >= n_angular_grids) continue;

				g_iy = d_iy + iy;
				if (g_iy < 0 || g_iy >= n_dist_grids) continue;

				nb_idx = g_ix + g_iy * n_angular_grids;

				for (auto idx : grid_indices[nb_idx]) {
					if (h_arr[idx] < dz_local) nearby_to_fit_idx.emplace_back(idx);
				}
			}
		}

		// perform the fit
		std::array<float, 3> local_grd_par = plane_fit(pos(nearby_to_fit_idx, Eigen::all));

		// Update the height for the points in the grid
		h_arr(grid_indices[i]) = pos(grid_indices[i], 2) - local_grd_par[0] * pos(grid_indices[i], 0)
		- local_grd_par[1] * pos(grid_indices[i], 1) - local_grd_par[2];

		// Label the local ground points
		size_t local_grd_pts = 0;
		for (auto idx : grid_indices[i]) {
			if (h_arr[idx] < dz_local) {
				// Compiling time to convert the type
				flags[idx] |= static_cast<uint8_t>(Flag::IS_GROUND);
				local_grd_pts++;
			}
		}

		// std::cout << local_grd_pts << " local ground points:" << "a="<< local_grd_par[0] <<", b="<<local_grd_par[1] << std::endl;

	}

	std::cout << "ground points: " << ((flags > 0).cast<int>()).sum() << std::endl;
}
