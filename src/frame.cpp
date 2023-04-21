#include "frame.h"
#include "util.h"
#include <algorithm>

Frame::Frame(const std::string& npz_filename) {
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

	for (size_t i = 0; i < firings; ++i) {
		if (n_points >= max_points) break;
		if (is_invalid[i]) continue;

		pos(n_points, 0) = xs[i];
		pos(n_points, 1) = ys[i];
		pos(n_points, 2) = zs[i];
		intensities[n_points] = rs[i];
		++n_points;
	}
}


void Frame::gridding(const json & cfg) {
	// Calculate the sqrt(x^2 + y^2)
	distXY.head(n_points) = (pos.topRows(n_points).col(0).array().square() 
	+ pos.topRows(n_points).col(1).array().square()).sqrt();

	// Calculate the angular array
	thetas.head(n_points) = (pos.topRows(n_points).col(0).array() / distXY.head(n_points)).asin();

	float max_theta = cfg["max_theta"].get<float>();
	float min_dist = cfg["min_dist"].get<float>();
	float inv_theta_grid_size = 1.0 / cfg["theta_grid_size"].get<float>();
	float inv_dist_grid_size = 1.0 / cfg["dist_grid_size"].get<float>();

	size_t n_dist_grids = cfg["n_dist_grids"].get<size_t>();
	size_t n_angular_grids = cfg["n_angular_grids"].get<size_t>();
	size_t grid_idx, ix, iy;

	float lidar_height = cfg["lidar_height"].get<float>();
	float max_slope = cfg["max_slope"].get<float>();
	float dz_local = cfg["dz_local"].get<float>();
	float dz_global = cfg["dz_global"].get<float>();
	float dz_cut;

	// Partition the points into the grids
	grid_indices.resize(n_dist_grids * n_angular_grids);
	// grd_candidates.resize(n_points);

	h_arr = pos.col(2) + lidar_height;

	// auto dz_cut_arr = distXY.topRows(n_points) * max_slope;
	// ix_arr = ((thetas + max_theta) / theta_grid_size).cast<int16_t>().cwiseMax(0).cwiseMin(n_angular_grids-1);
	// iy_arr = ((distXY - min_dist) / dist_grid_size).cast<int16_t>().cwiseMax(0).cwiseMin(n_dist_grids-1);

	// Loop through the points to get indices:
	for (size_t i = 0; i < n_points; ++i) {
		// Calculate the index in the for loop
		ix = static_cast<size_t>(std::min(std::max(thetas[i] + max_theta, 0.f) * inv_theta_grid_size, n_angular_grids-1.f));
		iy = static_cast<size_t>(std::min(std::max(distXY[i] - min_dist, 0.f) * inv_dist_grid_size, n_dist_grids-1.f));
		grid_idx = ix + iy * n_angular_grids;

		// Convert to the grid index
		// grid_idx = ix_arr[i] + iy_arr[i] * n_angular_grids;
		grid_indices[grid_idx].emplace_back(i);

		// Check if this point is a potential ground points
		// h_arr[i] = pos(i, 2) + lidar_height;

		// Get the z_cut for this point to be a candidate ground point
		// dz_cut = std::min(std::max(dz_cut_arr[i], dz_local), dz_global);
		dz_cut = std::max(std::min(distXY[i] * max_slope, dz_global), dz_local);
		// std::cout << h_arr[i] << ", " << dz_cut << std::endl;

		if ((h_arr[i] > -dz_cut) && (h_arr[i] < dz_cut)) {
			grd_candidates.emplace_back(i);
		}
	}
}

void Frame::ground_detection(const json & cfg) {
	//Reset the flags array
	flags.setZero();

	std::cout << "Ground candidates: " << grd_candidates.size() << std::endl;

	// Get grd_candidates
	std::array<float, 3> grd_par = plane_fit(pos.topRows(n_points)(grd_candidates, Eigen::all));
	std::cout << std::fixed << std::setprecision(4);
	std::cout << "a=" << grd_par[0] << ", " << "b=" << grd_par[1] << ", " << "c=" << grd_par[2] << std::endl;
}


void Frame::visualize(pcl::visualization::PCLVisualizer::Ptr viewer,
								 pcl::PointCloud<pcl::PointXYZI>::Ptr xyzi) {
	for (size_t i = 0; i < n_points; ++i) {
		xyzi->points[i].x = pos(i, 0);
		xyzi->points[i].y = pos(i, 1);
		xyzi->points[i].z = pos(i, 2);
		xyzi->points[i].intensity = intensities[i];
	}
	// Update the viewer
	viewer->removePointCloud("points");
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_handler(xyzi, "intensity");
  viewer->addPointCloud<pcl::PointXYZI>(xyzi, intensity_handler, "points");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "points");

}