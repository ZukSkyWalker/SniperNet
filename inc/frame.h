#pragma once

#include <Eigen/Dense>

#include "cnpy.h"
#include <json.hpp>

// This library is needed for using uint8_t, but already included in json.hpp and cnpy.h
// #include <cstdint>

enum class Flag : uint8_t {
	IS_NOISE = 1,
	IS_GROUND = 2,
	IS_BKG = 4,       // Points belonging to the static backgrounds
	IS_VEHICLE = 8,
	IS_SIGN = 16,
	IS_PED = 32,
	IS_BIKER = 64
};


const std::size_t max_points = 40000;
// const std::size_t n_grids = 1100;  // Make sure this number is larger than n_angular_grids * n_dist_grids

using json = nlohmann::json;

class Frame {
	public:
		Frame() = default;

		void load_cfg(const json & cfg);
		void load_pts(const std::string& npz_filename);

		void gridding();
		void ground_detection();

		// Matrix for matrix operation: x, y, z
		// Array for element wise operation
		Eigen::Array<float, max_points, 3> pos = Eigen::Array<float, max_points, 3>::Zero();
		Eigen::Array<float, max_points, 1> intensities = Eigen::Array<float, max_points, 1>::Zero();
		Eigen::Array<float, max_points, 1> h_arr = Eigen::Array<float, max_points, 1>::Zero();
		Eigen::Array<uint8_t, max_points, 1> flags = Eigen::Array<uint8_t, max_points, 1>::Zero();

		unsigned int n_points;

		// Configuration storage
		float max_theta, min_dist, inv_theta_grid_size, inv_dist_grid_size;
		float lidar_height, max_slope, dz_local, dz_global;
		size_t n_dist_grids, n_angular_grids;

		size_t min_grd_pts;

	private:
		// 1. Member functions
		// void ground_detection(const json & cfg);

		// 2. Private members
		Eigen::Array<float, max_points, 1> distXY = Eigen::Array<float, max_points, 1>::Zero();
		Eigen::Array<float, max_points, 1> thetas = Eigen::Array<float, max_points, 1>::Zero();

		// Prepare a container for the grid indices
		std::vector<std::vector<size_t>> grid_indices;

		// Prepare the vector of indices of groud candidates
		std::vector<size_t> grd_candidates;
};
