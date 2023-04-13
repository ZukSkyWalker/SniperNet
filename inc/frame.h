#pragma once

#include <Eigen/Dense>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "cnpy.h"

const std::size_t max_points = 40000;

class Frame {
	public:
		Frame(const std::string& npz_filename);
		void visualize(pcl::visualization::PCLVisualizer::Ptr viewer,
									 pcl::PointCloud<pcl::PointXYZI>::Ptr xyzi);

	private:
		// Matrix for matrix operation: x, y, z
		Eigen::MatrixXf pos = Eigen::MatrixXf::Zero(max_points, 3);

		Eigen::ArrayXf intensities = Eigen::ArrayXf::Zero(max_points);

		unsigned int n_points;
};
