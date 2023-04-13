#include "frame.h"

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

	// std::cout << "Number of the valid points: " << n_points << std::endl;

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