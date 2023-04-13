#include <iostream>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>

#include "cnpy.h"
#include "visualize.h"

const std::size_t max_points = 40000;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <directory_with_npz_files>" << std::endl;
    return 1;
  }

  std::string directory = argv[1];

  // Initialize point cloud object
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  cloud->points.resize(max_points);

  // Initialize the visualizer
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();

  // Start the rendering loop
  while (!viewer->wasStopped()) {
    for (size_t idx = 0; idx < 5000; idx++) {
      std::string npz_filename = directory + "rc_" + std::to_string(idx) + ".npz";
      visualize_points(npz_filename, viewer, cloud);

      // Add delay
      // std::this_thread::sleep_for(std::chrono::milliseconds(100));
      viewer->spinOnce(100);
      // viewer->spin();
    }
  }

  return 0;
}
