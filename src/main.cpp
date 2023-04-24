#include <iostream>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>

#include "frame.h"
#include "visualize.h"


int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <npz filename>" << std::endl;
    return 1;
  }

  // Load the JSON configuration
  std::ifstream config_file("cfg/cfg.json");
  if (!config_file.is_open()) {
    std::cerr << "Error: Could not open ../cfg/cfg.json" << std::endl;
    return 1;
  }

  // Parse the Json file
  nlohmann::json config;
  try {
    config_file >> config;
  } catch (nlohmann::json::parse_error& e) {
    std::cerr << "Error: JSON parse error: " << e.what() << std::endl;
    return 1;
  }

  std::string npz_filename = argv[1];

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
    // Create a Frame object by loading an npz file
    Frame frame;

    // Load configuration
    frame.load_cfg(config);

    // Load points
    auto t0 = std::chrono::steady_clock::now();
    frame.load_pts(npz_filename);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Loading Points: "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
              << "us" << std::endl;

    // Gridding
    t0 = t1;
    frame.gridding();
    t1 = std::chrono::steady_clock::now();
    std::cout << "Gridding: "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
              << "us" << std::endl;

    // Ground Detection
    t0 = t1;
    frame.ground_detection();
    t1 = std::chrono::steady_clock::now();
    std::cout << "Ground Detection: "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
              << "us" << std::endl;

    // Visualize the frame using the visualize member function
    visualize_frame(viewer, cloud, frame);

    // frame.visualize(viewer, cloud);

    // Add delay
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // viewer->spinOnce(100);
    viewer->spin();
  }

  return 0;
}
