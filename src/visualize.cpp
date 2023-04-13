#include <iostream>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "cnpy.h"
#include "visualize.h"

void visualize_points(const std::string& npz_filename,
											pcl::visualization::PCLVisualizer::Ptr viewer,
											pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
  // Load point cloud data from npz file
  try {
    // Load positions from the npz file
    cnpy::npz_t npz = cnpy::npz_load(npz_filename);
		size_t num_points = npz["x"].shape[0];
		
		int32_t* xs = npz["x"].data<int32_t>();
		int32_t* ys = npz["y"].data<int32_t>();
		int32_t* zs = npz["z"].data<int32_t>();
    uint16_t* rs = npz["r"].data<uint16_t>();

    // Directly update the point cloud with the positions from the npz file
    cloud->width = num_points;
    cloud->height = 1;

		for (size_t i = 0; i < num_points; ++i) {
      cloud->points[i].x = xs[i]*1e-3;
      cloud->points[i].y = ys[i]*1e-3;
      cloud->points[i].z = zs[i]*1e-3;
      cloud->points[i].intensity = rs[i]*1e-2;
    }
		std::cout << "Number of points " << num_points << std::endl;

		// Update the viewer with the new point cloud data
    viewer->removePointCloud("points");

    // Add the new point cloud to the viewer
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_handler(cloud, "intensity");
    viewer->addPointCloud<pcl::PointXYZI>(cloud, intensity_handler, "points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "point_cloud");

    // Spin the viewer once to update the point cloud display
    // viewer->spinOnce();
		// viewer->spin();

  } catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    throw;
  }
}
