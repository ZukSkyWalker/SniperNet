#include <iostream>
#include <string>
#include <vector>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "visualize.h"

using namespace std;

void visualize_points(const std::vector<float>& positions, size_t num_points) {
    // Create point cloud from positions
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(num_points);

    // std::cout << "Total " << num_points << " points" <<std::endl;

    for (size_t i = 0; i < num_points; ++i) {
        cloud->points[i].x = positions[i * 3];
        cloud->points[i].y = positions[i * 3 + 1];
        cloud->points[i].z = positions[i * 3 + 2];
    }

    // std::cout <<"Assigning Finished!" << std::endl;

    // Initialize visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 255, 255, 255);
    // std::cout <<"Visualizer initialized." << std::endl;

    // Add point cloud to viewer
    viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    // std::cout << "Added point cloud to viewer." << std::endl;

    // Start viewer
    while (!viewer->wasStopped()) {
        viewer->spin();
    }
}

