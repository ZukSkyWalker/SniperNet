#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <string>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


void visualize_points(const std::string& npz_filename,
                      pcl::visualization::PCLVisualizer::Ptr viewer,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

#endif // VISUALIZE_H
