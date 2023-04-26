#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <string>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


void visualize_frame_basic(pcl::visualization::PCLVisualizer::Ptr viewer,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr xyzi, const Frame & frm);


void visualize_frame(pcl::visualization::PCLVisualizer::Ptr viewer,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_xyzi,
                     pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_xyzi,
                     const Frame & frm);

#endif // VISUALIZE_H
