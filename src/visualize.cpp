#include "frame.h"
#include "visualize.h"
#include <iostream>
#include <string>


void visualize_frame(pcl::visualization::PCLVisualizer::Ptr viewer,
										 pcl::PointCloud<pcl::PointXYZI>::Ptr xyzi, const Frame & frm) {
  for (size_t i = 0; i < frm.n_points; ++i) {
		xyzi->points[i].x = frm.pos(i, 0);
		xyzi->points[i].y = frm.pos(i, 1);
		xyzi->points[i].z = frm.pos(i, 2);
		xyzi->points[i].intensity = frm.intensities[i];
	}
	// Update the viewer
	viewer->removePointCloud("points");
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_handler(xyzi, "intensity");
  viewer->addPointCloud<pcl::PointXYZI>(xyzi, intensity_handler, "points");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "points");
}
