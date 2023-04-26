#include "frame.h"
#include "visualize.h"
#include <iostream>
#include <string>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


void visualize_frame_basic(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZI>::Ptr xyzi, const Frame & frm) {
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


void visualize_frame(pcl::visualization::PCLVisualizer::Ptr viewer,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_xyzi,
                     pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_xyzi,
                     const Frame & frm) {
  ground_xyzi->clear();
  non_ground_xyzi->clear();

  for (size_t i = 0; i < frm.n_points; ++i) {
    pcl::PointXYZI point;
    point.x = frm.pos(i, 0);
    point.y = frm.pos(i, 1);
    point.z = frm.pos(i, 2);
    point.intensity = frm.intensities[i];

    if ((frm.flags[i] & static_cast<uint8_t>(Flag::IS_GROUND)) > 0) {
      uint8_t grey_value = static_cast<uint8_t>(std::min(point.intensity*255, 255.0f));
      pcl::PointXYZRGB point_rgb(grey_value, grey_value, grey_value);
      point_rgb.x = point.x;
      point_rgb.y = point.y;
      point_rgb.z = point.z;
      ground_xyzi->points.push_back(point_rgb);
    } else {
      non_ground_xyzi->points.push_back(point);
    }
  }

  ground_xyzi->width = ground_xyzi->points.size();
  ground_xyzi->height = 1;
  ground_xyzi->is_dense = true;

  // Update the viewer
  viewer->removePointCloud("ground_points");
  viewer->removePointCloud("non_ground_points");

  // Custom color handlers
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> ground_color_handler(ground_xyzi); // Grayscale for ground points based on intensity
  pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> non_ground_color_handler(non_ground_xyzi, "intensity");


  // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> ground_color_handler(ground_xyzi, 0, 255, 0); // Green for ground points
  // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> non_ground_color_handler(non_ground_xyzi, "intensity"); // Red for non-ground points

  viewer->addPointCloud<pcl::PointXYZRGB>(ground_xyzi, ground_color_handler, "ground_points");
  viewer->addPointCloud<pcl::PointXYZI>(non_ground_xyzi, non_ground_color_handler, "non_ground_points");

  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ground_points");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "non_ground_points");
}
