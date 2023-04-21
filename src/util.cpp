#include "util.h"
#include <iostream>
#include <Eigen/Dense>

std::array<float, 3> plane_fit(const Eigen::Array<float, Eigen::Dynamic, 3>& pos) {
  Eigen::ArrayXf mean = pos.colwise().mean();
  Eigen::ArrayXXf centered = pos.rowwise() - mean.transpose();

  float pxx = (centered.col(0) * centered.col(0)).sum();
  float pyy = (centered.col(1) * centered.col(1)).sum();
  float pxy = (centered.col(0) * centered.col(1)).sum();
  float pxz = (centered.col(0) * centered.col(2)).sum();
  float pyz = (centered.col(1) * centered.col(2)).sum();

  float dxy = pxx * pyy - pxy * pxy;
  float a = 0;
  if (dxy != 0) {
    a = (pxz * pyy - pyz * pxy) / dxy;
  }

  float b = 0;
  if (pxy != 0) {
    b = (pxz - a * pxx) / pxy;
  }

  float c = mean(2) - a * mean(0) - b * mean(1);

  return std::array<float, 3>{a, b, c};
}
