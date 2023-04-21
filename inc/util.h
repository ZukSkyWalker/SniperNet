#pragma once

#include <Eigen/Core>


std::array<float, 3> plane_fit(const Eigen::Array<float, Eigen::Dynamic, 3>& pos);
