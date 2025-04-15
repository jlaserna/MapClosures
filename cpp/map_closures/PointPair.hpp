#pragma once

#include <Eigen/Core>

namespace map_closures {
struct PointPair {
    PointPair() = default;
    PointPair(const Eigen::Vector3d &r, const Eigen::Vector3d &q);
    Eigen::Vector3d ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d query = Eigen::Vector3d::Zero();
};
}  // namespace map_closures