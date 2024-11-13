#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <utility>
#include <vector>

namespace map_closures {

struct PointPair {
    PointPair() = default;
    PointPair(const Eigen::Vector2d &r, const Eigen::Vector2d &q);
    Eigen::Vector2d ref = Eigen::Vector2d::Zero();
    Eigen::Vector2d query = Eigen::Vector2d::Zero();
};

}  // namespace map_closures