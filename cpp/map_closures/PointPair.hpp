#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <utility>
#include <vector>

namespace map_closures {

struct PointPair2D {
    PointPair2D() = default;
    PointPair2D(const Eigen::Vector2d &r, const Eigen::Vector2d &q);
    Eigen::Vector2d ref = Eigen::Vector2d::Zero();
    Eigen::Vector2d query = Eigen::Vector2d::Zero();
};

struct PointPair3D {
    PointPair3D() = default;
    PointPair3D(const Eigen::Vector3d &r, const Eigen::Vector3d &q);
    Eigen::Vector3d ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d query = Eigen::Vector3d::Zero();
};

}  // namespace map_closures
