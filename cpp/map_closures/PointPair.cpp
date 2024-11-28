#include "PointPair.hpp"

namespace map_closures {

PointPair2D::PointPair2D(const Eigen::Vector2d &r, const Eigen::Vector2d &q) : ref(r), query(q) {}
PointPair3D::PointPair3D(const Eigen::Vector3d &r, const Eigen::Vector3d &q) : ref(r), query(q) {}

}  // namespace map_closures