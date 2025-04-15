#include "PointPair.hpp"

#include <Eigen/Core>

namespace map_closures {
PointPair::PointPair(const Eigen::Vector3d &r, const Eigen::Vector3d &q) : ref(r), query(q) {}
}  // namespace map_closures