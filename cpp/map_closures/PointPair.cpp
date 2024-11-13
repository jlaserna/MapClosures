#include "PointPair.hpp"

namespace map_closures {

PointPair::PointPair(const Eigen::Vector2d &r, const Eigen::Vector2d &q) : ref(r), query(q) {}

}  // namespace map_closures