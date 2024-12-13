// MIT License
//
// Copyright (c) 2024 Javier Laserna
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <utility>
#include <vector>

#include "PointPair.hpp"

namespace map_closures {

std::tuple<Eigen::Isometry2d, int, std::vector<PointPair2D>> CliRegAlignment2D(
    const std::vector<PointPair2D> &keypoint_pairs);

std::tuple<Eigen::Isometry3d, int, std::vector<PointPair3D>> CliRegAlignment3D(
    const std::vector<PointPair3D> &keypoint_pairs,
    const double inliers3d_distance_threshold = 0.1);

}  // namespace map_closures
