// MIT License
//
// Copyright (c) 2024 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
// Ignacio Vizzo, Cyrill Stachniss.
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

#include "MapClosures.hpp"

#include <pcl/common/transforms.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

#include "AlignCliReg.hpp"
#include "AlignRansac.hpp"
#include "bshot.h"

namespace {
static constexpr int min_no_of_matches = 4;
static constexpr int no_of_local_maps_to_skip = 3;

// fixed parameters for BSHOT
static constexpr double voxel_grid_size = 0.5;
}  // namespace

namespace map_closures {
MapClosures::MapClosures() : config_(Config()) {
    bshot_extractor_ = std::make_shared<BSHOT::bshot_extractor>(voxel_grid_size);
}

MapClosures::MapClosures(const Config &config) : config_(config) {
    bshot_extractor_ = std::make_shared<BSHOT::bshot_extractor>(voxel_grid_size);
}

std::vector<ClosureCandidate> MapClosures::MatchAndAdd(
    const int query_id, const std::vector<Eigen::Vector3d> &local_map) {
    local_maps_.emplace(query_id, local_map);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_points(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : local_map) {
        pcl_points->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    std::vector<pcl::PointXYZ> bshot_keypoints;
    std::vector<BSHOT::BSHOTSignature352> bshot_descriptors;
    bshot_extractor_->detectAndCompute(pcl_points, bshot_descriptors, bshot_keypoints);

    MatchableVector matchables(bshot_keypoints.size());
    for (size_t i = 0; i < bshot_keypoints.size(); ++i) {
        matchables[i] = new Matchable(bshot_keypoints[i], bshot_descriptors[i].bits, query_id);
    }

    hbst_binary_tree_->matchAndAdd(matchables, descriptor_matches_,
                                   config_.hamming_distance_threshold,
                                   srrg_hbst::SplittingStrategy::SplitEven);

    auto is_far_enough = [](const int ref_id, const int query_id) {
        return std::abs(query_id - ref_id) > no_of_local_maps_to_skip;
    };

    std::vector<ClosureCandidate> closures;
    closures.reserve(query_id);
    std::for_each(descriptor_matches_.cbegin(), descriptor_matches_.cend(),
                  [&](const auto &descriptor_match) {
                      const auto ref_id = static_cast<int>(descriptor_match.first);
                      if (is_far_enough(ref_id, query_id)) {
                          const ClosureCandidate &closure = ValidateClosure(ref_id, query_id);
                          if (closure.number_of_inliers > min_no_of_matches) {
                              closures.emplace_back(closure);
                          }
                      }
                  });
    closures.shrink_to_fit();
    return closures;
}

ClosureCandidate MapClosures::ValidateClosure(const int reference_id, const int query_id) const {
    const Tree::MatchVector &matches = descriptor_matches_.at(reference_id);
    const size_t num_matches = matches.size();

    ClosureCandidate closure;
    if (num_matches > min_no_of_matches) {
        std::vector<PointPair> keypoint_pairs(num_matches);
        std::transform(matches.cbegin(), matches.cend(), keypoint_pairs.begin(),
                       [&](const Tree::Match &match) {
                           pcl::PointXYZ ref_point = match.object_references[0];
                           pcl::PointXYZ query_point = match.object_query;
                           return PointPair(
                               Eigen::Vector3d(ref_point.x, ref_point.y, ref_point.z),
                               Eigen::Vector3d(query_point.x, query_point.y, query_point.z));
                       });
        Eigen::Isometry3d pose3d;
        int number_of_inliers;
        std::vector<PointPair> inliers;
        const auto start = std::chrono::high_resolution_clock::now();
        std::tie(pose3d, number_of_inliers, inliers) =
            CliRegAlignment(keypoint_pairs, voxel_grid_size * 2);
        const auto end = std::chrono::high_resolution_clock::now();
        closure.alignment_time = std::chrono::duration<double, std::milli>(end - start).count();
        closure.source_id = reference_id;
        closure.target_id = query_id;
        closure.pose = pose3d.matrix();
        closure.number_of_inliers = number_of_inliers;
        closure.keypoint_pairs.reserve(keypoint_pairs.size());
        closure.inliers.reserve(inliers.size());
        std::transform(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                       std::back_inserter(closure.keypoint_pairs),
                       [](const PointPair &pair) { return std::make_pair(pair.ref, pair.query); });

        std::transform(inliers.cbegin(), inliers.cend(), std::back_inserter(closure.inliers),
                       [](const PointPair &pair) { return std::make_pair(pair.ref, pair.query); });
    }
    return closure;
}
}  // namespace map_closures