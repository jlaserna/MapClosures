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
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace {
// fixed parameters for OpenCV ORB Features
static constexpr float scale_factor = 1.00;
static constexpr int n_levels = 1;
static constexpr int first_level = 0;
static constexpr int WTA_K = 2;
static constexpr int nfeatures = 500;
static constexpr int edge_threshold = 31;
static constexpr int score_type = 0;
static constexpr int patch_size = 31;
static constexpr int fast_threshold = 35;
// fixed parameters for BSHOT
static constexpr double voxel_grid_size = 0.5;
}  // namespace

namespace map_closures {
MapClosures::MapClosures() : config_(Config()) {
    orb_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
    bshot_extractor_ = std::make_shared<BSHOT::bshot_extractor>(voxel_grid_size);
}

MapClosures::MapClosures(const Config &config) : config_(config) {
    orb_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
    bshot_extractor_ = std::make_shared<BSHOT::bshot_extractor>(voxel_grid_size);
}

ClosureCandidate2D MapClosures::MatchAndAdd2D(const int id,
                                              const std::vector<Eigen::Vector3d> &local_map) {
    local_maps_.emplace(id, local_map);
    DensityMap density_map =
        GenerateDensityMap(local_map, config_.density_map_resolution, config_.density_threshold);
    cv::Mat orb_descriptors;
    std::vector<cv::KeyPoint> orb_keypoints;
    orb_extractor_->detectAndCompute(density_map.grid, cv::noArray(), orb_keypoints,
                                     orb_descriptors);

    auto hbst_matchable = Tree::getMatchables(orb_descriptors, orb_keypoints, id);
    hbst_binary_tree_->matchAndAdd(hbst_matchable, descriptor_matches_,
                                   config_.hamming_distance_threshold,
                                   srrg_hbst::SplittingStrategy::SplitEven);
    density_maps_.emplace(id, std::move(density_map));
    std::vector<int> indices(descriptor_matches_.size());
    std::transform(descriptor_matches_.cbegin(), descriptor_matches_.cend(), indices.begin(),
                   [](const auto &descriptor_match) { return descriptor_match.first; });
    auto compare_closure_candidates = [](ClosureCandidate2D a,
                                         const ClosureCandidate2D &b) -> ClosureCandidate2D {
        return a.number_of_inliers > b.number_of_inliers ? a : b;
    };
    using iterator_type = std::vector<int>::const_iterator;
    const auto &closure = tbb::parallel_reduce(
        tbb::blocked_range<iterator_type>{indices.cbegin(), indices.cend()}, ClosureCandidate2D(),
        [&](const tbb::blocked_range<iterator_type> &r,
            ClosureCandidate2D candidate) -> ClosureCandidate2D {
            return std::transform_reduce(
                r.begin(), r.end(), candidate, compare_closure_candidates, [&](const auto &ref_id) {
                    const bool is_far_enough = std::abs(static_cast<int>(ref_id) - id) > 3;
                    return is_far_enough ? ValidateClosure2D(ref_id, id) : ClosureCandidate2D();
                });
        },
        compare_closure_candidates);
    return closure;
}

ClosureCandidate2D MapClosures::ValidateClosure2D(const int reference_id,
                                                  const int query_id) const {
    const Tree::MatchVector &matches = descriptor_matches_.at(reference_id);
    const size_t num_matches = matches.size();

    ClosureCandidate2D closure;
    if (num_matches > 2) {
        const auto &ref_map_lower_bound = density_maps_.at(reference_id).lower_bound;
        const auto &qry_map_lower_bound = density_maps_.at(query_id).lower_bound;
        auto to_world_point = [](const auto &p, const auto &offset) {
            return Eigen::Vector2d(p.y + offset.x(), p.x + offset.y());
        };
        std::vector<PointPair2D> keypoint_pairs(num_matches);
        std::transform(
            matches.cbegin(), matches.cend(), keypoint_pairs.begin(),
            [&](const Tree::Match &match) {
                auto ref_point = to_world_point(match.object_references[0].pt, ref_map_lower_bound);
                auto query_point = to_world_point(match.object_query.pt, qry_map_lower_bound);
                return PointPair2D(ref_point, query_point);
            });
        Eigen::Isometry2d pose2d;
        int number_of_inliers;
        std::vector<PointPair2D> inliers;
        const auto start = std::chrono::high_resolution_clock::now();
        switch (config_.alignment_algorithm) {
            case AlignmentAlgorithm::RANSAC: {
                std::tie(pose2d, number_of_inliers, inliers) = RansacAlignment2D(keypoint_pairs);
                break;
            }
            case AlignmentAlgorithm::CLIREG: {
                std::tie(pose2d, number_of_inliers, inliers) = CliRegAlignment2D(keypoint_pairs);
                break;
            }
        }
        const auto end = std::chrono::high_resolution_clock::now();
        closure.alignment_time = std::chrono::duration<double, std::milli>(end - start).count();
        closure.source_id = reference_id;
        closure.target_id = query_id;
        closure.pose.block<2, 2>(0, 0) = pose2d.linear();
        closure.pose.block<2, 1>(0, 3) = pose2d.translation() * config_.density_map_resolution;
        closure.number_of_inliers = number_of_inliers;
        closure.keypoint_pairs.reserve(keypoint_pairs.size());
        closure.inliers.reserve(inliers.size());
        auto to_map_point = [](const auto &p, const auto &offset) {
            return Eigen::Vector2d(p.y() - offset.y(), p.x() - offset.x());
        };
        std::transform(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                       std::back_inserter(closure.keypoint_pairs), [&](const PointPair2D &pair) {
                           return std::make_pair(to_map_point(pair.ref, ref_map_lower_bound),
                                                 to_map_point(pair.query, qry_map_lower_bound));
                       });

        std::transform(inliers.cbegin(), inliers.cend(), std::back_inserter(closure.inliers),
                       [&](const PointPair2D &pair) {
                           return std::make_pair(to_map_point(pair.ref, ref_map_lower_bound),
                                                 to_map_point(pair.query, qry_map_lower_bound));
                       });
    }
    return closure;
}

ClosureCandidate3D MapClosures::MatchAndAdd3D(const int id,
                                              const std::vector<Eigen::Vector3d> &local_map) {
    local_maps_.emplace(id, local_map);
    DensityMap density_map =
        GenerateDensityMap(local_map, config_.density_map_resolution, config_.density_threshold);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_points(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : local_map) {
        pcl_points->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    std::vector<BSHOT::BSHOTSignature352> bshot_descriptors;
    std::vector<pcl::PointXYZ> bshot_keypoints;
    bshot_extractor_->detectAndCompute(pcl_points, bshot_descriptors, bshot_keypoints);

    Matchable3DVector matchables(bshot_keypoints.size());
    for (size_t i = 0; i < bshot_keypoints.size(); ++i) {
        matchables[i] = new Matchable3D(bshot_keypoints[i], bshot_descriptors[i].bits, id);
    }

    hbst_binary_tree_3d_->matchAndAdd(matchables, descriptor_matches_3d_,
                                      config_.hamming_distance_threshold,
                                      srrg_hbst::SplittingStrategy::SplitEven);
    density_maps_.emplace(id, std::move(density_map));
    std::vector<int> indices(descriptor_matches_3d_.size());
    std::transform(descriptor_matches_3d_.cbegin(), descriptor_matches_3d_.cend(), indices.begin(),
                   [](const auto &descriptor_match) { return descriptor_match.first; });
    auto compare_closure_candidates = [](ClosureCandidate3D a,
                                         const ClosureCandidate3D &b) -> ClosureCandidate3D {
        return a.number_of_inliers > b.number_of_inliers ? a : b;
    };
    using iterator_type = std::vector<int>::const_iterator;
    const auto &closure = tbb::parallel_reduce(
        tbb::blocked_range<iterator_type>{indices.cbegin(), indices.cend()}, ClosureCandidate3D(),
        [&](const tbb::blocked_range<iterator_type> &r,
            ClosureCandidate3D candidate) -> ClosureCandidate3D {
            return std::transform_reduce(
                r.begin(), r.end(), candidate, compare_closure_candidates, [&](const auto &ref_id) {
                    const bool is_far_enough = std::abs(static_cast<int>(ref_id) - id) > 3;
                    return is_far_enough ? ValidateClosure3D(ref_id, id) : ClosureCandidate3D();
                });
        },
        compare_closure_candidates);
    return closure;
}

ClosureCandidate3D MapClosures::ValidateClosure3D(const int reference_id,
                                                  const int query_id) const {
    const Tree3D::MatchVector &matches = descriptor_matches_3d_.at(reference_id);
    const size_t num_matches = matches.size();

    ClosureCandidate3D closure;
    if (num_matches > 2) {
        std::vector<PointPair3D> keypoint_pairs(num_matches);
        std::transform(matches.cbegin(), matches.cend(), keypoint_pairs.begin(),
                       [&](const Tree3D::Match &match) {
                           pcl::PointXYZ ref_point = match.object_references[0];
                           pcl::PointXYZ query_point = match.object_query;
                           return PointPair3D(
                               Eigen::Vector3d(ref_point.x, ref_point.y, ref_point.z),
                               Eigen::Vector3d(query_point.x, query_point.y, query_point.z));
                       });
        Eigen::Isometry3d pose3d;
        int number_of_inliers;
        std::vector<PointPair3D> inliers;
        const auto start = std::chrono::high_resolution_clock::now();
        std::tie(pose3d, number_of_inliers, inliers) =
            CliRegAlignment3D(keypoint_pairs, voxel_grid_size * 2);
        const auto end = std::chrono::high_resolution_clock::now();
        closure.alignment_time = std::chrono::duration<double, std::milli>(end - start).count();
        closure.source_id = reference_id;
        closure.target_id = query_id;
        closure.pose = pose3d.matrix();
        closure.number_of_inliers = number_of_inliers;
        closure.keypoint_pairs.reserve(keypoint_pairs.size());
        closure.inliers.reserve(inliers.size());
        std::transform(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                       std::back_inserter(closure.keypoint_pairs), [](const PointPair3D &pair) {
                           return std::make_pair(pair.ref, pair.query);
                       });

        std::transform(
            inliers.cbegin(), inliers.cend(), std::back_inserter(closure.inliers),
            [](const PointPair3D &pair) { return std::make_pair(pair.ref, pair.query); });
    }
    return closure;
}
}  // namespace map_closures
