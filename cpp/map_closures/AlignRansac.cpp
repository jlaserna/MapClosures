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

#include "AlignRansac.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

namespace {
Eigen::Isometry2d KabschUmeyamaAlignment2D(
    const std::vector<map_closures::PointPair2D> &keypoint_pairs) {
    auto mean = std::reduce(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                            map_closures::PointPair2D(), [](auto lhs, const auto &rhs) {
                                lhs.ref += rhs.ref;
                                lhs.query += rhs.query;
                                return lhs;
                            });
    mean.query /= keypoint_pairs.size();
    mean.ref /= keypoint_pairs.size();
    auto covariance_matrix = std::transform_reduce(
        keypoint_pairs.cbegin(), keypoint_pairs.cend(), Eigen::Matrix2d().setZero(),
        std::plus<Eigen::Matrix2d>(), [&](const auto &keypoint_pair) {
            return (keypoint_pair.ref - mean.ref) *
                   ((keypoint_pair.query - mean.query).transpose());
        });

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(covariance_matrix,
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Isometry2d T = Eigen::Isometry2d::Identity();
    const Eigen::Matrix2d &&R = svd.matrixV() * svd.matrixU().transpose();
    T.linear() = R.determinant() > 0 ? R : -R;
    T.translation() = mean.query - R * mean.ref;

    return T;
}

Eigen::Isometry3d KabschUmeyamaAlignment3D(
    const std::vector<map_closures::PointPair3D> &keypoint_pairs) {
    auto mean = std::reduce(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                            map_closures::PointPair3D(), [](auto lhs, const auto &rhs) {
                                lhs.ref += rhs.ref;
                                lhs.query += rhs.query;
                                return lhs;
                            });
    mean.query /= keypoint_pairs.size();
    mean.ref /= keypoint_pairs.size();
    auto covariance_matrix = std::transform_reduce(
        keypoint_pairs.cbegin(), keypoint_pairs.cend(), Eigen::Matrix3d().setZero(),
        std::plus<Eigen::Matrix3d>(), [&](const auto &keypoint_pair) {
            return (keypoint_pair.ref - mean.ref) *
                   ((keypoint_pair.query - mean.query).transpose());
        });

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance_matrix,
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d &U = svd.matrixU();
    const Eigen::Matrix3d &V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0) {
        Eigen::Matrix3d V_corrected = V;
        V_corrected.col(2) *= -1;
        R = V_corrected * U.transpose();
    }

    Eigen::Vector3d t = mean.query - R * mean.ref;
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = R;
    T.translation() = t;

    return T;
}

static constexpr double inliers_distance_threshold = 3.0;

// RANSAC Parameters
static constexpr double inliers_ratio = 0.3;
static constexpr double probability_success = 0.999;
static constexpr int min_points = 2;
static constexpr int __RANSAC_TRIALS__ = std::ceil(
    std::log(1.0 - probability_success) / std::log(1.0 - std::pow(inliers_ratio, min_points)));
}  // namespace

namespace map_closures {

std::tuple<Eigen::Isometry2d, int, std::vector<PointPair2D>> RansacAlignment2D(
    const std::vector<PointPair2D> &keypoint_pairs) {
    const size_t max_inliers = keypoint_pairs.size();

    std::vector<PointPair2D> sample_keypoint_pairs(2);
    std::vector<int> inlier_indices;
    inlier_indices.reserve(max_inliers);

    std::vector<int> optimal_inlier_indices;
    optimal_inlier_indices.reserve(max_inliers);

    int iter = 0;
    while (iter++ < __RANSAC_TRIALS__) {
        inlier_indices.clear();

        std::sample(keypoint_pairs.begin(), keypoint_pairs.end(), sample_keypoint_pairs.begin(), 2,
                    std::mt19937{std::random_device{}()});
        auto T = KabschUmeyamaAlignment2D(sample_keypoint_pairs);

        int index = 0;
        std::for_each(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                      [&](const auto &keypoint_pair) {
                          if ((T * keypoint_pair.ref - keypoint_pair.query).norm() <
                              inliers_distance_threshold)
                              inlier_indices.emplace_back(index);
                          index++;
                      });

        if (inlier_indices.size() > optimal_inlier_indices.size()) {
            optimal_inlier_indices = inlier_indices;
        }
    }

    const int num_inliers = optimal_inlier_indices.size();
    std::vector<PointPair2D> inlier_keypoint_pairs(num_inliers);
    std::transform(optimal_inlier_indices.cbegin(), optimal_inlier_indices.cend(),
                   inlier_keypoint_pairs.begin(),
                   [&](const auto index) { return keypoint_pairs[index]; });
    auto T = KabschUmeyamaAlignment2D(inlier_keypoint_pairs);

    std::vector<PointPair2D> inliers(num_inliers);
    std::transform(optimal_inlier_indices.cbegin(), optimal_inlier_indices.cend(), inliers.begin(),
                   [&](const auto index) { return keypoint_pairs[index]; });

    return {T, num_inliers, inliers};
}

std::tuple<Eigen::Isometry3d, int, std::vector<PointPair3D>> RansacAlignment3D(
    const std::vector<PointPair3D> &keypoint_pairs,
    const double inliers3d_distance_threshold) {
    const size_t max_inliers = keypoint_pairs.size();

    std::vector<PointPair3D> sample_keypoint_pairs(3);
    std::vector<int> inlier_indices;
    inlier_indices.reserve(max_inliers);

    std::vector<int> optimal_inlier_indices;
    optimal_inlier_indices.reserve(max_inliers);

    constexpr double inliers_ratio = 0.3;
    constexpr double probability_success = 0.999;
    constexpr int min_points = 3;
    const int max_trials = std::ceil(std::log(1.0 - probability_success) /
                                     std::log(1.0 - std::pow(inliers_ratio, min_points)));

    int iter = 0;
    int adaptive_max_trials = max_trials;
    while (iter++ < adaptive_max_trials) {
        inlier_indices.clear();

        std::sample(keypoint_pairs.begin(), keypoint_pairs.end(),
                    sample_keypoint_pairs.begin(), 3, std::mt19937{std::random_device{}()});
        auto T = KabschUmeyamaAlignment3D(sample_keypoint_pairs);

        // Check if the transformation is valid
        if (std::abs(T.linear().determinant() - 1.0) > 0.1)
            continue;

        int index = 0;
        std::for_each(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                      [&](const auto &keypoint_pair) {
                          if ((T * keypoint_pair.ref - keypoint_pair.query).norm() <
                              inliers3d_distance_threshold)
                              inlier_indices.emplace_back(index);
                          index++;
                      });

        // Edge length validation
        bool valid = true;
        if (optimal_inlier_indices.size() >= 2) {
            for (size_t i = 0; i < optimal_inlier_indices.size(); ++i) {
                for (size_t j = i + 1; j < optimal_inlier_indices.size(); ++j) {
                    const auto &pi_ref = keypoint_pairs[optimal_inlier_indices[i]].ref;
                    const auto &pj_ref = keypoint_pairs[optimal_inlier_indices[j]].ref;
                    const auto &pi_query = keypoint_pairs[optimal_inlier_indices[i]].query;
                    const auto &pj_query = keypoint_pairs[optimal_inlier_indices[j]].query;

                    double dist_ref = (pi_ref - pj_ref).norm();
                    double dist_query = (pi_query - pj_query).norm();

                    if (dist_ref < 1e-6 || dist_query < 1e-6) continue; // avoid division by zero
                    double ratio = std::min(dist_ref, dist_query) / std::max(dist_ref, dist_query);
                    if (ratio < 0.9) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
            }
        }

        if (!valid) continue;

        if (inlier_indices.size() > optimal_inlier_indices.size()) {
            optimal_inlier_indices = inlier_indices;

            // Adaptive max trials
            double inlier_ratio = static_cast<double>(optimal_inlier_indices.size()) / keypoint_pairs.size();
            double updated_trials = std::log(1.0 - probability_success) /
                                    std::log(1.0 - std::pow(inlier_ratio, min_points));
            if (std::isfinite(updated_trials)) {
                adaptive_max_trials = std::min(adaptive_max_trials, static_cast<int>(std::ceil(updated_trials)));
            }
        }
    }

    const int num_inliers = optimal_inlier_indices.size();
    std::vector<PointPair3D> inlier_keypoint_pairs(num_inliers);
    std::transform(optimal_inlier_indices.cbegin(), optimal_inlier_indices.cend(),
                   inlier_keypoint_pairs.begin(),
                   [&](const auto index) { return keypoint_pairs[index]; });
    auto T = KabschUmeyamaAlignment3D(inlier_keypoint_pairs);

    std::vector<PointPair3D> inliers(num_inliers);
    std::transform(optimal_inlier_indices.cbegin(), optimal_inlier_indices.cend(),
                   inliers.begin(), [&](const auto index) { return keypoint_pairs[index]; });

    return {T, num_inliers, inliers};
}
}  // namespace map_closures
