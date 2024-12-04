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

#pragma once

#include <Eigen/Core>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AlignRansac2D.hpp"
#include "DensityMap.hpp"
#include "bshot.h"
#include "srrg_hbst/types/binary_tree.hpp"

static constexpr int descriptor_size_bits = 256;
using Matchable = srrg_hbst::BinaryMatchable<cv::KeyPoint, descriptor_size_bits>;
using Node = srrg_hbst::BinaryNode<Matchable>;
using Tree = srrg_hbst::BinaryTree<Node>;

static constexpr int descriptor_size_bits_3d = 352;
using Matchable3D = srrg_hbst::BinaryMatchable<pcl::PointXYZ, descriptor_size_bits_3d>;
using Matchable3DVector = std::vector<Matchable3D *>;
using Node3D = srrg_hbst::BinaryNode<Matchable3D>;
using Tree3D = srrg_hbst::BinaryTree<Node3D>;

namespace map_closures {
enum class AlignmentAlgorithm { RANSAC, CLIREG };
struct Config {
    float density_map_resolution = 0.5;
    float density_threshold = 0.05;
    int hamming_distance_threshold = 50;
    double voxel_grid_size = 0.1;
    AlignmentAlgorithm alignment_algorithm = AlignmentAlgorithm::RANSAC;
};

struct ClosureCandidate2D {
    int source_id = -1;
    int target_id = -1;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    size_t number_of_inliers = 0;
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> keypoint_pairs;
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> inliers;
    double alignment_time = 0.0;
};

struct ClosureCandidate3D {
    int source_id = -1;
    int target_id = -1;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    size_t number_of_inliers = 0;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> keypoint_pairs;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> inliers;
    double alignment_time = 0.0;
};

class MapClosures {
public:
    explicit MapClosures();
    explicit MapClosures(const Config &config);
    ~MapClosures() = default;

    ClosureCandidate2D MatchAndAdd2D(const int id, const std::vector<Eigen::Vector3d> &local_map);
    ClosureCandidate3D MatchAndAdd3D(const int id, const std::vector<Eigen::Vector3d> &local_map);
    ClosureCandidate2D ValidateClosure2D(const int reference_id, const int query_id) const;
    ClosureCandidate3D ValidateClosure3D(const int reference_id, const int query_id) const;

    const DensityMap &getDensityMapFromId(const int &map_id) const {
        return density_maps_.at(map_id);
    }

private:
    Config config_;
    Tree::MatchVectorMap descriptor_matches_;
    Tree3D::MatchVectorMap descriptor_matches_3d_;
    std::unordered_map<int, DensityMap> density_maps_;
    std::unordered_map<int, std::vector<Eigen::Vector3d>> local_maps_;
    std::unique_ptr<Tree> hbst_binary_tree_ = std::make_unique<Tree>();
    std::unique_ptr<Tree3D> hbst_binary_tree_3d_ = std::make_unique<Tree3D>();
    cv::Ptr<cv::DescriptorExtractor> orb_extractor_;
    BSHOT::bshot_extractor::Ptr bshot_extractor_;
};
}  // namespace map_closures
