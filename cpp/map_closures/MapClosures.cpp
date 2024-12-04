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

#include "AlignCliReg2D.hpp"
#include "AlignCliReg3D.hpp"
#include "AlignRansac2D.hpp"
#include "DensityMap.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

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
// fixed parameters for 3D registration
static constexpr double voxel_downsampling_rate = 64.0;
static constexpr double max_correspondence_distance = 1e16;
// fixed parameters for BSHOT
static constexpr double normal_radius = 0.2;
static constexpr double voxel_grid_size = 1.5;
static constexpr double shot_radius = 0.15;
}  // namespace

namespace map_closures {
MapClosures::MapClosures() : config_(Config()) {
    orb_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
    bshot_extractor_ =
        std::make_shared<BSHOT::bshot_extractor>(normal_radius, voxel_grid_size, shot_radius);
}

MapClosures::MapClosures(const Config &config) : config_(config) {
    orb_extractor_ =
        cv::ORB::create(nfeatures, scale_factor, n_levels, edge_threshold, first_level, WTA_K,
                        cv::ORB::ScoreType(score_type), patch_size, fast_threshold);
    bshot_extractor_ =
        std::make_shared<BSHOT::bshot_extractor>(normal_radius, voxel_grid_size, shot_radius);
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
    for (const auto &index : indices) {
        const Tree3D::MatchVector &matches = descriptor_matches_3d_.at(index);
        std::cout << "Matches size: " << matches.size() << " at index " << index << std::endl;
    }
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

double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud) {
    double resolution = 0.0;
    int numberOfPoints = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> squaredDistances(2);
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i) {
        if (!std::isfinite((*cloud)[i].x)) continue;

        // Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
        if (nres == 2) {
            resolution += sqrt(squaredDistances[1]);
            ++numberOfPoints;
        }
    }
    if (numberOfPoints != 0) resolution /= numberOfPoints;

    return resolution;
}

void voxelize(pcl::PointCloud<pcl::PointXYZ>::Ptr pc_src,
              pcl::PointCloud<pcl::PointXYZ>::Ptr pc_dst,
              double var_voxel_size) {
    static pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(pc_src);
    voxel_filter.setLeafSize(var_voxel_size, var_voxel_size, var_voxel_size);
    voxel_filter.filter(*pc_dst);
}

ClosureCandidate3D MapClosures::ValidateClosure3D(const int reference_id,
                                                  const int query_id) const {
    throw std::runtime_error("Not implemented yet");

    // Get the point clouds from the local maps
    const auto &reference_map = local_maps_.at(reference_id);
    const auto &query_map = local_maps_.at(query_id);

    // const auto start = std::chrono::high_resolution_clock::now();
    //  Create the point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr reference_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr query_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &point : reference_map) {
        reference_cloud->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    for (const auto &point : query_map) {
        query_cloud->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    std::vector<int> indices;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

    // Remove NaN points
    pcl::removeNaNFromPointCloud(*reference_cloud, *reference_cloud, indices);
    pcl::removeNaNFromPointCloud(*query_cloud, *query_cloud, indices);

    double reference_resolution = computeCloudResolution(reference_cloud);
    double query_resolution = computeCloudResolution(query_cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr reference_voxels(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr query_voxels(new pcl::PointCloud<pcl::PointXYZ>);
    voxelize(reference_cloud, reference_voxels, reference_resolution * voxel_downsampling_rate);
    voxelize(query_cloud, query_voxels, query_resolution * voxel_downsampling_rate);

    reference_resolution = computeCloudResolution(reference_voxels);
    query_resolution = computeCloudResolution(query_voxels);

    /*
    std::cout << "Reference resolution: " << reference_resolution << std::endl;
    std::cout << "Query resolution: " << query_resolution << std::endl;

    std::cout << "Reference voxels size: " << reference_voxels->size() << std::endl;
    std::cout << "Query voxels size: " << query_voxels->size() << std::endl;
    */

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    pcl::PointCloud<pcl::Normal>::Ptr reference_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr query_normals(new pcl::PointCloud<pcl::Normal>);
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setRadiusSearch(reference_resolution * 2);
    normal_estimator.setInputCloud(reference_voxels);
    normal_estimator.compute(*reference_normals);
    normal_estimator.setRadiusSearch(query_resolution * 2);
    normal_estimator.setInputCloud(query_voxels);
    normal_estimator.compute(*query_normals);

    // Remove NaN normals
    std::vector<int> reference_indices;
    pcl::removeNaNNormalsFromPointCloud(*reference_normals, *reference_normals, reference_indices);
    std::vector<int> query_indices;
    pcl::removeNaNNormalsFromPointCloud(*query_normals, *query_normals, query_indices);

    // Remove voxels with NaN normals
    pcl::PointCloud<pcl::PointXYZ>::Ptr reference_voxels_no_nan(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr query_voxels_no_nan(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < reference_indices.size(); i++) {
        reference_voxels_no_nan->push_back(reference_voxels->at(reference_indices[i]));
    }
    for (size_t i = 0; i < query_indices.size(); i++) {
        query_voxels_no_nan->push_back(query_voxels->at(query_indices[i]));
    }
    reference_voxels = reference_voxels_no_nan;
    query_voxels = query_voxels_no_nan;

    /*
    for (size_t i = 0; i < reference_voxels->size(); i++) {
        std::cout << "Reference voxel " << i << ": " << reference_voxels->at(i) << std::endl;
    }
    for (size_t i = 0; i < query_voxels->size(); i++) {
        std::cout << "Query voxel " << i << ": " << query_voxels->at(i) << std::endl;
    }
    for (size_t i = 0; i < reference_normals->size(); i++) {
        std::cout << "Reference normal " << i << ": " << reference_normals->at(i) << std::endl;
    }
    for (size_t i = 0; i < query_normals->size(); i++) {
        std::cout << "Query normal " << i << ": " << query_normals->at(i) << std::endl;
    }
    */

    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;
    pcl::PointCloud<pcl::PointXYZ>::Ptr reference_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr query_keypoints(new pcl::PointCloud<pcl::PointXYZ>);
    iss_detector.setThreshold21(0.975);
    iss_detector.setThreshold32(0.975);
    iss_detector.setMinNeighbors(5);
    iss_detector.setNumberOfThreads(4);
    iss_detector.setSalientRadius(6 * reference_resolution);
    iss_detector.setNonMaxRadius(4 * reference_resolution);
    iss_detector.setInputCloud(reference_voxels);
    iss_detector.compute(*reference_keypoints);
    iss_detector.setSalientRadius(6 * query_resolution);
    iss_detector.setNonMaxRadius(4 * query_resolution);
    iss_detector.setInputCloud(query_voxels);
    iss_detector.compute(*query_keypoints);

    /*
    for (size_t i = 0; i < reference_keypoints->size(); i++) {
        std::cout << "Reference keypoint " << i << ": " << reference_keypoints->at(i) <<
    std::endl;
    }
    for (size_t i = 0; i < query_keypoints->size(); i++) {
        std::cout << "Query keypoint " << i << ": " << query_keypoints->at(i) << std::endl;
    }
    std::cout << "Reference keypoints size: " << reference_keypoints->size() << std::endl;
    std::cout << "Query keypoints size: " << query_keypoints->size() << std::endl;
    */

    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimator;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr reference_features(
        new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr query_features(
        new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh_estimator.setSearchMethod(tree);
    fpfh_estimator.setRadiusSearch(reference_resolution * 5);
    fpfh_estimator.setInputCloud(reference_keypoints);
    fpfh_estimator.setInputNormals(reference_normals);
    fpfh_estimator.setSearchSurface(reference_voxels);
    fpfh_estimator.compute(*reference_features);
    fpfh_estimator.setRadiusSearch(query_resolution * 5);
    fpfh_estimator.setInputCloud(query_keypoints);
    fpfh_estimator.setInputNormals(query_normals);
    fpfh_estimator.setSearchSurface(query_voxels);
    fpfh_estimator.compute(*query_features);

    for (size_t i = 0; i < reference_features->size(); i++) {
        // std::cout << "Reference feature " << i << ": " << reference_features->at(i) << std::endl;
        if (!pcl::isFinite(reference_features->at(i))) {
            std::cerr << "Reference feature " << i << " is not finite" << std::endl;
        }
    }
    for (size_t i = 0; i < query_features->size(); i++) {
        // std::cout << "Query feature " << i << ": " << query_features->at(i) << std::endl;
        if (!pcl::isFinite(query_features->at(i))) {
            std::cerr << "Query feature " << i << " is not finite" << std::endl;
        }
    }
    // std::cout << "Reference features size: " << reference_features->size() << std::endl;
    // std::cout << "Query features size: " << query_features->size() << std::endl;

    pcl::KdTreeFLANN<pcl::FPFHSignature33> kdtree;
    kdtree.setInputCloud(reference_features);
    pcl::FPFHSignature33 query_feature;
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    for (size_t i = 0; i < query_features->size(); i++) {
        query_feature = query_features->at(i);
        // Validate the feature
        if (!pcl::isFinite(query_feature)) {
            std::cerr << "Query feature " << i << " is not finite" << std::endl;
            continue;
        }
        std::vector<int> indices;
        std::vector<float> squared_distances;
        // std::cout << "Feature " << i << " of " << query_features->size() << std::endl;
        // std::cout << "Feature data " << query_features->at(i) << std::endl;
        if (kdtree.nearestKSearch(query_feature, 1, indices, squared_distances) == 1) {
            if (!indices.empty() && !squared_distances.empty() &&
                squared_distances[0] < max_correspondence_distance) {
                pcl::Correspondence correspondence(static_cast<int>(i), indices[0],
                                                   squared_distances[0]);
                correspondences->push_back(correspondence);
            }
        }
    }

    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "Create correspondences: " << std::chrono::duration<double, std::milli>(end
    // - start).count() << " ms" << std::endl; std::cout << "Correspondences size: " <<
    // correspondences->size() << std::endl;

    Eigen::Isometry3d pose3d;
    int number_of_inliers;
    std::vector<PointPair3D> inliers;

    // Generate random inliers for testing
    const auto start = std::chrono::high_resolution_clock::now();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, correspondences->size() - 1);
    for (size_t i = 0; i < 2; i++) {
        const int index = dis(gen);
        const auto &correspondence = correspondences->at(index);
        const auto &ref_point = reference_keypoints->at(correspondence.index_match);
        const auto &query_point = query_keypoints->at(correspondence.index_query);
        inliers.push_back(
            PointPair3D(Eigen::Vector3d(ref_point.x, ref_point.y, ref_point.z),
                        Eigen::Vector3d(query_point.x, query_point.y, query_point.z)));
    }
    std::tie(pose3d, number_of_inliers) = std::make_tuple(Eigen::Isometry3d::Identity(), 2);
    const auto end = std::chrono::high_resolution_clock::now();

    // std::tie(pose3d, number_of_inliers, inliers) = CliRegAlignment3D(reference_keypoints,
    // query_keypoints, correspondences);

    // Create the closure candidate
    ClosureCandidate3D closure;
    closure.source_id = reference_id;
    closure.target_id = query_id;
    closure.pose = pose3d.matrix();
    closure.number_of_inliers = number_of_inliers;
    closure.keypoint_pairs.reserve(correspondences->size());
    closure.inliers.reserve(inliers.size());
    closure.alignment_time = std::chrono::duration<double, std::milli>(end - start).count();
    for (const auto &correspondence : *correspondences) {
        const auto &ref_point = reference_keypoints->at(correspondence.index_match);
        const auto &query_point = query_keypoints->at(correspondence.index_query);
        closure.keypoint_pairs.push_back(
            std::make_pair(Eigen::Vector3d(ref_point.x, ref_point.y, ref_point.z),
                           Eigen::Vector3d(query_point.x, query_point.y, query_point.z)));
    }
    for (const auto &inlier : inliers) {
        closure.inliers.push_back(std::make_pair(inlier.ref, inlier.query));
    }

    return closure;

    /*
        // Compute the transformation
        pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>
        transformation_estimator; Eigen::Matrix4f transformation;
        transformation_estimator.estimateRigidTransformation(*reference_cloud, *query_cloud,
        *correspondences, transformation);

        // Compute the inliers
        pcl::PointCloud<pcl::PointXYZ>::Ptr reference_cloud_transformed(new
       pcl::PointCloud<pcl::PointXYZ>); pcl::transformPointCloud(*reference_cloud,
       *reference_cloud_transformed, transformation); pcl::CorrespondencesPtr inliers(new
       pcl::Correspondences); for (size_t i = 0; i < correspondences->size(); i++) { const
       pcl::PointXYZ &reference_point =
       reference_cloud_transformed->at(correspondences->at(i).index_query); const pcl::PointXYZ
       &query_point = query_cloud->at(correspondences->at(i).index_match); if
       ((reference_point.getVector3fMap() - query_point.getVector3fMap()).norm() < 0.1) {
                inliers->push_back(correspondences->at(i));
            }
        }
    */
}
}  // namespace map_closures
