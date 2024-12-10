#pragma once

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <bitset>
#include <cassert>

namespace BSHOT {

class BSHOTSignature352 {
public:
    std::bitset<352> bits;

    BSHOTSignature352() { bits.reset(); }

    BSHOTSignature352(const pcl::SHOT352 &shot) {
        bits.reset();

        for (int i = 0; i < 88; i++) {
            float vec[4] = {shot.descriptor[i * 4 + 0], shot.descriptor[i * 4 + 1],
                            shot.descriptor[i * 4 + 2], shot.descriptor[i * 4 + 3]};

            float sum = vec[0] + vec[1] + vec[2] + vec[3];
            uint8_t bit = 0;

            if (sum > 0) {  // Skip unnecessary calculations if the vector is zero
                float threshold = 0.9f * sum;

                // Case B: A single element exceeds 90% of the sum
                for (int j = 0; j < 4; j++) {
                    if (vec[j] > threshold) {
                        bit = (1 << j);
                        break;
                    }
                }

                // Case C: Sum of pairs exceeds 90% of the sum
                if (bit == 0) {
                    for (int j = 0; j < 4; j++) {
                        for (int k = j + 1; k < 4; k++) {
                            if ((vec[j] + vec[k]) > threshold) {
                                bit = (1 << j) | (1 << k);
                                break;
                            }
                        }
                        if (bit != 0) break;
                    }
                }

                // Case D: Sum of three elements exceeds 90% of the sum
                if (bit == 0) {
                    for (int j = 0; j < 4; j++) {
                        for (int k = j + 1; k < 4; k++) {
                            for (int l = k + 1; l < 4; l++) {
                                if ((vec[j] + vec[k] + vec[l]) > threshold) {
                                    bit = (1 << j) | (1 << k) | (1 << l);
                                    break;
                                }
                            }
                            if (bit != 0) break;
                        }
                        if (bit != 0) break;
                    }
                }

                // Case E: All bits are set
                if (bit == 0) bit = 0xF;  // 0xF is 1111 in binary
            }

            // Set the bits directly using bitwise operations
            bits[i * 4 + 0] = (bit & 0x1);
            bits[i * 4 + 1] = (bit & 0x2) >> 1;
            bits[i * 4 + 2] = (bit & 0x4) >> 2;
            bits[i * 4 + 3] = (bit & 0x8) >> 3;
        }
    }

    BSHOTSignature352(const std::bitset<352> &bits) : bits(bits) {}

    BSHOTSignature352(const BSHOTSignature352 &bshot) : bits(bshot.bits) {}

    friend std::ostream &operator<<(std::ostream &os, const BSHOTSignature352 &bshot) {
        os << bshot.bits;
        return os;
    }
};  // class BSHOTSignature352

class bshot_extractor {
    double voxel_grid_size;

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud;
    pcl::PointCloud<pcl::Normal>::Ptr shot_normals;
    pcl::PointCloud<pcl::PointXYZ>::Ptr shot_keypoints;
    pcl::PointCloud<pcl::SHOT352>::Ptr shot_descriptors;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimator;
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shotEstimator;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
    pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;

public:
    typedef std::shared_ptr<bshot_extractor> Ptr;

    bshot_extractor(double voxel_grid_size) : voxel_grid_size(voxel_grid_size) {
        downsampled_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        shot_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
        shot_keypoints = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        shot_descriptors = pcl::PointCloud<pcl::SHOT352>::Ptr(new pcl::PointCloud<pcl::SHOT352>);
        tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void detectAndCompute(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          std::vector<BSHOTSignature352> &bshot_descriptors,
                          std::vector<pcl::PointXYZ> &keypoints) {
        // Downsample the cloud
        voxelGrid.setInputCloud(cloud);
        voxelGrid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
        voxelGrid.filter(*downsampled_cloud);

        double resolution = computeCloudResolution(downsampled_cloud);

        // Calculate the normals
        normalEstimator.setInputCloud(downsampled_cloud);
        normalEstimator.setRadiusSearch(resolution * 2);
        normalEstimator.setNumberOfThreads(12);
        normalEstimator.setSearchMethod(tree);
        normalEstimator.compute(*shot_normals);

        // Remove NaN normals
        std::vector<int> indices;
        pcl::removeNaNNormalsFromPointCloud(*shot_normals, *shot_normals, indices);

        // Remove the points with NaN normals
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud_no_nan(
            new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 0; i < indices.size(); i++) {
            downsampled_cloud_no_nan->push_back(downsampled_cloud->points[indices[i]]);
        }

        // Calculate the keypoints (ISS)
        pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> issEstimator;
        issEstimator.setInputCloud(downsampled_cloud_no_nan);
        issEstimator.setSearchMethod(tree);
        issEstimator.setSalientRadius(6 * resolution);
        issEstimator.setNonMaxRadius(4 * resolution);
        issEstimator.setThreshold21(0.975);
        issEstimator.setThreshold32(0.975);
        issEstimator.setMinNeighbors(5);
        issEstimator.setNumberOfThreads(12);
        issEstimator.compute(*shot_keypoints);

        // Calculate the SHOT descriptors
        shotEstimator.setInputCloud(shot_keypoints);
        shotEstimator.setSearchSurface(downsampled_cloud_no_nan);
        shotEstimator.setInputNormals(shot_normals);
        shotEstimator.setRadiusSearch(resolution * 4);
        shotEstimator.setSearchMethod(tree);
        shotEstimator.setNumberOfThreads(12);
        shotEstimator.compute(*shot_descriptors);

        // Convert the SHOT descriptors to BSHOT descriptors
        bshot_descriptors.clear();
        bshot_descriptors.reserve(shot_descriptors->size());
        for (const auto &shot : *shot_descriptors) {
            bshot_descriptors.push_back(BSHOTSignature352(shot));
        }

        // Convert the keypoints to pcl::PointXYZ
        keypoints.clear();
        keypoints.reserve(shot_keypoints->size());
        for (const auto &keypoint : *shot_keypoints) {
            keypoints.push_back(keypoint);
        }

        assert(bshot_descriptors.size() == keypoints.size());
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

};  // class bshot_extractor

}  // namespace BSHOT
