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

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_nan;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints;
    pcl::PointCloud<pcl::SHOT352>::Ptr descriptors;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimator;
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shotEstimator;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
    pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;

public:
    typedef std::shared_ptr<bshot_extractor> Ptr;

    bshot_extractor(double voxel_grid_size) : voxel_grid_size(voxel_grid_size) {
        cloud_no_nan = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
        keypoints = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        descriptors = pcl::PointCloud<pcl::SHOT352>::Ptr(new pcl::PointCloud<pcl::SHOT352>);
        tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void detectAndCompute(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          std::vector<BSHOTSignature352> &bshot_descriptors,
                          std::vector<pcl::PointXYZ> &keyPoints) {
        // Voxelize the input cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        voxelGrid.setInputCloud(cloud);
        voxelGrid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
        voxelGrid.filter(*voxel_cloud);

        // Compute the ISS keypoints
        pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> issDetector;
        issDetector.setInputCloud(voxel_cloud);
        issDetector.setSalientRadius(6 * voxel_grid_size);
        issDetector.setNonMaxRadius(4 * voxel_grid_size);
        issDetector.setThreshold21(0.975);
        issDetector.setThreshold32(0.975);
        issDetector.setMinNeighbors(5);
        issDetector.setNumberOfThreads(12);
        issDetector.compute(*keypoints);

        // Calculate the normals
        normalEstimator.setInputCloud(voxel_cloud);
        normalEstimator.setRadiusSearch(2 * voxel_grid_size);
        normalEstimator.setNumberOfThreads(12);
        normalEstimator.setSearchMethod(tree);
        normalEstimator.compute(*normals);

        // Remove NaN normals
        std::vector<int> indices;
        pcl::removeNaNNormalsFromPointCloud(*normals, *normals, indices);

        // Remove the points with NaN normals
        cloud_no_nan->clear();
        for (size_t i = 0; i < indices.size(); i++) {
            cloud_no_nan->push_back(voxel_cloud->points[indices[i]]);
        }

        // Calculate the SHOT descriptors
        shotEstimator.setInputCloud(keypoints);
        shotEstimator.setSearchSurface(cloud_no_nan);
        shotEstimator.setInputNormals(normals);
        shotEstimator.setRadiusSearch(4 * voxel_grid_size);
        shotEstimator.setSearchMethod(tree);
        shotEstimator.setNumberOfThreads(12);
        shotEstimator.compute(*descriptors);

        // Convert the SHOT descriptors to BSHOT descriptors
        bshot_descriptors.clear();
        bshot_descriptors.reserve(descriptors->size());
        for (const auto &shot : *descriptors) {
            bshot_descriptors.push_back(BSHOTSignature352(shot));
        }

        // Convert the keypoints to pcl::PointXYZ
        keyPoints.clear();
        keyPoints.reserve(keypoints->size());
        for (const auto &keypoint : *keypoints) {
            keyPoints.push_back(keypoint);
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
