#pragma once

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/voxel_grid.h>
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
            float vec[4] = {0};
            for (int j = 0; j < 4; j++) {
                vec[j] = shot.descriptor[i * 4 + j];
            }

            std::bitset<4> bit;
            bit.reset();

            float sum = vec[0] + vec[1] + vec[2] + vec[3];

            if (vec[0] == 0 and vec[1] == 0 and vec[2] == 0 and vec[3] == 0) {
                // bin[0] = bin[1] = bin[2] = bin[3] = 0;
            } else if (vec[0] > (0.9 * (sum))) {
                bit.set(0);
            } else if (vec[1] > (0.9 * (sum))) {
                bit.set(1);
            } else if (vec[2] > (0.9 * (sum))) {
                bit.set(2);
            } else if (vec[3] > (0.9 * (sum))) {
                bit.set(3);
            } else if ((vec[0] + vec[1]) > (0.9 * (sum))) {
                bit.set(0);
                bit.set(1);
            } else if ((vec[1] + vec[2]) > (0.9 * (sum))) {
                bit.set(1);
                bit.set(2);
            }

            else if ((vec[2] + vec[3]) > (0.9 * (sum))) {
                bit.set(2);
                bit.set(3);
            } else if ((vec[0] + vec[3]) > (0.9 * (sum))) {
                bit.set(0);
                bit.set(3);
            } else if ((vec[1] + vec[3]) > (0.9 * (sum))) {
                bit.set(1);
                bit.set(3);
            } else if ((vec[0] + vec[2]) > (0.9 * (sum))) {
                bit.set(0);
                bit.set(2);
            } else if ((vec[0] + vec[1] + vec[2]) > (0.9 * (sum))) {
                bit.set(0);
                bit.set(1);
                bit.set(2);
            } else if ((vec[1] + vec[2] + vec[3]) > (0.9 * (sum))) {
                bit.set(1);
                bit.set(2);
                bit.set(3);
            } else if ((vec[0] + vec[2] + vec[3]) > (0.9 * (sum))) {
                bit.set(0);
                bit.set(2);
                bit.set(3);
            } else if ((vec[0] + vec[1] + vec[3]) > (0.9 * (sum))) {
                bit.set(0);
                bit.set(1);
                bit.set(3);
            } else {
                bit.set(0);
                bit.set(1);
                bit.set(2);
                bit.set(3);
            }

            if (bit.test(0)) bits.set(i * 4);

            if (bit.test(1)) bits.set(i * 4 + 1);

            if (bit.test(2)) bits.set(i * 4 + 2);

            if (bit.test(3)) bits.set(i * 4 + 3);
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
    double normal_radius;
    double voxel_grid_size;
    double shot_radius;

    pcl::PointCloud<pcl::Normal>::Ptr shot_normals;
    pcl::PointCloud<pcl::PointXYZ>::Ptr shot_keypoints;
    pcl::PointCloud<pcl::SHOT352>::Ptr shot_descriptors;

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normalEstimator;
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shotEstimator;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
    pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;

public:
    typedef std::shared_ptr<bshot_extractor> Ptr;

    bshot_extractor(double normal_radius, double voxel_grid_size, double shot_radius)
        : normal_radius(normal_radius), voxel_grid_size(voxel_grid_size), shot_radius(shot_radius) {
        shot_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>);
        shot_keypoints = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        shot_descriptors = pcl::PointCloud<pcl::SHOT352>::Ptr(new pcl::PointCloud<pcl::SHOT352>);
        tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void detectAndCompute(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          std::vector<BSHOTSignature352> &bshot_descriptors,
                          std::vector<pcl::PointXYZ> &keypoints) {
        // Calculate the keypoints
        voxelGrid.setInputCloud(cloud);
        voxelGrid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
        voxelGrid.filter(*shot_keypoints);

        double resolution = computeCloudResolution(shot_keypoints);

        // Calculate the normals
        normalEstimator.setInputCloud(shot_keypoints);
        normalEstimator.setRadiusSearch(resolution * 2);
        normalEstimator.setNumberOfThreads(12);
        normalEstimator.setSearchMethod(tree);
        normalEstimator.compute(*shot_normals);

        // Remove NaN normals
        std::vector<int> indices;
        pcl::removeNaNNormalsFromPointCloud(*shot_normals, *shot_normals, indices);

        // Remove the points with NaN normals
        pcl::PointCloud<pcl::PointXYZ>::Ptr shot_keypoints_no_nan(
            new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 0; i < indices.size(); i++) {
            shot_keypoints_no_nan->push_back(shot_keypoints->at(indices[i]));
        }

        // Calculate the SHOT descriptors
        shotEstimator.setInputCloud(shot_keypoints_no_nan);
        // shotEstimator.setSearchSurface(shot_keypoints);
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
        keypoints.reserve(shot_keypoints_no_nan->size());
        for (const auto &keypoint : *shot_keypoints_no_nan) {
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
