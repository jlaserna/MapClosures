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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <tuple>
#include <vector>

#include "map_closures/MapClosures.hpp"
#include "stl_vector_eigen.h"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace py = pybind11;
using namespace py::literals;

namespace map_closures {
Config GetConfigFromYAML(const py::dict &yaml_cfg) {
    Config cpp_config;
    cpp_config.density_threshold = yaml_cfg["density_threshold"].cast<float>();
    cpp_config.density_map_resolution = yaml_cfg["density_map_resolution"].cast<float>();
    cpp_config.hamming_distance_threshold = yaml_cfg["hamming_distance_threshold"].cast<int>();
    cpp_config.alignment_algorithm =
        AlignmentAlgorithm(yaml_cfg["alignment_algorithm"].cast<int>());
    return cpp_config;
}

PYBIND11_MODULE(map_closures_pybind, m) {
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    py::class_<ClosureCandidate2D> closure_candidate(m, "_ClosureCandidate2D");
    closure_candidate.def(py::init<>())
        .def_readwrite("source_id", &ClosureCandidate2D::source_id)
        .def_readwrite("target_id", &ClosureCandidate2D::target_id)
        .def_readwrite("pose", &ClosureCandidate2D::pose)
        .def_readwrite("number_of_inliers", &ClosureCandidate2D::number_of_inliers)
        .def_readwrite("keypoint_pairs", &ClosureCandidate2D::keypoint_pairs)
        .def_readwrite("inliers", &ClosureCandidate2D::inliers)
        .def_readwrite("alignment_time", &ClosureCandidate2D::alignment_time);

    py::class_<ClosureCandidate3D> closure_candidate_3d(m, "_ClosureCandidate3D");
    closure_candidate_3d.def(py::init<>())
        .def_readwrite("source_id", &ClosureCandidate3D::source_id)
        .def_readwrite("target_id", &ClosureCandidate3D::target_id)
        .def_readwrite("pose", &ClosureCandidate3D::pose)
        .def_readwrite("number_of_inliers", &ClosureCandidate3D::number_of_inliers)
        .def_readwrite("keypoint_pairs", &ClosureCandidate3D::keypoint_pairs)
        .def_readwrite("inliers", &ClosureCandidate3D::inliers)
        .def_readwrite("alignment_time", &ClosureCandidate3D::alignment_time);

    py::class_<MapClosures, std::shared_ptr<MapClosures>> map_closures(m, "_MapClosures", "");
    map_closures
        .def(py::init([](const py::dict &cfg) {
                 auto config = GetConfigFromYAML(cfg);
                 return std::make_shared<MapClosures>(config);
             }),
             "config"_a)
        .def("_getDensityMapFromId",
             [](MapClosures &self, const int &map_id) {
                 const auto &density_map = self.getDensityMapFromId(map_id);
                 Eigen::MatrixXf density_map_eigen;
                 cv::cv2eigen(density_map.grid, density_map_eigen);
                 return density_map_eigen;
             })
        .def("_MatchAndAdd", [](MapClosures &self, int id, const std::vector<Eigen::Vector3d> &local_map) {
            auto result = self.MatchAndAdd(id, local_map);
            return std::visit([](auto &&arg) -> py::object {
                using T = std::decay_t<decltype(arg)>;
                return py::cast(arg);
            }, result);
        }, "map_id"_a, "local_map"_a)
        .def("_MatchAndAdd2D", &MapClosures::MatchAndAdd2D, "map_id"_a, "local_map"_a)
        .def("_MatchAndAdd3D", &MapClosures::MatchAndAdd3D, "map_id"_a, "local_map"_a)
        .def("_ValidateClosure2D", &MapClosures::ValidateClosure2D, "reference_id"_a, "query_id"_a)
        .def("_ValidateClosure3D", &MapClosures::ValidateClosure3D, "reference_id"_a, "query_id"_a);
}
}  // namespace map_closures
