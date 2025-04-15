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

#include "AlignCliReg.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "copt/runs/run_CLQ_manager.h"

enum ret_t { ERR = -1, OPT, UNKNOWN };
ret_t cliqueGraphRun(ugraph graph,
                     int alg,
                     int ord,
                     int AMTS,
                     double TIME_LIMIT,
                     double TIME_LIMIT_HEUR,
                     vint &vertices,
                     string filename,
                     bool verbose);

namespace {
    Eigen::Isometry3d KabschUmeyamaAlignment(
        const std::vector<map_closures::PointPair> &keypoint_pairs) {
        auto mean = std::reduce(keypoint_pairs.cbegin(), keypoint_pairs.cend(),
                                map_closures::PointPair(), [](auto lhs, const auto &rhs) {
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
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        const Eigen::Matrix3d &R = svd.matrixV() * svd.matrixU().transpose();
        T.linear() = R.determinant() > 0 ? R : -R;
        T.translation() = mean.query - R * mean.ref;
    
        return T;
    }
    
static constexpr bool verbose = false;
}  // namespace

namespace map_closures {
std::tuple<Eigen::Isometry3d, int, std::vector<PointPair>> CliRegAlignment(
    const std::vector<PointPair> &keypoint_pairs, const double inliers3d_distance_threshold) {
    const size_t max_inliers = keypoint_pairs.size();

    ugraph graph;

    graph.init(keypoint_pairs.size());

    for (int i = 0; i < keypoint_pairs.size(); i++) {
        for (int j = i + 1; j < keypoint_pairs.size(); j++) {
            float ref_dist = (keypoint_pairs[i].ref - keypoint_pairs[j].ref).norm();
            float query_dist = (keypoint_pairs[i].query - keypoint_pairs[j].query).norm();
            if (std::abs(ref_dist - query_dist) < inliers3d_distance_threshold) {
                graph.add_edge(i, j);
            }
        }
    }

    vint vertices;

    // Check if the graph is empty
    if (graph.number_of_vertices() == 0) {
        return {Eigen::Isometry3d::Identity(), 0, {}};
    }

    auto ret_status = cliqueGraphRun(graph, 2, 0, 0, 1, 1, vertices, "", verbose);

    if (ret_status == ERR) {
        return {Eigen::Isometry3d::Identity(), 0, {}};
    }

    std::vector<PointPair> inliers(vertices.size());
    std::transform(vertices.cbegin(), vertices.cend(), inliers.begin(),
                   [&](const auto index) { return keypoint_pairs[index]; });

    auto T = KabschUmeyamaAlignment(inliers);

    return {T, vertices.size(), inliers};
}

}  // namespace map_closures

ret_t cliqueGraphRun(ugraph graph,
                     int alg,
                     int ord,
                     int AMTS,
                     double TIME_LIMIT,
                     double TIME_LIMIT_HEUR,
                     vint &vertices,
                     string filename,
                     bool verbose) {
    infoCLQ info;
    ret_t status = ret_t::UNKNOWN;

    info.clear();
    stringstream sstr;

    ////
    // Parse Data
    ////
    info.name = graph.get_name();
    info.N = graph.number_of_vertices();
    info.M = graph.number_of_edges();
    info.TIME_LIMIT = TIME_LIMIT;
    info.TIME_LIMIT_HEUR = TIME_LIMIT_HEUR;
    info.id_AMTS = AMTS;
    info.id_sorting_alg_called = ord;
    info.id_alg = alg;

    ////
    // Print Data
    ////
    if (verbose) {
        sstr = std::stringstream();
        sstr << "starting the clique solver...runCLQ::clique()" << endl;
        info.print_params(sstr);
        LOG_INFO(sstr.str());
    }

    ////
    // Set Parameters
    ////
    clqo::param_t param;
    param.tout = info.TIME_LIMIT;
    param.tout_heur = all_info.TIME_LIMIT_HEUR;

    // Sorting strategy
    if (info.id_sorting_alg_called == 0) {
        param.init_order = clqo::NONE;
    } else if (info.id_sorting_alg_called == 1) {
        param.init_order = clqo::MIN_WIDTH;
    } else if (info.id_sorting_alg_called == 2) {
        param.init_order = clqo::RLF;
    } else if (info.id_sorting_alg_called == 3) {
        param.init_order = clqo::RLF_COND;
    } else if (info.id_sorting_alg_called == 4) {
        param.init_order = clqo::MAX_WIDTH;
    } else {
        LOG_ERROR("unknown ordering strategy, please select 1-RLF, 2-DEG or 3-RLF_COND...exiting");
        return ret_t::ERR;
    }

    // AMTS
    if (info.id_AMTS == 1) {
        param.init_preproc = clqo::UB_HEUR;
    } else if (info.id_AMTS == 0) {
        param.init_preproc = clqo::UB;
    } else {
        LOG_ERROR("unknown AMTS strategy, please select 1-ON or 0-OFF...exiting");
        return ret_t::ERR;
    }

    ////
    // Set Algorithm
    ////
    switch (info.id_alg) {
        case 1:
            param.alg = clqo::TEST1;  // TEST_ALL_SINGLE_VERTEX_ATTEMPT:	optimal for EVIL
                                      // (COLOR_SORT with exceptions, such as MANN)
            break;
        case 2:
            param.alg = clqo::
                TEST8;  // TEST_SELECTIVE_FIRST_FAIL_SINGLE_VERTEX_ATTEMPT_WITH_LAST_ISET_PREFILTER_NO_ISBOUND
            break;
        case 3:
            param.alg = clqo::
                TEST9;  // TEST_SELECTIVE_FIRST_FAIL_SINGLE_VERTEX_ATTEMPT_WITH_LAST_ISET_PREFILTER
            break;
        default:
            LOG_ERROR("unknown algorithm number, exiting....");
            return ret_t::ERR;
    }

    ////
    // Run Clique
    ////
    info.start_timer(infoCLQ::PARSE);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    CliqueXRD<ugraph, (WDIV(PMS_MAX_NUM_CONFLICTS) + 1) * WORD_SIZE /* for sparse should be high */>
        clq(&graph, param);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    info.read_timer(infoCLQ::PARSE);

    info.start_timer(infoCLQ::PREPROC);
    ////////////////////////////
    status = static_cast<ret_t>(clq.set_up());
    ////////////////////////////
    info.read_timer(infoCLQ::PREPROC);

    if (verbose) {
        LOG_INFO("SETUP FINISHED");
        LOG_INFO("RUNNING CLQ");
    }

    if (status == 0) {
        info.start_timer(infoCLQ::SEARCH);
        info.start_timer(infoCLQ::LAST_INCUMBENT);
        /////////////////////
        clq.run();
        /////////////////////
        if (!clq.tear_down()) {
            LOG_ERROR("bizarre solution... exiting");
            return ret_t::ERR;
        } else {
            if (!info.time_limit_reached) status = OPT;
        }
        info.read_timer(infoCLQ::SEARCH);
    } else {
        status = OPT;
        if (verbose) LOG_INFO("solved during preproc.....");
    }
    ////////////////////////////////////////

    Result &r = clq.get_result();
    info.nSteps = r.number_of_steps();
    info.optimum = r.get_upper_bound();
    info.sol = clq.decode_first_solution();

    ////
    // I/O
    ////
    if (verbose) {
        LOG_INFO("****************************");
        sstr = stringstream();
        LOG_INFO(info.name << "\tn:" << info.N << "\tm:" << info.M << "\tomega:" << info.optimum
                           << "\tts:" << info.time_search << "\ttpp:" << info.time_preproc
                           << "\tsteps:" << info.nSteps);
        com::stl::print_collection(info.sol, sstr);
        LOG_INFO(sstr.str());
        if (info.time_limit_reached) {
            LOG_INFO("TIME OUT");
        }
        LOG_INFO("****************************");
    }

    if (!filename.empty()) {
        ofstream f(filename, ofstream::app);
        if (!f) {
            LOG_ERROR("cannot open log file- runCLQ::clique(...)");
        }
        f << info;
        f.close();
    }

    vertices = info.sol;

    return status;
}