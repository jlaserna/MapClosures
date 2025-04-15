# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
# Ignacio Vizzo, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.linalg import inv, norm
from rich import box
from rich.console import Console
from rich.table import Table


@dataclass
class LocalMap:
    pointcloud: np.ndarray
    scan_indices: np.ndarray


class EvaluationMetrics:
    def __init__(self, true_positives, false_positives, false_negatives):
        self.tp = true_positives
        self.fp = false_positives
        self.fn = false_negatives

        try:
            self.precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            self.precision = np.nan

        try:
            self.recall = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            self.recall = np.nan

        try:
            self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        except ZeroDivisionError:
            self.F1 = np.nan

    def __call__(self):
        return np.r_[self.tp, self.fp, self.fn, self.precision, self.recall, self.F1]


class StubEvaluation(ABC):
    def __init__(self):
        pass

    def print(self):
        pass

    def append(self, *kwargs):
        pass

    def compute_metrics(self):
        pass

    def log_to_file(self, *kwargs):
        pass


class EvaluationPipeline(StubEvaluation):
    def __init__(
        self,
        gt_closures: np.ndarray,
        dataset_name: str,
        closure_distance_threshold: float,
        odom_poses: np.ndarray,
    ):
        self._dataset_name = dataset_name
        self._closure_distance_threshold = closure_distance_threshold

        self.odom_poses = odom_poses

        self.closure_indices_list: List[Set[Tuple]] = []
        self.closure_distances_list: List[List[float]] = []
        self.inliers_count_list: List = []

        self.metrics = np.zeros((16, len(np.arange(5, self._closure_distance_threshold + 1, 5)), 6))

        gt_closures = gt_closures if gt_closures.shape[1] == 2 else gt_closures.T
        self.gt_closures: Set[Tuple[int]] = set(map(lambda x: tuple(sorted(x)), gt_closures))

    def print(self):
        self._log_to_console()

    def _compute_closure_indices(
        self,
        ref_indices: np.ndarray,
        query_indices: np.ndarray,
        relative_tf: np.ndarray,
    ):
        # bring all poses to a common frame at the query map
        query_poses = (
            np.linalg.inv(self.odom_poses[query_indices[0]]) @ self.odom_poses[query_indices]
        )
        ref_poses = np.linalg.inv(self.odom_poses[ref_indices[0]]) @ self.odom_poses[ref_indices]
        query_locs = query_poses[:, :3, -1].squeeze()
        ref_locs = (relative_tf @ ref_poses)[:, :3, -1].squeeze()

        closure_indices = []
        closure_distances = []
        query_id_start = query_indices[0]
        ref_id_start = ref_indices[0]
        qq, rr = np.meshgrid(query_indices, ref_indices)
        distances = norm(query_locs[qq - query_id_start] - ref_locs[rr - ref_id_start], axis=2)
        ids = np.where(distances < self._closure_distance_threshold)
        for r_id, q_id, distance in zip(
            ids[0] + ref_id_start, ids[1] + query_id_start, distances[ids]
        ):
            closure_indices.append((r_id, q_id))
            closure_distances.append(distance)
        return np.asarray(closure_indices, int), np.asarray(closure_distances)

    def append(
        self, reference_map_scan_indices, query_map_scan_indices, relative_pose, inliers_count
    ):
        closure_indices, closure_distances = self._compute_closure_indices(
            reference_map_scan_indices, query_map_scan_indices, relative_pose
        )
        if len(closure_indices) > 0:
            self.closure_indices_list.append(closure_indices)
            self.closure_distances_list.append(closure_distances)
            self.inliers_count_list.append([inliers_count] * len(closure_indices))

    def compute_metrics(
        self,
    ):
        print("[INFO] Computing Loop Closure Evaluation Metrics")
        for i, inliers_threshold in enumerate(range(4, 20)):
            for j, distance_threshold in enumerate(
                np.arange(5, self._closure_distance_threshold + 1, 5)
            ):
                closures = set()
                for closure_indices, closure_distances, inliers_count in zip(
                    self.closure_indices_list, self.closure_distances_list, self.inliers_count_list
                ):
                    if inliers_count[0] >= inliers_threshold:
                        closures = closures.union(
                            set(
                                map(
                                    lambda x: tuple(x),
                                    closure_indices[
                                        np.where(closure_distances < distance_threshold)
                                    ],
                                )
                            )
                        )

            tp = len(self.gt_closures.intersection(closures))
            fp = len(closures) - tp
            fn = len(self.gt_closures) - tp
            self.metrics[i, j] = EvaluationMetrics(tp, fp, fn)()

    def _rich_table_pr(self, table_format: box.Box = box.HORIZONTALS) -> Table:
        table = Table(box=table_format)
        table.caption = f"Loop Closure Evaluation Metrics\n"
        table.add_column("Inliers \ Distance (m)", justify="center", style="cyan")
        for distance_threshold in np.arange(5, self._closure_distance_threshold + 1, 5):
            table.add_column(f"{distance_threshold}", justify="center", style="magenta")
        for i, row in enumerate(self.metrics):
            metrics = [f"{val[-3]:.3f}\n{val[-2]:.3f}\n{val[-1]:.3f}" for val in row]
            table.add_row(f"{i + 5}", *metrics)
        return table

    def _log_to_console(self):
        console = Console()
        console.print(self._rich_table_pr())

    def log_to_file(self, results_dir):
        with open(os.path.join(results_dir, "evaluation_metrics.txt"), "wt") as logfile:
            console = Console(file=logfile, width=100, force_jupyter=False)
            table = self._rich_table_pr(table_format=box.ASCII_DOUBLE_HEAD)
            console.print(table)
        np.savetxt(
            os.path.join(results_dir, "closure_indices.txt"),
            np.hstack(
                (
                    np.concatenate(self.closure_indices_list),
                    np.concatenate(self.closure_distances_list).reshape(-1, 1),
                    np.concatenate(self.inliers_count_list).reshape(-1, 1),
                )
            ),
        )
