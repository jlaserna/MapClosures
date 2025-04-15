# MIT License
#
# Copyright (c) 2023 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
# Cyrill Stachniss.
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
import datetime
import os
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm.auto import trange

from map_closures.config import load_config, write_config
from map_closures.map_closures import MapClosures
from map_closures.tools.evaluation import EvaluationPipeline, LocalMap, StubEvaluation
from map_closures.tools.gt_closures import generate_gt_closures
from map_closures.visualizer.visualizer import StubVisualizer, Visualizer


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


def pose_inv(T):
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, -1] = -T[:3, :3].T @ T[:3, -1]
    return T_inv


class MapClosurePipeline:
    def __init__(
        self,
        dataset,
        config_path: Path,
        results_dir: Path,
        eval: Optional[bool] = False,
        vis: Optional[bool] = False,
    ):
        self._dataset = dataset
        self._dataset_name = (
            self._dataset.sequence_id
            if hasattr(self._dataset, "sequence_id")
            else os.path.basename(self._dataset.data_dir)
        )
        self._vis = vis
        self._eval = eval
        self._results_dir = results_dir
        self._n_scans = len(self._dataset)

        self.closure_config = load_config(config_path)
        self.map_closures = MapClosures(self.closure_config)

        self.closures = []
        self.local_maps = []
        self.odom_poses = self._dataset.kiss_poses

        self.closure_overlap_threshold = 0.5
        self.gt_closures = (
            generate_gt_closures(self._dataset)
            if (self._eval and hasattr(self._dataset, "gt_poses"))
            else None
        )

        self.closure_distance_threshold = 25.0
        self.results = (
            EvaluationPipeline(
                self.gt_closures,
                self._dataset_name,
                self.closure_distance_threshold,
                self.odom_poses,
            )
            if self._eval
            else StubEvaluation()
        )

        self.visualizer = Visualizer(self.odom_poses) if self._vis else StubVisualizer()

    def run(self):
        self._run_pipeline()
        self.results.compute_metrics()
        self._save_config()
        self._log_to_file()
        self._log_to_console()

        return self.results

    def _run_pipeline(self):
        map_ref_pose = np.eye(4)

        for map_id in trange(
            0,
            self._n_scans,
            ncols=8,
            unit=" frames",
            dynamic_ncols=True,
            desc="Processing for Loop Closures",
        ):
            local_map = self._dataset[map_id]
            local_map_start_scan_range = self._dataset.local_map_scan_index_ranges[map_id]

            closures = self.map_closures.match_and_add(map_id, local_map)

            self.local_maps.append(
                LocalMap(
                    local_map,
                    np.copy(local_map_start_scan_range),
                )
            )
            self.visualizer.update_data(
                self.local_maps[-1].pointcloud,
                map_ref_pose,
            )

            for closure in closures:
                if closure.number_of_inliers > self.closure_config.inliers_threshold:
                    reference_local_map = self.local_maps[closure.source_id]
                    query_local_map = self.local_maps[closure.target_id]
                    self.closures.append(
                        np.r_[
                            closure.source_id,
                            closure.target_id,
                            reference_local_map.scan_indices[0],
                            query_local_map.scan_indices[0],
                            closure.pose.flatten(),
                            closure.number_of_inliers,
                            closure.alignment_time,
                        ]
                    )

                    self.results.append(
                        np.arange(*reference_local_map.scan_indices),
                        np.arange(*query_local_map.scan_indices),
                        closure.pose,
                        closure.number_of_inliers,
                    )

                    self.visualizer.update_closures(
                        np.asarray(closure.pose),
                        [closure.source_id, closure.target_id],
                        closure.alignment_time,
                    )
        self.visualizer.pause_vis()
        
    def _log_to_file(self):
        np.savetxt(os.path.join(self._results_dir, "map_closures.txt"), np.asarray(self.closures))
        np.savetxt(
            os.path.join(self._results_dir, "kiss_poses_kitti.txt"),
            np.asarray(self.odom_poses)[:, :3].reshape(-1, 12),
        )
        self.results.log_to_file(self._results_dir)

    def _log_to_console(self):
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(box=box.HORIZONTALS)
        table.caption = f"Loop Closures Detected Between Local Maps\n"
        table.add_column("# MapClosure", justify="left", style="cyan")
        table.add_column("Ref Map Index", justify="left", style="magenta")
        table.add_column("Query Map Index", justify="left", style="magenta")
        table.add_column("Inliers", justify="right", style="green")
        table.add_column("Alignment Time", justify="right", style="green")

        for i, closure in enumerate(self.closures):
            table.add_row(
                f"{i+1}",
                f"{int(closure[0])}",
                f"{int(closure[1])}",
                f"{int(closure[20])}",
                f"{closure[21]:.4f} ms",
            )
        console.print(table)

    def _save_config(self):
        self._results_dir = self._create_results_dir()
        write_config(self.closure_config, os.path.join(self._results_dir, "config"))

    def _create_results_dir(self) -> Path:
        def get_timestamp() -> str:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_dir = os.path.join(
            self._results_dir,
            f"{self._dataset_name}_results",
            get_timestamp(),
        )
        latest_dir = os.path.join(self._results_dir, f"{self._dataset_name}_results", "latest")

        os.makedirs(results_dir, exist_ok=True)
        os.unlink(latest_dir) if os.path.exists(latest_dir) or os.path.islink(latest_dir) else None
        os.symlink(results_dir, latest_dir)

        return results_dir
