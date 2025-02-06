# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino,
# Benedikt Mersch, Cyrill Stachniss.
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
import glob
import os
from pathlib import Path

import numpy as np
import open3d as o3d


class HeLiPRDataset:
    def __init__(self, data_dir: Path, sequence: str, *_, **__):
        self.sequence_id = f"{os.path.basename(data_dir)}_{sequence}"
        self.data_dir = os.path.realpath(data_dir)
        self.sequence_dir = os.path.join(self.data_dir, "LiDAR", sequence)
        self.local_maps_dir = os.path.join(self.sequence_dir, "MapClosures", "local_maps/")
        self.scan_files = sorted(glob.glob(self.local_maps_dir + "/*.ply"))

        self.kiss_poses = np.load(os.path.join(self.sequence_dir, "MapClosures", "kiss_poses.npy"))
        self.local_map_scan_index_ranges = np.load(
            os.path.join(self.sequence_dir, "MapClosures", "local_maps_scan_index_range.npy")
        )

        gt_poses_kitti = np.loadtxt(
            os.path.join(self.sequence_dir, "MapClosures", "gt_poses_kitti.txt")
        ).reshape(-1, 3, 4)
        self.gt_poses = np.tile(np.eye(4), (len(gt_poses_kitti), 1, 1))
        self.gt_poses[:, :3] = gt_poses_kitti

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        return self.read_point_cloud(idx)

    def read_point_cloud(self, idx: int):
        file_path = self.scan_files[idx]
        points = o3d.io.read_point_cloud(file_path).points
        return np.asarray(points, np.float64)