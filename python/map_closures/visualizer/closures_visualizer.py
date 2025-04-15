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
from dataclasses import dataclass, field

import numpy as np
import warnings

# Button names
PREV_CLOSURE = "Prev Closure [P]"
NEXT_CLOSURE = "Next Closure [N]"
ALIGN_CLOSURE = "Align [A]"

# Colors
SOURCE_COLOR = [0.8470, 0.1058, 0.3764]
TARGET_COLOR = [0.0, 0.3019, 0.2509]
TRAJECTORY_COLOR = [1.0, 0.0, 0.0]
INLIER_COLOR = [0.0, 1.0, 0.0]
OUTLIER_COLOR = [1.0, 0.0, 0.0]

# Size constants
SOURCE_PTS_SIZE = 0.06
TARGET_PTS_SIZE = 0.08
INLIER_PTS_SIZE = 0.3
OUTLIER_PTS_SIZE = 0.3
I = np.eye(4)


@dataclass
class LoopClosureData:
    size: int = 0
    current_id: int = 0
    closure_edges: list = field(default_factory=list)
    alignment_pose: list = field(default_factory=list)
    keypoints_pairs: list = field(default_factory=list)
    inliers: list = field(default_factory=list)
    alignment_time: list = field(default_factory=list)


@dataclass
class ClosuresStateMachine:
    toggle_view: bool = False
    global_view: bool = False

    view_query: bool = True
    view_reference: bool = True
    view_inliers: bool = True
    view_outliers: bool = False
    align: bool = True

    query_points_size: float = SOURCE_PTS_SIZE
    reference_points_size: float = TARGET_PTS_SIZE
    inliers_points_size: float = INLIER_PTS_SIZE
    outliers_points_size: float = OUTLIER_PTS_SIZE


class ClosuresVisualizer:
    def __init__(self, ps, gui, localmap_data):
        self._ps = ps
        self._gui = gui

        # Initialize GUI States
        self.states = ClosuresStateMachine()
        self.fig = None

        # Create data
        self.localmap_data = localmap_data
        self.data = LoopClosureData()
    

    def update_closures(self, alignment_pose, closure_edge, keypoints_pairs, inliers, alignment_time):
        self.data.closure_edges.append(closure_edge)
        self.data.alignment_pose.append(alignment_pose)
        self.data.size += 1
        self.data.current_id = self.data.size - 1
        if self.states.global_view:
            self._register_trajectory()
        if self.states.toggle_view:
            self._render_closure()
        self.data.keypoints_pairs.append(keypoints_pairs)
        self.data.inliers.append(inliers)
        self.data.alignment_time.append(alignment_time)

    def _query_map_callback(self):
        changed, self.states.query_points_size = self._gui.SliderFloat(
            "##query_size", self.states.query_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("query_map").set_radius(
                self.states.query_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self.states.view_query = self._gui.Checkbox("Query Map", self.states.view_query)
        if changed:
            self._ps.get_point_cloud("query_map").set_enabled(self.states.view_query)

    def _reference_map_callback(self):
        changed, self.states.reference_points_size = self._gui.SliderFloat(
            "##reference_size", self.states.reference_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("reference_map").set_radius(
                self.states.reference_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self.states.view_reference = self._gui.Checkbox(
            "Reference Map", self.states.view_reference
        )
        if changed:
            self._ps.get_point_cloud("reference_map").set_enabled(self.states.view_reference)
            
    def _inliers_callback(self):
        changed, self.states.inliers_points_size = self._gui.SliderFloat(
            "##inliers_size", self.states.inliers_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("query_map_inliers").set_radius(
                self.states.inliers_points_size, relative=False
            )
            self._ps.get_point_cloud("reference_map_inliers").set_radius(
                self.states.inliers_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self.states.view_inliers = self._gui.Checkbox(
            "Inliers", self.states.view_inliers
        )
        if changed:
            self._ps.get_point_cloud("query_map_inliers").set_enabled(self.states.view_inliers)
            self._ps.get_point_cloud("reference_map_inliers").set_enabled(self.states.view_inliers)
            self._ps.get_curve_network("inliers_edges").set_enabled(self.states.view_inliers)
            
    def _outliers_callback(self):
        changed, self.states.outliers_points_size = self._gui.SliderFloat(
            "##outliers_size", self.states.outliers_points_size, v_min=0.01, v_max=0.6
        )
        if changed:
            self._ps.get_point_cloud("query_map_outliers").set_radius(
                self.states.outliers_points_size, relative=False
            )
            self._ps.get_point_cloud("reference_map_outliers").set_radius(
                self.states.outliers_points_size, relative=False
            )
        self._gui.SameLine()
        changed, self.states.view_outliers = self._gui.Checkbox(
            "Outliers", self.states.view_outliers
        )
        if changed:
            self._ps.get_point_cloud("query_map_outliers").set_enabled(self.states.view_outliers)
            self._ps.get_point_cloud("reference_map_outliers").set_enabled(self.states.view_outliers)
            self._ps.get_curve_network("outliers_edges").set_enabled(self.states.view_outliers)

    def _alignment_callback(self):
        changed, self.states.align = self._gui.Checkbox(ALIGN_CLOSURE, self.states.align)
        if changed:
            self._render_closure()
        if self._gui.IsKeyPressed(self._gui.ImGuiKey_A):
            self.states.align = not self.states.align
            self._render_closure()

    def _closure_slider_callback(self):
        changed, self.data.current_id = self._gui.SliderInt(
            f"###Closure Index",
            self.data.current_id,
            v_min=0,
            v_max=self.data.size - 1,
            format="Closure ID: %d",
        )
        if changed:
            self._render_closure()

    def _previous_next_closure_callback(self):
        if self._gui.Button(PREV_CLOSURE) or self._gui.IsKeyPressed(self._gui.ImGuiKey_P):
            self.data.current_id = (self.data.current_id - 1) % self.data.size
        self._gui.SameLine()
        if self._gui.Button(NEXT_CLOSURE) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self.data.current_id = (self.data.current_id + 1) % self.data.size
        self._render_closure()
                
    def _update_connection_visibility(self):
        if self.update_connection_visibility:
            self.update_connection_visibility()

    def _render_closure(self):
        id = self.data.current_id
        [ref_id, query_id] = self.data.closure_edges[id]
        ref_map_pose = self.localmap_data.local_map_poses[ref_id]
        keypoints_pairs = self.data.keypoints_pairs[id]
        inliers = self.data.inliers[id]
        outliers = [pair for pair in keypoints_pairs if not any(np.array_equal(inlier, pair) for inlier in inliers)]
        query_map_pose = self.localmap_data.local_map_poses[query_id]
        query_map = self._ps.register_point_cloud(
            "query_map",
            self.localmap_data.local_maps[query_id],
            color=TARGET_COLOR,
            point_render_mode="quad",
        )
        query_map.set_radius(self.states.query_points_size, relative=False)
        query_map_inliers = self._ps.register_point_cloud(
            "query_map_inliers",
            np.array([inlier[1] for inlier in inliers]),
            color=INLIER_COLOR,
            point_render_mode="quad",
        )
        query_map_inliers.set_radius(0.5, relative=False)
        query_map_outliers = self._ps.register_point_cloud(
            "query_map_outliers",
            np.array([outlier[1] for outlier in outliers]),
            color=OUTLIER_COLOR,
            point_render_mode="quad",
        )
        query_map_outliers.set_radius(self.states.outliers_points_size, relative=False)
        reference_map = self._ps.register_point_cloud(
            "reference_map",
            self.localmap_data.local_maps[ref_id],
            color=SOURCE_COLOR,
            point_render_mode="quad",
        )
        reference_map.set_radius(self.states.reference_points_size, relative=False)
        reference_map_inliers = self._ps.register_point_cloud(
            "reference_map_inliers",
            np.array([inlier[0] for inlier in inliers]),
            color=INLIER_COLOR,
            point_render_mode="quad",
        )
        reference_map_inliers.set_radius(self.states.inliers_points_size, relative=False)
        reference_map_outliers = self._ps.register_point_cloud(
            "reference_map_outliers",
            np.array([outlier[0] for outlier in outliers]),
            color=OUTLIER_COLOR,
            point_render_mode="quad",
        )
        reference_map_outliers.set_radius(self.states.outliers_points_size, relative=False)
        self._ps.register_curve_network(
            "inliers_edges",
            np.array([inlier[0] for inlier in inliers] + [inlier[1] for inlier in inliers]),
            np.array([[i, i + len(inliers)] for i in range(len(inliers))]),
            color=INLIER_COLOR,
            radius=self.states.inliers_points_size / 1000,
        )
        self._ps.register_curve_network(
            "outliers_edges",
            np.array([outlier[0] for outlier in outliers] + [outlier[1] for outlier in outliers]),
            np.array([[i, i + len(outliers)] for i in range(len(outliers))]),
            color=OUTLIER_COLOR,
            radius=self.states.outliers_points_size / 2000,
            ),
        if self.states.global_view:
            query_map.set_transform(query_map_pose)
            if self.states.align:
                reference_map.set_transform(query_map_pose @ self.data.alignment_pose[id])
            else:
                reference_map.set_transform(ref_map_pose)
            query_map_inliers.set_enabled(False)
            reference_map_inliers.set_enabled(False)
            query_map_outliers.set_enabled(False)
            reference_map_outliers.set_enabled(False)
            self._ps.get_curve_network("inliers_edges").set_enabled(False)
            self._ps.get_curve_network("outliers_edges").set_enabled(False)
        else:
            query_map.set_transform(I)
            if self.states.align:
                reference_map.set_transform(self.data.alignment_pose[id])
            else:
                reference_map.set_transform(I)
            query_map_inliers.set_enabled(self.states.view_inliers)
            reference_map_inliers.set_enabled(self.states.view_inliers)
            query_map_outliers.set_enabled(self.states.view_outliers)
            reference_map_outliers.set_enabled(self.states.view_outliers)
            self._ps.get_curve_network("inliers_edges").set_enabled(self.states.view_inliers)
            self._ps.get_curve_network("outliers_edges").set_enabled(self.states.view_outliers)
        query_map.set_enabled(self.states.view_query)
        reference_map.set_enabled(self.states.view_reference)

    def _register_trajectory(self):
        closure_lines = self._ps.register_curve_network(
            "loop closures",
            np.array(self.localmap_data.local_map_poses)[:, :3, -1],
            np.array(self.data.closure_edges),
            color=TRAJECTORY_COLOR,
        )
        closure_lines.set_radius(0.1, relative=False)

    def _unregister_trajectory(self):
        self._ps.remove_curve_network("loop closures")
