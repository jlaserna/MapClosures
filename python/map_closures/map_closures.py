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
import numpy as np
from typing_extensions import TypeAlias

from map_closures.config import MapClosuresConfig
from map_closures.pybind import map_closures_pybind

ClosureCandidate2D: TypeAlias = map_closures_pybind._ClosureCandidate2D
ClosureCandidate3D: TypeAlias = map_closures_pybind._ClosureCandidate3D


class MapClosures:
    def __init__(self, config: MapClosuresConfig = MapClosuresConfig()):
        self._config = config
        self._pipeline = map_closures_pybind._MapClosures(self._config.model_dump())

    def match_and_add_2D(self, map_idx: int, local_map: np.ndarray) -> ClosureCandidate2D:
        pcd = map_closures_pybind._Vector3dVector(local_map)
        return self._pipeline._MatchAndAdd2D(map_idx, pcd)
    
    def match_and_add_3D(self, map_idx: int, local_map: np.ndarray) -> ClosureCandidate3D:
        pcd = map_closures_pybind._Vector3dVector(local_map)
        return self._pipeline._MatchAndAdd3D(map_idx, pcd)

    def get_density_map_from_id(self, map_id: int) -> np.ndarray:
        return self._pipeline._getDensityMapFromId(map_id)

    def validate_closure_2D(self, ref_idx: int, query_idx: int) -> ClosureCandidate2D:
        return self._pipeline._ValidateClosure2D(ref_idx, query_idx)
    
    def validate_closure_3D(self, ref_idx: int, query_idx: int) -> ClosureCandidate3D:
        return self._pipeline._ValidateClosure3D(ref_idx, query_idx)
