# MIT License
#
# Copyright (c) 2025 Javier Laserna, Saurabh Gupta
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

include(FetchContent)
FetchContent_Declare(
  pcl URL https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.10.1.tar.gz
  UPDATE_DISCONNECTED 1)
FetchContent_MakeAvailable(pcl)

add_library(PCL INTERFACE)
target_link_libraries(
  PCL
  INTERFACE pcl_features
            pcl_filters
            pcl_keypoints
            pcl_common
            pcl_search
            pcl_kdtree
            pcl_io
            pcl_registration)

target_include_directories(
  PCL
  INTERFACE ${pcl_SOURCE_DIR}
            ${pcl_SOURCE_DIR}/features/include
            ${pcl_SOURCE_DIR}/filters/include
            ${pcl_SOURCE_DIR}/keypoints/include
            ${pcl_SOURCE_DIR}/common/include
            ${pcl_SOURCE_DIR}/search/include
            ${pcl_SOURCE_DIR}/kdtree/include
            ${pcl_SOURCE_DIR}/io/include
            ${pcl_SOURCE_DIR}/registration/include
            ${pcl_BINARY_DIR}/include/)

set(PCL_LIBS PCL)
