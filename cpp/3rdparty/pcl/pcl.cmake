# MIT License
#
# Copyright (c) 2024 Javier Laserna
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
FetchContent_Declare(pcl URL https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.10.1.tar.gz UPDATE_DISCONNECTED 1)
FetchContent_GetProperties(pcl)
if(NOT pcl_POPULATED)
  FetchContent_Populate(pcl)
  if(${CMAKE_VERSION} GREATER_EQUAL 3.25)
    add_subdirectory(${pcl_SOURCE_DIR} ${pcl_BINARY_DIR} SYSTEM EXCLUDE_FROM_ALL)
  else()
    # Emulate the SYSTEM flag introduced in CMake 3.25. Withouth this flag the compiler will
    # consider this 3rdparty headers as source code and fail due the -Werror flag.
    add_subdirectory(${pcl_SOURCE_DIR} ${pcl_BINARY_DIR} EXCLUDE_FROM_ALL)
    get_target_property(pcl_include_dirs pcl INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(pcl PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${pcl_include_dirs}")
  endif()
endif()