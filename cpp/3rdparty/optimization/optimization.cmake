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

if(DEFINED ENV{GITHUB_TOKEN})
  set(GITHUB_TOKEN $ENV{GITHUB_TOKEN})
else()
  message(FATAL_ERROR "GITHUB_TOKEN is not set.")
endif()

FetchContent_Declare(
  optimization
  GIT_REPOSITORY https://$ENV{GITHUB_TOKEN}:x-oauth-basic@github.com/jlaserna/optimization.git
  GIT_TAG freeze/mapclosures)

FetchContent_GetProperties(optimization)
if(NOT optimization_POPULATED)
  message(STATUS "Fetching optimization")
  FetchContent_Populate(optimization)
  option(CPLEX_ON "Disable CPLEX" OFF)
  add_compile_options(-fPIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
  add_subdirectory(${optimization_SOURCE_DIR} ${optimization_BINARY_DIR} EXCLUDE_FROM_ALL)
  set(OPTIMIZATION_LIBRARIES copt graph bitscan utils CACHE INTERNAL "Optimization libraries")
  set(OPTIMIZATION_INCLUDE_DIRS ${optimization_SOURCE_DIR} CACHE INTERNAL
                                                                 "Optimization include directories")
endif()
