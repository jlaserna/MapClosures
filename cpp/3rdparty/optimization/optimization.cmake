include(FetchContent)

if(DEFINED ENV{GITHUB_TOKEN})
  set(GITHUB_TOKEN $ENV{GITHUB_TOKEN})
else()
  message(FATAL_ERROR "GITHUB_TOKEN is not set.")
endif()

FetchContent_Declare(
  optimization
  GIT_REPOSITORY https://$ENV{GITHUB_TOKEN}:x-oauth-basic@github.com/jlaserna/optimization.git
  GIT_TAG 38556b9d11a03708f83487615c0c40bbe189e4fe
)

FetchContent_GetProperties(optimization)
if(NOT optimization_POPULATED)
  message(STATUS "Fetching optimization")
  FetchContent_Populate(optimization)
  option(CPLEX_ON "Disable CPLEX" OFF)
  add_compile_options(-fPIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
  add_subdirectory(${optimization_SOURCE_DIR} ${optimization_BINARY_DIR} EXCLUDE_FROM_ALL)
  set(OPTIMIZATION_LIBRARIES copt graph bitscan utils CACHE INTERNAL "Optimization libraries")
  set(OPTIMIZATION_INCLUDE_DIRS ${optimization_SOURCE_DIR} CACHE INTERNAL "Optimization include directories")
endif()
