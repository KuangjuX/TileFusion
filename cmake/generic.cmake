# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD
    20
    CACHE STRING "The C++ standard whose features are requested." FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD
    20
    CACHE STRING "The CUDA standard whose features are requested." FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set host compiler flags. Enable all warnings and treat them as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

find_package(CUDAToolkit QUIET REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA on)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Python interpreter path: ${Python3_EXECUTABLE}")

set(TORCH_LIB_PREFIX "${Python3_SITEARCH}/torch")
if(NOT EXISTS ${TORCH_LIB_PREFIX})
  message(FATAL_ERROR "Torch library is not installed.")
else()
  list(APPEND CMAKE_PREFIX_PATH "${TORCH_LIB_PREFIX}/share/cmake/Torch")
endif()
find_package(Torch REQUIRED)

# let cmake automatically detect the current CUDA architecture to avoid
# generating device codes for all possible architectures
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  --Werror all-warnings")
# Set the CUDA_PROPAGATE_HOST_FLAGS to OFF to avoid passing host compiler flags
# to the device compiler
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# FIXME(haruhi): -std=c++20 has to be set explicitly here, Otherwise, linking
# against torchlibs will raise errors. it seems that the host compilation
# options are not passed to torchlibs.
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++20)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -std=c++20 -O0)
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -std=c++20 -O3)

if(${CUDA_VERSION_MAJOR} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(STATUS "CUDA_VERSION ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR} "
                 "is greater or equal than 11.0, enable -DENABLE_BF16 flag.")
endif()

message(STATUS "tilefusion: CUDA detected: " ${CUDA_VERSION})
message(STATUS "tilefusion: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "tilefusion: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})

if(ENABLE_DEBUG)
  message(STATUS "tilefusion: Debug mode enabled")
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DDEBUG")
endif()

function(cuda_test TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cuda_test "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  list(APPEND UT_SRCS "${PROJECT_SOURCE_DIR}/tests/cpp/test_unit.cc"
       "${PROJECT_SOURCE_DIR}/src/cuda_utils.cc"
       "${PROJECT_SOURCE_DIR}/tests/cpp/common/test_utils.cc" ${cuda_test_SRCS})

  cuda_add_executable(${TARGET_NAME} ${UT_SRCS})
  target_link_libraries(${TARGET_NAME} ${cuda_test_DEPS} gtest glog::glog)
  add_dependencies(${TARGET_NAME} gtest glog::glog)

  # add a unittest into ctest with the same name as the target
  add_test(${TARGET_NAME} ${TARGET_NAME})
endfunction(cuda_test)
