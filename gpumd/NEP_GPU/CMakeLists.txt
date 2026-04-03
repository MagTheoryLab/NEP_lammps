cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(nep_gpu LANGUAGES CXX CUDA)
else()
  enable_language(CUDA)
endif()

option(NEP_GPU_BUILD_SHARED "Build NEP GPU as a shared library" OFF)
option(NEP_GPU_ENABLE_TABLE "Enable NEP lookup tables (compile with USE_TABLE)" OFF)
option(NEP_GPU_USE_FAST_MATH "Enable CUDA fast math (--use_fast_math)" OFF)

# Default to a reasonable baseline if the parent project didn't set this.
if((NOT DEFINED CMAKE_CUDA_ARCHITECTURES) OR ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL ""))
  set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures")
endif()

get_filename_component(GPUMD_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

find_package(CUDAToolkit REQUIRED)

set(NEP_GPU_SOURCES
  ${GPUMD_ROOT}/NEP_GPU/src/nep_lmp_bridge.cu
  ${GPUMD_ROOT}/NEP_GPU/src/nep_gpu_lammps_model.cu
  ${GPUMD_ROOT}/NEP_GPU/src/nep_spin_lmp_bridge.cu

  ${GPUMD_ROOT}/src/model/box.cu
  ${GPUMD_ROOT}/src/force/potential.cu
  ${GPUMD_ROOT}/src/utilities/error.cu
  ${GPUMD_ROOT}/src/utilities/read_file.cu
)

if(NEP_GPU_BUILD_SHARED)
  add_library(nep_gpu SHARED ${NEP_GPU_SOURCES})
else()
  add_library(nep_gpu STATIC ${NEP_GPU_SOURCES})
endif()

add_library(NEP_GPU::nep_gpu ALIAS nep_gpu)

set_target_properties(nep_gpu PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(nep_gpu
  PUBLIC
    ${GPUMD_ROOT}/NEP_GPU/src
    ${GPUMD_ROOT}/src
    ${GPUMD_ROOT}/src/force
)

target_link_libraries(nep_gpu
  PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cufft
)

# Optional performance knobs (may trade precision for speed).
if(NEP_GPU_ENABLE_TABLE)
  target_compile_definitions(nep_gpu PRIVATE USE_TABLE)
endif()

if(NEP_GPU_USE_FAST_MATH AND (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"))
  target_compile_options(nep_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()

# Keep NVCC output clean on newer toolchains where C++14 is deprecated in Thrust/CUB,
# but do not force users to change their global warnings.
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  target_compile_options(nep_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=unrecognized_pragma>
  )
endif()






