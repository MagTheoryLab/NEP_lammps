# CMake integration for a non-upstream LAMMPS package: USER-NEP-GPU.
#
# This file is meant to be copied to:
#   <LAMMPS_SOURCE>/cmake/Modules/Packages/USER-NEP-GPU.cmake
#
# and LAMMPS's cmake/CMakeLists.txt must be patched to:
#   1) add USER-NEP-GPU to STANDARD_PACKAGES
#   2) add USER-NEP-GPU to the "packages with special needs" include list
#
# Then you can build with:
#   -DPKG_USER-NEP-GPU=ON
#
# The actual pair style sources live in <LAMMPS_SOURCE>/src/USER-NEP-GPU.
# The NEP GPU compute backend is built from the GPUMD/NEP_GPU library (this repo).

find_package(CUDAToolkit REQUIRED)

set(NEP_GPU_SOURCE_DIR "" CACHE PATH
  "Path to the GPUMD repo root that contains NEP_GPU/CMakeLists.txt (for building libnep_gpu)")
set(NEP_GPU_INCLUDE_DIR "" CACHE PATH
  "Path to the GPUMD repo root for headers (contains NEP_GPU/src/*.cuh and src/)")
set(NEP_GPU_LIBRARY "" CACHE FILEPATH
  "Optional: path to a prebuilt libnep_gpu.a/.so (if not building from source)")
option(NEP_GPU_BUILD_FROM_SOURCE "Build libnep_gpu from NEP_GPU_SOURCE_DIR" ON)

if(NEP_GPU_BUILD_FROM_SOURCE)
  if(NOT NEP_GPU_SOURCE_DIR)
    message(FATAL_ERROR
      "USER-NEP-GPU: NEP_GPU_BUILD_FROM_SOURCE=ON but NEP_GPU_SOURCE_DIR is empty. "
      "Set -DNEP_GPU_SOURCE_DIR=/path/to/GPUMD.")
  endif()
  if(NOT EXISTS "${NEP_GPU_SOURCE_DIR}/NEP_GPU/CMakeLists.txt")
    message(FATAL_ERROR
      "USER-NEP-GPU: ${NEP_GPU_SOURCE_DIR}/NEP_GPU/CMakeLists.txt not found. "
      "NEP_GPU_SOURCE_DIR must point to the GPUMD repo root.")
  endif()

  # Build the NEP GPU backend as a subproject.
  add_subdirectory("${NEP_GPU_SOURCE_DIR}/NEP_GPU" "${CMAKE_BINARY_DIR}/nep_gpu")
  target_link_libraries(lammps PUBLIC NEP_GPU::nep_gpu)
else()
  if(NOT NEP_GPU_LIBRARY)
    message(FATAL_ERROR
      "USER-NEP-GPU: NEP_GPU_BUILD_FROM_SOURCE=OFF but NEP_GPU_LIBRARY is empty. "
      "Set -DNEP_GPU_LIBRARY=/path/to/libnep_gpu.a (or .so).")
  endif()
  target_link_libraries(lammps PUBLIC
    "${NEP_GPU_LIBRARY}"
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cufft
  )
endif()

# Pair style sources include headers from the GPUMD tree (e.g. nep_gpu_lammps_model.h).
# When building from source, the NEP_GPU::nep_gpu target already exports these include
# directories. For the prebuilt-library mode, add them explicitly.
if(NOT NEP_GPU_INCLUDE_DIR AND NEP_GPU_SOURCE_DIR)
  set(NEP_GPU_INCLUDE_DIR "${NEP_GPU_SOURCE_DIR}")
endif()
if(NEP_GPU_INCLUDE_DIR)
  if(NOT EXISTS "${NEP_GPU_INCLUDE_DIR}/NEP_GPU/src/nep_gpu_lammps_model.h")
    message(FATAL_ERROR
      "USER-NEP-GPU: NEP_GPU_INCLUDE_DIR does not look like a GPUMD repo root: "
      "${NEP_GPU_INCLUDE_DIR}/NEP_GPU/src/nep_gpu_lammps_model.h not found.")
  endif()
  target_include_directories(lammps PRIVATE
    "${NEP_GPU_INCLUDE_DIR}/NEP_GPU/src"
    "${NEP_GPU_INCLUDE_DIR}/src"
    "${NEP_GPU_INCLUDE_DIR}/src/force"
  )
elseif(NOT NEP_GPU_BUILD_FROM_SOURCE)
  message(FATAL_ERROR
    "USER-NEP-GPU: building with a prebuilt NEP_GPU_LIBRARY requires NEP_GPU_INCLUDE_DIR "
    "so LAMMPS can find headers like nep_gpu_lammps_model.h.")
endif()

# NOTE: This repo ships both non-spin (`nep/gpu/kk`) and spin (`nep/spin/gpu/kk`)
# backends via the NEP_GPU library. The corresponding pair styles are part of
# the USER-NEP-GPU package sources copied into the LAMMPS tree.
