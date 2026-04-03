if(PKG_USER-NEP-GPU)
  set(USER-NEP-GPU_SOURCES
    ${LAMMPS_SOURCE_DIR}/USER-NEP-GPU/pair_nep_gpu.cpp
  )

  set(USER-NEP-GPU_HEADERS
    ${LAMMPS_SOURCE_DIR}/USER-NEP-GPU/pair_nep_gpu.h
  )

  get_property(LAMMPS_HEADERS GLOBAL PROPERTY HEADERS)
  list(APPEND LAMMPS_HEADERS ${USER-NEP-GPU_HEADERS})
  set_property(GLOBAL PROPERTY HEADERS "${LAMMPS_HEADERS}")

  get_target_property(LAMMPS_SOURCES lammps SOURCES)
  list(APPEND LAMMPS_SOURCES ${USER-NEP-GPU_SOURCES})
  set_property(TARGET lammps PROPERTY SOURCES "${LAMMPS_SOURCES}")

  # CMake-based build of USER-NEP-GPU requires an external NEP_GPU library
  # compiled from the GPUMD tree. Provide the GPUMD root via:
  #   -DNEP_GPU_ROOT=/path/to/GPUMD
  if(NOT DEFINED NEP_GPU_ROOT)
    message(FATAL_ERROR "NEP_GPU_ROOT must be set to the GPUMD root directory to enable USER-NEP-GPU")
  endif()

  # Header search paths so that pair_nep_gpu.cpp can find nep_gpu_model.cuh
  target_include_directories(lammps PRIVATE
    ${NEP_GPU_ROOT}/src
    ${NEP_GPU_ROOT}/NEP_GPU/src
  )

  # Find the pre-built NEP_GPU static library (libnep_gpu.a) in the GPUMD root.
  # Build it in the GPUMD tree first, then point NEP_GPU_ROOT to that tree.
  find_library(NEP_GPU_LIB nep_gpu
    HINTS ${NEP_GPU_ROOT} ${NEP_GPU_ROOT}/lib
    NO_DEFAULT_PATH
  )

  if(NOT NEP_GPU_LIB)
    message(FATAL_ERROR "Could not find libnep_gpu.a in ${NEP_GPU_ROOT} (expected name: libnep_gpu.a). Please build it in the GPUMD tree first.")
  endif()

  # Link LAMMPS against NEP_GPU and CUDA runtime libraries.
  target_link_libraries(lammps PRIVATE
    ${NEP_GPU_LIB}
    cudart
    cublas
    cusolver
    cufft
  )
endif()
