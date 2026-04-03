#pragma once

// Shared CUDA/HIP helpers for LAMMPS bridge code (non-spin + spin).
// Keep these `static` so multiple translation units can include this header safely.

#include "utilities/gpu_macro.cuh"

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <cstddef>

template <typename T>
static __global__ void fill_array_kernel(const size_t n, const T value, T* __restrict__ data)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((size_t) i < n) data[i] = value;
}

template <typename T>
static inline void fill_array_async(const size_t n, const T value, T* data, gpuStream_t stream)
{
  if (n == 0 || data == nullptr) return;
  const int block = 256;
  const int grid = (int) ((n + block - 1) / block);
  fill_array_kernel<<<grid, block, 0, stream>>>(n, value, data);
}

template <class VecT, typename T>
static inline void fill_vector_async(VecT& vec, const T value, gpuStream_t stream)
{
  fill_array_async(vec.size(), value, vec.data(), stream);
}

static inline gpuError_t stream_synchronize(gpuStream_t stream)
{
#ifdef USE_HIP
  return hipStreamSynchronize(stream);
#else
  return cudaStreamSynchronize(stream);
#endif
}

static __global__ void pack_xyz_aos_to_soa(
  const int n,
  const double* __restrict__ xyz_aos,
  double* __restrict__ xyz_soa)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int idx = 3 * i;
    xyz_soa[i] = xyz_aos[idx + 0];
    xyz_soa[i + n] = xyz_aos[idx + 1];
    xyz_soa[i + 2 * n] = xyz_aos[idx + 2];
  }
}

template <typename ForceT>
static __device__ inline void atomic_add_force(ForceT* addr, ForceT value);

template <>
__device__ inline void atomic_add_force<double>(double* addr, double value)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
  unsigned long long int* address_as_ull = (unsigned long long int*)addr;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
#else
  atomicAdd(addr, value);
#endif
}

template <>
__device__ inline void atomic_add_force<float>(float* addr, float value)
{
  atomicAdd(addr, value);
}

static __device__ __forceinline__ int map_owner_index(
  const int* __restrict__ owner,
  int idx,
  int nlocal)
{
  if (!owner) return idx;
  const int o = owner[idx];
  return (o >= 0 && o < nlocal) ? o : idx;
}

static __global__ void scatter_force_soa_to_aos_add(
  const int n,
  const double* __restrict__ fx,
  const double* __restrict__ fy,
  const double* __restrict__ fz,
  double* __restrict__ f_aos)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int idx = 3 * i;
    f_aos[idx + 0] += fx[i];
    f_aos[idx + 1] += fy[i];
    f_aos[idx + 2] += fz[i];
  }
}

static __global__ void scatter_force_soa_f_to_aos_add(
  const int n,
  const float* __restrict__ fx,
  const float* __restrict__ fy,
  const float* __restrict__ fz,
  double* __restrict__ f_aos)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int idx = 3 * i;
    f_aos[idx + 0] += static_cast<double>(fx[i]);
    f_aos[idx + 1] += static_cast<double>(fy[i]);
    f_aos[idx + 2] += static_cast<double>(fz[i]);
  }
}

// Non-spin bridge virial export: SoA stride = nlocal, AoS output is 9*nlocal.
static __global__ void virial_soa_to_aos9(
  const int nlocal,
  const double* __restrict__ v_soa, // 9*nlocal, SoA
  double* __restrict__ v_aos)       // 9*nlocal, AoS
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nlocal) {
    const int base = 9 * i;
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      v_aos[base + k] = v_soa[i + nlocal * k];
    }
  }
}

// Spin bridge virial export: SoA stride = natoms, but only local atoms are exported.
// Output is raw 9*nlocal AoS in the order (xx,yy,zz,xy,xz,yz,yx,zx,zy).
// Pair wrappers can symmetrize the off-diagonals when filling the 6-component
// LAMMPS stress/atom buffer, while cvatom can consume all 9 raw components.
static __global__ void virial_soa_to_aos9_local(
  const int nlocal,
  const int natoms,
  const double* __restrict__ v_soa, // 9*natoms, SoA
  double* __restrict__ v_aos)       // 9*nlocal, AoS
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nlocal) {
    const int base = 9 * i;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
      v_aos[base + k] = v_soa[i + natoms * k];
    }
  }
}
