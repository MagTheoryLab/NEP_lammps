/*
  LAMMPS-direct NEP_Spin GPU bridge.
  This file compiles a vendored subset of NEP_Spin CUDA code (kernels only) and
  provides a standalone entry point that consumes externally-built neighbor
  lists (LAMMPS/Kokkos).
*/

#include "nep_spin_lmp_bridge.cuh"

#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/kernel_timing.cuh"
#include "utils/nep_lmp_utils.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/nep_spin_utilities.cuh"
#include "utilities/read_file.cuh"

#include "neighbor.cuh"

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Pull in all NEP_Spin device kernels (but not the MD wrapper methods).
// We vendor the needed CUDA code under `NEP_GPU/src/` so NEP_GPU does not depend
// on the GPUMD MD translation unit `src/force/nep_spin.cu`.
// BEGIN embedded NEP_Spin kernels (vendored)
// Extracted from `src/force/nep_spin.cu` so `NEP_GPU/src/nep_spin_lmp_bridge.cu` is self-contained.
// END/BEGIN markers make future sync easier.

#if defined(USE_HIP)
#define NEP_SPIN_LDG(ptr) (*(ptr))
#else
#define NEP_SPIN_LDG(ptr) __ldg(ptr)
#endif

#define NEP_SPIN_DISPATCH_KMAX(KMAX, KERNEL, ...)                 \
  switch (KMAX) {                                                 \
    case 0: KERNEL<KMAX_PAIR, 0><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
    case 1: KERNEL<KMAX_PAIR, 1><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
    case 2: KERNEL<KMAX_PAIR, 2><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
    case 3: KERNEL<KMAX_PAIR, 3><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
    case 4: KERNEL<KMAX_PAIR, 4><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
    case 5: KERNEL<KMAX_PAIR, 5><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
    case 6: KERNEL<KMAX_PAIR, 6><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
    case 7: KERNEL<KMAX_PAIR, 7><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
    default: KERNEL<KMAX_PAIR, 8><<<grid_size, BLOCK_SIZE>>>(__VA_ARGS__); break; \
  }

// Pick a compile-time descriptor dimension bucket for better register usage.
// Mirrors the MD-side implementation in `src/force/nep_spin.cu`.
static inline int nep_spin_pick_dim_bucket(int dim)
{
  if (dim <= 64) return 64;
  if (dim <= 96) return 96;
  if (dim <= 128) return 128;
  return MAX_DIM;
}

enum class SpinV2RuntimePath {
  reference,
  fast,
  shadow
};

#ifdef USE_HIP
__device__ __constant__ float g_nep_spin_mref[NUM_ELEMENTS];
#else
__constant__ float g_nep_spin_mref[NUM_ELEMENTS];
#endif

static inline int spin_v2_pick_dim_bucket(int dim)
{
  return nep_spin_pick_dim_bucket(dim);
}

static inline bool nep_spin_env_true(const char* name)
{
  const char* value = std::getenv(name);
  if (!value || !value[0]) return false;
  return !(value[0] == '0' && value[1] == '\0');
}

static SpinV2RuntimePath get_spin_v2_runtime_path()
{
  const char* value = std::getenv("NEP_SPIN_GPU_LMP_V2_PATH");
  if (!value || !value[0]) return SpinV2RuntimePath::reference;
  if (std::strcmp(value, "fast") == 0) return SpinV2RuntimePath::fast;
  if (std::strcmp(value, "shadow") == 0) return SpinV2RuntimePath::shadow;
  if (std::strcmp(value, "reference") == 0) return SpinV2RuntimePath::reference;
  return SpinV2RuntimePath::reference;
}

static double get_spin_v2_shadow_tolerance()
{
  const char* value = std::getenv("NEP_SPIN_GPU_LMP_V2_SHADOW_TOL");
  if (!value || !value[0]) return 5.0e-6;
  return std::max(0.0, std::atof(value));
}

constexpr int MAX_SPIN_ABC = 24; // l_max_spin_angular <= 4
constexpr int SPIN3_N_TILE = 2;

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_base_offset(const ParaMB& paramb)
{
  return (paramb.n_max_radial + 1) + paramb.dim_angular;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_2body_count(const ParaMB& paramb)
{
  return paramb.n_max_spin_radial + 1;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_3body_count(const ParaMB& paramb)
{
  return paramb.n_max_spin_angular + 1;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_3body_abc_count(const ParaMB& paramb)
{
  return (paramb.l_max_spin_angular + 1) * (paramb.l_max_spin_angular + 1) - 1;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block0_offset(const ParaMB& paramb)
{
  return nep_spin_base_offset(paramb);
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block1_offset(const ParaMB& paramb)
{
  return nep_spin_block0_offset(paramb) + paramb.spin_pmax;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block2_offset(const ParaMB& paramb)
{
  return nep_spin_block1_offset(paramb) + 4 * nep_spin_2body_count(paramb);
}

static __device__ __forceinline__ float nep_spin_type_mref(const int t1)
{
  return g_nep_spin_mref[t1];
}

static __device__ __forceinline__ float nep_spin_type_yref(const int t1, const int basis_mode)
{
  const float mref = nep_spin_type_mref(t1);
  return (basis_mode == 2) ? mref : mref * mref;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_total_dim(const ParaMB& paramb)
{
  const int nspin3 = nep_spin_3body_count(paramb);
  const int lmax = paramb.l_max_spin_angular;
  const int core_count = nspin3 * lmax;
  const int g1_count = (lmax >= 2) ? nspin3 : 0;
  const int across_count = (nspin3 * (nspin3 - 1) / 2) * lmax;
  return paramb.spin_pmax + 4 * nep_spin_2body_count(paramb) + 5 * core_count + 2 * g1_count +
         across_count;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block1_index(
  const ParaMB& paramb, const int family, const int n)
{
  return nep_spin_block1_offset(paramb) + family * nep_spin_2body_count(paramb) + n;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block2_core_count(const ParaMB& paramb)
{
  return nep_spin_3body_count(paramb) * paramb.l_max_spin_angular;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block2_g1_count(const ParaMB& paramb)
{
  return paramb.l_max_spin_angular >= 2 ? nep_spin_3body_count(paramb) : 0;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block2_pair_lex_index(
  const ParaMB& paramb, const int n1, const int n2)
{
  const int nspin3 = nep_spin_3body_count(paramb);
  if (!(0 <= n1 && n1 < n2 && n2 < nspin3)) return -1;
  return n1 * (nspin3 - 1) - (n1 * (n1 - 1)) / 2 + (n2 - n1 - 1);
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block2_core_index(
  const ParaMB& paramb, const int family, const int n, const int L_minus_1)
{
  const int family_stride = nep_spin_block2_core_count(paramb);
  return nep_spin_block2_offset(paramb) + family * family_stride + n * paramb.l_max_spin_angular +
         L_minus_1;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block2_g1_index(
  const ParaMB& paramb, const int family, const int n)
{
  return nep_spin_block2_offset(paramb) + 5 * nep_spin_block2_core_count(paramb) +
         family * nep_spin_block2_g1_count(paramb) + n;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_block2_across_index(
  const ParaMB& paramb, const int n1, const int n2, const int L_minus_1)
{
  return nep_spin_block2_offset(paramb) + 5 * nep_spin_block2_core_count(paramb) +
         2 * nep_spin_block2_g1_count(paramb) +
         nep_spin_block2_pair_lex_index(paramb, n1, n2) * paramb.l_max_spin_angular + L_minus_1;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_c_index_2body(
  const ParaMB& paramb, const int n, const int k, const int t1, const int t2)
{
  return paramb.c_spin_2body_offset +
         (n * (paramb.basis_size_spin_radial + 1) + k) * paramb.num_types_sq + t1 * paramb.num_types +
         t2;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_c_index_3body(
  const ParaMB& paramb, const int n, const int k, const int t1, const int t2)
{
  return paramb.c_spin_3body_offset +
         (n * (paramb.basis_size_spin_angular + 1) + k) * paramb.num_types_sq + t1 * paramb.num_types +
         t2;
}

template <int L>
static __device__ __forceinline__ float find_q_cross_one(const float* a, const float* b)
{
  const int start = L * L - 1;
  const int num_terms = 2 * L + 1;
  float q = C3B[start] * a[start] * b[start];
  for (int k = 1; k < num_terms; ++k) {
    q += 2.0f * C3B[start + k] * a[start + k] * b[start + k];
  }
  return q;
}

static __device__ __forceinline__ float compute_q4b_l2(const float* a)
{
  return C4B[0] * a[0] * a[0] * a[0] + C4B[1] * a[0] * (a[1] * a[1] + a[2] * a[2]) +
         C4B[2] * a[0] * (a[3] * a[3] + a[4] * a[4]) + C4B[3] * a[3] * (a[2] * a[2] - a[1] * a[1]) +
         C4B[4] * a[1] * a[2] * a[4];
}

static __device__ __forceinline__ void compute_grad_q4b_l2(const float* a, float* grad)
{
  grad[0] = 3.0f * C4B[0] * a[0] * a[0] + C4B[1] * (a[1] * a[1] + a[2] * a[2]) +
            C4B[2] * (a[3] * a[3] + a[4] * a[4]);
  grad[1] = 2.0f * C4B[1] * a[0] * a[1] - 2.0f * C4B[3] * a[3] * a[1] + C4B[4] * a[2] * a[4];
  grad[2] = 2.0f * C4B[1] * a[0] * a[2] + 2.0f * C4B[3] * a[3] * a[2] + C4B[4] * a[1] * a[4];
  grad[3] = 2.0f * C4B[2] * a[0] * a[3] + C4B[3] * (a[2] * a[2] - a[1] * a[1]);
  grad[4] = 2.0f * C4B[2] * a[0] * a[4] + C4B[4] * a[1] * a[2];
}

static __device__ __forceinline__ void accumulate_mix_q4b_l2(
  const float* s0, const float* sc, float& q_mix, float* grad_s0, float* grad_sc)
{
  float grad_q4b[5];
  compute_grad_q4b_l2(sc, grad_q4b);
  q_mix = 0.0f;
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    grad_s0[i] = grad_q4b[i] / 3.0f;
    q_mix += s0[i] * grad_s0[i];
  }

  grad_sc[0] =
    (s0[0] * (6.0f * C4B[0] * sc[0]) + s0[1] * (2.0f * C4B[1] * sc[1]) +
     s0[2] * (2.0f * C4B[1] * sc[2]) + s0[3] * (2.0f * C4B[2] * sc[3]) +
     s0[4] * (2.0f * C4B[2] * sc[4])) / 3.0f;
  grad_sc[1] =
    (s0[0] * (2.0f * C4B[1] * sc[1]) + s0[1] * (2.0f * C4B[1] * sc[0] - 2.0f * C4B[3] * sc[3]) +
     s0[2] * (C4B[4] * sc[4]) + s0[3] * (-2.0f * C4B[3] * sc[1]) + s0[4] * (C4B[4] * sc[2])) / 3.0f;
  grad_sc[2] =
    (s0[0] * (2.0f * C4B[1] * sc[2]) + s0[1] * (C4B[4] * sc[4]) +
     s0[2] * (2.0f * C4B[1] * sc[0] + 2.0f * C4B[3] * sc[3]) +
     s0[3] * (2.0f * C4B[3] * sc[2]) + s0[4] * (C4B[4] * sc[1])) / 3.0f;
  grad_sc[3] =
    (s0[0] * (2.0f * C4B[2] * sc[3]) + s0[1] * (-2.0f * C4B[3] * sc[1]) +
     s0[2] * (2.0f * C4B[3] * sc[2]) + s0[3] * (2.0f * C4B[2] * sc[0])) / 3.0f;
  grad_sc[4] =
    (s0[0] * (2.0f * C4B[2] * sc[4]) + s0[1] * (C4B[4] * sc[2]) + s0[2] * (C4B[4] * sc[1]) +
     s0[4] * (2.0f * C4B[2] * sc[0])) / 3.0f;
}

template <int L>
static __device__ __forceinline__ void fill_ylm_one(
  const float x12, const float y12, const float z12, float* ylm);

template <int L>
static __device__ __forceinline__ void accumulate_spin3body_one(
  const float xhat,
  const float yhat,
  const float zhat,
  const float w0,
  const float wc,
  const float wAx,
  const float wAy,
  const float wAz,
  const float wD,
  float* s0,
  float* sc,
  float* Ax,
  float* Ay,
  float* Az,
  float* D)
{
  float ylm[2 * L + 1];
  fill_ylm_one<L>(xhat, yhat, zhat, ylm);
  constexpr int start = L * L - 1;
  #pragma unroll
  for (int k = 0; k < 2 * L + 1; ++k) {
    const float y = ylm[k];
    s0[start + k] += w0 * y;
    sc[start + k] += wc * y;
    Ax[start + k] += wAx * y;
    Ay[start + k] += wAy * y;
    Az[start + k] += wAz * y;
    D[start + k] += wD * y;
  }
}

template <int LMAX>
static __device__ __forceinline__ void accumulate_spin3body_all_lmax(
  const float d12,
  const float x12,
  const float y12,
  const float z12,
  const float w0,
  const float wc,
  const float wAx,
  const float wAy,
  const float wAz,
  const float wD,
  float* s0,
  float* sc,
  float* Ax,
  float* Ay,
  float* Az,
  float* D)
{
  const float d12inv = 1.0f / d12;
  const float xhat = x12 * d12inv;
  const float yhat = y12 * d12inv;
  const float zhat = z12 * d12inv;
  if constexpr (LMAX >= 1) {
    accumulate_spin3body_one<1>(xhat, yhat, zhat, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
  }
  if constexpr (LMAX >= 2) {
    accumulate_spin3body_one<2>(xhat, yhat, zhat, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
  }
  if constexpr (LMAX >= 3) {
    accumulate_spin3body_one<3>(xhat, yhat, zhat, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
  }
  if constexpr (LMAX >= 4) {
    accumulate_spin3body_one<4>(xhat, yhat, zhat, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
  }
}

static __device__ __forceinline__ void accumulate_spin3body_all(
  const int L_max,
  const float d12,
  const float x12,
  const float y12,
  const float z12,
  const float w0,
  const float wc,
  const float wAx,
  const float wAy,
  const float wAz,
  const float wD,
  float* s0,
  float* sc,
  float* Ax,
  float* Ay,
  float* Az,
  float* D)
{
  switch (L_max) {
    case 4:
      accumulate_spin3body_all_lmax<4>(d12, x12, y12, z12, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
      break;
    case 3:
      accumulate_spin3body_all_lmax<3>(d12, x12, y12, z12, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
      break;
    case 2:
      accumulate_spin3body_all_lmax<2>(d12, x12, y12, z12, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
      break;
    default:
      accumulate_spin3body_all_lmax<1>(d12, x12, y12, z12, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
      break;
  }
}

template <int L>
static __device__ __forceinline__ void fill_ylm_one(
  const float x12, const float y12, const float z12, float* ylm)
{
  float z_pow[L + 1] = {1.0f};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z12 * z_pow[n - 1];
  }
  float real_part = x12;
  float imag_part = y12;
  int y_index = 0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = ((L + n1) % 2 == 0) ? 0 : 1;
    float z_factor = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      if (L == 1) z_factor += Z_COEFFICIENT_1[n1][n2] * z_pow[n2];
      if (L == 2) z_factor += Z_COEFFICIENT_2[n1][n2] * z_pow[n2];
      if (L == 3) z_factor += Z_COEFFICIENT_3[n1][n2] * z_pow[n2];
      if (L == 4) z_factor += Z_COEFFICIENT_4[n1][n2] * z_pow[n2];
    }
    if (n1 == 0) {
      ylm[y_index++] = z_factor;
    } else {
      ylm[y_index++] = real_part * z_factor;
      ylm[y_index++] = imag_part * z_factor;
      complex_product(x12, y12, real_part, imag_part);
    }
  }
}

template <int L>
static __device__ __forceinline__ float dot_packed_terms(const float* a, const float* b)
{
  float out = 0.0f;
  #pragma unroll
  for (int k = 0; k < 2 * L + 1; ++k) {
    out += a[k] * b[k];
  }
  return out;
}

template <int L>
static __device__ __forceinline__ void accumulate_spin3body_force_one_L(
  const float d12inv,
  const float gn,
  const float gnp,
  const float si_dot_sj,
  const float phi_dmi,
  const float* sj,
  const float* rhat,
  const float* dEs0L,
  const float* dEscL,
  const float* dEAxL,
  const float* dEAyL,
  const float* dEAzL,
  const float* dEDL,
  float* f12,
  float& projc,
  float& projD,
  float& projAx,
  float& projAy,
  float& projAz)
{
  float ylm[2 * L + 1];
  float dEeff[2 * L + 1];
  fill_ylm_one<L>(rhat[0], rhat[1], rhat[2], ylm);
  #pragma unroll
  for (int k = 0; k < 2 * L + 1; ++k) {
    dEeff[k] =
      dEs0L[k] + si_dot_sj * dEscL[k] + sj[0] * dEAxL[k] + sj[1] * dEAyL[k] + sj[2] * dEAzL[k] +
      phi_dmi * dEDL[k];
  }
  accumulate_f12_one<L>(d12inv, gn, gnp, dEeff, rhat, f12);
  projc += dot_packed_terms<L>(dEscL, ylm);
  projD += dot_packed_terms<L>(dEDL, ylm);
  projAx += dot_packed_terms<L>(dEAxL, ylm);
  projAy += dot_packed_terms<L>(dEAyL, ylm);
  projAz += dot_packed_terms<L>(dEAzL, ylm);
}

template <typename ForceT>
static __device__ __forceinline__ void add_force_and_virial(
  const int n2,
  const float* r12,
  const float* f12,
  float& fi_x,
  float& fi_y,
  float& fi_z,
  float& v_xx,
  float& v_yy,
  float& v_zz,
  float& v_xy,
  float& v_yz,
  float& v_zx,
  ForceT* g_fx,
  ForceT* g_fy,
  ForceT* g_fz)
{
  fi_x += f12[0];
  fi_y += f12[1];
  fi_z += f12[2];
  atomic_add_force(&g_fx[n2], static_cast<ForceT>(-f12[0]));
  atomic_add_force(&g_fy[n2], static_cast<ForceT>(-f12[1]));
  atomic_add_force(&g_fz[n2], static_cast<ForceT>(-f12[2]));
  v_xx -= r12[0] * f12[0];
  v_yy -= r12[1] * f12[1];
  v_zz -= r12[2] * f12[2];
  v_xy -= r12[0] * f12[1];
  v_yz -= r12[1] * f12[2];
  v_zx -= r12[2] * f12[0];
}

static __global__ void find_descriptors_radial_spinbase(
  const int N,
  const int nlocal,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  float q[MAX_NUM_N] = {0.0f};
  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + N * i1;
    const int n2 = g_NL[index];
    const float x12 = g_x12[index];
    const float y12 = g_y12[index];
    const float z12 = g_z12[index];
    const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
    const int t2 = g_type[n2];
    const float rc = (paramb.rc_radial_by_type[t1] + paramb.rc_radial_by_type[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(bs, rcinv, d12, fc12, fn12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12 = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      q[n] += gn12;
    }
  }
  for (int n = 0; n <= paramb.n_max_radial; ++n) {
    g_descriptors[n1 + n * N] = q[n];
  }
}

static __global__ void find_descriptors_angular_spinbase(
  const int N,
  const int nlocal,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  float q[MAX_DIM_ANGULAR] = {0.0f};
  int bs = paramb.basis_size_angular;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      const int index = n1 + N * i1;
      const int n2 = g_NL[index];
      const float x12 = g_x12[index];
      const float y12 = g_y12[index];
      const float z12 = g_z12[index];
      const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
      const int t2 = g_type[n2];
      const float rc = (paramb.rc_angular_by_type[t1] + paramb.rc_angular_by_type[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(bs, rcinv, d12, fc12, fn12);
      float gn12 = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
    }
    find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q);
    for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
      g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
    }
  }
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    for (int l = 0; l < paramb.num_L; ++l) {
      const int ln = l * (paramb.n_max_angular + 1) + n;
      g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
    }
  }
}

static __global__ void find_descriptors_spin_onsite(
  const int N,
  const int nlocal,
  const NEP_Spin::ParaMB paramb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_spin,
  float* __restrict__ g_descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal || paramb.spin_pmax <= 0) return;
  const int offset = nep_spin_block0_offset(paramb);
  const int t1 = g_type[n1];
  const float sx = __ldg(&g_spin[n1]);
  const float sy = __ldg(&g_spin[n1 + N]);
  const float sz = __ldg(&g_spin[n1 + N * 2]);
  const float si2 = sx * sx + sy * sy + sz * sz;
  if (si2 <= 1.0e-12f) {
    for (int p = 0; p < paramb.spin_pmax; ++p) {
      g_descriptors[n1 + (offset + p) * N] = 0.0f;
    }
    return;
  }
  if (paramb.spin_onsite_basis_mode == 0) {
    float m2p = si2;
    for (int p = 0; p < paramb.spin_pmax; ++p) {
      g_descriptors[n1 + (offset + p) * N] = m2p;
      m2p *= si2;
    }
    return;
  }
  float y = si2;
  float yref = nep_spin_type_yref(t1, paramb.spin_onsite_basis_mode);
  if (paramb.spin_onsite_basis_mode == 2) {
    y = sqrtf(si2);
  }
  if (yref <= 0.0f) yref = 1.0f;
  float x = (y - yref) / (y + yref + 1.0e-12f);
  x = fminf(1.0f, fmaxf(-1.0f, x));
  float Tp[9] = {1.0f};
  if (paramb.spin_pmax >= 1) Tp[1] = x;
  for (int p = 2; p <= paramb.spin_pmax; ++p) {
    Tp[p] = 2.0f * x * Tp[p - 1] - Tp[p - 2];
  }
  for (int p = 1; p <= paramb.spin_pmax; ++p) {
    g_descriptors[n1 + (offset + p - 1) * N] = Tp[p];
  }
}

static __global__ void find_descriptors_spin_2body(
  const int N,
  const int nlocal,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  float* __restrict__ g_descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  const int nspin = nep_spin_2body_count(paramb);
  const float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
  float q_ex[MAX_NUM_N] = {0.0f};
  float q_dmi[MAX_NUM_N] = {0.0f};
  float q_ani[MAX_NUM_N] = {0.0f};
  float q_sia[MAX_NUM_N] = {0.0f};
  int bs = paramb.basis_size_spin_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = i1 * N + n1;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (!(d12 > 0.0f)) continue;
    const float inv_d12 = 1.0f / d12;
    const float rhat[3] = {r12[0] * inv_d12, r12[1] * inv_d12, r12[2] * inv_d12};
    const float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
    const float si_dot_sj = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
    const float si_dot_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const float sj_dot_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
    const float cross[3] = {
      si[1] * sj[2] - si[2] * sj[1],
      si[2] * sj[0] - si[0] * sj[2],
      si[0] * sj[1] - si[1] * sj[0]};
    const float phi_ex = si_dot_sj;
    const float phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
    const float phi_ani = si_dot_r * sj_dot_r;
    const float phi_sia = si_dot_r * si_dot_r;
    const float rc = (paramb.rc_radial_by_type[t1] + paramb.rc_radial_by_type[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(bs, rcinv, d12, fc12, fn12);
    for (int n = 0; n < nspin; ++n) {
      float gn = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        gn += fn12[k] * annmb.c[nep_spin_c_index_2body(paramb, n, k, t1, t2)];
      }
      q_ex[n] += gn * phi_ex;
      q_dmi[n] += gn * phi_dmi;
      q_ani[n] += gn * phi_ani;
      q_sia[n] += gn * phi_sia;
    }
  }
  for (int n = 0; n < nspin; ++n) {
    g_descriptors[n1 + nep_spin_block1_index(paramb, 0, n) * N] = q_ex[n];
    g_descriptors[n1 + nep_spin_block1_index(paramb, 1, n) * N] = q_dmi[n];
    g_descriptors[n1 + nep_spin_block1_index(paramb, 2, n) * N] = q_ani[n];
    g_descriptors[n1 + nep_spin_block1_index(paramb, 3, n) * N] = q_sia[n];
  }
}

static __global__ void find_descriptors_spin_3body(
  const int N,
  const int nlocal,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  float* __restrict__ g_descriptors,
  float* __restrict__ g_sum_fxyz_0,
  float* __restrict__ g_sum_fxyz_c,
  float* __restrict__ g_sum_fxyz_Ax,
  float* __restrict__ g_sum_fxyz_Ay,
  float* __restrict__ g_sum_fxyz_Az,
  float* __restrict__ g_sum_fxyz_D)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  const int nspin3 = nep_spin_3body_count(paramb);
  const int abc_count = nep_spin_3body_abc_count(paramb);
  const float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
  int bs = paramb.basis_size_spin_angular;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int n = 0; n < nspin3; ++n) {
    float s0[MAX_SPIN_ABC] = {0.0f};
    float sc[MAX_SPIN_ABC] = {0.0f};
    float Ax[MAX_SPIN_ABC] = {0.0f};
    float Ay[MAX_SPIN_ABC] = {0.0f};
    float Az[MAX_SPIN_ABC] = {0.0f};
    float D[MAX_SPIN_ABC] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      const int index = i1 * N + n1;
      const int n2 = g_NL[index];
      const int t2 = g_type[n2];
      const float x12 = g_x12[index];
      const float y12 = g_y12[index];
      const float z12 = g_z12[index];
      const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
      if (!(d12 > 0.0f)) continue;
      const float rc = (paramb.rc_angular_by_type[t1] + paramb.rc_angular_by_type[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(bs, rcinv, d12, fc12, fn12);
      float gn = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        gn += fn12[k] * annmb.c[nep_spin_c_index_3body(paramb, n, k, t1, t2)];
      }
      const float sjx = __ldg(&g_spin[n2]);
      const float sjy = __ldg(&g_spin[n2 + N]);
      const float sjz = __ldg(&g_spin[n2 + N * 2]);
      const float gn_x_si_dot_sj = gn * (si[0] * sjx + si[1] * sjy + si[2] * sjz);
      const float d12inv = 1.0f / d12;
      const float rhat[3] = {x12 * d12inv, y12 * d12inv, z12 * d12inv};
      const float cross[3] = {
        si[1] * sjz - si[2] * sjy,
        si[2] * sjx - si[0] * sjz,
        si[0] * sjy - si[1] * sjx};
      const float phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
      accumulate_spin3body_all(
        paramb.l_max_spin_angular,
        d12,
        x12,
        y12,
        z12,
        gn,
        gn_x_si_dot_sj,
        gn * sjx,
        gn * sjy,
        gn * sjz,
        gn * phi_dmi,
        s0,
        sc,
        Ax,
        Ay,
        Az,
        D);
    }
    for (int abc = 0; abc < abc_count; ++abc) {
      const int offset = (n * abc_count + abc) * N + n1;
      g_sum_fxyz_0[offset] = s0[abc];
      g_sum_fxyz_c[offset] = sc[abc];
      g_sum_fxyz_Ax[offset] = Ax[abc];
      g_sum_fxyz_Ay[offset] = Ay[abc];
      g_sum_fxyz_Az[offset] = Az[abc];
      g_sum_fxyz_D[offset] = D[abc];
    }
    for (int L = 1; L <= paramb.l_max_spin_angular; ++L) {
      float q2 = 0.0f, q3 = 0.0f, q4 = 0.0f, qD0 = 0.0f, qDc = 0.0f;
      if (L == 1) {
        q2 = find_q_one<1>(sc);
        q3 = find_q_one<1>(Ax) + find_q_one<1>(Ay) + find_q_one<1>(Az);
        q4 = find_q_cross_one<1>(s0, sc);
        qD0 = find_q_cross_one<1>(s0, D);
        qDc = find_q_cross_one<1>(sc, D);
      } else if (L == 2) {
        q2 = find_q_one<2>(sc);
        q3 = find_q_one<2>(Ax) + find_q_one<2>(Ay) + find_q_one<2>(Az);
        q4 = find_q_cross_one<2>(s0, sc);
        qD0 = find_q_cross_one<2>(s0, D);
        qDc = find_q_cross_one<2>(sc, D);
      } else if (L == 3) {
        q2 = find_q_one<3>(sc);
        q3 = find_q_one<3>(Ax) + find_q_one<3>(Ay) + find_q_one<3>(Az);
        q4 = find_q_cross_one<3>(s0, sc);
        qD0 = find_q_cross_one<3>(s0, D);
        qDc = find_q_cross_one<3>(sc, D);
      } else if (L == 4) {
        q2 = find_q_one<4>(sc);
        q3 = find_q_one<4>(Ax) + find_q_one<4>(Ay) + find_q_one<4>(Az);
        q4 = find_q_cross_one<4>(s0, sc);
        qD0 = find_q_cross_one<4>(s0, D);
        qDc = find_q_cross_one<4>(sc, D);
      }
      g_descriptors[n1 + nep_spin_block2_core_index(paramb, 0, n, L - 1) * N] = q2;
      g_descriptors[n1 + nep_spin_block2_core_index(paramb, 1, n, L - 1) * N] = q3;
      g_descriptors[n1 + nep_spin_block2_core_index(paramb, 2, n, L - 1) * N] = q4;
      g_descriptors[n1 + nep_spin_block2_core_index(paramb, 3, n, L - 1) * N] = qD0;
      g_descriptors[n1 + nep_spin_block2_core_index(paramb, 4, n, L - 1) * N] = qDc;
    }
    if (paramb.l_max_spin_angular >= 2) {
      float grad_s0_mix[5];
      float grad_sc_mix[5];
      const float q4b = compute_q4b_l2(sc + 3);
      float qmix = 0.0f;
      accumulate_mix_q4b_l2(s0 + 3, sc + 3, qmix, grad_s0_mix, grad_sc_mix);
      g_descriptors[n1 + nep_spin_block2_g1_index(paramb, 0, n) * N] = q4b;
      g_descriptors[n1 + nep_spin_block2_g1_index(paramb, 1, n) * N] = qmix;
    }
  }

  for (int n1_pair = 0; n1_pair < nspin3; ++n1_pair) {
    for (int n2_pair = n1_pair + 1; n2_pair < nspin3; ++n2_pair) {
      const int pair_index = nep_spin_block2_pair_lex_index(paramb, n1_pair, n2_pair);
      if (pair_index < 0) continue;
      for (int L = 1; L <= paramb.l_max_spin_angular; ++L) {
        float qAcross = 0.0f;
        const int start = L * L - 1;
        const int terms = 2 * L + 1;
        for (int k = 0; k < terms; ++k) {
          const int abc = start + k;
          const float weight = (k == 0 ? 1.0f : 2.0f) * C3B[abc];
          const int idx1 = (n1_pair * abc_count + abc) * N + n1;
          const int idx2 = (n2_pair * abc_count + abc) * N + n1;
          qAcross += weight *
                     (g_sum_fxyz_Ax[idx1] * g_sum_fxyz_Ax[idx2] +
                      g_sum_fxyz_Ay[idx1] * g_sum_fxyz_Ay[idx2] +
                      g_sum_fxyz_Az[idx1] * g_sum_fxyz_Az[idx2]);
        }
        g_descriptors[n1 + nep_spin_block2_across_index(paramb, n1_pair, n2_pair, L - 1) * N] = qAcross;
      }
    }
  }
}

static __global__ void zero_spin_descriptor_block(
  const int N, const int spin_dim, float* g_descriptors, const int spin_offset)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  for (int d = 0; d < spin_dim; ++d) {
    g_descriptors[n1 + (spin_offset + d) * N] = 0.0f;
  }
}

static __global__ void zero_spin_sum_block(const int size, float* data)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) data[idx] = 0.0f;
}

static __global__ void apply_ann_spin(
  const int N,
  const int nlocal,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal) return;
  const int type = g_type[n1];
  float q[MAX_DIM];
  float Fp[MAX_DIM] = {0.0f};
  for (int d = 0; d < annmb.dim; ++d) {
    q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
  }
  float F = 0.0f;
  apply_ann_one_layer(
    annmb.dim,
    annmb.num_neurons1,
    annmb.w0[type],
    annmb.b0[type],
    annmb.w1[type],
    annmb.b1,
    q,
    F,
    Fp);
  g_pe[n1] = F;
  for (int d = 0; d < annmb.dim; ++d) {
    g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
  }
}

static __global__ void find_mforce_spin_onsite(
  const int N,
  const int nlocal,
  const NEP_Spin::ParaMB paramb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* __restrict__ g_mx,
  double* __restrict__ g_my,
  double* __restrict__ g_mz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal || paramb.spin_pmax <= 0) return;
  const int t1 = g_type[n1];
  const float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
  const float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= 1.0e-12f) return;
  const float msign = paramb.mforce_sign;
  const int offset = nep_spin_block0_offset(paramb);
  float mx = 0.0f, my = 0.0f, mz = 0.0f;
  if (paramb.spin_onsite_basis_mode == 0) {
    float m2pow = 1.0f;
    for (int p = 1; p <= paramb.spin_pmax; ++p) {
      const float Fp_p = __ldg(&g_Fp[n1 + (offset + p - 1) * N]);
      const float coeff = msign * Fp_p * (2.0f * p) * m2pow;
      mx += coeff * si[0];
      my += coeff * si[1];
      mz += coeff * si[2];
      m2pow *= si2;
    }
  } else {
    float y = si2;
    float yref = nep_spin_type_yref(t1, paramb.spin_onsite_basis_mode);
    const float si_norm = sqrtf(si2);
    const float inv_si_norm = 1.0f / (si_norm + 1.0e-12f);
    if (paramb.spin_onsite_basis_mode == 2) {
      y = si_norm;
    }
    if (yref <= 0.0f) yref = 1.0f;
    const float denom = y + yref;
    const float inv_denom = 1.0f / (denom + 1.0e-12f);
    float x = (y - yref) * inv_denom;
    x = fminf(1.0f, fmaxf(-1.0f, x));
    const float dx_dy = (2.0f * yref) * inv_denom * inv_denom;
    float dy_dsi[3] = {2.0f * si[0], 2.0f * si[1], 2.0f * si[2]};
    if (paramb.spin_onsite_basis_mode == 2) {
      dy_dsi[0] = si[0] * inv_si_norm;
      dy_dsi[1] = si[1] * inv_si_norm;
      dy_dsi[2] = si[2] * inv_si_norm;
    }
    float Tp[9] = {1.0f};
    float dTp[9] = {0.0f};
    if (paramb.spin_pmax >= 1) {
      Tp[1] = x;
      dTp[1] = 1.0f;
    }
    for (int p = 2; p <= paramb.spin_pmax; ++p) {
      Tp[p] = 2.0f * x * Tp[p - 1] - Tp[p - 2];
      dTp[p] = 2.0f * Tp[p - 1] + 2.0f * x * dTp[p - 1] - dTp[p - 2];
    }
    for (int p = 1; p <= paramb.spin_pmax; ++p) {
      const float Fp_p = __ldg(&g_Fp[n1 + (offset + p - 1) * N]);
      const float coeff = msign * Fp_p * dTp[p] * dx_dy;
      mx += coeff * dy_dsi[0];
      my += coeff * dy_dsi[1];
      mz += coeff * dy_dsi[2];
    }
  }
  atomic_add_force(&g_mx[n1], static_cast<double>(mx));
  atomic_add_force(&g_my[n1], static_cast<double>(my));
  atomic_add_force(&g_mz[n1], static_cast<double>(mz));
}

static __global__ void find_force_spin_2body(
  const int N,
  const int nlocal,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_mx,
  double* g_my,
  double* g_mz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  const int nspin = nep_spin_2body_count(paramb);
  const float si[3] = {__ldg(&g_spin[n1]), __ldg(&g_spin[n1 + N]), __ldg(&g_spin[n1 + N * 2])};
  const float msign = paramb.mforce_sign;
  float fi_acc_x = 0.0f, fi_acc_y = 0.0f, fi_acc_z = 0.0f;
  float v_xx = 0.0f, v_yy = 0.0f, v_zz = 0.0f, v_xy = 0.0f, v_yz = 0.0f, v_zx = 0.0f;
  float mi_x = 0.0f, mi_y = 0.0f, mi_z = 0.0f;
  int bs = paramb.basis_size_spin_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = i1 * N + n1;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (!(d12 > 0.0f)) continue;
    const float inv_d12 = 1.0f / d12;
    const float rhat[3] = {r12[0] * inv_d12, r12[1] * inv_d12, r12[2] * inv_d12};
    const float sj[3] = {__ldg(&g_spin[n2]), __ldg(&g_spin[n2 + N]), __ldg(&g_spin[n2 + N * 2])};
    const float si_dot_sj = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
    const float si_dot_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const float sj_dot_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
    const float cross[3] = {
      si[1] * sj[2] - si[2] * sj[1],
      si[2] * sj[0] - si[0] * sj[2],
      si[0] * sj[1] - si[1] * sj[0]};
    const float phi_ex = si_dot_sj;
    const float phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
    const float phi_ani = si_dot_r * sj_dot_r;
    const float phi_sia = si_dot_r * si_dot_r;
    const float rc = (paramb.rc_radial_by_type[t1] + paramb.rc_radial_by_type[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
    float f12[3] = {0.0f, 0.0f, 0.0f};
    float mj_x = 0.0f, mj_y = 0.0f, mj_z = 0.0f;
    for (int n = 0; n < nspin; ++n) {
      float gn = 0.0f, gnp = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        const float c = annmb.c[nep_spin_c_index_2body(paramb, n, k, t1, t2)];
        gn += fn12[k] * c;
        gnp += fnp12[k] * c;
      }
      const float Fex = __ldg(&g_Fp[n1 + nep_spin_block1_index(paramb, 0, n) * N]);
      const float Fdmi = __ldg(&g_Fp[n1 + nep_spin_block1_index(paramb, 1, n) * N]);
      const float Fani = __ldg(&g_Fp[n1 + nep_spin_block1_index(paramb, 2, n) * N]);
      const float Fsia = __ldg(&g_Fp[n1 + nep_spin_block1_index(paramb, 3, n) * N]);
      for (int d = 0; d < 3; ++d) {
        f12[d] += Fex * (phi_ex * gnp * rhat[d]);
      }
      if (Fdmi != 0.0f) {
        for (int d = 0; d < 3; ++d) {
          f12[d] += Fdmi * (gn * (cross[d] - phi_dmi * rhat[d]) * inv_d12 + phi_dmi * gnp * rhat[d]);
        }
      }
      if (Fani != 0.0f) {
        const float vec_ani[3] = {
          si_dot_r * (sj[0] - sj_dot_r * rhat[0]) + sj_dot_r * (si[0] - si_dot_r * rhat[0]),
          si_dot_r * (sj[1] - sj_dot_r * rhat[1]) + sj_dot_r * (si[1] - si_dot_r * rhat[1]),
          si_dot_r * (sj[2] - sj_dot_r * rhat[2]) + sj_dot_r * (si[2] - si_dot_r * rhat[2])};
        for (int d = 0; d < 3; ++d) {
          f12[d] += Fani * (gn * vec_ani[d] * inv_d12 + phi_ani * gnp * rhat[d]);
        }
      }
      if (Fsia != 0.0f) {
        const float vec_sia[3] = {
          2.0f * si_dot_r * (si[0] - si_dot_r * rhat[0]),
          2.0f * si_dot_r * (si[1] - si_dot_r * rhat[1]),
          2.0f * si_dot_r * (si[2] - si_dot_r * rhat[2])};
        for (int d = 0; d < 3; ++d) {
          f12[d] += Fsia * (gn * vec_sia[d] * inv_d12 + phi_sia * gnp * rhat[d]);
        }
      }
      mi_x += msign * (Fex * gn * sj[0] + Fdmi * gn * (sj[1] * rhat[2] - sj[2] * rhat[1]) +
                       Fani * gn * sj_dot_r * rhat[0] + Fsia * gn * 2.0f * si_dot_r * rhat[0]);
      mi_y += msign * (Fex * gn * sj[1] + Fdmi * gn * (sj[2] * rhat[0] - sj[0] * rhat[2]) +
                       Fani * gn * sj_dot_r * rhat[1] + Fsia * gn * 2.0f * si_dot_r * rhat[1]);
      mi_z += msign * (Fex * gn * sj[2] + Fdmi * gn * (sj[0] * rhat[1] - sj[1] * rhat[0]) +
                       Fani * gn * sj_dot_r * rhat[2] + Fsia * gn * 2.0f * si_dot_r * rhat[2]);
      mj_x += msign * (Fex * gn * si[0] + Fdmi * gn * (rhat[1] * si[2] - rhat[2] * si[1]) +
                       Fani * gn * si_dot_r * rhat[0]);
      mj_y += msign * (Fex * gn * si[1] + Fdmi * gn * (rhat[2] * si[0] - rhat[0] * si[2]) +
                       Fani * gn * si_dot_r * rhat[1]);
      mj_z += msign * (Fex * gn * si[2] + Fdmi * gn * (rhat[0] * si[1] - rhat[1] * si[0]) +
                       Fani * gn * si_dot_r * rhat[2]);
    }
    add_force_and_virial(
      n2, r12, f12, fi_acc_x, fi_acc_y, fi_acc_z, v_xx, v_yy, v_zz, v_xy, v_yz, v_zx, g_fx, g_fy, g_fz);
    atomic_add_force(&g_mx[n2], static_cast<double>(mj_x));
    atomic_add_force(&g_my[n2], static_cast<double>(mj_y));
    atomic_add_force(&g_mz[n2], static_cast<double>(mj_z));
  }
  if (neighbor_number > 0) {
    atomic_add_force(&g_fx[n1], static_cast<double>(fi_acc_x));
    atomic_add_force(&g_fy[n1], static_cast<double>(fi_acc_y));
    atomic_add_force(&g_fz[n1], static_cast<double>(fi_acc_z));
  }
  atomic_add_force(&g_mx[n1], static_cast<double>(mi_x));
  atomic_add_force(&g_my[n1], static_cast<double>(mi_y));
  atomic_add_force(&g_mz[n1], static_cast<double>(mi_z));
  g_virial[n1 + N * 0] += static_cast<double>(v_xx);
  g_virial[n1 + N * 1] += static_cast<double>(v_yy);
  g_virial[n1 + N * 2] += static_cast<double>(v_zz);
  g_virial[n1 + N * 3] += static_cast<double>(v_xy);
  g_virial[n1 + N * 4] += static_cast<double>(v_yz);
  g_virial[n1 + N * 5] += static_cast<double>(v_zx);
}

static __global__ void find_force_spin_3body(
  const int N,
  const int nlocal,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz_0,
  const float* __restrict__ g_sum_fxyz_c,
  const float* __restrict__ g_sum_fxyz_Ax,
  const float* __restrict__ g_sum_fxyz_Ay,
  const float* __restrict__ g_sum_fxyz_Az,
  const float* __restrict__ g_sum_fxyz_D,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_mx,
  double* g_my,
  double* g_mz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  const int nspin3 = nep_spin_3body_count(paramb);
  const float si[3] = {
    __ldg(&g_spin[n1]),
    __ldg(&g_spin[n1 + N]),
    __ldg(&g_spin[n1 + N * 2])};
  const float msign = paramb.mforce_sign;
  float fi_acc_x = 0.0f, fi_acc_y = 0.0f, fi_acc_z = 0.0f;
  float v_xx = 0.0f, v_yy = 0.0f, v_zz = 0.0f, v_xy = 0.0f, v_yz = 0.0f, v_zx = 0.0f;
  float mi_x = 0.0f, mi_y = 0.0f, mi_z = 0.0f;
  const int sum_stride = nep_spin_3body_abc_count(paramb);
  const int g1_count = nep_spin_block2_g1_count(paramb);
  int bs = paramb.basis_size_spin_angular;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int n = 0; n < nspin3; ++n) {
    float dEs0_1[3] = {0.0f}, dEsc_1[3] = {0.0f}, dEAx_1[3] = {0.0f}, dEAy_1[3] = {0.0f},
          dEAz_1[3] = {0.0f}, dED_1[3] = {0.0f};
    float dEs0_2[5] = {0.0f}, dEsc_2[5] = {0.0f}, dEAx_2[5] = {0.0f}, dEAy_2[5] = {0.0f},
          dEAz_2[5] = {0.0f}, dED_2[5] = {0.0f};
    float dEs0_3[7] = {0.0f}, dEsc_3[7] = {0.0f}, dEAx_3[7] = {0.0f}, dEAy_3[7] = {0.0f},
          dEAz_3[7] = {0.0f}, dED_3[7] = {0.0f};
    float dEs0_4[9] = {0.0f}, dEsc_4[9] = {0.0f}, dEAx_4[9] = {0.0f}, dEAy_4[9] = {0.0f},
          dEAz_4[9] = {0.0f}, dED_4[9] = {0.0f};

    auto fill_one_L = [&](const int L,
                          float* dEs0L,
                          float* dEscL,
                          float* dEAxL,
                          float* dEAyL,
                          float* dEAzL,
                          float* dEDL) {
      const float G2 = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 0, n, L - 1) * N]);
      const float G3 = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 1, n, L - 1) * N]);
      const float G4 = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 2, n, L - 1) * N]);
      const float GD0 = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 3, n, L - 1) * N]);
      const float GDc = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 4, n, L - 1) * N]);
      const int start = L * L - 1;
      const int terms = 2 * L + 1;
      for (int k = 0; k < terms; ++k) {
        const int abc = start + k;
        const float weight = (k == 0 ? 1.0f : 2.0f) * C3B[abc];
        const int idx = (n * sum_stride + abc) * N + n1;
        const float s0 = g_sum_fxyz_0[idx];
        const float sc = g_sum_fxyz_c[idx];
        const float Ax = g_sum_fxyz_Ax[idx];
        const float Ay = g_sum_fxyz_Ay[idx];
        const float Az = g_sum_fxyz_Az[idx];
        const float D = g_sum_fxyz_D[idx];
        dEs0L[k] = G4 * weight * sc + GD0 * weight * D;
        dEscL[k] = 2.0f * G2 * weight * sc + G4 * weight * s0 + GDc * weight * D;
        dEAxL[k] = 2.0f * G3 * weight * Ax;
        dEAyL[k] = 2.0f * G3 * weight * Ay;
        dEAzL[k] = 2.0f * G3 * weight * Az;
        dEDL[k] = GD0 * weight * s0 + GDc * weight * sc;
      }
      if (L == 2 && g1_count > 0) {
        const float G4b = __ldg(&g_Fp[n1 + nep_spin_block2_g1_index(paramb, 0, n) * N]);
        const float Gmix = __ldg(&g_Fp[n1 + nep_spin_block2_g1_index(paramb, 1, n) * N]);
        if (G4b != 0.0f || Gmix != 0.0f) {
          float sc2[5];
          float s02[5];
          float grad_q4b[5];
          float grad_s0_mix[5];
          float grad_sc_mix[5];
          for (int k = 0; k < 5; ++k) {
            const int idx = (n * sum_stride + 3 + k) * N + n1;
            sc2[k] = g_sum_fxyz_c[idx];
            s02[k] = g_sum_fxyz_0[idx];
          }
          compute_grad_q4b_l2(sc2, grad_q4b);
          float qmix_dummy = 0.0f;
          accumulate_mix_q4b_l2(s02, sc2, qmix_dummy, grad_s0_mix, grad_sc_mix);
          for (int k = 0; k < 5; ++k) {
            dEscL[k] += G4b * grad_q4b[k] + Gmix * grad_sc_mix[k];
            dEs0L[k] += Gmix * grad_s0_mix[k];
          }
        }
      }
    };
    if (paramb.l_max_spin_angular >= 1) fill_one_L(1, dEs0_1, dEsc_1, dEAx_1, dEAy_1, dEAz_1, dED_1);
    if (paramb.l_max_spin_angular >= 2) fill_one_L(2, dEs0_2, dEsc_2, dEAx_2, dEAy_2, dEAz_2, dED_2);
    if (paramb.l_max_spin_angular >= 3) fill_one_L(3, dEs0_3, dEsc_3, dEAx_3, dEAy_3, dEAz_3, dED_3);
    if (paramb.l_max_spin_angular >= 4) fill_one_L(4, dEs0_4, dEsc_4, dEAx_4, dEAy_4, dEAz_4, dED_4);

    for (int other = 0; other < nspin3; ++other) {
      if (other == n) continue;
      const int low = other < n ? other : n;
      const int high = other < n ? n : other;
      for (int L = 1; L <= paramb.l_max_spin_angular; ++L) {
        const float GAcross = __ldg(&g_Fp[n1 + nep_spin_block2_across_index(paramb, low, high, L - 1) * N]);
        if (GAcross == 0.0f) continue;
        const int start = L * L - 1;
        const int terms = 2 * L + 1;
        for (int k = 0; k < terms; ++k) {
          const int abc = start + k;
          const float weight = (k == 0 ? 1.0f : 2.0f) * C3B[abc];
          const int idx = (other * sum_stride + abc) * N + n1;
          const float Ax_other = g_sum_fxyz_Ax[idx];
          const float Ay_other = g_sum_fxyz_Ay[idx];
          const float Az_other = g_sum_fxyz_Az[idx];
          if (L == 1) {
            dEAx_1[k] += GAcross * weight * Ax_other;
            dEAy_1[k] += GAcross * weight * Ay_other;
            dEAz_1[k] += GAcross * weight * Az_other;
          } else if (L == 2) {
            dEAx_2[k] += GAcross * weight * Ax_other;
            dEAy_2[k] += GAcross * weight * Ay_other;
            dEAz_2[k] += GAcross * weight * Az_other;
          } else if (L == 3) {
            dEAx_3[k] += GAcross * weight * Ax_other;
            dEAy_3[k] += GAcross * weight * Ay_other;
            dEAz_3[k] += GAcross * weight * Az_other;
          } else if (L == 4) {
            dEAx_4[k] += GAcross * weight * Ax_other;
            dEAy_4[k] += GAcross * weight * Ay_other;
            dEAz_4[k] += GAcross * weight * Az_other;
          }
        }
      }
    }

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      const int index = i1 * N + n1;
      const int n2 = g_NL[index];
      const int t2 = g_type[n2];
      const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      if (!(d12 > 0.0f)) continue;
      const float rc = (paramb.rc_angular_by_type[t1] + paramb.rc_angular_by_type[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12, fcp12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
      float gn = 0.0f, gnp = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        const float c = annmb.c[nep_spin_c_index_3body(paramb, n, k, t1, t2)];
        gn += fn12[k] * c;
        gnp += fnp12[k] * c;
      }
      const float sj[3] = {
        __ldg(&g_spin[n2]),
        __ldg(&g_spin[n2 + N]),
        __ldg(&g_spin[n2 + N * 2])};
      const float si_dot_sj = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
      const float cross[3] = {
        si[1] * sj[2] - si[2] * sj[1],
        si[2] * sj[0] - si[0] * sj[2],
        si[0] * sj[1] - si[1] * sj[0]};
      const float d12inv = 1.0f / d12;
      const float rhat[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};
      const float phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
      float f12[3] = {0.0f, 0.0f, 0.0f};
      float projc = 0.0f, projD = 0.0f, projAx = 0.0f, projAy = 0.0f, projAz = 0.0f;
      if (paramb.l_max_spin_angular >= 1) {
        accumulate_spin3body_force_one_L<1>(
          d12inv, gn, gnp, si_dot_sj, phi_dmi, sj, rhat, dEs0_1, dEsc_1, dEAx_1, dEAy_1, dEAz_1, dED_1,
          f12, projc, projD, projAx, projAy, projAz);
      }
      if (paramb.l_max_spin_angular >= 2) {
        accumulate_spin3body_force_one_L<2>(
          d12inv, gn, gnp, si_dot_sj, phi_dmi, sj, rhat, dEs0_2, dEsc_2, dEAx_2, dEAy_2, dEAz_2, dED_2,
          f12, projc, projD, projAx, projAy, projAz);
      }
      if (paramb.l_max_spin_angular >= 3) {
        accumulate_spin3body_force_one_L<3>(
          d12inv, gn, gnp, si_dot_sj, phi_dmi, sj, rhat, dEs0_3, dEsc_3, dEAx_3, dEAy_3, dEAz_3, dED_3,
          f12, projc, projD, projAx, projAy, projAz);
      }
      if (paramb.l_max_spin_angular >= 4) {
        accumulate_spin3body_force_one_L<4>(
          d12inv, gn, gnp, si_dot_sj, phi_dmi, sj, rhat, dEs0_4, dEsc_4, dEAx_4, dEAy_4, dEAz_4, dED_4,
          f12, projc, projD, projAx, projAy, projAz);
      }
      if (projD != 0.0f) {
        for (int d = 0; d < 3; ++d) {
          f12[d] += gn * projD * (cross[d] - phi_dmi * rhat[d]) * d12inv;
        }
      }
      add_force_and_virial(
        n2, r12, f12, fi_acc_x, fi_acc_y, fi_acc_z, v_xx, v_yy, v_zz, v_xy, v_yz, v_zx, g_fx, g_fy, g_fz);
      const float cross_sj_r[3] = {
        sj[1] * rhat[2] - sj[2] * rhat[1],
        sj[2] * rhat[0] - sj[0] * rhat[2],
        sj[0] * rhat[1] - sj[1] * rhat[0]};
      const float cross_r_si[3] = {
        rhat[1] * si[2] - rhat[2] * si[1],
        rhat[2] * si[0] - rhat[0] * si[2],
        rhat[0] * si[1] - rhat[1] * si[0]};
      mi_x += msign * gn * projc * sj[0];
      mi_y += msign * gn * projc * sj[1];
      mi_z += msign * gn * projc * sj[2];
      mi_x += msign * gn * projD * cross_sj_r[0];
      mi_y += msign * gn * projD * cross_sj_r[1];
      mi_z += msign * gn * projD * cross_sj_r[2];
      atomic_add_force(&g_mx[n2], static_cast<double>(msign * gn * (projc * si[0] + projAx + projD * cross_r_si[0])));
      atomic_add_force(&g_my[n2], static_cast<double>(msign * gn * (projc * si[1] + projAy + projD * cross_r_si[1])));
      atomic_add_force(&g_mz[n2], static_cast<double>(msign * gn * (projc * si[2] + projAz + projD * cross_r_si[2])));
    }
  }
  if (neighbor_number > 0) {
    atomic_add_force(&g_fx[n1], static_cast<double>(fi_acc_x));
    atomic_add_force(&g_fy[n1], static_cast<double>(fi_acc_y));
    atomic_add_force(&g_fz[n1], static_cast<double>(fi_acc_z));
  }
  atomic_add_force(&g_mx[n1], static_cast<double>(mi_x));
  atomic_add_force(&g_my[n1], static_cast<double>(mi_y));
  atomic_add_force(&g_mz[n1], static_cast<double>(mi_z));
  g_virial[n1 + N * 0] += static_cast<double>(v_xx);
  g_virial[n1 + N * 1] += static_cast<double>(v_yy);
  g_virial[n1 + N * 2] += static_cast<double>(v_zz);
  g_virial[n1 + N * 3] += static_cast<double>(v_xy);
  g_virial[n1 + N * 4] += static_cast<double>(v_yz);
  g_virial[n1 + N * 5] += static_cast<double>(v_zx);
}

// Fallback atomicAdd(double*) for pre-Pascal architectures
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __inline__ double atomicAdd(double* address, double val)
{
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// Local periodic table symbols used to map header element names
// to atomic numbers (Z-1 index into COVALENT_RADIUS).
static const char* kElementSymbols[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

// Zero magnetic force buffer
static __global__ void zero_mforce_spin(const int N, double* g_mx, double* g_my, double* g_mz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_mx[n1] = 0.0;
    g_my[n1] = 0.0;
    g_mz[n1] = 0.0;
  }
}

// Per-atom ANN application helper: takes pre-scaled descriptors q[d].
__device__ void apply_ann_spin_one_atom(
  int n1,
  int N,
  const NEP_Spin::ParaMB& paramb,
  const NEP_Spin::ANN& annmb,
  const int* __restrict__ g_type,
  float* __restrict__ q,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
{
  (void)paramb;

  int type = g_type[n1];
  float F = 0.0f;
  float Fp_loc[MAX_DIM];
  for (int d = 0; d < annmb.dim; ++d) {
    Fp_loc[d] = 0.0f;
  }

  apply_ann_one_layer(
    annmb.dim,
    annmb.num_neurons1,
    annmb.w0[type],
    annmb.b0[type],
    annmb.w1[type],
    annmb.b1,
    q,
    F,
    Fp_loc);

  g_pe[n1] = static_cast<double>(F);

  for (int d = 0; d < annmb.dim; ++d) {
    g_Fp[n1 + d * N] = Fp_loc[d] * g_q_scaler[d];
  }
}

// Templated variant to keep the Fp scratchpad sized to the active descriptor bucket
// (avoid Fp_loc[MAX_DIM] local memory for dim<=128).
template <int kMaxDim>
__device__ __forceinline__ void apply_ann_spin_one_atom_tmpl(
  int n1,
  int N,
  const NEP_Spin::ParaMB& paramb,
  const NEP_Spin::ANN& annmb,
  const int* __restrict__ g_type,
  float* __restrict__ q,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
{
  (void)paramb;

  int type = g_type[n1];
  float F = 0.0f;
  float Fp_loc[kMaxDim];
  for (int d = 0; d < annmb.dim; ++d) {
    Fp_loc[d] = 0.0f;
  }

  apply_ann_one_layer(
    annmb.dim,
    annmb.num_neurons1,
    annmb.w0[type],
    annmb.b0[type],
    annmb.w1[type],
    annmb.b1,
    q,
    F,
    Fp_loc);

  g_pe[n1] = static_cast<double>(F);
  for (int d = 0; d < annmb.dim; ++d) {
    g_Fp[n1 + d * N] = Fp_loc[d] * g_q_scaler[d];
  }
}

template <bool kDoAnn>
struct NepSpinApplyAnn;

template <>
struct NepSpinApplyAnn<true> {
  __device__ __forceinline__ static void run(
    int n1,
    int N,
    const NEP_Spin::ParaMB& paramb,
    const NEP_Spin::ANN& annmb,
    const int* __restrict__ g_type,
    float* __restrict__ q,
    const float* __restrict__ g_q_scaler,
    double* g_pe,
    float* g_Fp)
  {
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] *= g_q_scaler[d];
    }
    apply_ann_spin_one_atom(n1, N, paramb, annmb, g_type, q, g_q_scaler, g_pe, g_Fp);
  }
};

template <>
struct NepSpinApplyAnn<false> {
  __device__ __forceinline__ static void run(
    int /*n1*/,
    int /*N*/,
    const NEP_Spin::ParaMB& /*paramb*/,
    const NEP_Spin::ANN& /*annmb*/,
    const int* __restrict__ /*g_type*/,
    float* __restrict__ /*q*/,
    const float* __restrict__ /*g_q_scaler*/,
    double* /*g_pe*/,
    float* /*g_Fp*/)
  {
  }
};

// ----------------------------------------------------------------------
// Small-box: fused baseline + spin descriptors + ANN.
template <bool kDoAnn, int kMaxDim>
static __device__ __forceinline__ void compute_all_q_and_ann_small_md_body(
  int N,
  int N1,
  int N2,
  const int* g_NN_r,
  const int* g_NL_r,
  const int* g_NN_a,
  const int* g_NL_a,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_r,
  const float* __restrict__ g_y12_r,
  const float* __restrict__ g_z12_r,
  const float* __restrict__ g_x12_a,
  const float* __restrict__ g_y12_a,
  const float* __restrict__ g_z12_a,
  const float* __restrict__ g_spin,
  float* g_descriptors,
  float* g_sum_fxyz,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp,
  float* __restrict__ fn12_ws)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) {
    return;
  }

  const int t1 = g_type[n1];

  float q[kMaxDim];
  for (int d = 0; d < annmb.dim; ++d) q[d] = 0.0f;
  float* __restrict__ fn12 = fn12_ws;

  // ------------------------------------------------------------------
  // Baseline radial descriptors (small-box, using stored x12_r)
  // ------------------------------------------------------------------
  int neighbor_number_r = g_NN_r[n1];
  int bs_rad = paramb.basis_size_radial;
  if (bs_rad >= MAX_NUM_N) {
    bs_rad = MAX_NUM_N - 1;
  }

  // Spin model parameters (kmax-driven)
  const bool enable_spin = (g_spin != nullptr);
  const int nspin = nep_spin_nspin(paramb);
  const int radial_offset = paramb.n_max_radial + 1;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  const int kmax_ex = spin_blocks.kmax_ex;
  const int kmax_dmi = spin_blocks.kmax_dmi;
  const int kmax_ani = spin_blocks.kmax_ani;
  const int kmax_sia = spin_blocks.kmax_sia;

  const int ex_blocks = spin_blocks.ex_blocks;
  const int dmi_blocks = spin_blocks.dmi_blocks;
  const int ani_blocks = spin_blocks.ani_blocks;
  const int sia_blocks = spin_blocks.sia_blocks;

  const int dmi_block0 = spin_blocks.dmi_block0;
  const int ani_block0 = spin_blocks.ani_block0;
  const int sia_block0 = spin_blocks.sia_block0;
  const int pair_blocks = spin_blocks.pair_blocks;

  const int kmax_pair = spin_blocks.kmax_pair;

  const int spin_pmax = nep_spin_clamp_pmax(paramb.spin_pmax);
  const int spin_offset_loc = radial_offset + paramb.dim_angular;
  const int onsite_offset_loc = spin_offset_loc + nspin * pair_blocks;

  float si[3] = {0.0f, 0.0f, 0.0f};
  float si2 = 0.0f;
  if (enable_spin) {
    si[0] = g_spin[n1];
    si[1] = g_spin[n1 + N];
    si[2] = g_spin[n1 + 2 * N];
    si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  }
  bool central_has_spin = enable_spin && (si2 > kSpinZeroEpsSph);
  float si_norm = central_has_spin ? sqrtf(si2) : 0.0f;

  constexpr int kMaxK = 8;

  const SpinCMode mode_spin =
    enable_spin ? nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride) : SPIN_C_SHARED_LATTICE;

  const int stride_n = (bs_rad + 1) * paramb.num_types_sq;
  const int stride_k = paramb.num_types_sq;
  int c_step = 0;
  const float* c_base_init = annmb.c;
  if (mode_spin != SPIN_C_SHARED_LATTICE) {
    c_base_init += paramb.c_spin_offset;
    if (mode_spin == SPIN_C_PER_BLOCK) c_step = paramb.c_spin_block_stride;
  }
  const float* c_base_t1 = c_base_init + t1 * paramb.num_types;

  for (int i1 = 0; i1 < neighbor_number_r; ++i1) {
    int index = n1 + N * i1;
    int n2 = g_NL_r[index];
    int t2 = g_type[n2];
    const float* c_base_pair = c_base_t1 + t2;

    float x12 = g_x12_r[index];
    float y12 = g_y12_r[index];
    float z12 = g_z12_r[index];
    float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
    if (d12 <= 0.0f) continue;
    float d12inv = 1.0f / d12;

    float fc12;
    const float rc = (paramb.rc_radial_by_type[t1] + paramb.rc_radial_by_type[t2]) * 0.5f;
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);

    find_fn(bs_rad, rcinv, d12, fc12, fn12);

    // Baseline radial q
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12 = 0.0f;
      for (int k = 0; k <= bs_rad; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      q[n] += gn12;
    }

    // Spin spherical blocks reuse same fn12
    if (central_has_spin && pair_blocks > 0) {
      float rhat[3] = {x12 * d12inv, y12 * d12inv, z12 * d12inv};
      float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];

      float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + 2 * N]};
      float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
      const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);

      float sj_r = 0.0f;
      float dmi = 0.0f;
      float Tk[kMaxK + 1] = {0.0f};
      Tk[0] = 1.0f;
      float ex_invariant[kMaxK + 1] = {0.0f};

      float ani_scalar = 0.0f;
      const float sia_scalar = si_r * si_r;

      if (neighbor_has_spin) {
        sj_r = nep_spin_dot3(sj, rhat);
        const float sdot = nep_spin_dot3(si, sj);

        float sixsj[3];
        nep_spin_cross3(si, sj, sixsj);
        dmi = nep_spin_dot3(sixsj, rhat);

        ani_scalar = si_r * sj_r;

        const float sj_norm = sqrtf(sj2);
        const float denom = si_norm * sj_norm;
        const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));
        nep_spin_fill_Tk<kMaxK>(c, kmax_pair, Tk);

        if (ex_blocks > 0) {
          const float phi = nep_spin_ex_phi(paramb.spin_ex_phi_mode, si_norm, sj_norm, denom);
          nep_spin_fill_ex_invariant<kMaxK>(phi, Tk, kmax_ex, ex_invariant);
        }
      }

      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr = c_base_pair + n * stride_n;
        if (neighbor_has_spin) {
          if (ex_blocks > 0) {
            const float* c_kk_ptr = c_n_ptr;
            for (int kk = 0; kk <= kmax_ex; ++kk) {
              float gn = 0.0f;
              const float* c_kb_ptr = c_kk_ptr;
              for (int k = 0; k <= bs_rad; ++k) {
                gn += fn12[k] * NEP_SPIN_LDG(c_kb_ptr);
                c_kb_ptr += stride_k;
              }
              c_kk_ptr += c_step;
              q[spin_offset_loc + kk * nspin + n] += gn * ex_invariant[kk];
            }
          }
          if (dmi_blocks > 0) {
            const float* c_kk_ptr = c_n_ptr + dmi_block0 * c_step;
            for (int kk = 0; kk <= kmax_dmi; ++kk) {
              const int block = dmi_block0 + kk;
              float gn = 0.0f;
              const float* c_kb_ptr = c_kk_ptr;
              for (int k = 0; k <= bs_rad; ++k) {
                gn += fn12[k] * NEP_SPIN_LDG(c_kb_ptr);
                c_kb_ptr += stride_k;
              }
              c_kk_ptr += c_step;
              q[spin_offset_loc + block * nspin + n] += gn * (dmi * Tk[kk]);
            }
          }
          if (ani_blocks > 0) {
            const float* c_kk_ptr = c_n_ptr + ani_block0 * c_step;
            for (int kk = 0; kk <= kmax_ani; ++kk) {
              const int block = ani_block0 + kk;
              float gn = 0.0f;
              const float* c_kb_ptr = c_kk_ptr;
              for (int k = 0; k <= bs_rad; ++k) {
                gn += fn12[k] * NEP_SPIN_LDG(c_kb_ptr);
                c_kb_ptr += stride_k;
              }
              c_kk_ptr += c_step;
              q[spin_offset_loc + block * nspin + n] += gn * (ani_scalar * Tk[kk]);
            }
          }
        }
        if (sia_blocks > 0) {
          const float* c_kk_ptr = c_n_ptr + sia_block0 * c_step;
          for (int kk = 0; kk <= kmax_sia; ++kk) {
            if (kk > 0 && !neighbor_has_spin) {
              c_kk_ptr += c_step;
              continue;
            }
            const int block = sia_block0 + kk;
            float gn = 0.0f;
            const float* c_kb_ptr = c_kk_ptr;
            for (int k = 0; k <= bs_rad; ++k) {
              gn += fn12[k] * NEP_SPIN_LDG(c_kb_ptr);
              c_kb_ptr += stride_k;
            }
            c_kk_ptr += c_step;
            q[spin_offset_loc + block * nspin + n] += gn * (sia_scalar * Tk[kk]);
          }
        }
      }
    }
  } // end radial loop

  // ------------------------------------------------------------------
  // Baseline angular descriptors (small-box, using x12_a)
  // ------------------------------------------------------------------
  int neighbor_number_a = g_NN_a[n1];
  int bs_ang = paramb.basis_size_angular;
  if (bs_ang >= MAX_NUM_N) {
    bs_ang = MAX_NUM_N - 1;
  }

  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number_a; ++i1) {
      int index = n1 + N * i1;
      int n2 = g_NL_a[index];
      float x12 = g_x12_a[index];
      float y12 = g_y12_a[index];
      float z12 = g_z12_a[index];
      float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
      if (d12 <= 0.0f) continue;
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      find_fn(bs_ang, rcinv, d12, fc12, fn12);
      float gn12 = 0.0f;
      for (int k = 0; k <= bs_ang; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
    }

    find_q(
      paramb.L_max,
      paramb.num_L,
      paramb.n_max_angular + 1,
      n,
      s,
      q + radial_offset);

    for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
      g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
    }
  }

  // ------------------------------------------------------------------
  // On-site longitudinal descriptors (p=1..spin_pmax)
  // ------------------------------------------------------------------
  if (central_has_spin && spin_pmax > 0) {
    int basis_mode = paramb.spin_onsite_basis_mode;
    if (basis_mode == 0) {
      float m2p = si2;
      for (int p = 1; p <= spin_pmax; ++p) {
        q[onsite_offset_loc + (p - 1)] = m2p;
        m2p *= si2;
      }
    } else {
      float y = si2;
      float yref = nep_spin_type_yref(t1, basis_mode);
      if (basis_mode == 2) {
        y = si_norm;
      }
      if (yref <= 0.0f) yref = 1.0f;
      float denom = y + yref;
      float x = (y - yref) / (denom + 1.0e-12f);
      x = fminf(1.0f, fmaxf(-1.0f, x));

      float Tp[kMaxK + 1] = {0.0f};
      Tp[0] = 1.0f;
      if (spin_pmax >= 1) Tp[1] = x;
      for (int p = 2; p <= spin_pmax; ++p) {
        Tp[p] = 2.0f * x * Tp[p - 1] - Tp[p - 2];
      }
      for (int p = 1; p <= spin_pmax; ++p) {
        q[onsite_offset_loc + (p - 1)] = Tp[p];
      }
    }
  }

  // Optionally export descriptors to global buffer for debugging or analysis.
  if (g_descriptors != nullptr) {
    for (int d = 0; d < annmb.dim; ++d) {
      g_descriptors[n1 + d * N] = q[d];
    }
  }

  if (kDoAnn) {
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] *= g_q_scaler[d];
    }
    apply_ann_spin_one_atom_tmpl<kMaxDim>(n1, N, paramb, annmb, g_type, q, g_q_scaler, g_pe, g_Fp);
  }
}

static __global__ void compute_all_q_and_ann_small_md(
  int N,
  int N1,
  int N2,
  const int* g_NN_r,
  const int* g_NL_r,
  const int* g_NN_a,
  const int* g_NL_a,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_r,
  const float* __restrict__ g_y12_r,
  const float* __restrict__ g_z12_r,
  const float* __restrict__ g_x12_a,
  const float* __restrict__ g_y12_a,
  const float* __restrict__ g_z12_a,
  const float* __restrict__ g_spin,
  float* g_descriptors,
  float* g_sum_fxyz,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
{
  extern __shared__ float sh_fn12[];
  float* fn12_ws = sh_fn12 + threadIdx.x * MAX_NUM_N;
  compute_all_q_and_ann_small_md_body<true, MAX_DIM>(
    N,
    N1,
    N2,
    g_NN_r,
    g_NL_r,
    g_NN_a,
    g_NL_a,
    paramb,
    annmb,
    g_type,
    g_x12_r,
    g_y12_r,
    g_z12_r,
    g_x12_a,
    g_y12_a,
    g_z12_a,
    g_spin,
    g_descriptors,
    g_sum_fxyz,
    g_q_scaler,
    g_pe,
    g_Fp,
    fn12_ws);
}

template <int kMaxDim>
static __global__ void compute_all_q_and_ann_small_md_tmpl(
  int N,
  int N1,
  int N2,
  const int* g_NN_r,
  const int* g_NL_r,
  const int* g_NN_a,
  const int* g_NL_a,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_r,
  const float* __restrict__ g_y12_r,
  const float* __restrict__ g_z12_r,
  const float* __restrict__ g_x12_a,
  const float* __restrict__ g_y12_a,
  const float* __restrict__ g_z12_a,
  const float* __restrict__ g_spin,
  float* g_descriptors,
  float* g_sum_fxyz,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
{
  extern __shared__ float sh_fn12[];
  float* fn12_ws = sh_fn12 + threadIdx.x * MAX_NUM_N;
  compute_all_q_and_ann_small_md_body<true, kMaxDim>(
    N,
    N1,
    N2,
    g_NN_r,
    g_NL_r,
    g_NN_a,
    g_NL_a,
    paramb,
    annmb,
    g_type,
    g_x12_r,
    g_y12_r,
    g_z12_r,
    g_x12_a,
    g_y12_a,
    g_z12_a,
    g_spin,
    g_descriptors,
    g_sum_fxyz,
    g_q_scaler,
    g_pe,
    g_Fp,
    fn12_ws);
}

// ----------------------------------------------------------------------
// Baseline NEP forces (radial + angular), double precision
// ----------------------------------------------------------------------
static __global__ void find_force_radial_spinbase_md(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  float* fnp12 = fn12 + MAX_NUM_N;
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  int t1 = g_type[n1];
  const bool gather = (N == (N2 - N1));
  float Fp1_cache[MAX_NUM_N];
  for (int n = 0; n <= paramb.n_max_radial; ++n) {
    Fp1_cache[n] = g_Fp[n1 + n * N];
  }
  // Accumulate this thread's contribution to the central atom in registers and
  // update `n1` once. In scatter mode, other threads may atomicAdd into `n1`
  // when this atom appears as a neighbor, so the final update is atomic.
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    int index = i1 * N + n1;
    int n2 = g_NL[index];
    int t2 = g_type[n2];
    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 == 0.0f) continue;
    float d12inv = 1.0f / d12;
    float f12[3] = {0.0f, 0.0f, 0.0f};
    float f21[3] = {0.0f, 0.0f, 0.0f};

    float fc12, fcp12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = min(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
         COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    int bs = paramb.basis_size_radial;
    if (bs >= MAX_NUM_N) {
      bs = MAX_NUM_N - 1;
    }
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gnp12 = 0.0f;
      float gnp21 = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2;
        gnp12 += fnp12[k] * annmb.c[c_index];
        if (gather) {
          int c_index_21 = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index_21 += t2 * paramb.num_types + t1;
          gnp21 += fnp12[k] * annmb.c[c_index_21];
        }
      }
      float tmp12 = Fp1_cache[n] * gnp12 * d12inv;
      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
      if (gather) {
        float tmp21 = g_Fp[n2 + n * N] * gnp21 * d12inv;
        f21[0] -= tmp21 * r12[0];
        f21[1] -= tmp21 * r12[1];
        f21[2] -= tmp21 * r12[2];
      }
    }

    double s_sxx = 0.0;
    double s_sxy = 0.0;
    double s_sxz = 0.0;
    double s_syx = 0.0;
    double s_syy = 0.0;
    double s_syz = 0.0;
    double s_szx = 0.0;
    double s_szy = 0.0;
    double s_szz = 0.0;

    s_sxx -= r12[0] * f12[0];
    s_syy -= r12[1] * f12[1];
    s_szz -= r12[2] * f12[2];
    s_sxy -= r12[0] * f12[1];
    s_sxz -= r12[0] * f12[2];
    s_syz -= r12[1] * f12[2];
    s_syx -= r12[1] * f12[0];
    s_szx -= r12[2] * f12[0];
    s_szy -= r12[2] * f12[1];

    s_fx += (f12[0] - f21[0]);
    s_fy += (f12[1] - f21[1]);
    s_fz += (f12[2] - f21[2]);
    if (!gather) {
      atomicAdd(&g_fx[n2], static_cast<double>(-f12[0]));
      atomicAdd(&g_fy[n2], static_cast<double>(-f12[1]));
      atomicAdd(&g_fz[n2], static_cast<double>(-f12[2]));
    }

    // Save virial (9 components per atom, unsymmetrized as in NEP small box)
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    atomicAdd(&g_virial[n1 + 0 * N], s_sxx);
    atomicAdd(&g_virial[n1 + 1 * N], s_syy);
    atomicAdd(&g_virial[n1 + 2 * N], s_szz);
    atomicAdd(&g_virial[n1 + 3 * N], s_sxy);
    atomicAdd(&g_virial[n1 + 4 * N], s_sxz);
    atomicAdd(&g_virial[n1 + 5 * N], s_syz);
    atomicAdd(&g_virial[n1 + 6 * N], s_syx);
    atomicAdd(&g_virial[n1 + 7 * N], s_szx);
    atomicAdd(&g_virial[n1 + 8 * N], s_szy);
  }

  if (gather) {
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
  } else {
    atomicAdd(&g_fx[n1], s_fx);
    atomicAdd(&g_fy[n1], s_fy);
    atomicAdd(&g_fz[n1], s_fz);
  }
}

static __global__ void find_force_angular_spinbase_md(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  float* fnp12 = fn12 + MAX_NUM_N;
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  float Fp_loc[MAX_DIM_ANGULAR];
  for (int d = 0; d < paramb.dim_angular; ++d) {
    Fp_loc[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
  }

  const int sum_stride = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  const float* sum_fxyz_base = g_sum_fxyz + n1;

  int t1 = g_type[n1];
  const bool gather = (N == (N2 - N1));
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;

  for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
    int index = i1 * N + n1;
    int n2 = g_NL_angular[index];
    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 == 0.0f) continue;
    float f12[3] = {0.0f, 0.0f, 0.0f};
    float f21[3] = {0.0f, 0.0f, 0.0f};

    float fc12, fcp12;
    int t2 = g_type[n2];
    const float rc = (paramb.rc_angular_by_type[t1] + paramb.rc_angular_by_type[t2]) * 0.5f;
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);

    float Fp2_loc[MAX_DIM_ANGULAR];
    const float* sum_fxyz_base2 = nullptr;
    float r21[3] = {0.0f, 0.0f, 0.0f};
    if (gather) {
      for (int d = 0; d < paramb.dim_angular; ++d) {
        Fp2_loc[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n2];
      }
      sum_fxyz_base2 = g_sum_fxyz + n2;
      r21[0] = -r12[0];
      r21[1] = -r12[1];
      r21[2] = -r12[2];
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;
      float gn21 = 0.0f;
      float gnp21 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
        gnp12 += fnp12[k] * annmb.c[c_index];
        if (gather) {
          const int c_index_21 = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq +
            t2 * paramb.num_types + t1 + paramb.num_c_radial;
          gn21 += fn12[k] * annmb.c[c_index_21];
          gnp21 += fnp12[k] * annmb.c[c_index_21];
        }
      }
      accumulate_f12_packed(
        paramb.L_max,
        paramb.num_L,
        n,
        paramb.n_max_angular + 1,
        d12,
        r12,
        gn12,
        gnp12,
        Fp_loc,
        sum_fxyz_base,
          sum_stride,
          N,
          f12);

      if (gather) {
        accumulate_f12_packed(
          paramb.L_max,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r21,
          gn21,
          gnp21,
          Fp2_loc,
          sum_fxyz_base2,
          sum_stride,
          N,
          f21);
      }
    }

    double s_sxx = 0.0;
    double s_sxy = 0.0;
    double s_sxz = 0.0;
    double s_syx = 0.0;
    double s_syy = 0.0;
    double s_syz = 0.0;
    double s_szx = 0.0;
    double s_szy = 0.0;
    double s_szz = 0.0;

    s_sxx -= r12[0] * f12[0];
    s_syy -= r12[1] * f12[1];
    s_szz -= r12[2] * f12[2];
    s_sxy -= r12[0] * f12[1];
    s_sxz -= r12[0] * f12[2];
    s_syz -= r12[1] * f12[2];
    s_syx -= r12[1] * f12[0];
    s_szx -= r12[2] * f12[0];
    s_szy -= r12[2] * f12[1];

    s_fx += (f12[0] - f21[0]);
    s_fy += (f12[1] - f21[1]);
    s_fz += (f12[2] - f21[2]);
    if (!gather) {
      atomicAdd(&g_fx[n2], static_cast<double>(-f12[0]));
      atomicAdd(&g_fy[n2], static_cast<double>(-f12[1]));
      atomicAdd(&g_fz[n2], static_cast<double>(-f12[2]));
    }

    atomicAdd(&g_virial[n1 + 0 * N], s_sxx);
    atomicAdd(&g_virial[n1 + 1 * N], s_syy);
    atomicAdd(&g_virial[n1 + 2 * N], s_szz);
    atomicAdd(&g_virial[n1 + 3 * N], s_sxy);
    atomicAdd(&g_virial[n1 + 4 * N], s_sxz);
    atomicAdd(&g_virial[n1 + 5 * N], s_syz);
    atomicAdd(&g_virial[n1 + 6 * N], s_syx);
    atomicAdd(&g_virial[n1 + 7 * N], s_szx);
    atomicAdd(&g_virial[n1 + 8 * N], s_szy);
  }

  if (gather) {
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
  } else {
    atomicAdd(&g_fx[n1], s_fx);
    atomicAdd(&g_fy[n1], s_fy);
    atomicAdd(&g_fz[n1], s_fz);
  }
}

// Split kernels (MD/x12, no neighbor atomics): EX/DMI/ANI/SIA blocks.
// These avoid large-box MIC recomputation by consuming precomputed neighbor vectors (x12/y12/z12).
template <int KMAX_PAIR, int KMAX_TERM>
static __global__ void find_force_radial_spin_spherical_md_noatomic_ex_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  float* fnp12 = fn12 + MAX_NUM_N;
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  constexpr int kmax_ex = KMAX_TERM;
  const bool gather = (N == (N2 - N1));

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

  // Cache central Fp for this atom once (hot loop otherwise reloads from global per neighbor).
  float Fp1_cache[(KMAX_TERM + 1) * MAX_NUM_N];
  for (int kk = 0; kk <= kmax_ex; ++kk) {
    for (int n = 0; n < nspin; ++n) {
      const int fp_idx = kk * nspin + n;
      Fp1_cache[kk * MAX_NUM_N + n] = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
    }
  }

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int stride_k = (mode == SPIN_C_SHARED_LATTICE) ? 0 : paramb.num_types_sq;
  const int stride_n = (mode == SPIN_C_SHARED_LATTICE) ? 0 : (paramb.basis_size_radial + 1) * stride_k;
  const int stride_block = paramb.c_spin_block_stride;
  const int c_step = (mode == SPIN_C_PER_BLOCK) ? stride_block : 0;
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) c_base_init += paramb.c_spin_offset;
  const float* c_base_init_t1 = c_base_init + t1 * paramb.num_types;

  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
  float s_virial_xy = 0.0f, s_virial_xz = 0.0f, s_virial_yz = 0.0f;
  float s_virial_yx = 0.0f, s_virial_zx = 0.0f, s_virial_zy = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;

    float d12inv = 1.0f / d12;

    float fc12, fcp12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    if (sj2 <= kSpinZeroEpsSph) continue;

    const float sdot = nep_spin_dot3(si, sj);
    const float sj_norm = sqrtf(sj2);
    const float denom = si_norm * sj_norm;
    const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

    float Tk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    nep_spin_fill_Tk<KMAX_TERM>(c, kmax_ex, Tk);

    float ex_invariant[KMAX_TERM + 1] = {0.0f};
    const float phi = nep_spin_ex_phi(paramb.spin_ex_phi_mode, si_norm, sj_norm, denom);
    nep_spin_fill_ex_invariant<KMAX_TERM>(phi, Tk, kmax_ex, ex_invariant);

    float fi_self[3] = {0.0f, 0.0f, 0.0f};
    float fi_other[3] = {0.0f, 0.0f, 0.0f};

    const float* c_base_ptr_12 = c_base_init_t1 + t2;
    const float* c_base_ptr_21 = nullptr;
    if (!same_type) {
      c_base_ptr_21 = c_base_init + t2 * paramb.num_types + t1;
    }

    for (int n = 0; n < nspin; ++n) {
      const float* c_n_ptr_12 = c_base_ptr_12 + n * stride_n;
      const float* c_n_ptr_21 = same_type ? nullptr : (c_base_ptr_21 + n * stride_n);

      for (int kk = 0; kk <= kmax_ex; ++kk) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        const float* c_k_ptr_12 = c_n_ptr_12 + kk * c_step;
        const float* c_k_ptr_21 = same_type ? nullptr : (c_n_ptr_21 + kk * c_step);

        for (int kb = 0; kb <= bs; ++kb) {
          const float fnpk = fnp12[kb];
          gnp12 += fnpk * NEP_SPIN_LDG(c_k_ptr_12);
          c_k_ptr_12 += stride_k;
          if (!same_type) {
            gnp21 += fnpk * NEP_SPIN_LDG(c_k_ptr_21);
            c_k_ptr_21 += stride_k;
          }
        }
        if (same_type) gnp21 = gnp12;
        const int fp_idx = kk * nspin + n;
        const float Fp1 = Fp1_cache[kk * MAX_NUM_N + n];
        const float inv = ex_invariant[kk];
        const float tmp_self = Fp1 * gnp12 * d12inv * inv;
        fi_self[0] += tmp_self * r12[0];
        fi_self[1] += tmp_self * r12[1];
        fi_self[2] += tmp_self * r12[2];
        if (gather) {
          const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
          const float tmp_other = Fp2 * gnp21 * d12inv * inv;
          fi_other[0] -= tmp_other * r12[0];
          fi_other[1] -= tmp_other * r12[1];
          fi_other[2] -= tmp_other * r12[2];
        }
      }
    }

    if (gather) {
      s_fx += fi_self[0] - fi_other[0];
      s_fy += fi_self[1] - fi_other[1];
      s_fz += fi_self[2] - fi_other[2];
    } else {
      s_fx += fi_self[0];
      s_fy += fi_self[1];
      s_fz += fi_self[2];

      atomicAdd(&g_fx[n2], static_cast<double>(-fi_self[0]));
      atomicAdd(&g_fy[n2], static_cast<double>(-fi_self[1]));
      atomicAdd(&g_fz[n2], static_cast<double>(-fi_self[2]));
    }

    s_virial_xx -= r12[0] * fi_self[0];
    s_virial_yy -= r12[1] * fi_self[1];
    s_virial_zz -= r12[2] * fi_self[2];
    s_virial_xy -= r12[0] * fi_self[1];
    s_virial_xz -= r12[0] * fi_self[2];
    s_virial_yz -= r12[1] * fi_self[2];
    s_virial_yx -= r12[1] * fi_self[0];
    s_virial_zx -= r12[2] * fi_self[0];
    s_virial_zy -= r12[2] * fi_self[1];
  }

  if (gather) {
    g_fx[n1] += static_cast<double>(s_fx);
    g_fy[n1] += static_cast<double>(s_fy);
    g_fz[n1] += static_cast<double>(s_fz);
  } else {
    atomicAdd(&g_fx[n1], static_cast<double>(s_fx));
    atomicAdd(&g_fy[n1], static_cast<double>(s_fy));
    atomicAdd(&g_fz[n1], static_cast<double>(s_fz));
  }

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 4 * N] += s_virial_xz;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 6 * N] += s_virial_yx;
  g_virial[n1 + 7 * N] += s_virial_zx;
  g_virial[n1 + 8 * N] += s_virial_zy;
}

template <int KMAX_PAIR, int KMAX_TERM>
static __global__ void find_force_radial_spin_spherical_md_noatomic_dmi_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  float* fnp12 = fn12 + MAX_NUM_N;
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_dmi = KMAX_TERM;
  const bool gather = (N == (N2 - N1));

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int dmi_block0 = spin_blocks.dmi_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

  // Cache central Fp once (avoid reloading per neighbor).
  float Fp1_cache[(KMAX_TERM + 1) * MAX_NUM_N];
  for (int kk = 0; kk <= kmax_dmi; ++kk) {
    const int block = dmi_block0 + kk;
    for (int n = 0; n < nspin; ++n) {
      const int fp_idx = block * nspin + n;
      Fp1_cache[kk * MAX_NUM_N + n] = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
    }
  }

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int stride_k = (mode == SPIN_C_SHARED_LATTICE) ? 0 : paramb.num_types_sq;
  const int stride_n = (mode == SPIN_C_SHARED_LATTICE) ? 0 : (paramb.basis_size_radial + 1) * stride_k;
  const int stride_block = paramb.c_spin_block_stride;
  const int c_step = (mode == SPIN_C_PER_BLOCK) ? stride_block : 0;
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) c_base_init += paramb.c_spin_offset;
  const float* c_base_init_t1 = c_base_init + t1 * paramb.num_types;

  auto J_apply = [&](const float v[3], float out[3], float d12inv, const float rhat[3]) {
    float rdotv = rhat[0] * v[0] + rhat[1] * v[1] + rhat[2] * v[2];
    out[0] = (v[0] - rhat[0] * rdotv) * d12inv;
    out[1] = (v[1] - rhat[1] * rdotv) * d12inv;
    out[2] = (v[2] - rhat[2] * rdotv) * d12inv;
  };

  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
  float s_virial_xy = 0.0f, s_virial_xz = 0.0f, s_virial_yz = 0.0f;
  float s_virial_yx = 0.0f, s_virial_zx = 0.0f, s_virial_zy = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;

    float d12inv = 1.0f / d12;
    float rhat[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

    float fc12, fcp12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    if (sj2 <= kSpinZeroEpsSph) continue;

    const float sdot = nep_spin_dot3(si, sj);
    const float sj_norm = sqrtf(sj2);
    const float denom = si_norm * sj_norm;
    const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

    float Tk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    nep_spin_fill_Tk<KMAX_TERM>(c, kmax_dmi, Tk);

    float sixsj[3];
    nep_spin_cross3(si, sj, sixsj);
    const float dmi = nep_spin_dot3(sixsj, rhat);

    float Jac_dmi[3] = {0.0f, 0.0f, 0.0f};
    J_apply(sixsj, Jac_dmi, d12inv, rhat);

    float fi_self[3] = {0.0f, 0.0f, 0.0f};
    float fi_other[3] = {0.0f, 0.0f, 0.0f};

    const float* c_base_ptr_12 = c_base_init_t1 + t2;
    const float* c_base_ptr_21 = nullptr;
    if (!same_type) {
      c_base_ptr_21 = c_base_init + t2 * paramb.num_types + t1;
    }

    for (int n = 0; n < nspin; ++n) {
      const float* c_n_ptr_12 = c_base_ptr_12 + n * stride_n;
      const float* c_n_ptr_21 = same_type ? nullptr : (c_base_ptr_21 + n * stride_n);

      for (int kk = 0; kk <= kmax_dmi; ++kk) {
        const int block = dmi_block0 + kk;
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        float gn21 = 0.0f;
        float gnp21 = 0.0f;
        const float* c_k_ptr_12 = c_n_ptr_12 + block * c_step;
        const float* c_k_ptr_21 = same_type ? nullptr : (c_n_ptr_21 + block * c_step);

        for (int kb = 0; kb <= bs; ++kb) {
          const float fnk = fn12[kb];
          const float fnpk = fnp12[kb];
          const float c12 = NEP_SPIN_LDG(c_k_ptr_12);
          c_k_ptr_12 += stride_k;
          gn12 += fnk * c12;
          gnp12 += fnpk * c12;
          if (!same_type) {
            const float c21 = NEP_SPIN_LDG(c_k_ptr_21);
            c_k_ptr_21 += stride_k;
            gn21 += fnk * c21;
            gnp21 += fnpk * c21;
          }
        }
        if (same_type) {
          gn21 = gn12;
          gnp21 = gnp12;
        }
        const int fp_idx = block * nspin + n;
        const float Fp1 = Fp1_cache[kk * MAX_NUM_N + n];
        const float Tk_k = Tk[kk];
        const float inv = dmi * Tk_k;
        const float tmp_self = Fp1 * gnp12 * d12inv * inv;
        fi_self[0] += tmp_self * r12[0];
        fi_self[1] += tmp_self * r12[1];
        fi_self[2] += tmp_self * r12[2];

        const float coeff_self = Fp1 * gn12 * Tk_k;
        fi_self[0] += coeff_self * Jac_dmi[0];
        fi_self[1] += coeff_self * Jac_dmi[1];
        fi_self[2] += coeff_self * Jac_dmi[2];
        if (gather) {
          const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
          const float tmp_other = Fp2 * gnp21 * d12inv * inv;
          fi_other[0] -= tmp_other * r12[0];
          fi_other[1] -= tmp_other * r12[1];
          fi_other[2] -= tmp_other * r12[2];
          const float coeff_other = Fp2 * gn21 * Tk_k;
          fi_other[0] -= coeff_other * Jac_dmi[0];
          fi_other[1] -= coeff_other * Jac_dmi[1];
          fi_other[2] -= coeff_other * Jac_dmi[2];
        }
      }
    }

    if (gather) {
      s_fx += fi_self[0] - fi_other[0];
      s_fy += fi_self[1] - fi_other[1];
      s_fz += fi_self[2] - fi_other[2];
    } else {
      s_fx += fi_self[0];
      s_fy += fi_self[1];
      s_fz += fi_self[2];

      atomicAdd(&g_fx[n2], static_cast<double>(-fi_self[0]));
      atomicAdd(&g_fy[n2], static_cast<double>(-fi_self[1]));
      atomicAdd(&g_fz[n2], static_cast<double>(-fi_self[2]));
    }

    s_virial_xx -= r12[0] * fi_self[0];
    s_virial_yy -= r12[1] * fi_self[1];
    s_virial_zz -= r12[2] * fi_self[2];
    s_virial_xy -= r12[0] * fi_self[1];
    s_virial_xz -= r12[0] * fi_self[2];
    s_virial_yz -= r12[1] * fi_self[2];
    s_virial_yx -= r12[1] * fi_self[0];
    s_virial_zx -= r12[2] * fi_self[0];
    s_virial_zy -= r12[2] * fi_self[1];
  }

  if (gather) {
    g_fx[n1] += static_cast<double>(s_fx);
    g_fy[n1] += static_cast<double>(s_fy);
    g_fz[n1] += static_cast<double>(s_fz);
  } else {
    atomicAdd(&g_fx[n1], static_cast<double>(s_fx));
    atomicAdd(&g_fy[n1], static_cast<double>(s_fy));
    atomicAdd(&g_fz[n1], static_cast<double>(s_fz));
  }

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 4 * N] += s_virial_xz;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 6 * N] += s_virial_yx;
  g_virial[n1 + 7 * N] += s_virial_zx;
  g_virial[n1 + 8 * N] += s_virial_zy;
}

template <int KMAX_PAIR, int KMAX_TERM>
static __global__ void find_force_radial_spin_spherical_md_noatomic_ani_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  float* fnp12 = fn12 + MAX_NUM_N;
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_ani = KMAX_TERM;
  const bool gather = (N == (N2 - N1));

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int ani_block0 = spin_blocks.ani_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

  // Cache central Fp once (avoid reloading per neighbor).
  float Fp1_cache[(KMAX_TERM + 1) * MAX_NUM_N];
  for (int kk = 0; kk <= kmax_ani; ++kk) {
    const int block = ani_block0 + kk;
    for (int n = 0; n < nspin; ++n) {
      const int fp_idx = block * nspin + n;
      Fp1_cache[kk * MAX_NUM_N + n] = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
    }
  }

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int stride_k = (mode == SPIN_C_SHARED_LATTICE) ? 0 : paramb.num_types_sq;
  const int stride_n = (mode == SPIN_C_SHARED_LATTICE) ? 0 : (paramb.basis_size_radial + 1) * stride_k;
  const int stride_block = paramb.c_spin_block_stride;
  const int c_step = (mode == SPIN_C_PER_BLOCK) ? stride_block : 0;
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) c_base_init += paramb.c_spin_offset;
  const float* c_base_init_t1 = c_base_init + t1 * paramb.num_types;

  auto J_apply = [&](const float v[3], float out[3], float d12inv, const float rhat[3]) {
    float rdotv = rhat[0] * v[0] + rhat[1] * v[1] + rhat[2] * v[2];
    out[0] = (v[0] - rhat[0] * rdotv) * d12inv;
    out[1] = (v[1] - rhat[1] * rdotv) * d12inv;
    out[2] = (v[2] - rhat[2] * rdotv) * d12inv;
  };

  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
  float s_virial_xy = 0.0f, s_virial_xz = 0.0f, s_virial_yz = 0.0f;
  float s_virial_yx = 0.0f, s_virial_zx = 0.0f, s_virial_zy = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;

    float d12inv = 1.0f / d12;
    float rhat[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

    float fc12, fcp12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    if (sj2 <= kSpinZeroEpsSph) continue;

    const float sdot = nep_spin_dot3(si, sj);
    const float sj_norm = sqrtf(sj2);
    const float denom = si_norm * sj_norm;
    const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

    float Tk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    nep_spin_fill_Tk<KMAX_TERM>(c, kmax_ani, Tk);

    const float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const float sj_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
    const float ani_scalar = si_r * sj_r;

    float sumv[3] = {si[0] * sj_r + sj[0] * si_r, si[1] * sj_r + sj[1] * si_r, si[2] * sj_r + sj[2] * si_r};
    float Jac_ani[3] = {0.0f, 0.0f, 0.0f};
    J_apply(sumv, Jac_ani, d12inv, rhat);

    float fi_self[3] = {0.0f, 0.0f, 0.0f};
    float fi_other[3] = {0.0f, 0.0f, 0.0f};

    const float* c_base_ptr_12 = c_base_init_t1 + t2;
    const float* c_base_ptr_21 = nullptr;
    if (!same_type) {
      c_base_ptr_21 = c_base_init + t2 * paramb.num_types + t1;
    }

    for (int n = 0; n < nspin; ++n) {
      const float* c_n_ptr_12 = c_base_ptr_12 + n * stride_n;
      const float* c_n_ptr_21 = same_type ? nullptr : (c_base_ptr_21 + n * stride_n);

      for (int kk = 0; kk <= kmax_ani; ++kk) {
        const int block = ani_block0 + kk;
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        float gn21 = 0.0f;
        float gnp21 = 0.0f;
        const float* c_k_ptr_12 = c_n_ptr_12 + block * c_step;
        const float* c_k_ptr_21 = same_type ? nullptr : (c_n_ptr_21 + block * c_step);

        for (int kb = 0; kb <= bs; ++kb) {
          const float fnk = fn12[kb];
          const float fnpk = fnp12[kb];
          const float c12 = NEP_SPIN_LDG(c_k_ptr_12);
          c_k_ptr_12 += stride_k;
          gn12 += fnk * c12;
          gnp12 += fnpk * c12;
          if (!same_type) {
            const float c21 = NEP_SPIN_LDG(c_k_ptr_21);
            c_k_ptr_21 += stride_k;
            gn21 += fnk * c21;
            gnp21 += fnpk * c21;
          }
        }
        if (same_type) {
          gn21 = gn12;
          gnp21 = gnp12;
        }
        const int fp_idx = block * nspin + n;
        const float Fp1 = Fp1_cache[kk * MAX_NUM_N + n];
        const float Tk_k = Tk[kk];
        const float inv = ani_scalar * Tk_k;
        const float tmp_self = Fp1 * gnp12 * d12inv * inv;
        fi_self[0] += tmp_self * r12[0];
        fi_self[1] += tmp_self * r12[1];
        fi_self[2] += tmp_self * r12[2];

        const float coeff_self = Fp1 * gn12 * Tk_k;
        fi_self[0] += coeff_self * Jac_ani[0];
        fi_self[1] += coeff_self * Jac_ani[1];
        fi_self[2] += coeff_self * Jac_ani[2];
        if (gather) {
          const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
          const float tmp_other = Fp2 * gnp21 * d12inv * inv;
          fi_other[0] -= tmp_other * r12[0];
          fi_other[1] -= tmp_other * r12[1];
          fi_other[2] -= tmp_other * r12[2];
          const float coeff_other = Fp2 * gn21 * Tk_k;
          fi_other[0] -= coeff_other * Jac_ani[0];
          fi_other[1] -= coeff_other * Jac_ani[1];
          fi_other[2] -= coeff_other * Jac_ani[2];
        }
      }
    }

    if (gather) {
      s_fx += fi_self[0] - fi_other[0];
      s_fy += fi_self[1] - fi_other[1];
      s_fz += fi_self[2] - fi_other[2];
    } else {
      s_fx += fi_self[0];
      s_fy += fi_self[1];
      s_fz += fi_self[2];

      atomicAdd(&g_fx[n2], static_cast<double>(-fi_self[0]));
      atomicAdd(&g_fy[n2], static_cast<double>(-fi_self[1]));
      atomicAdd(&g_fz[n2], static_cast<double>(-fi_self[2]));
    }

    s_virial_xx -= r12[0] * fi_self[0];
    s_virial_yy -= r12[1] * fi_self[1];
    s_virial_zz -= r12[2] * fi_self[2];
    s_virial_xy -= r12[0] * fi_self[1];
    s_virial_xz -= r12[0] * fi_self[2];
    s_virial_yz -= r12[1] * fi_self[2];
    s_virial_yx -= r12[1] * fi_self[0];
    s_virial_zx -= r12[2] * fi_self[0];
    s_virial_zy -= r12[2] * fi_self[1];
  }

  if (gather) {
    g_fx[n1] += static_cast<double>(s_fx);
    g_fy[n1] += static_cast<double>(s_fy);
    g_fz[n1] += static_cast<double>(s_fz);
  } else {
    atomicAdd(&g_fx[n1], static_cast<double>(s_fx));
    atomicAdd(&g_fy[n1], static_cast<double>(s_fy));
    atomicAdd(&g_fz[n1], static_cast<double>(s_fz));
  }

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 4 * N] += s_virial_xz;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 6 * N] += s_virial_yx;
  g_virial[n1 + 7 * N] += s_virial_zx;
  g_virial[n1 + 8 * N] += s_virial_zy;
}

template <int KMAX_PAIR, int KMAX_TERM>
static __global__ void find_force_radial_spin_spherical_md_noatomic_sia_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  float* fnp12 = fn12 + MAX_NUM_N;
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_sia = KMAX_TERM;
  const bool gather = (N == (N2 - N1));

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int sia_block0 = spin_blocks.sia_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

  // Cache central Fp once (avoid reloading per neighbor).
  float Fp1_cache[(KMAX_TERM + 1) * MAX_NUM_N];
  for (int kk = 0; kk <= kmax_sia; ++kk) {
    const int block = sia_block0 + kk;
    for (int n = 0; n < nspin; ++n) {
      const int fp_idx = block * nspin + n;
      Fp1_cache[kk * MAX_NUM_N + n] = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
    }
  }

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int stride_k = (mode == SPIN_C_SHARED_LATTICE) ? 0 : paramb.num_types_sq;
  const int stride_n = (mode == SPIN_C_SHARED_LATTICE) ? 0 : (paramb.basis_size_radial + 1) * stride_k;
  const int stride_block = paramb.c_spin_block_stride;
  const int c_step = (mode == SPIN_C_PER_BLOCK) ? stride_block : 0;
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) c_base_init += paramb.c_spin_offset;
  const float* c_base_init_t1 = c_base_init + t1 * paramb.num_types;

  auto J_apply = [&](const float v[3], float out[3], float d12inv, const float rhat[3]) {
    float rdotv = rhat[0] * v[0] + rhat[1] * v[1] + rhat[2] * v[2];
    out[0] = (v[0] - rhat[0] * rdotv) * d12inv;
    out[1] = (v[1] - rhat[1] * rdotv) * d12inv;
    out[2] = (v[2] - rhat[2] * rdotv) * d12inv;
  };

  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
  float s_virial_xy = 0.0f, s_virial_xz = 0.0f, s_virial_yz = 0.0f;
  float s_virial_yx = 0.0f, s_virial_zx = 0.0f, s_virial_zy = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;

    float d12inv = 1.0f / d12;
    float rhat[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

    float fc12, fcp12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);

    const float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const float sia_scalar_i = si_r * si_r;
    float sj_r = 0.0f;
    float sia_scalar_j = 0.0f;
    float Jac_sia_j[3] = {0.0f, 0.0f, 0.0f};
    if (gather && neighbor_has_spin) {
      sj_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
      sia_scalar_j = sj_r * sj_r;
      J_apply(sj, Jac_sia_j, d12inv, rhat);
    }

    float Tk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    if (neighbor_has_spin) {
      const float sdot = nep_spin_dot3(si, sj);
      const float sj_norm = sqrtf(sj2);
      const float denom = si_norm * sj_norm;
      const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));
      nep_spin_fill_Tk<KMAX_TERM>(c, kmax_sia, Tk);
    }

    float Jac_sia_i[3] = {0.0f, 0.0f, 0.0f};
    J_apply(si, Jac_sia_i, d12inv, rhat);

    float fi_self[3] = {0.0f, 0.0f, 0.0f};
    float fi_other[3] = {0.0f, 0.0f, 0.0f};

    const float* c_base_ptr_12 = c_base_init_t1 + t2;
    const float* c_base_ptr_21 = nullptr;
    if (neighbor_has_spin && !same_type) {
      c_base_ptr_21 = c_base_init + t2 * paramb.num_types + t1;
    }

    for (int n = 0; n < nspin; ++n) {
      const float* c_n_ptr_12 = c_base_ptr_12 + n * stride_n;
      const float* c_n_ptr_21 = (neighbor_has_spin && !same_type) ? (c_base_ptr_21 + n * stride_n) : nullptr;

      for (int kk = 0; kk <= kmax_sia; ++kk) {
        if (kk > 0 && !neighbor_has_spin) continue;
        const int block = sia_block0 + kk;
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        float gn21 = 0.0f;
        float gnp21 = 0.0f;
        const float* c_k_ptr_12 = c_n_ptr_12 + block * c_step;
        const float* c_k_ptr_21 = (neighbor_has_spin && !same_type) ? (c_n_ptr_21 + block * c_step) : nullptr;

        for (int kb = 0; kb <= bs; ++kb) {
          const float fnk = fn12[kb];
          const float fnpk = fnp12[kb];
          const float c12 = NEP_SPIN_LDG(c_k_ptr_12);
          c_k_ptr_12 += stride_k;
          gn12 += fnk * c12;
          gnp12 += fnpk * c12;
          if (neighbor_has_spin && !same_type) {
            const float c21 = NEP_SPIN_LDG(c_k_ptr_21);
            c_k_ptr_21 += stride_k;
            gn21 += fnk * c21;
            gnp21 += fnpk * c21;
          }
        }
        if (neighbor_has_spin && same_type) {
          gn21 = gn12;
          gnp21 = gnp12;
        }
        const int fp_idx = block * nspin + n;
        const float Fp1 = Fp1_cache[kk * MAX_NUM_N + n];
        const float Tk_k = Tk[kk];
        const float inv_i = sia_scalar_i * Tk_k;
        const float tmp_self = Fp1 * gnp12 * d12inv * inv_i;
        fi_self[0] += tmp_self * r12[0];
        fi_self[1] += tmp_self * r12[1];
        fi_self[2] += tmp_self * r12[2];
        const float coeff_self = 2.0f * Fp1 * gn12 * Tk_k * si_r;
        fi_self[0] += coeff_self * Jac_sia_i[0];
        fi_self[1] += coeff_self * Jac_sia_i[1];
        fi_self[2] += coeff_self * Jac_sia_i[2];
        if (gather && neighbor_has_spin) {
          const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
          const float inv_j = sia_scalar_j * Tk_k;
          const float tmp_other = Fp2 * gnp21 * d12inv * inv_j;
          fi_other[0] -= tmp_other * r12[0];
          fi_other[1] -= tmp_other * r12[1];
          fi_other[2] -= tmp_other * r12[2];
          const float coeff_other = 2.0f * Fp2 * gn21 * Tk_k * sj_r;
          fi_other[0] -= coeff_other * Jac_sia_j[0];
          fi_other[1] -= coeff_other * Jac_sia_j[1];
          fi_other[2] -= coeff_other * Jac_sia_j[2];
        }
      }
    }

    if (gather) {
      s_fx += fi_self[0] - fi_other[0];
      s_fy += fi_self[1] - fi_other[1];
      s_fz += fi_self[2] - fi_other[2];
    } else {
      s_fx += fi_self[0];
      s_fy += fi_self[1];
      s_fz += fi_self[2];

      atomicAdd(&g_fx[n2], static_cast<double>(-fi_self[0]));
      atomicAdd(&g_fy[n2], static_cast<double>(-fi_self[1]));
      atomicAdd(&g_fz[n2], static_cast<double>(-fi_self[2]));
    }

    s_virial_xx -= r12[0] * fi_self[0];
    s_virial_yy -= r12[1] * fi_self[1];
    s_virial_zz -= r12[2] * fi_self[2];
    s_virial_xy -= r12[0] * fi_self[1];
    s_virial_xz -= r12[0] * fi_self[2];
    s_virial_yz -= r12[1] * fi_self[2];
    s_virial_yx -= r12[1] * fi_self[0];
    s_virial_zx -= r12[2] * fi_self[0];
    s_virial_zy -= r12[2] * fi_self[1];
  }

  if (gather) {
    g_fx[n1] += static_cast<double>(s_fx);
    g_fy[n1] += static_cast<double>(s_fy);
    g_fz[n1] += static_cast<double>(s_fz);
  } else {
    atomicAdd(&g_fx[n1], static_cast<double>(s_fx));
    atomicAdd(&g_fy[n1], static_cast<double>(s_fy));
    atomicAdd(&g_fz[n1], static_cast<double>(s_fz));
  }

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 4 * N] += s_virial_xz;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 6 * N] += s_virial_yx;
  g_virial[n1 + 7 * N] += s_virial_zx;
  g_virial[n1 + 8 * N] += s_virial_zy;
}

static __device__ __forceinline__ void nep_spin_J_apply_md(
  const float v[3],
  float out[3],
  float d12inv,
  const float rhat[3])
{
  float rdotv = rhat[0] * v[0] + rhat[1] * v[1] + rhat[2] * v[2];
  out[0] = (v[0] - rhat[0] * rdotv) * d12inv;
  out[1] = (v[1] - rhat[1] * rdotv) * d12inv;
  out[2] = (v[2] - rhat[2] * rdotv) * d12inv;
}

// Fused force kernel: accumulates EX + DMI + ANI + SIA contributions in a single
// pass over the neighbor list to reduce redundant work.
template <int KMAX_PAIR>
static __global__ void find_force_radial_spin_spherical_md_noatomic_fused_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  float* fnp12 = fn12 + MAX_NUM_N;

  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const bool gather = (N == (N2 - N1));
  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  const int kmax_ex = spin_blocks.kmax_ex;
  const int kmax_dmi = spin_blocks.kmax_dmi;
  const int kmax_ani = spin_blocks.kmax_ani;
  const int kmax_sia = spin_blocks.kmax_sia;
  const int kmax_pair = spin_blocks.kmax_pair;
  const bool do_ex = (kmax_ex >= 0);
  const bool do_dmi = (kmax_dmi >= 0);
  const bool do_ani = (kmax_ani >= 0);
  const bool do_sia = (kmax_sia >= 0);
  if (!(do_ex || do_dmi || do_ani || do_sia)) return;

  const int dmi_block0 = spin_blocks.dmi_block0;
  const int ani_block0 = spin_blocks.ani_block0;
  const int sia_block0 = spin_blocks.sia_block0;

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int stride_k = (mode == SPIN_C_SHARED_LATTICE) ? 0 : paramb.num_types_sq;
  const int stride_n = (mode == SPIN_C_SHARED_LATTICE) ? 0 : (paramb.basis_size_radial + 1) * stride_k;
  const int stride_block = paramb.c_spin_block_stride;
  const int c_step = (mode == SPIN_C_PER_BLOCK) ? stride_block : 0;
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) c_base_init += paramb.c_spin_offset;
  const float* c_base_init_t1 = c_base_init + t1 * paramb.num_types;

  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_virial_xx = 0.0f, s_virial_yy = 0.0f, s_virial_zz = 0.0f;
  float s_virial_xy = 0.0f, s_virial_xz = 0.0f, s_virial_yz = 0.0f;
  float s_virial_yx = 0.0f, s_virial_zx = 0.0f, s_virial_zy = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;
    float d12inv = 1.0f / d12;
    float rhat[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

    float fc12, fcp12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
    if (!neighbor_has_spin && !do_sia) continue;

    float Tk[KMAX_PAIR + 1] = {0.0f};
    Tk[0] = 1.0f;

    float ex_invariant[KMAX_PAIR + 1] = {0.0f};

    float Jac_dmi[3] = {0.0f, 0.0f, 0.0f};
    float Jac_ani[3] = {0.0f, 0.0f, 0.0f};
    float Jac_sia_i[3] = {0.0f, 0.0f, 0.0f};
    float Jac_sia_j[3] = {0.0f, 0.0f, 0.0f};

    float dmi = 0.0f;
    float ani_scalar = 0.0f;
    float si_r = 0.0f;
    float sj_r = 0.0f;
    float sia_scalar_i = 0.0f;
    float sia_scalar_j = 0.0f;

    if (neighbor_has_spin) {
      const float sdot = nep_spin_dot3(si, sj);
      const float sj_norm = sqrtf(sj2);
      const float denom = si_norm * sj_norm;
      const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));
      nep_spin_fill_Tk<KMAX_PAIR>(c, kmax_pair, Tk);

      if (do_ex) {
        const float phi = nep_spin_ex_phi(paramb.spin_ex_phi_mode, si_norm, sj_norm, denom);
        nep_spin_fill_ex_invariant<KMAX_PAIR>(phi, Tk, kmax_ex, ex_invariant);
      }
      if (do_dmi) {
        float sixsj[3];
        nep_spin_cross3(si, sj, sixsj);
        dmi = nep_spin_dot3(sixsj, rhat);
        nep_spin_J_apply_md(sixsj, Jac_dmi, d12inv, rhat);
      }
      if (do_ani) {
        si_r = nep_spin_dot3(si, rhat);
        sj_r = nep_spin_dot3(sj, rhat);
        ani_scalar = si_r * sj_r;
        float sumv[3] = {si[0] * sj_r + sj[0] * si_r, si[1] * sj_r + sj[1] * si_r, si[2] * sj_r + sj[2] * si_r};
        nep_spin_J_apply_md(sumv, Jac_ani, d12inv, rhat);
      }
      if (do_sia && gather) {
        sj_r = nep_spin_dot3(sj, rhat);
        sia_scalar_j = sj_r * sj_r;
        nep_spin_J_apply_md(sj, Jac_sia_j, d12inv, rhat);
      }
    }

    if (do_sia) {
      si_r = nep_spin_dot3(si, rhat);
      sia_scalar_i = si_r * si_r;
      nep_spin_J_apply_md(si, Jac_sia_i, d12inv, rhat);
    }

    float fi_self_total[3] = {0.0f, 0.0f, 0.0f};
    float fi_other_total[3] = {0.0f, 0.0f, 0.0f};

    const float* c_base_ptr_12 = c_base_init_t1 + t2;
    const float* c_base_ptr_21 = (!same_type) ? (c_base_init + t2 * paramb.num_types + t1) : nullptr;

    // EX term
    if (do_ex && neighbor_has_spin) {
      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr_12 = c_base_ptr_12 + n * stride_n;
        const float* c_n_ptr_21 = same_type ? nullptr : (c_base_ptr_21 + n * stride_n);
        for (int kk = 0; kk <= kmax_ex; ++kk) {
          float gnp12 = 0.0f;
          float gnp21 = 0.0f;
          const float* c_k_ptr_12 = c_n_ptr_12 + kk * c_step;
          const float* c_k_ptr_21 = same_type ? nullptr : (c_n_ptr_21 + kk * c_step);
          for (int kb = 0; kb <= bs; ++kb) {
            const float fnpk = fnp12[kb];
            gnp12 += fnpk * NEP_SPIN_LDG(c_k_ptr_12);
            c_k_ptr_12 += stride_k;
            if (gather && !same_type) {
              gnp21 += fnpk * NEP_SPIN_LDG(c_k_ptr_21);
              c_k_ptr_21 += stride_k;
            }
          }
          if (!gather || same_type) gnp21 = gnp12;
          const int fp_idx = kk * nspin + n;
          const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
          const float inv = ex_invariant[kk];
          const float tmp_self = Fp1 * gnp12 * d12inv * inv;
          fi_self_total[0] += tmp_self * r12[0];
          fi_self_total[1] += tmp_self * r12[1];
          fi_self_total[2] += tmp_self * r12[2];
          if (gather) {
            const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
            const float tmp_other = Fp2 * gnp21 * d12inv * inv;
            fi_other_total[0] -= tmp_other * r12[0];
            fi_other_total[1] -= tmp_other * r12[1];
            fi_other_total[2] -= tmp_other * r12[2];
          }
        }
      }
    }

    // DMI term
    if (do_dmi && neighbor_has_spin) {
      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr_12 = c_base_ptr_12 + n * stride_n;
        const float* c_n_ptr_21 = same_type ? nullptr : (c_base_ptr_21 + n * stride_n);
        for (int kk = 0; kk <= kmax_dmi; ++kk) {
          const int block = dmi_block0 + kk;
          float gn12 = 0.0f;
          float gnp12 = 0.0f;
          float gn21 = 0.0f;
          float gnp21 = 0.0f;
          const float* c_k_ptr_12 = c_n_ptr_12 + block * c_step;
          const float* c_k_ptr_21 = same_type ? nullptr : (c_n_ptr_21 + block * c_step);
          for (int kb = 0; kb <= bs; ++kb) {
            const float fnk = fn12[kb];
            const float fnpk = fnp12[kb];
            const float c12 = NEP_SPIN_LDG(c_k_ptr_12);
            c_k_ptr_12 += stride_k;
            gn12 += fnk * c12;
            gnp12 += fnpk * c12;
            if (gather && !same_type) {
              const float c21 = NEP_SPIN_LDG(c_k_ptr_21);
              c_k_ptr_21 += stride_k;
              gn21 += fnk * c21;
              gnp21 += fnpk * c21;
            }
          }
          if (!gather || same_type) {
            gn21 = gn12;
            gnp21 = gnp12;
          }
          const int fp_idx = block * nspin + n;
          const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
          const float Tk_k = Tk[kk];
          const float inv = dmi * Tk_k;
          const float tmp_self = Fp1 * gnp12 * d12inv * inv;
          fi_self_total[0] += tmp_self * r12[0];
          fi_self_total[1] += tmp_self * r12[1];
          fi_self_total[2] += tmp_self * r12[2];

          const float coeff_self = Fp1 * gn12 * Tk_k;
          fi_self_total[0] += coeff_self * Jac_dmi[0];
          fi_self_total[1] += coeff_self * Jac_dmi[1];
          fi_self_total[2] += coeff_self * Jac_dmi[2];
          if (gather) {
            const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
            const float tmp_other = Fp2 * gnp21 * d12inv * inv;
            fi_other_total[0] -= tmp_other * r12[0];
            fi_other_total[1] -= tmp_other * r12[1];
            fi_other_total[2] -= tmp_other * r12[2];
            const float coeff_other = Fp2 * gn21 * Tk_k;
            fi_other_total[0] -= coeff_other * Jac_dmi[0];
            fi_other_total[1] -= coeff_other * Jac_dmi[1];
            fi_other_total[2] -= coeff_other * Jac_dmi[2];
          }
        }
      }
    }

    // ANI term
    if (do_ani && neighbor_has_spin) {
      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr_12 = c_base_ptr_12 + n * stride_n;
        const float* c_n_ptr_21 = same_type ? nullptr : (c_base_ptr_21 + n * stride_n);
        for (int kk = 0; kk <= kmax_ani; ++kk) {
          const int block = ani_block0 + kk;
          float gn12 = 0.0f;
          float gnp12 = 0.0f;
          float gn21 = 0.0f;
          float gnp21 = 0.0f;
          const float* c_k_ptr_12 = c_n_ptr_12 + block * c_step;
          const float* c_k_ptr_21 = same_type ? nullptr : (c_n_ptr_21 + block * c_step);
          for (int kb = 0; kb <= bs; ++kb) {
            const float fnk = fn12[kb];
            const float fnpk = fnp12[kb];
            const float c12 = NEP_SPIN_LDG(c_k_ptr_12);
            c_k_ptr_12 += stride_k;
            gn12 += fnk * c12;
            gnp12 += fnpk * c12;
            if (gather && !same_type) {
              const float c21 = NEP_SPIN_LDG(c_k_ptr_21);
              c_k_ptr_21 += stride_k;
              gn21 += fnk * c21;
              gnp21 += fnpk * c21;
            }
          }
          if (!gather || same_type) {
            gn21 = gn12;
            gnp21 = gnp12;
          }
          const int fp_idx = block * nspin + n;
          const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
          const float Tk_k = Tk[kk];
          const float inv = ani_scalar * Tk_k;
          const float tmp_self = Fp1 * gnp12 * d12inv * inv;
          fi_self_total[0] += tmp_self * r12[0];
          fi_self_total[1] += tmp_self * r12[1];
          fi_self_total[2] += tmp_self * r12[2];

          const float coeff_self = Fp1 * gn12 * Tk_k;
          fi_self_total[0] += coeff_self * Jac_ani[0];
          fi_self_total[1] += coeff_self * Jac_ani[1];
          fi_self_total[2] += coeff_self * Jac_ani[2];
          if (gather) {
            const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
            const float tmp_other = Fp2 * gnp21 * d12inv * inv;
            fi_other_total[0] -= tmp_other * r12[0];
            fi_other_total[1] -= tmp_other * r12[1];
            fi_other_total[2] -= tmp_other * r12[2];
            const float coeff_other = Fp2 * gn21 * Tk_k;
            fi_other_total[0] -= coeff_other * Jac_ani[0];
            fi_other_total[1] -= coeff_other * Jac_ani[1];
            fi_other_total[2] -= coeff_other * Jac_ani[2];
          }
        }
      }
    }

    // SIA term
    if (do_sia) {
      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr_12 = c_base_ptr_12 + n * stride_n;
        const float* c_n_ptr_21 = (gather && neighbor_has_spin && !same_type) ? (c_base_ptr_21 + n * stride_n) : nullptr;
        for (int kk = 0; kk <= kmax_sia; ++kk) {
          if (kk > 0 && !neighbor_has_spin) continue;
          const int block = sia_block0 + kk;
          float gn12 = 0.0f;
          float gnp12 = 0.0f;
          float gn21 = 0.0f;
          float gnp21 = 0.0f;
          const float* c_k_ptr_12 = c_n_ptr_12 + block * c_step;
          const float* c_k_ptr_21 = (gather && neighbor_has_spin && !same_type) ? (c_n_ptr_21 + block * c_step) : nullptr;
          for (int kb = 0; kb <= bs; ++kb) {
            const float fnk = fn12[kb];
            const float fnpk = fnp12[kb];
            const float c12 = NEP_SPIN_LDG(c_k_ptr_12);
            c_k_ptr_12 += stride_k;
            gn12 += fnk * c12;
            gnp12 += fnpk * c12;
            if (gather && neighbor_has_spin && !same_type) {
              const float c21 = NEP_SPIN_LDG(c_k_ptr_21);
              c_k_ptr_21 += stride_k;
              gn21 += fnk * c21;
              gnp21 += fnpk * c21;
            }
          }
          if (gather && neighbor_has_spin && same_type) {
            gn21 = gn12;
            gnp21 = gnp12;
          }
          const int fp_idx = block * nspin + n;
          const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
          const float Tk_k = Tk[kk];
          const float inv_i = sia_scalar_i * Tk_k;
          const float tmp_self = Fp1 * gnp12 * d12inv * inv_i;
          fi_self_total[0] += tmp_self * r12[0];
          fi_self_total[1] += tmp_self * r12[1];
          fi_self_total[2] += tmp_self * r12[2];
          const float coeff_self = 2.0f * Fp1 * gn12 * Tk_k * si_r;
          fi_self_total[0] += coeff_self * Jac_sia_i[0];
          fi_self_total[1] += coeff_self * Jac_sia_i[1];
          fi_self_total[2] += coeff_self * Jac_sia_i[2];
          if (gather && neighbor_has_spin) {
            const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
            const float inv_j = sia_scalar_j * Tk_k;
            const float tmp_other = Fp2 * gnp21 * d12inv * inv_j;
            fi_other_total[0] -= tmp_other * r12[0];
            fi_other_total[1] -= tmp_other * r12[1];
            fi_other_total[2] -= tmp_other * r12[2];
            const float coeff_other = 2.0f * Fp2 * gn21 * Tk_k * sj_r;
            fi_other_total[0] -= coeff_other * Jac_sia_j[0];
            fi_other_total[1] -= coeff_other * Jac_sia_j[1];
            fi_other_total[2] -= coeff_other * Jac_sia_j[2];
          }
        }
      }
    }

    if (gather) {
      s_fx += fi_self_total[0] - fi_other_total[0];
      s_fy += fi_self_total[1] - fi_other_total[1];
      s_fz += fi_self_total[2] - fi_other_total[2];
    } else {
      s_fx += fi_self_total[0];
      s_fy += fi_self_total[1];
      s_fz += fi_self_total[2];
      atomicAdd(&g_fx[n2], static_cast<double>(-fi_self_total[0]));
      atomicAdd(&g_fy[n2], static_cast<double>(-fi_self_total[1]));
      atomicAdd(&g_fz[n2], static_cast<double>(-fi_self_total[2]));
    }

    s_virial_xx -= r12[0] * fi_self_total[0];
    s_virial_yy -= r12[1] * fi_self_total[1];
    s_virial_zz -= r12[2] * fi_self_total[2];
    s_virial_xy -= r12[0] * fi_self_total[1];
    s_virial_xz -= r12[0] * fi_self_total[2];
    s_virial_yz -= r12[1] * fi_self_total[2];
    s_virial_yx -= r12[1] * fi_self_total[0];
    s_virial_zx -= r12[2] * fi_self_total[0];
    s_virial_zy -= r12[2] * fi_self_total[1];
  }

  if (gather) {
    g_fx[n1] += static_cast<double>(s_fx);
    g_fy[n1] += static_cast<double>(s_fy);
    g_fz[n1] += static_cast<double>(s_fz);
  } else {
    atomicAdd(&g_fx[n1], static_cast<double>(s_fx));
    atomicAdd(&g_fy[n1], static_cast<double>(s_fy));
    atomicAdd(&g_fz[n1], static_cast<double>(s_fz));
  }

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 4 * N] += s_virial_xz;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 6 * N] += s_virial_yx;
  g_virial[n1 + 7 * N] += s_virial_zx;
  g_virial[n1 + 8 * N] += s_virial_zy;
}

// Split kernels (MD/x12, no neighbor atomics): EX/DMI/ANI/SIA blocks.
// These avoid large-box MIC recomputation by consuming precomputed neighbor vectors (x12/y12/z12).
// On-site longitudinal terms only (p=1..spin_pmax).
static __global__ void find_mforce_onsite_spin_spherical_md(
  const int N,
  const int N1,
  const int N2,
  const NEP_Spin::ParaMB paramb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_mx,
  double* g_my,
  double* g_mz,
  int onsite_offset)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const int spin_pmax = nep_spin_clamp_pmax(paramb.spin_pmax);
  if (spin_pmax <= 0) return;
  const int t1 = g_type[n1];

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;

  float si_norm = sqrtf(si2);
  float inv_si_norm = 1.0f / (si_norm + 1.0e-12f);

  float mfx_i = 0.0f;
  float mfy_i = 0.0f;
  float mfz_i = 0.0f;

  int basis_mode = paramb.spin_onsite_basis_mode;
  if (basis_mode == 0) {
    float m2 = si2;
    float m2pow = 1.0f;
    for (int p = 1; p <= spin_pmax; ++p) {
      float Fp_p = g_Fp[n1 + (onsite_offset + (p - 1)) * N];
      float coeff = paramb.mforce_sign * Fp_p * (2.0f * p) * m2pow;
      mfx_i += coeff * si[0];
      mfy_i += coeff * si[1];
      mfz_i += coeff * si[2];
      m2pow *= m2;
    }
  } else {
    float y = si2;
    float yref = nep_spin_type_yref(t1, basis_mode);
    if (basis_mode == 2) {
      y = si_norm;
    }
    if (yref <= 0.0f) yref = 1.0f;

    float denom = y + yref;
    float inv_denom = 1.0f / (denom + 1.0e-12f);
    float x = (y - yref) * inv_denom;
    x = fminf(1.0f, fmaxf(-1.0f, x));

    float dx_dy = (2.0f * yref) * inv_denom * inv_denom;

    constexpr int kMaxK = 8;
    float Tp[kMaxK + 1] = {0.0f};
    float dTp[kMaxK + 1] = {0.0f};
    Tp[0] = 1.0f;
    dTp[0] = 0.0f;
    if (spin_pmax >= 1) {
      Tp[1] = x;
      dTp[1] = 1.0f;
    }
    for (int p = 2; p <= spin_pmax; ++p) {
      Tp[p] = 2.0f * x * Tp[p - 1] - Tp[p - 2];
      dTp[p] = 2.0f * Tp[p - 1] + 2.0f * x * dTp[p - 1] - dTp[p - 2];
    }

    float dy_dsi[3] = {0.0f, 0.0f, 0.0f};
    if (basis_mode == 2) {
      dy_dsi[0] = inv_si_norm * si[0];
      dy_dsi[1] = inv_si_norm * si[1];
      dy_dsi[2] = inv_si_norm * si[2];
    } else {
      dy_dsi[0] = 2.0f * si[0];
      dy_dsi[1] = 2.0f * si[1];
      dy_dsi[2] = 2.0f * si[2];
    }

    for (int p = 1; p <= spin_pmax; ++p) {
      float Fp_p = g_Fp[n1 + (onsite_offset + (p - 1)) * N];
      float coeff = paramb.mforce_sign * Fp_p * dTp[p] * dx_dy;
      mfx_i += coeff * dy_dsi[0];
      mfy_i += coeff * dy_dsi[1];
      mfz_i += coeff * dy_dsi[2];
    }
  }

  g_mx[n1] += static_cast<double>(mfx_i);
  g_my[n1] += static_cast<double>(mfy_i);
  g_mz[n1] += static_cast<double>(mfz_i);
}

// Split kernels (MD/x12, NEP-style no neighbor atomics): per-interaction mforce.
// These avoid large-box MIC recomputation by consuming precomputed neighbor vectors (x12/y12/z12).
template <int KMAX_PAIR, int KMAX_TERM>
static __global__ void find_mforce_radial_spin_spherical_md_noatomic_ex_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_mx,
  double* g_my,
  double* g_mz,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  constexpr int kmax_ex = KMAX_TERM;
  const bool gather = (N == (N2 - N1));

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);
  float inv_si_norm = 1.0f / (si_norm + 1.0e-12f);

  // Cache central Fp once (avoid reloading per neighbor).
  float Fp1_cache[(KMAX_TERM + 1) * MAX_NUM_N];
  for (int kk = 0; kk <= kmax_ex; ++kk) {
    for (int n = 0; n < nspin; ++n) {
      const int fp_idx = kk * nspin + n;
      Fp1_cache[kk * MAX_NUM_N + n] = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
    }
  }

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int stride_k = paramb.num_types_sq;
  const int stride_n = (paramb.basis_size_radial + 1) * stride_k;
  const int stride_block = paramb.c_spin_block_stride;
  const int c_step = (mode == SPIN_C_PER_BLOCK) ? stride_block : 0;
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) c_base_init += paramb.c_spin_offset;
  const float* c_base_t1 = c_base_init + t1 * paramb.num_types;

  float mfx_i = 0.0f;
  float mfy_i = 0.0f;
  float mfz_i = 0.0f;
  const float msign = paramb.mforce_sign;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const float* c_base_pair12 = c_base_t1 + t2;
    const float* c_base_pair21 = c_base_init + t2 * paramb.num_types + t1;
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;

    float fc12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);
    find_fn(bs, rcinv, d12, fc12, fn12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    if (sj2 <= kSpinZeroEpsSph) continue;

    const float sdot = nep_spin_dot3(si, sj);
    const float sj_norm = sqrtf(sj2);
    const float inv_sj_norm = 1.0f / (sj_norm + 1.0e-12f);
    const float denom = si_norm * sj_norm;
    const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

    float Tk[KMAX_TERM + 1] = {0.0f};
    float dTk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    dTk[0] = 0.0f;
    nep_spin_fill_Tk_and_dTk<KMAX_TERM>(c, kmax_ex, Tk, dTk);

    float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
    float dc_dsj[3] = {0.0f, 0.0f, 0.0f};
    nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);

    float phi = 0.0f;
    float dphi_dsi[3] = {0.0f, 0.0f, 0.0f};
    float dphi_dsj[3] = {0.0f, 0.0f, 0.0f};
    nep_spin_ex_phi_and_grads(
      paramb.spin_ex_phi_mode, si, sj, si_norm, sj_norm, inv_si_norm, inv_sj_norm, phi, dphi_dsi, dphi_dsj);

    float mfx_j = 0.0f;
    float mfy_j = 0.0f;
    float mfz_j = 0.0f;

    for (int n = 0; n < nspin; ++n) {
      const float* c_n_ptr12 = c_base_pair12 + n * stride_n;
      const float* c_n_ptr21 = c_base_pair21 + n * stride_n;
      const float* c_kk_ptr12 = c_n_ptr12;
      const float* c_kk_ptr21 = c_n_ptr21;
      for (int kk = 0; kk <= KMAX_TERM; ++kk) {
        float gn12 = 0.0f;
        float gn21 = 0.0f;
        const float* c_kb_ptr12 = c_kk_ptr12;
        const float* c_kb_ptr21 = c_kk_ptr21;
        for (int kb = 0; kb <= bs; ++kb) {
          const float fnk = fn12[kb];
          gn12 += fnk * NEP_SPIN_LDG(c_kb_ptr12);
          if (!same_type) {
            gn21 += fnk * NEP_SPIN_LDG(c_kb_ptr21);
          }
          c_kb_ptr12 += stride_k;
          c_kb_ptr21 += stride_k;
        }
        c_kk_ptr12 += c_step;
        c_kk_ptr21 += c_step;

        if (same_type) gn21 = gn12;
        const int fp_idx = kk * nspin + n;
        const float Fp1 = Fp1_cache[kk * MAX_NUM_N + n];
        const float coeff = msign * (Fp1 * gn12);
        const float coeff_other = gather ? (msign * (NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]) * gn21)) : 0.0f;
        const float grad_i[3] = {
          dphi_dsi[0] * Tk[kk] + phi * dTk[kk] * dc_dsi[0],
          dphi_dsi[1] * Tk[kk] + phi * dTk[kk] * dc_dsi[1],
          dphi_dsi[2] * Tk[kk] + phi * dTk[kk] * dc_dsi[2]};
        mfx_i += coeff * grad_i[0];
        mfy_i += coeff * grad_i[1];
        mfz_i += coeff * grad_i[2];
        if (gather) {
          mfx_i += coeff_other * grad_i[0];
          mfy_i += coeff_other * grad_i[1];
          mfz_i += coeff_other * grad_i[2];
        } else {
          const float grad_j[3] = {
            dphi_dsj[0] * Tk[kk] + phi * dTk[kk] * dc_dsj[0],
            dphi_dsj[1] * Tk[kk] + phi * dTk[kk] * dc_dsj[1],
            dphi_dsj[2] * Tk[kk] + phi * dTk[kk] * dc_dsj[2]};
          mfx_j += coeff * grad_j[0];
          mfy_j += coeff * grad_j[1];
          mfz_j += coeff * grad_j[2];
        }
      }
    }

    if (!gather) {
      atomicAdd(&g_mx[n2], static_cast<double>(mfx_j));
      atomicAdd(&g_my[n2], static_cast<double>(mfy_j));
      atomicAdd(&g_mz[n2], static_cast<double>(mfz_j));
    }
  }

  if (gather) {
    g_mx[n1] += static_cast<double>(mfx_i);
    g_my[n1] += static_cast<double>(mfy_i);
    g_mz[n1] += static_cast<double>(mfz_i);
  } else {
    atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
    atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
    atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
  }
}

template <int KMAX_PAIR, int KMAX_TERM>
static __global__ void find_mforce_radial_spin_spherical_md_noatomic_dmi_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_mx,
  double* g_my,
  double* g_mz,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_dmi = KMAX_TERM;
  const bool gather = (N == (N2 - N1));

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int dmi_block0 = spin_blocks.dmi_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

  // Cache central Fp once (avoid reloading per neighbor).
  float Fp1_cache[(KMAX_TERM + 1) * MAX_NUM_N];
  for (int kk = 0; kk <= kmax_dmi; ++kk) {
    const int block = dmi_block0 + kk;
    for (int n = 0; n < nspin; ++n) {
      const int fp_idx = block * nspin + n;
      Fp1_cache[kk * MAX_NUM_N + n] = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
    }
  }

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int stride_k = paramb.num_types_sq;
  const int stride_n = (paramb.basis_size_radial + 1) * stride_k;
  const int stride_block = paramb.c_spin_block_stride;
  const int c_step = (mode == SPIN_C_PER_BLOCK) ? stride_block : 0;
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) c_base_init += paramb.c_spin_offset;
  const float* c_base_t1 = c_base_init + t1 * paramb.num_types;

  float mfx_i = 0.0f;
  float mfy_i = 0.0f;
  float mfz_i = 0.0f;
  const float msign = paramb.mforce_sign;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;

    float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

    float fc12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);
    find_fn(bs, rcinv, d12, fc12, fn12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    if (sj2 <= kSpinZeroEpsSph) continue;

    const float sdot = nep_spin_dot3(si, sj);
    const float sj_norm = sqrtf(sj2);
    const float denom = si_norm * sj_norm;
    const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

    float Tk[KMAX_TERM + 1] = {0.0f};
    float dTk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    dTk[0] = 0.0f;
    nep_spin_fill_Tk_and_dTk<KMAX_TERM>(c, kmax_dmi, Tk, dTk);

    float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
    float dc_dsj[3] = {0.0f, 0.0f, 0.0f};
    nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);

    float sixsj[3];
    nep_spin_cross3(si, sj, sixsj);
    const float dmi = nep_spin_dot3(sixsj, rhat);

    float sjxr[3];
    nep_spin_cross3(sj, rhat, sjxr);
    float rxsi[3];
    nep_spin_cross3(rhat, si, rxsi);

    const float* c_base_pair12 = c_base_t1 + t2;
    const float* c_base_pair21 = c_base_init + t2 * paramb.num_types + t1;

    float mfx_j = 0.0f;
    float mfy_j = 0.0f;
    float mfz_j = 0.0f;

    for (int n = 0; n < nspin; ++n) {
      const float* c_n_ptr12 = c_base_pair12 + n * stride_n + dmi_block0 * c_step;
      const float* c_n_ptr21 = c_base_pair21 + n * stride_n + dmi_block0 * c_step;
      const float* c_kk_ptr12 = c_n_ptr12;
      const float* c_kk_ptr21 = c_n_ptr21;
      for (int kk = 0; kk <= KMAX_TERM; ++kk) {
        const int block = dmi_block0 + kk;
        float gn12 = 0.0f;
        float gn21 = 0.0f;
        const float* c_kb_ptr12 = c_kk_ptr12;
        const float* c_kb_ptr21 = c_kk_ptr21;
        for (int kb = 0; kb <= bs; ++kb) {
          const float fnk = fn12[kb];
          gn12 += fnk * NEP_SPIN_LDG(c_kb_ptr12);
          if (!same_type) {
            gn21 += fnk * NEP_SPIN_LDG(c_kb_ptr21);
          }
          c_kb_ptr12 += stride_k;
          c_kb_ptr21 += stride_k;
        }
        c_kk_ptr12 += c_step;
        c_kk_ptr21 += c_step;

        if (same_type) gn21 = gn12;
        const int fp_idx = block * nspin + n;
        const float Fp1 = Fp1_cache[kk * MAX_NUM_N + n];
        const float coeff = msign * (Fp1 * gn12);
        const float coeff_other = gather ? (msign * (NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]) * gn21)) : 0.0f;
        const float grad_i[3] = {
          sjxr[0] * Tk[kk] + dmi * dTk[kk] * dc_dsi[0],
          sjxr[1] * Tk[kk] + dmi * dTk[kk] * dc_dsi[1],
          sjxr[2] * Tk[kk] + dmi * dTk[kk] * dc_dsi[2]};
        mfx_i += coeff * grad_i[0];
        mfy_i += coeff * grad_i[1];
        mfz_i += coeff * grad_i[2];
        if (gather) {
          mfx_i += coeff_other * grad_i[0];
          mfy_i += coeff_other * grad_i[1];
          mfz_i += coeff_other * grad_i[2];
        } else {
          const float grad_j[3] = {
            rxsi[0] * Tk[kk] + dmi * dTk[kk] * dc_dsj[0],
            rxsi[1] * Tk[kk] + dmi * dTk[kk] * dc_dsj[1],
            rxsi[2] * Tk[kk] + dmi * dTk[kk] * dc_dsj[2]};
          mfx_j += coeff * grad_j[0];
          mfy_j += coeff * grad_j[1];
          mfz_j += coeff * grad_j[2];
        }
      }
    }

    if (!gather) {
      atomicAdd(&g_mx[n2], static_cast<double>(mfx_j));
      atomicAdd(&g_my[n2], static_cast<double>(mfy_j));
      atomicAdd(&g_mz[n2], static_cast<double>(mfz_j));
    }
  }

  if (gather) {
    g_mx[n1] += static_cast<double>(mfx_i);
    g_my[n1] += static_cast<double>(mfy_i);
    g_mz[n1] += static_cast<double>(mfz_i);
  } else {
    atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
    atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
    atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
  }
}

template <int KMAX_PAIR, int KMAX_TERM>
static __global__ void find_mforce_radial_spin_spherical_md_noatomic_ani_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_mx,
  double* g_my,
  double* g_mz,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_ani = KMAX_TERM;
  const bool gather = (N == (N2 - N1));

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int ani_block0 = spin_blocks.ani_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

  // Cache central Fp once (avoid reloading per neighbor).
  float Fp1_cache[(KMAX_TERM + 1) * MAX_NUM_N];
  for (int kk = 0; kk <= kmax_ani; ++kk) {
    const int block = ani_block0 + kk;
    for (int n = 0; n < nspin; ++n) {
      const int fp_idx = block * nspin + n;
      Fp1_cache[kk * MAX_NUM_N + n] = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
    }
  }

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int c_stride_block = paramb.c_spin_block_stride;
  const int num_types_sq = paramb.num_types_sq;
  int stride_k = num_types_sq;
  int stride_n = (paramb.basis_size_radial + 1) * num_types_sq;
  int c_step = 0;
  if (mode == SPIN_C_PER_BLOCK) {
    c_step = c_stride_block;
  }
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) {
    c_base_init += paramb.c_spin_offset;
  }
  const float* c_base_t1 = c_base_init + t1 * paramb.num_types;

  float mfx_i = 0.0f;
  float mfy_i = 0.0f;
  float mfz_i = 0.0f;
  const float msign = paramb.mforce_sign;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;

    float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

    float fc12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);
    find_fn(bs, rcinv, d12, fc12, fn12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    if (sj2 <= kSpinZeroEpsSph) continue;

    const float sdot = nep_spin_dot3(si, sj);
    const float sj_norm = sqrtf(sj2);
    const float denom = si_norm * sj_norm;
    const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

    float Tk[KMAX_TERM + 1] = {0.0f};
    float dTk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    dTk[0] = 0.0f;
    nep_spin_fill_Tk_and_dTk<KMAX_TERM>(c, kmax_ani, Tk, dTk);

    float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
    float dc_dsj[3] = {0.0f, 0.0f, 0.0f};
    nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);

    const float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const float sj_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
    const float ani_scalar = si_r * sj_r;

    const float* c_base_pair12 = c_base_t1 + t2;
    const float* c_base_pair21 = c_base_init + t2 * paramb.num_types + t1;

    float mfx_j = 0.0f;
    float mfy_j = 0.0f;
    float mfz_j = 0.0f;

    for (int n = 0; n < nspin; ++n) {
      const float* c_n_ptr12 = c_base_pair12 + n * stride_n + ani_block0 * c_step;
      const float* c_n_ptr21 = c_base_pair21 + n * stride_n + ani_block0 * c_step;
      const float* c_kk_ptr12 = c_n_ptr12;
      const float* c_kk_ptr21 = c_n_ptr21;
      for (int kk = 0; kk <= KMAX_TERM; ++kk) {
        const int block = ani_block0 + kk;
        float gn12 = 0.0f;
        float gn21 = 0.0f;
        const float* c_kb_ptr12 = c_kk_ptr12;
        const float* c_kb_ptr21 = c_kk_ptr21;
        for (int kb = 0; kb <= bs; ++kb) {
          const float fnk = fn12[kb];
          gn12 += fnk * NEP_SPIN_LDG(c_kb_ptr12);
          if (!same_type) {
            gn21 += fnk * NEP_SPIN_LDG(c_kb_ptr21);
          }
          c_kb_ptr12 += stride_k;
          c_kb_ptr21 += stride_k;
        }
        c_kk_ptr12 += c_step;
        c_kk_ptr21 += c_step;
        
        if (same_type) gn21 = gn12;
        const int fp_idx = block * nspin + n;
        const float Fp1 = Fp1_cache[kk * MAX_NUM_N + n];
        const float coeff = msign * (Fp1 * gn12);
        const float coeff_other = gather ? (msign * (NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]) * gn21)) : 0.0f;
        const float grad_i[3] = {
          (sj_r * rhat[0]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[0],
          (sj_r * rhat[1]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[1],
          (sj_r * rhat[2]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[2]};
        mfx_i += coeff * grad_i[0];
        mfy_i += coeff * grad_i[1];
        mfz_i += coeff * grad_i[2];
        if (gather) {
          mfx_i += coeff_other * grad_i[0];
          mfy_i += coeff_other * grad_i[1];
          mfz_i += coeff_other * grad_i[2];
        } else {
          const float grad_j[3] = {
            (si_r * rhat[0]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsj[0],
            (si_r * rhat[1]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsj[1],
            (si_r * rhat[2]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsj[2]};
          mfx_j += coeff * grad_j[0];
          mfy_j += coeff * grad_j[1];
          mfz_j += coeff * grad_j[2];
        }
      }
    }

    if (!gather) {
      atomicAdd(&g_mx[n2], static_cast<double>(mfx_j));
      atomicAdd(&g_my[n2], static_cast<double>(mfy_j));
      atomicAdd(&g_mz[n2], static_cast<double>(mfz_j));
    }
  }

  if (gather) {
    g_mx[n1] += static_cast<double>(mfx_i);
    g_my[n1] += static_cast<double>(mfy_i);
    g_mz[n1] += static_cast<double>(mfz_i);
  } else {
    atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
    atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
    atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
  }
}

template <int KMAX_PAIR, int KMAX_TERM>
static __global__ void find_mforce_radial_spin_spherical_md_noatomic_sia_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_mx,
  double* g_my,
  double* g_mz,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_sia = KMAX_TERM;
  const bool gather = (N == (N2 - N1));

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int sia_block0 = spin_blocks.sia_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

  // Cache central Fp once (avoid reloading per neighbor).
  float Fp1_cache[(KMAX_TERM + 1) * MAX_NUM_N];
  for (int kk = 0; kk <= kmax_sia; ++kk) {
    const int block = sia_block0 + kk;
    for (int n = 0; n < nspin; ++n) {
      const int fp_idx = block * nspin + n;
      Fp1_cache[kk * MAX_NUM_N + n] = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
    }
  }

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int c_stride_block = paramb.c_spin_block_stride;
  const int num_types_sq = paramb.num_types_sq;
  int stride_k = num_types_sq;
  int stride_n = (paramb.basis_size_radial + 1) * num_types_sq;
  int c_step = 0;
  if (mode == SPIN_C_PER_BLOCK) {
    c_step = c_stride_block;
  }
  const float* c_base_init = annmb.c;
  if (mode != SPIN_C_SHARED_LATTICE) {
    c_base_init += paramb.c_spin_offset;
  }
  const float* c_base_t1 = c_base_init + t1 * paramb.num_types;

  float mfx_i = 0.0f;
  float mfy_i = 0.0f;
  float mfz_i = 0.0f;
  const float msign = paramb.mforce_sign;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;

    float rhat[3] = {r12[0] / d12, r12[1] / d12, r12[2] / d12};

    float fc12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);
    find_fn(bs, rcinv, d12, fc12, fn12);

    const float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const float sia_scalar = si_r * si_r;

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
    float sj_r = 0.0f;
    float sia_scalar_j = 0.0f;
    if (gather && neighbor_has_spin) {
      sj_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
      sia_scalar_j = sj_r * sj_r;
    }

    float Tk[KMAX_TERM + 1] = {0.0f};
    float dTk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    dTk[0] = 0.0f;
    float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
    float dc_dsj[3] = {0.0f, 0.0f, 0.0f};
    if (neighbor_has_spin) {
      const float sdot = nep_spin_dot3(si, sj);
      const float sj_norm = sqrtf(sj2);
      const float denom = si_norm * sj_norm;
      const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));
      nep_spin_fill_Tk_and_dTk<KMAX_TERM>(c, kmax_sia, Tk, dTk);
      nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
    }

    const float* c_base_pair12 = c_base_t1 + t2;
    const float* c_base_pair21 = c_base_init + t2 * paramb.num_types + t1;

    float mfx_j = 0.0f;
    float mfy_j = 0.0f;
    float mfz_j = 0.0f;

    for (int n = 0; n < nspin; ++n) {
      const float* c_n_ptr12 = c_base_pair12 + n * stride_n + sia_block0 * c_step;
      const float* c_n_ptr21 = c_base_pair21 + n * stride_n + sia_block0 * c_step;
      const float* c_kk_ptr12 = c_n_ptr12;
      const float* c_kk_ptr21 = c_n_ptr21;
      for (int kk = 0; kk <= KMAX_TERM; ++kk) {
        if (kk > 0 && !neighbor_has_spin) {
          c_kk_ptr12 += c_step;
          c_kk_ptr21 += c_step;
          continue;
        }
        const int block = sia_block0 + kk;
        float gn12 = 0.0f;
        float gn21 = 0.0f;
        const float* c_kb_ptr12 = c_kk_ptr12;
        const float* c_kb_ptr21 = c_kk_ptr21;
        for (int kb = 0; kb <= bs; ++kb) {
          const float fnk = fn12[kb];
          gn12 += fnk * NEP_SPIN_LDG(c_kb_ptr12);
          if (neighbor_has_spin && kk > 0 && !same_type) {
            gn21 += fnk * NEP_SPIN_LDG(c_kb_ptr21);
          }
          c_kb_ptr12 += stride_k;
          c_kb_ptr21 += stride_k;
        }
        c_kk_ptr12 += c_step;
        c_kk_ptr21 += c_step;
        if (gather && same_type) {
          gn21 = gn12;
        }

        const int fp_idx = block * nspin + n;
        const float Fp1 = Fp1_cache[kk * MAX_NUM_N + n];
        const float coeff1 = msign * Fp1 * gn12;
        const float grad_i[3] = {
          (2.0f * si_r * rhat[0]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[0],
          (2.0f * si_r * rhat[1]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[1],
          (2.0f * si_r * rhat[2]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[2]};
        mfx_i += coeff1 * grad_i[0];
        mfy_i += coeff1 * grad_i[1];
        mfz_i += coeff1 * grad_i[2];
        if (gather) {
          if (neighbor_has_spin && kk > 0) {
            const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
            const float coeff2 = msign * Fp2 * gn21;
            const float w = sia_scalar_j * dTk[kk];
            mfx_i += coeff2 * (w * dc_dsi[0]);
            mfy_i += coeff2 * (w * dc_dsi[1]);
            mfz_i += coeff2 * (w * dc_dsi[2]);
          }
        } else {
          const float grad_j[3] = {
            sia_scalar * dTk[kk] * dc_dsj[0],
            sia_scalar * dTk[kk] * dc_dsj[1],
            sia_scalar * dTk[kk] * dc_dsj[2]};
          mfx_j += coeff1 * grad_j[0];
          mfy_j += coeff1 * grad_j[1];
          mfz_j += coeff1 * grad_j[2];
        }
      }
    }

    if (!gather) {
      atomicAdd(&g_mx[n2], static_cast<double>(mfx_j));
      atomicAdd(&g_my[n2], static_cast<double>(mfy_j));
      atomicAdd(&g_mz[n2], static_cast<double>(mfz_j));
    }
  }

  if (gather) {
    g_mx[n1] += static_cast<double>(mfx_i);
    g_my[n1] += static_cast<double>(mfy_i);
    g_mz[n1] += static_cast<double>(mfz_i);
  } else {
    atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
    atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
    atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
  }
}

// Fused mforce kernel: accumulates EX + DMI + ANI + SIA magnetic-force terms in
// a single pass over the neighbor list (small-box MD variant).
template <int KMAX_PAIR>
static __global__ void find_mforce_radial_spin_spherical_md_noatomic_fused_k(
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_Fp,
  double* g_mx,
  double* g_my,
  double* g_mz,
  int spin_offset)
{
  extern __shared__ float sh_fn[];
  float* fn12 = sh_fn + threadIdx.x * (2 * MAX_NUM_N);

  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  const int kmax_ex = spin_blocks.kmax_ex;
  const int kmax_dmi = spin_blocks.kmax_dmi;
  const int kmax_ani = spin_blocks.kmax_ani;
  const int kmax_sia = spin_blocks.kmax_sia;
  const int kmax_pair = spin_blocks.kmax_pair;
  const bool do_ex = (kmax_ex >= 0);
  const bool do_dmi = (kmax_dmi >= 0);
  const bool do_ani = (kmax_ani >= 0);
  const bool do_sia = (kmax_sia >= 0);
  if (!(do_ex || do_dmi || do_ani || do_sia)) return;

  const bool gather = (N == (N2 - N1));
  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);

  const int dmi_block0 = spin_blocks.dmi_block0;
  const int ani_block0 = spin_blocks.ani_block0;
  const int sia_block0 = spin_blocks.sia_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);
  float inv_si_norm = 1.0f / (si_norm + 1.0e-12f);

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;

  const SpinCMode mode = nep_spin_get_c_mode(paramb.num_c_spin, paramb.c_spin_block_stride);
  const int num_types_sq = paramb.num_types_sq;
  const int stride_k = num_types_sq;
  const int stride_n = (paramb.basis_size_radial + 1) * num_types_sq;
  const int stride_block = paramb.c_spin_block_stride;
  const int c_step = (mode == SPIN_C_PER_BLOCK) ? stride_block : 0;
  const float* c_base_init = (mode == SPIN_C_SHARED_LATTICE) ? annmb.c : (annmb.c + paramb.c_spin_offset);
  const float* c_base_t1 = c_base_init + t1 * paramb.num_types;

  float mfx_i = 0.0f;
  float mfy_i = 0.0f;
  float mfz_i = 0.0f;
  const float msign = paramb.mforce_sign;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = n1 + i1 * N;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const bool same_type = (t1 == t2);

    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 <= 0.0f) continue;
    float d12inv = 1.0f / d12;
    float rhat[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

    float fc12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = fminf(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);
    find_fn(bs, rcinv, d12, fc12, fn12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);
    if (!neighbor_has_spin && !do_sia) continue;

    float Tk[KMAX_PAIR + 1] = {0.0f};
    float dTk[KMAX_PAIR + 1] = {0.0f};
    Tk[0] = 1.0f;
    dTk[0] = 0.0f;

    float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
    float dc_dsj[3] = {0.0f, 0.0f, 0.0f};

    float phi = 0.0f;
    float dphi_dsi[3] = {0.0f, 0.0f, 0.0f};
    float dphi_dsj[3] = {0.0f, 0.0f, 0.0f};

    float sjxr[3] = {0.0f, 0.0f, 0.0f};
    float rxsi[3] = {0.0f, 0.0f, 0.0f};
    float dmi = 0.0f;

    float si_r = 0.0f;
    float sj_r = 0.0f;
    float ani_scalar = 0.0f;

    float sia_scalar = 0.0f;
    float sia_scalar_j = 0.0f;

    float ex_invariant[KMAX_PAIR + 1] = {0.0f};

    if (neighbor_has_spin) {
      const float sdot = nep_spin_dot3(si, sj);
      const float sj_norm = sqrtf(sj2);
      const float inv_sj_norm = 1.0f / (sj_norm + 1.0e-12f);
      const float denom = si_norm * sj_norm;
      const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));

      nep_spin_fill_Tk_and_dTk<KMAX_PAIR>(c, kmax_pair, Tk, dTk);
      nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);

      if (do_ex) {
        nep_spin_ex_phi_and_grads(
          paramb.spin_ex_phi_mode, si, sj, si_norm, sj_norm, inv_si_norm, inv_sj_norm, phi, dphi_dsi, dphi_dsj);
        nep_spin_fill_ex_invariant<KMAX_PAIR>(phi, Tk, kmax_ex, ex_invariant);
      }
      if (do_dmi) {
        float sixsj[3];
        nep_spin_cross3(si, sj, sixsj);
        dmi = nep_spin_dot3(sixsj, rhat);
        nep_spin_cross3(sj, rhat, sjxr);
        nep_spin_cross3(rhat, si, rxsi);
      }
      if (do_ani) {
        si_r = nep_spin_dot3(si, rhat);
        sj_r = nep_spin_dot3(sj, rhat);
        ani_scalar = si_r * sj_r;
      }
      if (do_sia && gather) {
        sj_r = nep_spin_dot3(sj, rhat);
        sia_scalar_j = sj_r * sj_r;
      }
    }

    if (do_sia) {
      si_r = nep_spin_dot3(si, rhat);
      sia_scalar = si_r * si_r;
    }

    float mfx_j = 0.0f;
    float mfy_j = 0.0f;
    float mfz_j = 0.0f;

    const float* c_base_pair12 = c_base_t1 + t2;
    const float* c_base_pair21 = c_base_init + t2 * paramb.num_types + t1;

    // EX term
    if (do_ex && neighbor_has_spin) {
      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr12 = c_base_pair12 + n * stride_n;
        const float* c_n_ptr21 = c_base_pair21 + n * stride_n;
        const float* c_kk_ptr12 = c_n_ptr12;
        const float* c_kk_ptr21 = c_n_ptr21;
        for (int kk = 0; kk <= kmax_ex; ++kk) {
          float gn12 = 0.0f;
          float gn21 = 0.0f;
          const float* c_kb_ptr12 = c_kk_ptr12;
          const float* c_kb_ptr21 = c_kk_ptr21;
          for (int kb = 0; kb <= bs; ++kb) {
            const float fnk = fn12[kb];
            gn12 += fnk * NEP_SPIN_LDG(c_kb_ptr12);
            if (!same_type) {
              gn21 += fnk * NEP_SPIN_LDG(c_kb_ptr21);
            }
            c_kb_ptr12 += stride_k;
            c_kb_ptr21 += stride_k;
          }
          c_kk_ptr12 += c_step;
          c_kk_ptr21 += c_step;
          if (same_type) gn21 = gn12;

          const int fp_idx = kk * nspin + n;
          const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
          const float coeff = msign * (Fp1 * gn12);
          const float coeff_other = gather ? (msign * (NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]) * gn21)) : 0.0f;
          const float grad_i[3] = {
            dphi_dsi[0] * Tk[kk] + phi * dTk[kk] * dc_dsi[0],
            dphi_dsi[1] * Tk[kk] + phi * dTk[kk] * dc_dsi[1],
            dphi_dsi[2] * Tk[kk] + phi * dTk[kk] * dc_dsi[2]};
          mfx_i += coeff * grad_i[0];
          mfy_i += coeff * grad_i[1];
          mfz_i += coeff * grad_i[2];
          if (gather) {
            mfx_i += coeff_other * grad_i[0];
            mfy_i += coeff_other * grad_i[1];
            mfz_i += coeff_other * grad_i[2];
          } else {
            const float grad_j[3] = {
              dphi_dsj[0] * Tk[kk] + phi * dTk[kk] * dc_dsj[0],
              dphi_dsj[1] * Tk[kk] + phi * dTk[kk] * dc_dsj[1],
              dphi_dsj[2] * Tk[kk] + phi * dTk[kk] * dc_dsj[2]};
            mfx_j += coeff * grad_j[0];
            mfy_j += coeff * grad_j[1];
            mfz_j += coeff * grad_j[2];
          }
        }
      }
    }

    // DMI term
    if (do_dmi && neighbor_has_spin) {
      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr12 = c_base_pair12 + n * stride_n + dmi_block0 * c_step;
        const float* c_n_ptr21 = c_base_pair21 + n * stride_n + dmi_block0 * c_step;
        const float* c_kk_ptr12 = c_n_ptr12;
        const float* c_kk_ptr21 = c_n_ptr21;
        for (int kk = 0; kk <= kmax_dmi; ++kk) {
          const int block = dmi_block0 + kk;
          float gn12 = 0.0f;
          float gn21 = 0.0f;
          const float* c_kb_ptr12 = c_kk_ptr12;
          const float* c_kb_ptr21 = c_kk_ptr21;
          for (int kb = 0; kb <= bs; ++kb) {
            const float fnk = fn12[kb];
            gn12 += fnk * NEP_SPIN_LDG(c_kb_ptr12);
            if (!same_type) {
              gn21 += fnk * NEP_SPIN_LDG(c_kb_ptr21);
            }
            c_kb_ptr12 += stride_k;
            c_kb_ptr21 += stride_k;
          }
          c_kk_ptr12 += c_step;
          c_kk_ptr21 += c_step;
          if (same_type) gn21 = gn12;

          const int fp_idx = block * nspin + n;
          const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
          const float coeff = msign * (Fp1 * gn12);
          const float coeff_other = gather ? (msign * (NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]) * gn21)) : 0.0f;
          const float grad_i[3] = {
            sjxr[0] * Tk[kk] + dmi * dTk[kk] * dc_dsi[0],
            sjxr[1] * Tk[kk] + dmi * dTk[kk] * dc_dsi[1],
            sjxr[2] * Tk[kk] + dmi * dTk[kk] * dc_dsi[2]};
          mfx_i += coeff * grad_i[0];
          mfy_i += coeff * grad_i[1];
          mfz_i += coeff * grad_i[2];
          if (gather) {
            mfx_i += coeff_other * grad_i[0];
            mfy_i += coeff_other * grad_i[1];
            mfz_i += coeff_other * grad_i[2];
          } else {
            const float grad_j[3] = {
              rxsi[0] * Tk[kk] + dmi * dTk[kk] * dc_dsj[0],
              rxsi[1] * Tk[kk] + dmi * dTk[kk] * dc_dsj[1],
              rxsi[2] * Tk[kk] + dmi * dTk[kk] * dc_dsj[2]};
            mfx_j += coeff * grad_j[0];
            mfy_j += coeff * grad_j[1];
            mfz_j += coeff * grad_j[2];
          }
        }
      }
    }

    // ANI term
    if (do_ani && neighbor_has_spin) {
      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr12 = c_base_pair12 + n * stride_n + ani_block0 * c_step;
        const float* c_n_ptr21 = c_base_pair21 + n * stride_n + ani_block0 * c_step;
        const float* c_kk_ptr12 = c_n_ptr12;
        const float* c_kk_ptr21 = c_n_ptr21;
        for (int kk = 0; kk <= kmax_ani; ++kk) {
          const int block = ani_block0 + kk;
          float gn12 = 0.0f;
          float gn21 = 0.0f;
          const float* c_kb_ptr12 = c_kk_ptr12;
          const float* c_kb_ptr21 = c_kk_ptr21;
          for (int kb = 0; kb <= bs; ++kb) {
            const float fnk = fn12[kb];
            gn12 += fnk * NEP_SPIN_LDG(c_kb_ptr12);
            if (!same_type) {
              gn21 += fnk * NEP_SPIN_LDG(c_kb_ptr21);
            }
            c_kb_ptr12 += stride_k;
            c_kb_ptr21 += stride_k;
          }
          c_kk_ptr12 += c_step;
          c_kk_ptr21 += c_step;
          if (same_type) gn21 = gn12;

          const int fp_idx = block * nspin + n;
          const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
          const float coeff = msign * (Fp1 * gn12);
          const float coeff_other = gather ? (msign * (NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]) * gn21)) : 0.0f;
          const float grad_i[3] = {
            (sj_r * rhat[0]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[0],
            (sj_r * rhat[1]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[1],
            (sj_r * rhat[2]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[2]};
          mfx_i += coeff * grad_i[0];
          mfy_i += coeff * grad_i[1];
          mfz_i += coeff * grad_i[2];
          if (gather) {
            mfx_i += coeff_other * grad_i[0];
            mfy_i += coeff_other * grad_i[1];
            mfz_i += coeff_other * grad_i[2];
          } else {
            const float grad_j[3] = {
              (si_r * rhat[0]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsj[0],
              (si_r * rhat[1]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsj[1],
              (si_r * rhat[2]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsj[2]};
            mfx_j += coeff * grad_j[0];
            mfy_j += coeff * grad_j[1];
            mfz_j += coeff * grad_j[2];
          }
        }
      }
    }

    // SIA term
    if (do_sia) {
      for (int n = 0; n < nspin; ++n) {
        const float* c_n_ptr12 = c_base_pair12 + n * stride_n + sia_block0 * c_step;
        const float* c_n_ptr21 = c_base_pair21 + n * stride_n + sia_block0 * c_step;
        const float* c_kk_ptr12 = c_n_ptr12;
        const float* c_kk_ptr21 = c_n_ptr21;
        for (int kk = 0; kk <= kmax_sia; ++kk) {
          if (kk > 0 && !neighbor_has_spin) {
            c_kk_ptr12 += c_step;
            c_kk_ptr21 += c_step;
            continue;
          }
          const int block = sia_block0 + kk;
          float gn12 = 0.0f;
          float gn21 = 0.0f;
          const float* c_kb_ptr12 = c_kk_ptr12;
          const float* c_kb_ptr21 = c_kk_ptr21;
          for (int kb = 0; kb <= bs; ++kb) {
            const float fnk = fn12[kb];
            gn12 += fnk * NEP_SPIN_LDG(c_kb_ptr12);
            if (neighbor_has_spin && kk > 0 && !same_type) {
              gn21 += fnk * NEP_SPIN_LDG(c_kb_ptr21);
            }
            c_kb_ptr12 += stride_k;
            c_kb_ptr21 += stride_k;
          }
          c_kk_ptr12 += c_step;
          c_kk_ptr21 += c_step;
          if (gather && same_type) gn21 = gn12;

          const int fp_idx = block * nspin + n;
          const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
          const float coeff1 = msign * (Fp1 * gn12);
          const float grad_i[3] = {
            (2.0f * si_r * rhat[0]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[0],
            (2.0f * si_r * rhat[1]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[1],
            (2.0f * si_r * rhat[2]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[2]};
          mfx_i += coeff1 * grad_i[0];
          mfy_i += coeff1 * grad_i[1];
          mfz_i += coeff1 * grad_i[2];

          if (gather) {
            if (neighbor_has_spin && kk > 0) {
              const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
              const float coeff2 = msign * (Fp2 * gn21);
              const float w = sia_scalar_j * dTk[kk];
              mfx_i += coeff2 * (w * dc_dsi[0]);
              mfy_i += coeff2 * (w * dc_dsi[1]);
              mfz_i += coeff2 * (w * dc_dsi[2]);
            }
          } else {
            const float grad_j[3] = {
              sia_scalar * dTk[kk] * dc_dsj[0],
              sia_scalar * dTk[kk] * dc_dsj[1],
              sia_scalar * dTk[kk] * dc_dsj[2]};
            mfx_j += coeff1 * grad_j[0];
            mfy_j += coeff1 * grad_j[1];
            mfz_j += coeff1 * grad_j[2];
          }
        }
      }
    }

    if (!gather) {
      atomicAdd(&g_mx[n2], static_cast<double>(mfx_j));
      atomicAdd(&g_my[n2], static_cast<double>(mfy_j));
      atomicAdd(&g_mz[n2], static_cast<double>(mfz_j));
    }
  }

  if (gather) {
    g_mx[n1] += static_cast<double>(mfx_i);
    g_my[n1] += static_cast<double>(mfy_i);
    g_mz[n1] += static_cast<double>(mfz_i);
  } else {
    atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
    atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
    atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
  }
}

// Split kernels (NEP-style, no neighbor atomics): per-interaction mforce.
// Spin spherical magnetic forces (spin derivatives) - large-box MD variant, NEP-style (no neighbor atomics).

// END embedded NEP_Spin kernels


namespace {

static __global__ void pack_sp4_aos_to_spin_soa(
  const int n,
  const double* __restrict__ sp4_aos,
  float* __restrict__ spin_soa)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int idx = 4 * i;
    const double sx = sp4_aos[idx + 0];
    const double sy = sp4_aos[idx + 1];
    const double sz = sp4_aos[idx + 2];
    const double sm = sp4_aos[idx + 3];
    spin_soa[i]       = static_cast<float>(sx * sm);
    spin_soa[i + n]   = static_cast<float>(sy * sm);
    spin_soa[i + 2*n] = static_cast<float>(sz * sm);
  }
}

static __global__ void build_x12_from_nl(
  const int natoms,
  const int nlocal,
  const int force_mic_with_ghosts,
  const Box box,
  const int* __restrict__ NN,
  const int* __restrict__ NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  float* __restrict__ x12,
  float* __restrict__ y12,
  float* __restrict__ z12)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlocal) return;

  // When consuming LAMMPS-provided neighbor lists, ghost coordinates already
  // encode the correct periodic image, so dx = x[j]-x[i] is a nearest-image
  // displacement. In that case, applying MIC again is redundant and can
  // change the physics if multiple periodic images are present.
  //
  // Historically we only applied MIC when there were no ghosts at all.
  // In practice, LAMMPS ghost images are usually already nearest-image, but
  // we allow forcing MIC when ghosts are present for robustness/debugging.
  const bool need_mic = (natoms == nlocal) || (force_mic_with_ghosts != 0);

  const double xi = x[i];
  const double yi = y[i];
  const double zi = z[i];

  const int nn = NN[i];

  for (int s = 0; s < nn; ++s) {
    const int idx = i + natoms * s;
    const int j = NL[idx];
    double dx = x[j] - xi;
    double dy = y[j] - yi;
    double dz = z[j] - zi;
    if (need_mic) apply_mic(box, dx, dy, dz);
    x12[idx] = static_cast<float>(dx);
    y12[idx] = static_cast<float>(dy);
    z12[idx] = static_cast<float>(dz);
  }
}

template <bool kWriteDescriptors, int kMaxDim>
static __device__ __forceinline__ void compute_all_q_and_ann_spin_v2_body(
  const int N,
  const int nlocal,
  const int* g_NN_r,
  const int* g_NL_r,
  const int* g_NN_a,
  const int* g_NL_a,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_r,
  const float* __restrict__ g_y12_r,
  const float* __restrict__ g_z12_r,
  const float* __restrict__ g_x12_a,
  const float* __restrict__ g_y12_a,
  const float* __restrict__ g_z12_a,
  const float* __restrict__ g_spin,
  float* g_descriptors,
  float* g_sum_fxyz,
  float* g_sum_fxyz_0,
  float* g_sum_fxyz_c,
  float* g_sum_fxyz_Ax,
  float* g_sum_fxyz_Ay,
  float* g_sum_fxyz_Az,
  float* g_sum_fxyz_D,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp,
  float* __restrict__ fn12_ws)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= nlocal) return;

  const int t1 = g_type[n1];
  float q[kMaxDim];
  for (int d = 0; d < annmb.dim; ++d) q[d] = 0.0f;
  float* __restrict__ fn12 = fn12_ws;
  const float si[3] = {
    __ldg(&g_spin[n1]),
    __ldg(&g_spin[n1 + N]),
    __ldg(&g_spin[n1 + N * 2])};
  const float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  const bool central_has_spin = (si2 > 1.0e-12f);

  const int radial_offset = paramb.n_max_radial + 1;
  const int onsite_offset = nep_spin_block0_offset(paramb);
  const int nspin2 = nep_spin_2body_count(paramb);
  const int nspin3 = nep_spin_3body_count(paramb);
  const int abc_count = nep_spin_3body_abc_count(paramb);

  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  const int neighbor_number_r = g_NN_r[n1];
  for (int i1 = 0; i1 < neighbor_number_r; ++i1) {
    const int index = n1 + N * i1;
    const int n2 = g_NL_r[index];
    const int t2 = g_type[n2];
    const float x12 = g_x12_r[index];
    const float y12 = g_y12_r[index];
    const float z12 = g_z12_r[index];
    const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
    if (!(d12 > 0.0f)) continue;
    const float inv_d12 = 1.0f / d12;
    const float rc = (paramb.rc_radial_by_type[t1] + paramb.rc_radial_by_type[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    find_fn(bs, rcinv, d12, fc12, fn12);

    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12 = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      q[n] += gn12;
    }

    if (central_has_spin && nspin2 > 0) {
      const float sj[3] = {
        __ldg(&g_spin[n2]),
        __ldg(&g_spin[n2 + N]),
        __ldg(&g_spin[n2 + N * 2])};
      const float rhat[3] = {x12 * inv_d12, y12 * inv_d12, z12 * inv_d12};
      const float si_dot_sj = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
      const float si_dot_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
      const float sj_dot_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
      const float cross[3] = {
        si[1] * sj[2] - si[2] * sj[1],
        si[2] * sj[0] - si[0] * sj[2],
        si[0] * sj[1] - si[1] * sj[0]};
      const float phi_ex = si_dot_sj;
      const float phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
      const float phi_ani = si_dot_r * sj_dot_r;
      const float phi_sia = si_dot_r * si_dot_r;
      for (int n = 0; n < nspin2; ++n) {
        float gn = 0.0f;
        for (int k = 0; k <= bs; ++k) {
          gn += fn12[k] * annmb.c[nep_spin_c_index_2body(paramb, n, k, t1, t2)];
        }
        q[nep_spin_block1_index(paramb, 0, n)] += gn * phi_ex;
        q[nep_spin_block1_index(paramb, 1, n)] += gn * phi_dmi;
        q[nep_spin_block1_index(paramb, 2, n)] += gn * phi_ani;
        q[nep_spin_block1_index(paramb, 3, n)] += gn * phi_sia;
      }
    }
  }

  if (central_has_spin && paramb.spin_pmax > 0) {
    if (paramb.spin_onsite_basis_mode == 0) {
      float m2p = si2;
      for (int p = 0; p < paramb.spin_pmax; ++p) {
        q[onsite_offset + p] = m2p;
        m2p *= si2;
      }
    } else {
      float y = si2;
      float yref = nep_spin_type_yref(t1, paramb.spin_onsite_basis_mode);
      if (paramb.spin_onsite_basis_mode == 2) {
        y = sqrtf(si2);
      }
      if (yref <= 0.0f) yref = 1.0f;
      float x = (y - yref) / (y + yref + 1.0e-12f);
      x = fminf(1.0f, fmaxf(-1.0f, x));
      float Tp[9] = {1.0f};
      if (paramb.spin_pmax >= 1) Tp[1] = x;
      for (int p = 2; p <= paramb.spin_pmax; ++p) {
        Tp[p] = 2.0f * x * Tp[p - 1] - Tp[p - 2];
      }
      for (int p = 1; p <= paramb.spin_pmax; ++p) {
        q[onsite_offset + p - 1] = Tp[p];
      }
    }
  }

  int bs_ang = paramb.basis_size_angular;
  if (bs_ang >= MAX_NUM_N) bs_ang = MAX_NUM_N - 1;
  const int neighbor_number_a = g_NN_a[n1];
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number_a; ++i1) {
      const int index = n1 + N * i1;
      const int n2 = g_NL_a[index];
      const float x12 = g_x12_a[index];
      const float y12 = g_y12_a[index];
      const float z12 = g_z12_a[index];
      const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
      if (!(d12 > 0.0f)) continue;
      const int t2 = g_type[n2];
      const float rc = (paramb.rc_angular_by_type[t1] + paramb.rc_angular_by_type[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      find_fn(bs_ang, rcinv, d12, fc12, fn12);
      float gn12 = 0.0f;
      for (int k = 0; k <= bs_ang; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
      }
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
    }
    find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + radial_offset);
    for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
      g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
    }
  }

  bs_ang = paramb.basis_size_spin_angular;
  if (bs_ang >= MAX_NUM_N) bs_ang = MAX_NUM_N - 1;
  if (central_has_spin && nspin3 > 0) {
    for (int n_base = 0; n_base < nspin3; n_base += SPIN3_N_TILE) {
      float s0[SPIN3_N_TILE][MAX_SPIN_ABC] = {{0.0f}};
      float sc[SPIN3_N_TILE][MAX_SPIN_ABC] = {{0.0f}};
      float Ax[SPIN3_N_TILE][MAX_SPIN_ABC] = {{0.0f}};
      float Ay[SPIN3_N_TILE][MAX_SPIN_ABC] = {{0.0f}};
      float Az[SPIN3_N_TILE][MAX_SPIN_ABC] = {{0.0f}};
      float D[SPIN3_N_TILE][MAX_SPIN_ABC] = {{0.0f}};
      for (int i1 = 0; i1 < neighbor_number_a; ++i1) {
        const int index = n1 + N * i1;
        const int n2 = g_NL_a[index];
        const int t2 = g_type[n2];
        const float x12 = g_x12_a[index];
        const float y12 = g_y12_a[index];
        const float z12 = g_z12_a[index];
        const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
        if (!(d12 > 0.0f)) continue;
        const float rc = (paramb.rc_angular_by_type[t1] + paramb.rc_angular_by_type[t2]) * 0.5f;
        const float rcinv = 1.0f / rc;
        float fc12;
        find_fc(rc, rcinv, d12, fc12);
        find_fn(bs_ang, rcinv, d12, fc12, fn12);
        const float sjx = __ldg(&g_spin[n2]);
        const float sjy = __ldg(&g_spin[n2 + N]);
        const float sjz = __ldg(&g_spin[n2 + N * 2]);
        const float d12inv = 1.0f / d12;
        const float rhat[3] = {x12 * d12inv, y12 * d12inv, z12 * d12inv};
        const float cross[3] = {
          si[1] * sjz - si[2] * sjy,
          si[2] * sjx - si[0] * sjz,
          si[0] * sjy - si[1] * sjx};
        const float phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
        for (int tn = 0; tn < SPIN3_N_TILE; ++tn) {
          const int n = n_base + tn;
          if (n >= nspin3) break;
          float gn = 0.0f;
          for (int k = 0; k <= bs_ang; ++k) {
            gn += fn12[k] * annmb.c[nep_spin_c_index_3body(paramb, n, k, t1, t2)];
          }
          const float gn_x_si_dot_sj = gn * (si[0] * sjx + si[1] * sjy + si[2] * sjz);
          accumulate_spin3body_all(
            paramb.l_max_spin_angular,
            d12,
            x12,
            y12,
            z12,
            gn,
            gn_x_si_dot_sj,
            gn * sjx,
            gn * sjy,
            gn * sjz,
            gn * phi_dmi,
            s0[tn],
            sc[tn],
            Ax[tn],
            Ay[tn],
            Az[tn],
            D[tn]);
        }
      }
      for (int tn = 0; tn < SPIN3_N_TILE; ++tn) {
        const int n = n_base + tn;
        if (n >= nspin3) break;
        for (int abc = 0; abc < abc_count; ++abc) {
          const int offset = (n * abc_count + abc) * N + n1;
          g_sum_fxyz_0[offset] = s0[tn][abc];
          g_sum_fxyz_c[offset] = sc[tn][abc];
          g_sum_fxyz_Ax[offset] = Ax[tn][abc];
          g_sum_fxyz_Ay[offset] = Ay[tn][abc];
          g_sum_fxyz_Az[offset] = Az[tn][abc];
          g_sum_fxyz_D[offset] = D[tn][abc];
        }
        for (int L = 1; L <= paramb.l_max_spin_angular; ++L) {
          float q2 = 0.0f, q3 = 0.0f, q4 = 0.0f, qD0 = 0.0f, qDc = 0.0f;
          if (L == 1) {
            q2 = find_q_one<1>(sc[tn]);
            q3 = find_q_one<1>(Ax[tn]) + find_q_one<1>(Ay[tn]) + find_q_one<1>(Az[tn]);
            q4 = find_q_cross_one<1>(s0[tn], sc[tn]);
            qD0 = find_q_cross_one<1>(s0[tn], D[tn]);
            qDc = find_q_cross_one<1>(sc[tn], D[tn]);
          } else if (L == 2) {
            q2 = find_q_one<2>(sc[tn]);
            q3 = find_q_one<2>(Ax[tn]) + find_q_one<2>(Ay[tn]) + find_q_one<2>(Az[tn]);
            q4 = find_q_cross_one<2>(s0[tn], sc[tn]);
            qD0 = find_q_cross_one<2>(s0[tn], D[tn]);
            qDc = find_q_cross_one<2>(sc[tn], D[tn]);
          } else if (L == 3) {
            q2 = find_q_one<3>(sc[tn]);
            q3 = find_q_one<3>(Ax[tn]) + find_q_one<3>(Ay[tn]) + find_q_one<3>(Az[tn]);
            q4 = find_q_cross_one<3>(s0[tn], sc[tn]);
            qD0 = find_q_cross_one<3>(s0[tn], D[tn]);
            qDc = find_q_cross_one<3>(sc[tn], D[tn]);
          } else if (L == 4) {
            q2 = find_q_one<4>(sc[tn]);
            q3 = find_q_one<4>(Ax[tn]) + find_q_one<4>(Ay[tn]) + find_q_one<4>(Az[tn]);
            q4 = find_q_cross_one<4>(s0[tn], sc[tn]);
            qD0 = find_q_cross_one<4>(s0[tn], D[tn]);
            qDc = find_q_cross_one<4>(sc[tn], D[tn]);
          }
          q[nep_spin_block2_core_index(paramb, 0, n, L - 1)] = q2;
          q[nep_spin_block2_core_index(paramb, 1, n, L - 1)] = q3;
          q[nep_spin_block2_core_index(paramb, 2, n, L - 1)] = q4;
          q[nep_spin_block2_core_index(paramb, 3, n, L - 1)] = qD0;
          q[nep_spin_block2_core_index(paramb, 4, n, L - 1)] = qDc;
        }
        if (paramb.l_max_spin_angular >= 2) {
          float grad_s0_mix[5];
          float grad_sc_mix[5];
          const float q4b = compute_q4b_l2(sc[tn] + 3);
          float qmix = 0.0f;
          accumulate_mix_q4b_l2(s0[tn] + 3, sc[tn] + 3, qmix, grad_s0_mix, grad_sc_mix);
          q[nep_spin_block2_g1_index(paramb, 0, n)] = q4b;
          q[nep_spin_block2_g1_index(paramb, 1, n)] = qmix;
        }
      }
    }
    for (int n1_pair = 0; n1_pair < nspin3; ++n1_pair) {
      for (int n2_pair = n1_pair + 1; n2_pair < nspin3; ++n2_pair) {
        for (int L = 1; L <= paramb.l_max_spin_angular; ++L) {
          float qAcross = 0.0f;
          const int start = L * L - 1;
          const int terms = 2 * L + 1;
          for (int k = 0; k < terms; ++k) {
            const int abc = start + k;
            const float weight = (k == 0 ? 1.0f : 2.0f) * C3B[abc];
            const int idx1 = (n1_pair * abc_count + abc) * N + n1;
            const int idx2 = (n2_pair * abc_count + abc) * N + n1;
            qAcross += weight *
                       (g_sum_fxyz_Ax[idx1] * g_sum_fxyz_Ax[idx2] +
                        g_sum_fxyz_Ay[idx1] * g_sum_fxyz_Ay[idx2] +
                        g_sum_fxyz_Az[idx1] * g_sum_fxyz_Az[idx2]);
          }
          q[nep_spin_block2_across_index(paramb, n1_pair, n2_pair, L - 1)] = qAcross;
        }
      }
    }
  }

  if constexpr (kWriteDescriptors) {
    for (int d = 0; d < annmb.dim; ++d) {
      g_descriptors[n1 + d * N] = q[d];
    }
  }

  for (int d = 0; d < annmb.dim; ++d) {
    q[d] *= g_q_scaler[d];
  }
  apply_ann_spin_one_atom_tmpl<kMaxDim>(n1, N, paramb, annmb, g_type, q, g_q_scaler, g_pe, g_Fp);
}

template <int kMaxDim, bool kWriteDescriptors>
static __global__ void compute_all_q_and_ann_spin_v2_tmpl(
  const int N,
  const int nlocal,
  const int* g_NN_r,
  const int* g_NL_r,
  const int* g_NN_a,
  const int* g_NL_a,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_r,
  const float* __restrict__ g_y12_r,
  const float* __restrict__ g_z12_r,
  const float* __restrict__ g_x12_a,
  const float* __restrict__ g_y12_a,
  const float* __restrict__ g_z12_a,
  const float* __restrict__ g_spin,
  float* g_descriptors,
  float* g_sum_fxyz,
  float* g_sum_fxyz_0,
  float* g_sum_fxyz_c,
  float* g_sum_fxyz_Ax,
  float* g_sum_fxyz_Ay,
  float* g_sum_fxyz_Az,
  float* g_sum_fxyz_D,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
{
  extern __shared__ float sh_fn12[];
  float* fn12_ws = sh_fn12 + threadIdx.x * MAX_NUM_N;
  compute_all_q_and_ann_spin_v2_body<kWriteDescriptors, kMaxDim>(
    N,
    nlocal,
    g_NN_r,
    g_NL_r,
    g_NN_a,
    g_NL_a,
    paramb,
    annmb,
    g_type,
    g_x12_r,
    g_y12_r,
    g_z12_r,
    g_x12_a,
    g_y12_a,
    g_z12_a,
    g_spin,
    g_descriptors,
    g_sum_fxyz,
    g_sum_fxyz_0,
    g_sum_fxyz_c,
    g_sum_fxyz_Ax,
    g_sum_fxyz_Ay,
    g_sum_fxyz_Az,
    g_sum_fxyz_D,
    g_q_scaler,
    g_pe,
    g_Fp,
    fn12_ws);
}

static __global__ void repack_neighbors_local_to_natoms_stride(
  const int nlocal,
  const int natoms,
  const int* __restrict__ nn_in,
  const int* __restrict__ nl_in,
  int* __restrict__ nn_out,
  int* __restrict__ nl_out)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlocal) return;

  const int nn = nn_in[i];
  nn_out[i] = nn;
  for (int s = 0; s < nn; ++s) {
    nl_out[i + natoms * s] = nl_in[i + nlocal * s];
  }
}

static __global__ void scatter_mforce_soa_to_fm_aos_add_scale(
  const int n,
  const double* __restrict__ mx,
  const double* __restrict__ my,
  const double* __restrict__ mz,
  const double inv_hbar,
  double* __restrict__ fm_aos)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int idx = 3 * i;
    fm_aos[idx + 0] += inv_hbar * mx[i];
    fm_aos[idx + 1] += inv_hbar * my[i];
    fm_aos[idx + 2] += inv_hbar * mz[i];
  }
}

static __global__ void reduce_ev_totals_spin(
  const int nlocal,
  const int natoms,
  const double* __restrict__ pe,     // length natoms (local part valid)
  const double* __restrict__ virial, // length 9*natoms (SoA)
  double* __restrict__ totals)       // length 7
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlocal) return;

  if (pe) atomicAdd(&totals[0], pe[i]);
  if (virial) {
    const double vxx = virial[i + natoms * 0];
    const double vyy = virial[i + natoms * 1];
    const double vzz = virial[i + natoms * 2];
    const double vxy = 0.5 * (virial[i + natoms * 3] + virial[i + natoms * 6]);
    const double vxz = 0.5 * (virial[i + natoms * 4] + virial[i + natoms * 7]);
    const double vyz = 0.5 * (virial[i + natoms * 5] + virial[i + natoms * 8]);

    // LAMMPS ordering: (xx,yy,zz,xy,xz,yz)
    atomicAdd(&totals[1], vxx);
    atomicAdd(&totals[2], vyy);
    atomicAdd(&totals[3], vzz);
    atomicAdd(&totals[4], vxy);
    atomicAdd(&totals[5], vxz);
    atomicAdd(&totals[6], vyz);
  }
}

static __global__ void reduce_virial_raw9_spin(
  const int nlocal,
  const int natoms,
  const double* __restrict__ virial, // length 9*natoms (SoA)
  double* __restrict__ totals9)      // length 9
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlocal) return;
  if (!virial) return;

  #pragma unroll
  for (int k = 0; k < 9; ++k) {
    atomicAdd(&totals9[k], virial[i + natoms * k]);
  }
}

template <typename T>
static __global__ void nep_spin_scan_nonfinite_linear(
  const T* __restrict__ data,
  const int n,
  int* __restrict__ first_idx,
  int* __restrict__ count)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const double v = static_cast<double>(data[idx]);
  if (!isfinite(v)) {
    atomicMin(first_idx, idx);
    atomicAdd(count, 1);
  }
}

template <typename T>
static __global__ void nep_spin_scan_nonfinite_soa(
  const T* __restrict__ data,
  const int nlocal,
  const int natoms,
  const int ncomp,
  int* __restrict__ first_linear, // linear in [0, nlocal*ncomp)
  int* __restrict__ count)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = nlocal * ncomp;
  if (idx >= total) return;
  const int i = idx % nlocal;
  const int c = idx / nlocal;
  const double v = static_cast<double>(data[i + natoms * c]);
  if (!isfinite(v)) {
    atomicMin(first_linear, idx);
    atomicAdd(count, 1);
  }
}

static void nep_spin_backend_dump_atom_neighbors(
  const char* stage,
  const int i,
  const int nlocal,
  const int natoms,
  const int* type_dev,
  const double* pos_soa,   // 3*natoms, SoA
  const float* spin_soa,   // 3*natoms, SoA
  const int* NN_radial_dev,
  const int* NL_radial_dev,
  const float* x12_r,
  const float* y12_r,
  const float* z12_r,
  gpuStream_t st)
{
  (void)st;
  if (i < 0 || i >= nlocal) return;

  int type_i = -1;
  double xi = 0.0, yi = 0.0, zi = 0.0;
  float sxi = 0.0f, syi = 0.0f, szi = 0.0f;
  CHECK(gpuMemcpy(&type_i, type_dev + i, sizeof(int), gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(&xi, pos_soa + i, sizeof(double), gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(&yi, pos_soa + (i + natoms), sizeof(double), gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(&zi, pos_soa + (i + 2 * natoms), sizeof(double), gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(&sxi, spin_soa + i, sizeof(float), gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(&syi, spin_soa + (i + natoms), sizeof(float), gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(&szi, spin_soa + (i + 2 * natoms), sizeof(float), gpuMemcpyDeviceToHost));
  const double smag = std::sqrt(double(sxi) * double(sxi) + double(syi) * double(syi) + double(szi) * double(szi));

  int nnr = 0;
  CHECK(gpuMemcpy(&nnr, NN_radial_dev + i, sizeof(int), gpuMemcpyDeviceToHost));

  std::fprintf(
    stderr,
    "[NEP_SPIN_GPU backend] context at %s: i=%d type=%d x=(%.16g %.16g %.16g) S=(%.8g %.8g %.8g) |S|=%.8g nn_r=%d\n",
    stage,
    i,
    type_i,
    xi,
    yi,
    zi,
    (double)sxi,
    (double)syi,
    (double)szi,
    smag,
    nnr);

  const int max_show = 4;
  const int show = (nnr < max_show) ? nnr : max_show;
  for (int s = 0; s < show; ++s) {
    const int idx = i + natoms * s;
    int j = -1;
    CHECK(gpuMemcpy(&j, NL_radial_dev + idx, sizeof(int), gpuMemcpyDeviceToHost));
    if (j < 0 || j >= natoms) {
      std::fprintf(stderr, "[NEP_SPIN_GPU backend]   NL_r[%d]=%d (out of range)\n", s, j);
      continue;
    }

    double xj = 0.0, yj = 0.0, zj = 0.0;
    CHECK(gpuMemcpy(&xj, pos_soa + j, sizeof(double), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(&yj, pos_soa + (j + natoms), sizeof(double), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(&zj, pos_soa + (j + 2 * natoms), sizeof(double), gpuMemcpyDeviceToHost));
    const double dx = xj - xi;
    const double dy = yj - yi;
    const double dz = zj - zi;
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

    float dx12 = 0.0f, dy12 = 0.0f, dz12 = 0.0f;
    CHECK(gpuMemcpy(&dx12, x12_r + idx, sizeof(float), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(&dy12, y12_r + idx, sizeof(float), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(&dz12, z12_r + idx, sizeof(float), gpuMemcpyDeviceToHost));
    const double r12 = std::sqrt(double(dx12) * double(dx12) + double(dy12) * double(dy12) + double(dz12) * double(dz12));

    float sxj = 0.0f, syj = 0.0f, szj = 0.0f;
    CHECK(gpuMemcpy(&sxj, spin_soa + j, sizeof(float), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(&syj, spin_soa + (j + natoms), sizeof(float), gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(&szj, spin_soa + (j + 2 * natoms), sizeof(float), gpuMemcpyDeviceToHost));
    const double smag_j = std::sqrt(double(sxj) * double(sxj) + double(syj) * double(syj) + double(szj) * double(szj));

    std::fprintf(
      stderr,
      "[NEP_SPIN_GPU backend]   s=%d j=%d r_pos=%.10g r_x12=%.10g  Sj=(%.6g %.6g %.6g) |Sj|=%.6g\n",
      s,
      j,
      r,
      r12,
      (double)sxj,
      (double)syj,
      (double)szj,
      smag_j);
  }
}

template <typename T>
static bool nep_spin_report_nonfinite_soa_once(
  const char* stage,
  const char* name,
  const T* data,
  const int nlocal,
  const int natoms,
  const int ncomp,
  int* out_first_linear, // optional
  int* out_count,        // optional
  gpuStream_t st)
{
  int* d_first = nullptr;
  int* d_count = nullptr;
  CHECK(gpuMalloc(&d_first, sizeof(int)));
  CHECK(gpuMalloc(&d_count, sizeof(int)));
  const int h_first_init = INT_MAX;
  const int h_zero = 0;
  // `gpu_macro.cuh` does not define gpuMemcpyAsync/gpuStreamSynchronize; keep this synchronous.
  // This code is only enabled via NEP_SPIN_GPU_LMP_BACKEND_CHECK_NAN.
  CHECK(gpuMemcpy(d_first, &h_first_init, sizeof(int), gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_count, &h_zero, sizeof(int), gpuMemcpyHostToDevice));

  const int total = nlocal * ncomp;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  nep_spin_scan_nonfinite_soa<<<grid, block, 0, st>>>(data, nlocal, natoms, ncomp, d_first, d_count);
  GPU_CHECK_KERNEL

  int h_first = 0;
  int h_count = 0;
  CHECK(gpuDeviceSynchronize());
  CHECK(gpuMemcpy(&h_first, d_first, sizeof(int), gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(&h_count, d_count, sizeof(int), gpuMemcpyDeviceToHost));

  CHECK(gpuFree(d_first));
  CHECK(gpuFree(d_count));

  if (out_first_linear) *out_first_linear = h_first;
  if (out_count) *out_count = h_count;

  if (h_count <= 0 || h_first == INT_MAX) return false;

  const int i = h_first % nlocal;
  const int c = h_first / nlocal;
  T h_val{};
  CHECK(gpuMemcpy(&h_val, data + (i + natoms * c), sizeof(T), gpuMemcpyDeviceToHost));
  std::fprintf(
    stderr,
    "[NEP_SPIN_GPU backend] non-finite at %s: %s i=%d comp=%d count=%d val=%g (nlocal=%d natoms=%d)\n",
    stage,
    name,
    i,
    c,
    h_count,
    static_cast<double>(h_val),
    nlocal,
    natoms);
  return true;
}

static bool nep_spin_report_nonfinite_potential_once(
  const char* stage,
  const double* pe,
  const int nlocal,
  int* out_first, // optional
  int* out_count, // optional
  gpuStream_t st)
{
  int* d_first = nullptr;
  int* d_count = nullptr;
  CHECK(gpuMalloc(&d_first, sizeof(int)));
  CHECK(gpuMalloc(&d_count, sizeof(int)));
  const int h_first_init = INT_MAX;
  const int h_zero = 0;
  // `gpu_macro.cuh` does not define gpuMemcpyAsync/gpuStreamSynchronize; keep this synchronous.
  // This code is only enabled via NEP_SPIN_GPU_LMP_BACKEND_CHECK_NAN.
  CHECK(gpuMemcpy(d_first, &h_first_init, sizeof(int), gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_count, &h_zero, sizeof(int), gpuMemcpyHostToDevice));

  const int block = 256;
  const int grid = (nlocal + block - 1) / block;
  nep_spin_scan_nonfinite_linear<<<grid, block, 0, st>>>(pe, nlocal, d_first, d_count);
  GPU_CHECK_KERNEL

  int h_first = 0;
  int h_count = 0;
  CHECK(gpuDeviceSynchronize());
  CHECK(gpuMemcpy(&h_first, d_first, sizeof(int), gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(&h_count, d_count, sizeof(int), gpuMemcpyDeviceToHost));

  CHECK(gpuFree(d_first));
  CHECK(gpuFree(d_count));

  if (out_first) *out_first = h_first;
  if (out_count) *out_count = h_count;

  if (h_count <= 0 || h_first == INT_MAX) return false;

  double h_val = 0.0;
  CHECK(gpuMemcpy(&h_val, pe + h_first, sizeof(double), gpuMemcpyDeviceToHost));
  std::fprintf(
    stderr,
    "[NEP_SPIN_GPU backend] non-finite at %s: potential i=%d count=%d val=%g (nlocal=%d)\n",
    stage,
    h_first,
    h_count,
    h_val,
    nlocal);
  return true;
}

} // namespace

NEP_Spin_LMP::NEP_Spin_LMP(const char* file_potential, int max_atoms)
  : max_atoms_(max_atoms > 0 ? max_atoms : 0)
{
  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    spin_mref_host_[i] = 1.0f;
  }
  read_potential_file(file_potential);
}

NEP_Spin_LMP::~NEP_Spin_LMP() = default;

float NEP_Spin_LMP::get_rc_radial() const { return paramb_.rc_radial; }
float NEP_Spin_LMP::get_rc_angular() const { return paramb_.rc_angular; }
int NEP_Spin_LMP::get_num_types() const { return paramb_.num_types; }
int NEP_Spin_LMP::get_MN_radial() const { return MN_radial_; }
int NEP_Spin_LMP::get_MN_angular() const { return MN_angular_; }
int NEP_Spin_LMP::get_descriptor_dim() const { return annmb_.dim; }
int NEP_Spin_LMP::get_current_natoms() const { return current_natoms_; }
bool NEP_Spin_LMP::has_last_descriptors() const { return last_descriptors_valid_; }

void NEP_Spin_LMP::copy_last_descriptors_to_host(float* host_data, size_t size)
{
  if (!last_descriptors_valid_) {
    PRINT_INPUT_ERROR(
      "NEP_Spin_LMP::copy_last_descriptors_to_host: descriptors were not materialized by the last spin v2 runtime path.\n");
  }
  const size_t expected = static_cast<size_t>(annmb_.dim) * static_cast<size_t>(current_natoms_);
  if (!host_data) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP::copy_last_descriptors_to_host received null host buffer.\n");
  }
  if (size < expected) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP::copy_last_descriptors_to_host host buffer is too small.\n");
  }
  if (expected == 0) return;
  nep_data_.descriptors.copy_to_host(host_data, expected);
}

void NEP_Spin_LMP::copy_last_Fp_to_host(float* host_data, size_t size)
{
  const size_t expected = static_cast<size_t>(annmb_.dim) * static_cast<size_t>(current_natoms_);
  if (!host_data) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP::copy_last_Fp_to_host received null host buffer.\n");
  }
  if (size < expected) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP::copy_last_Fp_to_host host buffer is too small.\n");
  }
  if (expected == 0) return;
  nep_data_.Fp.copy_to_host(host_data, expected);
}

void NEP_Spin_LMP::copy_q_scaler_to_host(float* host_data, size_t size)
{
  const size_t expected = static_cast<size_t>(annmb_.dim);
  if (!host_data) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP::copy_q_scaler_to_host received null host buffer.\n");
  }
  if (size < expected) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP::copy_q_scaler_to_host host buffer is too small.\n");
  }
  if (expected == 0) return;
  nep_data_.q_scaler.copy_to_host(host_data, expected);
}

static double max_abs_diff_host(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size()) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP shadow compare: float host buffer size mismatch.\n");
  }
  double max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return max_abs;
}

static double max_abs_diff_host(const std::vector<double>& a, const std::vector<double>& b)
{
  if (a.size() != b.size()) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP shadow compare: double host buffer size mismatch.\n");
  }
  double max_abs = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_abs = std::max(max_abs, std::abs(a[i] - b[i]));
  }
  return max_abs;
}

void NEP_Spin_LMP::ensure_capacity(int natoms)
{
  if (natoms == current_natoms_) return;
  current_natoms_ = natoms;
  if (natoms > max_atoms_) max_atoms_ = natoms;

  pos_soa_.resize(static_cast<size_t>(natoms) * 3);
  spin_soa_.resize(static_cast<size_t>(natoms) * 3);

  nn_radial_.resize(natoms);
  nn_angular_.resize(natoms);
  nl_radial_.resize(static_cast<size_t>(natoms) * static_cast<size_t>(MN_radial_));
  nl_angular_.resize(static_cast<size_t>(natoms) * static_cast<size_t>(MN_angular_));
  x12_r_.resize(static_cast<size_t>(natoms) * static_cast<size_t>(MN_radial_));
  y12_r_.resize(static_cast<size_t>(natoms) * static_cast<size_t>(MN_radial_));
  z12_r_.resize(static_cast<size_t>(natoms) * static_cast<size_t>(MN_radial_));
  x12_a_.resize(static_cast<size_t>(natoms) * static_cast<size_t>(MN_angular_));
  y12_a_.resize(static_cast<size_t>(natoms) * static_cast<size_t>(MN_angular_));
  z12_a_.resize(static_cast<size_t>(natoms) * static_cast<size_t>(MN_angular_));

  potential_.resize(natoms);
  force_soa_.resize(static_cast<size_t>(natoms) * 3);
  virial_soa_.resize(static_cast<size_t>(natoms) * 9);
  mforce_soa_.resize(static_cast<size_t>(natoms) * 3);

  // Descriptor buffers are stored with stride = natoms (SoA) to match kernels.
  nep_data_.descriptors.resize(static_cast<size_t>(natoms) * annmb_.dim);
  nep_data_.Fp.resize(static_cast<size_t>(natoms) * annmb_.dim);
  nep_data_.sum_fxyz.resize(
    static_cast<size_t>(natoms) *
    static_cast<size_t>(paramb_.n_max_angular + 1) *
    static_cast<size_t>(((paramb_.L_max + 1) * (paramb_.L_max + 1) - 1)));
  const size_t spin_sum_size =
    static_cast<size_t>(natoms) *
    static_cast<size_t>(nep_spin_3body_count(paramb_)) *
    static_cast<size_t>(nep_spin_3body_abc_count(paramb_));
  nep_data_.sum_fxyz_0.resize(spin_sum_size);
  nep_data_.sum_fxyz_c.resize(spin_sum_size);
  nep_data_.sum_fxyz_Ax.resize(spin_sum_size);
  nep_data_.sum_fxyz_Ay.resize(spin_sum_size);
  nep_data_.sum_fxyz_Az.resize(spin_sum_size);
  nep_data_.sum_fxyz_D.resize(spin_sum_size);
}

void NEP_Spin_LMP::compute_with_neighbors_device(
  Box& box,
  int nlocal,
  int natoms,
  const int* type_dev,
  const double* xyz_aos_dev,
  const double* sp4_aos_dev,
  void* stream,
  const int* NN_radial_dev,
  const int* NL_radial_dev,
  const int* NN_angular_dev,
  const int* NL_angular_dev,
  double* force_aos_dev,
  double* fm_aos_dev,
  double inv_hbar,
  double* potential_dev,
  double* virial_aos_dev,
  bool need_energy,
  bool need_virial,
  double& eng_out,
  double virial_out[6],
  double* virial_raw9_out)
{
  eng_out = 0.0;
  for (int k = 0; k < 6; ++k) virial_out[k] = 0.0;
  if (virial_raw9_out) for (int k = 0; k < 9; ++k) virial_raw9_out[k] = 0.0;
  if (natoms <= 0 || nlocal <= 0) return;

  if (!type_dev || !xyz_aos_dev || !sp4_aos_dev ||
      !NN_radial_dev || !NL_radial_dev || !NN_angular_dev || !NL_angular_dev) {
    PRINT_INPUT_ERROR("NEP_Spin_LMP::compute_with_neighbors_device: null device pointer.\n");
  }

  gpuStream_t st = stream ? reinterpret_cast<gpuStream_t>(stream) : (gpuStream_t)0;

  ensure_capacity(natoms);

  const bool debug_check_nan = (std::getenv("NEP_SPIN_GPU_LMP_BACKEND_CHECK_NAN") != nullptr);
  const bool debug_print_params = (std::getenv("NEP_SPIN_GPU_LMP_BACKEND_DEBUG") != nullptr);
  const bool want_fm = (fm_aos_dev != nullptr) && (inv_hbar != 0.0);
  const bool timing_enabled = (std::getenv("NEP_SPIN_GPU_LMP_BACKEND_TIMING") != nullptr);
  const bool force_mic_with_ghosts =
    (std::getenv("NEP_SPIN_GPU_LMP_BACKEND_FORCE_MIC") != nullptr) ||
    (std::getenv("NEP_SPIN_GPU_LMP_FORCE_MIC") != nullptr) ||
    (std::getenv("NEP_GPU_LMP_STRICT_BOX") != nullptr);
  static int reported_backend_nan = 0;
  static int printed_backend_params = 0;
  static int timing_call_counter = 0;

  if (debug_print_params && !printed_backend_params) {
    printed_backend_params = 1;
    std::fprintf(
      stderr,
      "[NEP_SPIN_GPU backend] params: rc_r=%.8g rc_a=%.8g nmax_r=%d nmax_a=%d bs_r=%d bs_a=%d L_max=%d dim_ang=%d ann_dim=%d MN_r=%d MN_a=%d MAX_NUM_N=%d MAX_DIM=%d force_mic_with_ghosts=%d\n",
      (double)paramb_.rc_radial,
      (double)paramb_.rc_angular,
      paramb_.n_max_radial,
      paramb_.n_max_angular,
      paramb_.basis_size_radial,
      paramb_.basis_size_angular,
      paramb_.L_max,
      paramb_.dim_angular,
      annmb_.dim,
      MN_radial_,
      MN_angular_,
      MAX_NUM_N,
      MAX_DIM,
      force_mic_with_ghosts ? 1 : 0);
  }

  int timing_every = 1;
  if (const char* env = std::getenv("NEP_SPIN_GPU_LMP_BACKEND_TIMING_EVERY")) {
    const int v = std::atoi(env);
    if (v > 1) timing_every = v;
  }
  const int timing_call_id = ++timing_call_counter;
  const bool do_timing = timing_enabled && (timing_every > 0) && (((timing_call_id - 1) % timing_every) == 0);
  gpuEvent_t timing_ev0{};
  gpuEvent_t timing_ev1{};
  if (do_timing) {
    CHECK(gpuEventCreate(&timing_ev0));
    CHECK(gpuEventCreate(&timing_ev1));
  }
  auto time_region_ms = [&](const char* label, const auto& fn) {
    if (!do_timing) {
      fn();
      return;
    }
    const auto wall_t0 = std::chrono::high_resolution_clock::now();
    CHECK(gpuEventRecord(timing_ev0, st));
    fn();
    CHECK(gpuEventRecord(timing_ev1, st));
    CHECK(gpuEventSynchronize(timing_ev1));
    const auto wall_t1 = std::chrono::high_resolution_clock::now();
    float ms = 0.0f;
    CHECK(gpuEventElapsedTime(&ms, timing_ev0, timing_ev1));
    const double wall_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(wall_t1 - wall_t0).count();
    std::fprintf(
      stderr,
      "[NEP_SPIN_GPU backend timing] call=%d natoms=%d nlocal=%d want_fm=%d need_e=%d need_v=%d %s: gpu=%.3f ms wall=%.3f ms\n",
      timing_call_id,
      natoms,
      nlocal,
      want_fm ? 1 : 0,
      need_energy ? 1 : 0,
      need_virial ? 1 : 0,
      label,
      static_cast<double>(ms),
      wall_ms);
  };

  // Pack positions AoS->SoA and spin sp(4)->S(3) into SoA float.
  time_region_ms("pack_aos_to_soa", [&]() {
    const int block = 256;
    const int grid = (natoms + block - 1) / block;
    pack_xyz_aos_to_soa<<<grid, block, 0, st>>>(natoms, xyz_aos_dev, pos_soa_.data());
    pack_sp4_aos_to_spin_soa<<<grid, block, 0, st>>>(natoms, sp4_aos_dev, spin_soa_.data());
    GPU_CHECK_KERNEL
  });

  if (debug_check_nan && !reported_backend_nan) {
    if (nep_spin_report_nonfinite_soa_once("after_pack", "pos_soa", pos_soa_.data(), nlocal, natoms, 3, nullptr, nullptr, st) ||
        nep_spin_report_nonfinite_soa_once("after_pack", "spin_soa", spin_soa_.data(), nlocal, natoms, 3, nullptr, nullptr, st)) {
      reported_backend_nan = 1;
    }
  }

  // Repack LAMMPS/Kokkos neighbor lists from nlocal stride to the backend's
  // natoms stride so the kernels can keep a single indexing convention.
  time_region_ms("repack_neighbors", [&]() {
    nn_radial_.fill(0);
    nn_angular_.fill(0);
    const int block = 128;
    const int grid = (nlocal + block - 1) / block;
    repack_neighbors_local_to_natoms_stride<<<grid, block, 0, st>>>(
      nlocal, natoms, NN_radial_dev, NL_radial_dev, nn_radial_.data(), nl_radial_.data());
    repack_neighbors_local_to_natoms_stride<<<grid, block, 0, st>>>(
      nlocal, natoms, NN_angular_dev, NL_angular_dev, nn_angular_.data(), nl_angular_.data());
    GPU_CHECK_KERNEL
  });

  // Build x12 vectors from repacked NN/NL and positions (for local atoms only).
  time_region_ms("build_x12_from_nl", [&]() {
    const int block = 128;
    const int grid = (nlocal + block - 1) / block;
    const double* x = pos_soa_.data();
    const double* y = pos_soa_.data() + natoms;
    const double* z = pos_soa_.data() + 2 * natoms;
    build_x12_from_nl<<<grid, block, 0, st>>>(
      natoms, nlocal, force_mic_with_ghosts ? 1 : 0, box,
      nn_radial_.data(), nl_radial_.data(),
      x, y, z,
      x12_r_.data(), y12_r_.data(), z12_r_.data());
    build_x12_from_nl<<<grid, block, 0, st>>>(
      natoms, nlocal, force_mic_with_ghosts ? 1 : 0, box,
      nn_angular_.data(), nl_angular_.data(),
      x, y, z,
      x12_a_.data(), y12_a_.data(), z12_a_.data());
    GPU_CHECK_KERNEL
  });

  // Note: we intentionally do not scan x12 buffers here to avoid excessive overhead.
  const SpinV2RuntimePath runtime_path = get_spin_v2_runtime_path();
  const bool force_descriptor_export =
    nep_spin_env_true("NEP_SPIN_GPU_LMP_EXPORT_DESCRIPTORS") ||
    nep_spin_env_true("NEP_SPIN_GPU_LMP_BACKEND_EXPORT_DESCRIPTORS");
  const double shadow_tol = get_spin_v2_shadow_tolerance();
  last_descriptors_valid_ = false;

  const int N1 = 0;
  const int N2 = nlocal;
  const int BLOCK_SIZE = 128;
  const int FAST_BLOCK_SIZE = 64;
  const int grid_size = (nlocal + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int fast_grid_size = (nlocal + FAST_BLOCK_SIZE - 1) / FAST_BLOCK_SIZE;
  const size_t base_pair_shmem_bytes = static_cast<size_t>(BLOCK_SIZE) * (2 * MAX_NUM_N) * sizeof(float);
  const size_t fast_forward_shmem_bytes =
    static_cast<size_t>(FAST_BLOCK_SIZE) * static_cast<size_t>(MAX_NUM_N) * sizeof(float);

  auto reset_state = [&]() {
    force_soa_.fill(0.0);
    virial_soa_.fill(0.0);
    const int block = 256;
    const int grid = (natoms + block - 1) / block;
    zero_mforce_spin<<<grid, block, 0, st>>>(
      natoms,
      mforce_soa_.data(),
      mforce_soa_.data() + natoms,
      mforce_soa_.data() + 2 * natoms);
    GPU_CHECK_KERNEL
  };

  auto check_after_forward = [&](const char* stage_tag) {
    if (!(debug_check_nan && !reported_backend_nan)) return;
    int fp_comp_limit = 16;
    if (fp_comp_limit > annmb_.dim) fp_comp_limit = annmb_.dim;
    if (const char* env = std::getenv("NEP_SPIN_GPU_LMP_BACKEND_FP_COMP_LIMIT")) {
      const int req = std::atoi(env);
      if (req > 0) fp_comp_limit = std::min(req, annmb_.dim);
    }

    int bad_pe_i = -1;
    int bad_pe_count = 0;
    const bool pe_bad = nep_spin_report_nonfinite_potential_once(
      stage_tag, potential_.data(), nlocal, &bad_pe_i, &bad_pe_count, st);

    int bad_fp_first = -1;
    int bad_fp_count = 0;
    const bool fp_bad = (fp_comp_limit > 0) &&
      nep_spin_report_nonfinite_soa_once(
        stage_tag, "Fp", nep_data_.Fp.data(), nlocal, natoms, fp_comp_limit, &bad_fp_first, &bad_fp_count, st);

    if (pe_bad) {
      nep_spin_backend_dump_atom_neighbors(
        stage_tag,
        bad_pe_i,
        nlocal,
        natoms,
        type_dev,
        pos_soa_.data(),
        spin_soa_.data(),
        nn_radial_.data(),
        nl_radial_.data(),
        x12_r_.data(),
        y12_r_.data(),
        z12_r_.data(),
        st);
    }
    if (pe_bad || fp_bad) reported_backend_nan = 1;
  };

  auto run_forward_reference = [&]() {
    last_descriptors_valid_ = true;
    time_region_ms("descriptor_radial_base", [&]() {
      find_descriptors_radial_spinbase<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms,
        nlocal,
        nn_radial_.data(),
        nl_radial_.data(),
        paramb_,
        annmb_,
        type_dev,
        x12_r_.data(),
        y12_r_.data(),
        z12_r_.data(),
        nep_data_.descriptors.data());
      GPU_CHECK_KERNEL
    });

    time_region_ms("descriptor_angular_base", [&]() {
      find_descriptors_angular_spinbase<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms,
        nlocal,
        nn_angular_.data(),
        nl_angular_.data(),
        paramb_,
        annmb_,
        type_dev,
        x12_a_.data(),
        y12_a_.data(),
        z12_a_.data(),
        nep_data_.descriptors.data(),
        nep_data_.sum_fxyz.data());
      GPU_CHECK_KERNEL
    });

    time_region_ms("descriptor_spin_onsite", [&]() {
      find_descriptors_spin_onsite<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms, nlocal, paramb_, type_dev, spin_soa_.data(), nep_data_.descriptors.data());
      GPU_CHECK_KERNEL
    });

    time_region_ms("descriptor_spin_2body", [&]() {
      find_descriptors_spin_2body<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms,
        nlocal,
        nn_radial_.data(),
        nl_radial_.data(),
        paramb_,
        annmb_,
        type_dev,
        x12_r_.data(),
        y12_r_.data(),
        z12_r_.data(),
        spin_soa_.data(),
        nep_data_.descriptors.data());
      GPU_CHECK_KERNEL
    });

    time_region_ms("descriptor_spin_3body", [&]() {
      find_descriptors_spin_3body<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms,
        nlocal,
        nn_angular_.data(),
        nl_angular_.data(),
        paramb_,
        annmb_,
        type_dev,
        x12_a_.data(),
        y12_a_.data(),
        z12_a_.data(),
        spin_soa_.data(),
        nep_data_.descriptors.data(),
        nep_data_.sum_fxyz_0.data(),
        nep_data_.sum_fxyz_c.data(),
        nep_data_.sum_fxyz_Ax.data(),
        nep_data_.sum_fxyz_Ay.data(),
        nep_data_.sum_fxyz_Az.data(),
        nep_data_.sum_fxyz_D.data());
      GPU_CHECK_KERNEL
    });

    time_region_ms("apply_ann_spin", [&]() {
      apply_ann_spin<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms,
        nlocal,
        paramb_,
        annmb_,
        type_dev,
        nep_data_.descriptors.data(),
        nep_data_.q_scaler.data(),
        potential_.data(),
        nep_data_.Fp.data());
      GPU_CHECK_KERNEL
    });

    check_after_forward("after_q_ann_reference");
  };

  auto run_forward_fast = [&](const bool materialize_descriptors) {
    last_descriptors_valid_ = materialize_descriptors;
    time_region_ms("forward_spin_v2_fast", [&]() {
      const int bucket = spin_v2_pick_dim_bucket(annmb_.dim);
      if (bucket <= 64) {
        if (materialize_descriptors) {
          compute_all_q_and_ann_spin_v2_tmpl<64, true><<<fast_grid_size, FAST_BLOCK_SIZE, fast_forward_shmem_bytes, st>>>(
            natoms, nlocal, nn_radial_.data(), nl_radial_.data(), nn_angular_.data(), nl_angular_.data(),
            paramb_, annmb_, type_dev, x12_r_.data(), y12_r_.data(), z12_r_.data(),
            x12_a_.data(), y12_a_.data(), z12_a_.data(), spin_soa_.data(),
            nep_data_.descriptors.data(), nep_data_.sum_fxyz.data(),
            nep_data_.sum_fxyz_0.data(), nep_data_.sum_fxyz_c.data(), nep_data_.sum_fxyz_Ax.data(),
            nep_data_.sum_fxyz_Ay.data(), nep_data_.sum_fxyz_Az.data(), nep_data_.sum_fxyz_D.data(),
            nep_data_.q_scaler.data(), potential_.data(), nep_data_.Fp.data());
        } else {
          compute_all_q_and_ann_spin_v2_tmpl<64, false><<<fast_grid_size, FAST_BLOCK_SIZE, fast_forward_shmem_bytes, st>>>(
            natoms, nlocal, nn_radial_.data(), nl_radial_.data(), nn_angular_.data(), nl_angular_.data(),
            paramb_, annmb_, type_dev, x12_r_.data(), y12_r_.data(), z12_r_.data(),
            x12_a_.data(), y12_a_.data(), z12_a_.data(), spin_soa_.data(),
            nep_data_.descriptors.data(), nep_data_.sum_fxyz.data(),
            nep_data_.sum_fxyz_0.data(), nep_data_.sum_fxyz_c.data(), nep_data_.sum_fxyz_Ax.data(),
            nep_data_.sum_fxyz_Ay.data(), nep_data_.sum_fxyz_Az.data(), nep_data_.sum_fxyz_D.data(),
            nep_data_.q_scaler.data(), potential_.data(), nep_data_.Fp.data());
        }
      } else if (bucket <= 96) {
        if (materialize_descriptors) {
          compute_all_q_and_ann_spin_v2_tmpl<96, true><<<fast_grid_size, FAST_BLOCK_SIZE, fast_forward_shmem_bytes, st>>>(
            natoms, nlocal, nn_radial_.data(), nl_radial_.data(), nn_angular_.data(), nl_angular_.data(),
            paramb_, annmb_, type_dev, x12_r_.data(), y12_r_.data(), z12_r_.data(),
            x12_a_.data(), y12_a_.data(), z12_a_.data(), spin_soa_.data(),
            nep_data_.descriptors.data(), nep_data_.sum_fxyz.data(),
            nep_data_.sum_fxyz_0.data(), nep_data_.sum_fxyz_c.data(), nep_data_.sum_fxyz_Ax.data(),
            nep_data_.sum_fxyz_Ay.data(), nep_data_.sum_fxyz_Az.data(), nep_data_.sum_fxyz_D.data(),
            nep_data_.q_scaler.data(), potential_.data(), nep_data_.Fp.data());
        } else {
          compute_all_q_and_ann_spin_v2_tmpl<96, false><<<fast_grid_size, FAST_BLOCK_SIZE, fast_forward_shmem_bytes, st>>>(
            natoms, nlocal, nn_radial_.data(), nl_radial_.data(), nn_angular_.data(), nl_angular_.data(),
            paramb_, annmb_, type_dev, x12_r_.data(), y12_r_.data(), z12_r_.data(),
            x12_a_.data(), y12_a_.data(), z12_a_.data(), spin_soa_.data(),
            nep_data_.descriptors.data(), nep_data_.sum_fxyz.data(),
            nep_data_.sum_fxyz_0.data(), nep_data_.sum_fxyz_c.data(), nep_data_.sum_fxyz_Ax.data(),
            nep_data_.sum_fxyz_Ay.data(), nep_data_.sum_fxyz_Az.data(), nep_data_.sum_fxyz_D.data(),
            nep_data_.q_scaler.data(), potential_.data(), nep_data_.Fp.data());
        }
      } else if (bucket <= 128) {
        if (materialize_descriptors) {
          compute_all_q_and_ann_spin_v2_tmpl<128, true><<<fast_grid_size, FAST_BLOCK_SIZE, fast_forward_shmem_bytes, st>>>(
            natoms, nlocal, nn_radial_.data(), nl_radial_.data(), nn_angular_.data(), nl_angular_.data(),
            paramb_, annmb_, type_dev, x12_r_.data(), y12_r_.data(), z12_r_.data(),
            x12_a_.data(), y12_a_.data(), z12_a_.data(), spin_soa_.data(),
            nep_data_.descriptors.data(), nep_data_.sum_fxyz.data(),
            nep_data_.sum_fxyz_0.data(), nep_data_.sum_fxyz_c.data(), nep_data_.sum_fxyz_Ax.data(),
            nep_data_.sum_fxyz_Ay.data(), nep_data_.sum_fxyz_Az.data(), nep_data_.sum_fxyz_D.data(),
            nep_data_.q_scaler.data(), potential_.data(), nep_data_.Fp.data());
        } else {
          compute_all_q_and_ann_spin_v2_tmpl<128, false><<<fast_grid_size, FAST_BLOCK_SIZE, fast_forward_shmem_bytes, st>>>(
            natoms, nlocal, nn_radial_.data(), nl_radial_.data(), nn_angular_.data(), nl_angular_.data(),
            paramb_, annmb_, type_dev, x12_r_.data(), y12_r_.data(), z12_r_.data(),
            x12_a_.data(), y12_a_.data(), z12_a_.data(), spin_soa_.data(),
            nep_data_.descriptors.data(), nep_data_.sum_fxyz.data(),
            nep_data_.sum_fxyz_0.data(), nep_data_.sum_fxyz_c.data(), nep_data_.sum_fxyz_Ax.data(),
            nep_data_.sum_fxyz_Ay.data(), nep_data_.sum_fxyz_Az.data(), nep_data_.sum_fxyz_D.data(),
            nep_data_.q_scaler.data(), potential_.data(), nep_data_.Fp.data());
        }
      } else {
        if (materialize_descriptors) {
          compute_all_q_and_ann_spin_v2_tmpl<MAX_DIM, true><<<fast_grid_size, FAST_BLOCK_SIZE, fast_forward_shmem_bytes, st>>>(
            natoms, nlocal, nn_radial_.data(), nl_radial_.data(), nn_angular_.data(), nl_angular_.data(),
            paramb_, annmb_, type_dev, x12_r_.data(), y12_r_.data(), z12_r_.data(),
            x12_a_.data(), y12_a_.data(), z12_a_.data(), spin_soa_.data(),
            nep_data_.descriptors.data(), nep_data_.sum_fxyz.data(),
            nep_data_.sum_fxyz_0.data(), nep_data_.sum_fxyz_c.data(), nep_data_.sum_fxyz_Ax.data(),
            nep_data_.sum_fxyz_Ay.data(), nep_data_.sum_fxyz_Az.data(), nep_data_.sum_fxyz_D.data(),
            nep_data_.q_scaler.data(), potential_.data(), nep_data_.Fp.data());
        } else {
          compute_all_q_and_ann_spin_v2_tmpl<MAX_DIM, false><<<fast_grid_size, FAST_BLOCK_SIZE, fast_forward_shmem_bytes, st>>>(
            natoms, nlocal, nn_radial_.data(), nl_radial_.data(), nn_angular_.data(), nl_angular_.data(),
            paramb_, annmb_, type_dev, x12_r_.data(), y12_r_.data(), z12_r_.data(),
            x12_a_.data(), y12_a_.data(), z12_a_.data(), spin_soa_.data(),
            nep_data_.descriptors.data(), nep_data_.sum_fxyz.data(),
            nep_data_.sum_fxyz_0.data(), nep_data_.sum_fxyz_c.data(), nep_data_.sum_fxyz_Ax.data(),
            nep_data_.sum_fxyz_Ay.data(), nep_data_.sum_fxyz_Az.data(), nep_data_.sum_fxyz_D.data(),
            nep_data_.q_scaler.data(), potential_.data(), nep_data_.Fp.data());
        }
      }
      GPU_CHECK_KERNEL
    });

    check_after_forward("after_q_ann_fast");
  };

  auto run_backward = [&]() {
    time_region_ms("force_base_radial", [&]() {
      find_force_radial_spinbase_md<<<grid_size, BLOCK_SIZE, base_pair_shmem_bytes, st>>>(
        natoms,
        N1,
        N2,
        nn_radial_.data(),
        nl_radial_.data(),
        paramb_,
        annmb_,
        type_dev,
        x12_r_.data(),
        y12_r_.data(),
        z12_r_.data(),
        nep_data_.Fp.data(),
        force_soa_.data(),
        force_soa_.data() + natoms,
        force_soa_.data() + 2 * natoms,
        virial_soa_.data());
      GPU_CHECK_KERNEL
    });

    time_region_ms("force_base_angular", [&]() {
      find_force_angular_spinbase_md<<<grid_size, BLOCK_SIZE, base_pair_shmem_bytes, st>>>(
        natoms,
        N1,
        N2,
        nn_angular_.data(),
        nl_angular_.data(),
        paramb_,
        annmb_,
        type_dev,
        x12_a_.data(),
        y12_a_.data(),
        z12_a_.data(),
        nep_data_.Fp.data(),
        nep_data_.sum_fxyz.data(),
        force_soa_.data(),
        force_soa_.data() + natoms,
        force_soa_.data() + 2 * natoms,
        virial_soa_.data());
      GPU_CHECK_KERNEL
    });

    time_region_ms("mforce_spin_onsite", [&]() {
      find_mforce_spin_onsite<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms,
        nlocal,
        paramb_,
        type_dev,
        spin_soa_.data(),
        nep_data_.Fp.data(),
        mforce_soa_.data(),
        mforce_soa_.data() + natoms,
        mforce_soa_.data() + 2 * natoms);
      GPU_CHECK_KERNEL
    });

    time_region_ms("force_spin_2body", [&]() {
      find_force_spin_2body<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms,
        nlocal,
        nn_radial_.data(),
        nl_radial_.data(),
        paramb_,
        annmb_,
        type_dev,
        x12_r_.data(),
        y12_r_.data(),
        z12_r_.data(),
        spin_soa_.data(),
        nep_data_.Fp.data(),
        force_soa_.data(),
        force_soa_.data() + natoms,
        force_soa_.data() + 2 * natoms,
        virial_soa_.data(),
        mforce_soa_.data(),
        mforce_soa_.data() + natoms,
        mforce_soa_.data() + 2 * natoms);
      GPU_CHECK_KERNEL
    });

    time_region_ms("force_spin_3body", [&]() {
      find_force_spin_3body<<<grid_size, BLOCK_SIZE, 0, st>>>(
        natoms,
        nlocal,
        nn_angular_.data(),
        nl_angular_.data(),
        paramb_,
        annmb_,
        type_dev,
        x12_a_.data(),
        y12_a_.data(),
        z12_a_.data(),
        spin_soa_.data(),
        nep_data_.Fp.data(),
        nep_data_.sum_fxyz_0.data(),
        nep_data_.sum_fxyz_c.data(),
        nep_data_.sum_fxyz_Ax.data(),
        nep_data_.sum_fxyz_Ay.data(),
        nep_data_.sum_fxyz_Az.data(),
        nep_data_.sum_fxyz_D.data(),
        force_soa_.data(),
        force_soa_.data() + natoms,
        force_soa_.data() + 2 * natoms,
        virial_soa_.data(),
        mforce_soa_.data(),
        mforce_soa_.data() + natoms,
        mforce_soa_.data() + 2 * natoms);
      GPU_CHECK_KERNEL
    });

    if (debug_check_nan && !reported_backend_nan) {
      if ((want_fm && nep_spin_report_nonfinite_soa_once("after_spin_kernels", "mforce_soa", mforce_soa_.data(), nlocal, natoms, 3, nullptr, nullptr, st)) ||
          nep_spin_report_nonfinite_soa_once("after_spin_kernels", "force_soa", force_soa_.data(), nlocal, natoms, 3, nullptr, nullptr, st)) {
        reported_backend_nan = 1;
      }
    }
  };

  struct ShadowState {
    std::vector<float> fp;
    std::vector<double> potential;
    std::vector<double> force;
    std::vector<double> mforce;
    std::vector<double> virial;
  };

  auto capture_shadow_state = [&]() {
    ShadowState state;
    state.fp.resize(static_cast<size_t>(annmb_.dim) * static_cast<size_t>(natoms));
    state.potential.resize(static_cast<size_t>(nlocal));
    state.force.resize(static_cast<size_t>(natoms) * 3);
    state.mforce.resize(static_cast<size_t>(natoms) * 3);
    state.virial.resize(static_cast<size_t>(natoms) * 9);
    nep_data_.Fp.copy_to_host(state.fp.data(), state.fp.size());
    potential_.copy_to_host(state.potential.data(), state.potential.size());
    force_soa_.copy_to_host(state.force.data(), state.force.size());
    mforce_soa_.copy_to_host(state.mforce.data(), state.mforce.size());
    virial_soa_.copy_to_host(state.virial.data(), state.virial.size());
    return state;
  };

  auto validate_shadow = [&](const ShadowState& ref_state, const ShadowState& fast_state) {
    const double fp_diff = max_abs_diff_host(ref_state.fp, fast_state.fp);
    const double pe_diff = max_abs_diff_host(ref_state.potential, fast_state.potential);
    const double f_diff = max_abs_diff_host(ref_state.force, fast_state.force);
    const double fm_diff = max_abs_diff_host(ref_state.mforce, fast_state.mforce);
    const double v_diff = max_abs_diff_host(ref_state.virial, fast_state.virial);
    const double max_diff = std::max(fp_diff, std::max(pe_diff, std::max(f_diff, std::max(fm_diff, v_diff))));
    if (max_diff > shadow_tol) {
      char message[512];
      std::snprintf(
        message,
        sizeof(message),
        "NEP_Spin_LMP shadow compare failed: tol=%g fp=%g pe=%g force=%g mforce=%g virial=%g\n",
        shadow_tol,
        fp_diff,
        pe_diff,
        f_diff,
        fm_diff,
        v_diff);
      PRINT_INPUT_ERROR(message);
    }
  };

  if (runtime_path == SpinV2RuntimePath::shadow) {
    time_region_ms("reset_state_reference", [&]() { reset_state(); });
    run_forward_reference();
    run_backward();
#ifdef USE_HIP
    CHECK(hipStreamSynchronize(st));
#else
    CHECK(cudaStreamSynchronize(st));
#endif
    const ShadowState reference_state = capture_shadow_state();

    time_region_ms("reset_state_fast", [&]() { reset_state(); });
    run_forward_fast(force_descriptor_export);
    run_backward();
#ifdef USE_HIP
    CHECK(hipStreamSynchronize(st));
#else
    CHECK(cudaStreamSynchronize(st));
#endif
    const ShadowState fast_state = capture_shadow_state();
    validate_shadow(reference_state, fast_state);
  } else {
    time_region_ms("reset_state", [&]() { reset_state(); });
    if (runtime_path == SpinV2RuntimePath::fast) {
      run_forward_fast(force_descriptor_export);
    } else {
      run_forward_reference();
    }
    run_backward();
  }

  // Scatter into LAMMPS AoS outputs (accumulating).
  if (force_aos_dev) {
    time_region_ms("scatter_force", [&]() {
      const int block = 256;
      const int grid = (natoms + block - 1) / block;
      scatter_force_soa_to_aos_add<<<grid, block, 0, st>>>(
        natoms,
        force_soa_.data(),
        force_soa_.data() + natoms,
        force_soa_.data() + 2 * natoms,
        force_aos_dev);
      GPU_CHECK_KERNEL
    });
  }

  if (want_fm) {
    time_region_ms("scatter_mforce", [&]() {
      const int block = 256;
      const int grid = (natoms + block - 1) / block;
      scatter_mforce_soa_to_fm_aos_add_scale<<<grid, block, 0, st>>>(
        natoms,
        mforce_soa_.data(),
        mforce_soa_.data() + natoms,
        mforce_soa_.data() + 2 * natoms,
        inv_hbar,
        fm_aos_dev);
      GPU_CHECK_KERNEL
    });
  }

  // Export per-atom outputs if requested.
  if (potential_dev) {
    time_region_ms("export_potential", [&]() {
      CHECK(gpuMemcpy(potential_dev, potential_.data(), sizeof(double) * nlocal, gpuMemcpyDeviceToDevice));
    });
  }
  if (virial_aos_dev) {
    time_region_ms("export_virial_aos", [&]() {
      const int block = 256;
      const int grid = (nlocal + block - 1) / block;
      virial_soa_to_aos9_local<<<grid, block, 0, st>>>(nlocal, natoms, virial_soa_.data(), virial_aos_dev);
      GPU_CHECK_KERNEL
    });
  }

  // Reduce totals on device and copy 7 scalars back to host.
  if (need_energy || need_virial) {
    totals_.resize(7);
    totals_.fill(0.0);
    const int block = 256;
    const int grid = (nlocal + block - 1) / block;
    time_region_ms("reduce_ev_totals", [&]() {
      reduce_ev_totals_spin<<<grid, block, 0, st>>>(
        nlocal,
        natoms,
        need_energy ? potential_.data() : nullptr,
        need_virial ? virial_soa_.data() : nullptr,
        totals_.data());
      GPU_CHECK_KERNEL
    });

    double host7[7];
    time_region_ms("copy_totals_d2h", [&]() { totals_.copy_to_host(host7); });
    eng_out = need_energy ? host7[0] : 0.0;
    virial_out[0] = need_virial ? host7[1] : 0.0;
    virial_out[1] = need_virial ? host7[2] : 0.0;
    virial_out[2] = need_virial ? host7[3] : 0.0;
    virial_out[3] = need_virial ? host7[4] : 0.0;
    virial_out[4] = need_virial ? host7[5] : 0.0;
    virial_out[5] = need_virial ? host7[6] : 0.0;
  }

  // Optional: also return raw (non-symmetrized) 3x3 tensor totals for debugging/comparison
  // with GPUMD's main_nep virial convention (which records xy,yz,zx without symmetrizing).
  if (virial_raw9_out) {
    totals_raw9_.resize(9);
    totals_raw9_.fill(0.0);
    const int block = 256;
    const int grid = (nlocal + block - 1) / block;
    time_region_ms("reduce_virial_raw9", [&]() {
      reduce_virial_raw9_spin<<<grid, block, 0, st>>>(nlocal, natoms, virial_soa_.data(), totals_raw9_.data());
      GPU_CHECK_KERNEL
    });

    double host9[9];
    time_region_ms("copy_virial_raw9_d2h", [&]() { totals_raw9_.copy_to_host(host9); });
    for (int k = 0; k < 9; ++k) virial_raw9_out[k] = host9[k];
  }

  if (do_timing) {
    CHECK(gpuEventDestroy(timing_ev0));
    CHECK(gpuEventDestroy(timing_ev1));
  }

  // Ensure all device work is complete before returning to a Kokkos caller.
  CHECK(gpuDeviceSynchronize());
}

void NEP_Spin_LMP::read_potential_file(const char* file_potential)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open spin potential file.\n");
  }

  auto require_int_in_range = [&](const std::string& token, const char* what, int lo, int hi) {
    const int value = get_int_from_token(token, __FILE__, __LINE__);
    if (value < lo || value > hi) {
      std::string msg = std::string(what) + " is out of range.\n";
      PRINT_INPUT_ERROR(msg.c_str());
    }
    return value;
  };

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    PRINT_INPUT_ERROR("The first line of nep*_spin must be 'nep[3/4]_spin <num_types> <elements...>'.\n");
  }
  if (tokens[0] != "nep3_spin" && tokens[0] != "nep4_spin") {
    PRINT_INPUT_ERROR("Only nep3_spin and nep4_spin are supported.\n");
  }
  paramb_.version = (tokens[0] == "nep3_spin") ? 3 : 4;
  paramb_.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (paramb_.num_types <= 0 || paramb_.num_types > NUM_ELEMENTS) {
    PRINT_INPUT_ERROR("Invalid number of types in spin model header.\n");
  }
  if (static_cast<int>(tokens.size()) != paramb_.num_types + 2) {
    PRINT_INPUT_ERROR("Spin model header element count does not match num_types.\n");
  }
  paramb_.num_types_sq = paramb_.num_types * paramb_.num_types;
  for (int n = 0; n < paramb_.num_types; ++n) {
    int atomic_number = -1;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == kElementSymbols[m]) {
        atomic_number = m;
        break;
      }
    }
    if (atomic_number < 0) {
      PRINT_INPUT_ERROR("Unknown element symbol in spin model header.\n");
    }
    paramb_.atomic_numbers[n] = atomic_number;
  }

  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "spin_mode") {
    PRINT_INPUT_ERROR("Second line of nep*_spin must be 'spin_mode <mode> 3'.\n");
  }
  spin_mode_ = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (spin_mode_ != 1) {
    PRINT_INPUT_ERROR("Only spin_mode 1 is supported for NEP-Spin v2.\n");
  }
  if (get_int_from_token(tokens[2], __FILE__, __LINE__) != 3) {
    PRINT_INPUT_ERROR("spin_header_lines must be 3 for NEP-Spin v2.\n");
  }

  tokens = get_tokens(input);
  if (!tokens.empty() && (tokens[0] == "spin_feature" || tokens[0] == "spin_expansion")) {
    PRINT_INPUT_ERROR("Legacy spin_feature/spin_expansion headers are not supported. Regenerate nep.txt with NEP-Spin v2.\n");
  }
  const bool use_single_mref = (tokens.size() == 4);
  const bool use_per_type_mref = (paramb_.num_types > 0 && static_cast<int>(tokens.size()) == 3 + paramb_.num_types);
  if ((!use_single_mref && !use_per_type_mref) || tokens[0] != "spin_onsite") {
    PRINT_INPUT_ERROR(
      "Third line of nep*_spin must be 'spin_onsite <pmax> <basis_mode> <mref>' or "
      "'spin_onsite <pmax> <basis_mode> <mref_type1> ... <mref_typeN>'.\n");
  }
  paramb_.spin_pmax = require_int_in_range(tokens[1], "spin_onsite pmax", 0, 8);
  paramb_.spin_onsite_basis_mode = require_int_in_range(tokens[2], "spin_onsite basis_mode", 0, 2);
  const int mref_count = use_single_mref ? 1 : paramb_.num_types;
  for (int i = 0; i < mref_count; ++i) {
    const float parsed_mref = get_double_from_token(tokens[3 + i], __FILE__, __LINE__);
    if (!(parsed_mref > 0.0f)) {
      PRINT_INPUT_ERROR("spin_onsite mref should be > 0.\n");
    }
    if (use_single_mref) {
      for (int t = 0; t < NUM_ELEMENTS; ++t) {
        spin_mref_host_[t] = parsed_mref;
      }
      break;
    }
    spin_mref_host_[i] = parsed_mref;
  }
  CHECK(gpuMemcpyToSymbol(g_nep_spin_mref, spin_mref_host_, sizeof(float) * NUM_ELEMENTS));

  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "spin_2body") {
    PRINT_INPUT_ERROR("Fourth line of nep*_spin must be 'spin_2body <n_max_spin_radial> <basis_size_spin_radial>'.\n");
  }
  paramb_.n_max_spin_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.basis_size_spin_radial = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (paramb_.n_max_spin_radial < 0 || paramb_.basis_size_spin_radial < 0) {
    PRINT_INPUT_ERROR("spin_2body parameters must be >= 0.\n");
  }

  tokens = get_tokens(input);
  if (tokens.size() != 4 || tokens[0] != "spin_3body") {
    PRINT_INPUT_ERROR("Fifth line of nep*_spin must be 'spin_3body <n_max_spin_angular> <l_max_spin_angular> <basis_size_spin_angular>'.\n");
  }
  paramb_.n_max_spin_angular = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.l_max_spin_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb_.basis_size_spin_angular = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (paramb_.n_max_spin_angular < 0 || paramb_.basis_size_spin_angular < 0) {
    PRINT_INPUT_ERROR("spin_3body n_max and basis_size must be >= 0.\n");
  }
  if (paramb_.l_max_spin_angular < 1 || paramb_.l_max_spin_angular > 4) {
    PRINT_INPUT_ERROR("spin_3body l_max_spin_angular should be in [1,4].\n");
  }

  tokens = get_tokens(input);
  if (tokens.empty() || tokens[0] != "cutoff") {
    PRINT_INPUT_ERROR("Missing cutoff line in spin potential.\n");
  }
  const int cutoff_tokens = static_cast<int>(tokens.size());
  if (cutoff_tokens != 5 && cutoff_tokens != paramb_.num_types * 2 + 3) {
    PRINT_INPUT_ERROR("cutoff line must be 'cutoff rc_r rc_a MN_r MN_a' or the multi-cutoff variant.\n");
  }
  paramb_.use_typewise_cutoff = false;
  paramb_.rc_radial = 0.0f;
  paramb_.rc_angular = 0.0f;
  if (cutoff_tokens == 5) {
    const float rc_r = get_double_from_token(tokens[1], __FILE__, __LINE__);
    const float rc_a = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (!(rc_r > 0.0f) || !(rc_a > 0.0f)) {
      PRINT_INPUT_ERROR("cutoff values must be > 0.\n");
    }
    for (int t = 0; t < paramb_.num_types; ++t) {
      paramb_.rc_radial_by_type[t] = rc_r;
      paramb_.rc_angular_by_type[t] = rc_a;
    }
    paramb_.rc_radial = rc_r;
    paramb_.rc_angular = rc_a;
    MN_radial_ = get_int_from_token(tokens[3], __FILE__, __LINE__);
    MN_angular_ = get_int_from_token(tokens[4], __FILE__, __LINE__);
  } else {
    paramb_.use_typewise_cutoff = true;
    for (int t = 0; t < paramb_.num_types; ++t) {
      const float rc_r = get_double_from_token(tokens[1 + 2 * t], __FILE__, __LINE__);
      const float rc_a = get_double_from_token(tokens[2 + 2 * t], __FILE__, __LINE__);
      if (!(rc_r > 0.0f) || !(rc_a > 0.0f)) {
        PRINT_INPUT_ERROR("typewise cutoff values must be > 0.\n");
      }
      paramb_.rc_radial_by_type[t] = rc_r;
      paramb_.rc_angular_by_type[t] = rc_a;
      paramb_.rc_radial = std::max(paramb_.rc_radial, rc_r);
      paramb_.rc_angular = std::max(paramb_.rc_angular, rc_a);
    }
    MN_radial_ = get_int_from_token(tokens[1 + paramb_.num_types * 2], __FILE__, __LINE__);
    MN_angular_ = get_int_from_token(tokens[2 + paramb_.num_types * 2], __FILE__, __LINE__);
  }
  if (MN_radial_ <= 0 || MN_angular_ <= 0) {
    PRINT_INPUT_ERROR("MN_radial and MN_angular must be > 0.\n");
  }

  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "n_max") {
    PRINT_INPUT_ERROR("Missing n_max line in spin potential.\n");
  }
  paramb_.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (paramb_.n_max_radial < 0 || paramb_.n_max_angular < 0) {
    PRINT_INPUT_ERROR("n_max values must be >= 0.\n");
  }

  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "basis_size") {
    PRINT_INPUT_ERROR("Missing basis_size line in spin potential.\n");
  }
  paramb_.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  if (paramb_.basis_size_radial < 0 || paramb_.basis_size_angular < 0) {
    PRINT_INPUT_ERROR("basis_size values must be >= 0.\n");
  }

  tokens = get_tokens(input);
  if (tokens.size() != 4 || tokens[0] != "l_max") {
    PRINT_INPUT_ERROR("Missing l_max line in spin potential.\n");
  }
  paramb_.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  const int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  const int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
  paramb_.num_L = paramb_.L_max;
  if (L_max_4body == 2) paramb_.num_L += 1;
  if (L_max_5body == 1) paramb_.num_L += 1;
  paramb_.dim_angular = (paramb_.n_max_angular + 1) * paramb_.num_L;

  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "ANN") {
    PRINT_INPUT_ERROR("Missing ANN line in spin potential.\n");
  }
  annmb_.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (annmb_.num_neurons1 <= 0) {
    PRINT_INPUT_ERROR("ANN num_neurons1 must be > 0.\n");
  }

  if (paramb_.basis_size_radial + 1 > MAX_NUM_N ||
      paramb_.basis_size_angular + 1 > MAX_NUM_N ||
      paramb_.n_max_radial + 1 > MAX_NUM_N ||
      paramb_.n_max_spin_radial + 1 > MAX_NUM_N ||
      paramb_.basis_size_spin_radial + 1 > MAX_NUM_N ||
      paramb_.n_max_spin_angular + 1 > MAX_NUM_N ||
      paramb_.basis_size_spin_angular + 1 > MAX_NUM_N) {
    PRINT_INPUT_ERROR("spin model exceeds compiled MAX_NUM_N.\n");
  }

  annmb_.dim =
    (paramb_.n_max_radial + 1) +
    paramb_.dim_angular +
    nep_spin_total_dim(paramb_);
  if (annmb_.dim > MAX_DIM) {
    PRINT_INPUT_ERROR("spin model descriptor dimension exceeds compiled MAX_DIM.\n");
  }

  if (paramb_.version == 3) {
    annmb_.num_para_ann = (annmb_.dim + 2) * annmb_.num_neurons1 + 1;
  } else if (paramb_.version == 4) {
    annmb_.num_para_ann = (annmb_.dim + 2) * annmb_.num_neurons1 * paramb_.num_types + 1;
  } else {
    PRINT_INPUT_ERROR("Only NEP3/NEP4 spin models are supported on GPU.\n");
  }

  paramb_.num_c_radial =
    paramb_.num_types_sq * (paramb_.n_max_radial + 1) * (paramb_.basis_size_radial + 1);
  paramb_.num_c_angular =
    paramb_.num_types_sq * (paramb_.n_max_angular + 1) * (paramb_.basis_size_angular + 1);
  paramb_.num_c_spin_2body =
    paramb_.num_types_sq * (paramb_.n_max_spin_radial + 1) * (paramb_.basis_size_spin_radial + 1);
  paramb_.num_c_spin_3body =
    paramb_.num_types_sq * (paramb_.n_max_spin_angular + 1) * (paramb_.basis_size_spin_angular + 1);
  paramb_.c_spin_2body_offset = paramb_.num_c_radial + paramb_.num_c_angular;
  paramb_.c_spin_3body_offset = paramb_.c_spin_2body_offset + paramb_.num_c_spin_2body;
  paramb_.num_c_spin = paramb_.num_c_spin_2body + paramb_.num_c_spin_3body;
  paramb_.c_spin_offset = paramb_.c_spin_2body_offset;
  paramb_.c_spin_block_stride = paramb_.num_c_spin_2body;
  annmb_.num_para =
    annmb_.num_para_ann +
    paramb_.num_c_radial +
    paramb_.num_c_angular +
    paramb_.num_c_spin_2body +
    paramb_.num_c_spin_3body;

  paramb_.rcinv_radial = 1.0f / paramb_.rc_radial;
  paramb_.rcinv_angular = 1.0f / paramb_.rc_angular;

  std::vector<float> parameters(annmb_.num_para);
  for (int n = 0; n < annmb_.num_para; ++n) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected EOF while reading spin model parameters.\n");
    }
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_data_.parameters.resize(annmb_.num_para);
  nep_data_.parameters.copy_from_host(parameters.data());
  update_potential(nep_data_.parameters.data(), annmb_);

  std::vector<float> q_scaler_host(annmb_.dim);
  for (int d = 0; d < annmb_.dim; ++d) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected EOF while reading spin model q_scaler.\n");
    }
    q_scaler_host[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_data_.q_scaler.resize(annmb_.dim);
  nep_data_.q_scaler.copy_from_host(q_scaler_host.data());
}

void NEP_Spin_LMP::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb_.num_types; ++t) {
    if (t > 0 && paramb_.version == 3) {
      // For NEP3, reuse the same ANN parameters for all types
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
  }
  ann.b1 = pointer;
  pointer += 1;
  ann.c = pointer;
}
