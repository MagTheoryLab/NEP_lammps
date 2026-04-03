/*
  NEP-Spin v2 training-side implementation.
  Baseline nonmagnetic radial/angular blocks stay aligned with src/main_nep/nep.cu.
*/

#include "dataset.cuh"
#include "nep_spin.cuh"
#include "mic.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/kernel_timing.cuh"
#include "utilities/nep_utilities.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int MAX_SPIN_ABC = 24; // l_max_spin_angular <= 4

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
static __host__ __device__ __forceinline__ int nep_spin_block2_pair_count(const ParaMB& paramb)
{
  const int nspin3 = nep_spin_3body_count(paramb);
  return nspin3 * (nspin3 - 1) / 2;
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
static __host__ __device__ __forceinline__ int nep_spin_block2_pair_lex_index(
  const ParaMB& paramb, const int n1, const int n2)
{
  const int nspin3 = nep_spin_3body_count(paramb);
  if (!(0 <= n1 && n1 < n2 && n2 < nspin3)) return -1;
  return n1 * (nspin3 - 1) - (n1 * (n1 - 1)) / 2 + (n2 - n1 - 1);
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
         (n * (paramb.basis_size_spin_radial + 1) + k) * paramb.num_types_sq + t1 * paramb.num_types + t2;
}

template <typename ParaMB>
static __host__ __device__ __forceinline__ int nep_spin_c_index_3body(
  const ParaMB& paramb, const int n, const int k, const int t1, const int t2)
{
  return paramb.c_spin_3body_offset +
         (n * (paramb.basis_size_spin_angular + 1) + k) * paramb.num_types_sq + t1 * paramb.num_types + t2;
}

template <int L>
static __device__ __forceinline__ SpinReal find_q_one_spin(const SpinReal* s)
{
  const int start = L * L - 1;
  const int num_terms = 2 * L + 1;
  SpinReal q = 0.0;
  for (int k = 1; k < num_terms; ++k) {
    q += C3B[start + k] * s[start + k] * s[start + k];
  }
  q *= 2.0;
  q += C3B[start] * s[start] * s[start];
  return q;
}

template <int L, typename T>
static __device__ __forceinline__ T find_q_cross_one(const T* a, const T* b)
{
  const int start = L * L - 1;
  const int num_terms = 2 * L + 1;
  T q = C3B[start] * a[start] * b[start];
  for (int k = 1; k < num_terms; ++k) {
    q += 2.0 * C3B[start + k] * a[start + k] * b[start + k];
  }
  return q;
}

template <typename T>
static __device__ __forceinline__ T compute_q4b_l2(const T* a)
{
  return C4B[0] * a[0] * a[0] * a[0] + C4B[1] * a[0] * (a[1] * a[1] + a[2] * a[2]) +
         C4B[2] * a[0] * (a[3] * a[3] + a[4] * a[4]) + C4B[3] * a[3] * (a[2] * a[2] - a[1] * a[1]) +
         C4B[4] * a[1] * a[2] * a[4];
}

template <typename T>
static __device__ __forceinline__ void compute_grad_q4b_l2(const T* a, T* grad)
{
  grad[0] = 3.0 * C4B[0] * a[0] * a[0] + C4B[1] * (a[1] * a[1] + a[2] * a[2]) +
            C4B[2] * (a[3] * a[3] + a[4] * a[4]);
  grad[1] = 2.0 * C4B[1] * a[0] * a[1] - 2.0 * C4B[3] * a[3] * a[1] + C4B[4] * a[2] * a[4];
  grad[2] = 2.0 * C4B[1] * a[0] * a[2] + 2.0 * C4B[3] * a[3] * a[2] + C4B[4] * a[1] * a[4];
  grad[3] = 2.0 * C4B[2] * a[0] * a[3] + C4B[3] * (a[2] * a[2] - a[1] * a[1]);
  grad[4] = 2.0 * C4B[2] * a[0] * a[4] + C4B[4] * a[1] * a[2];
}

template <typename T>
static __device__ __forceinline__ void accumulate_mix_q4b_l2(
  const T* s0, const T* sc, T& q_mix, T* grad_s0, T* grad_sc)
{
  T grad_q4b[5];
  compute_grad_q4b_l2(sc, grad_q4b);
  q_mix = 0.0;
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    grad_s0[i] = grad_q4b[i] / 3.0;
    q_mix += s0[i] * grad_s0[i];
  }

  grad_sc[0] =
    (s0[0] * (6.0 * C4B[0] * sc[0]) + s0[1] * (2.0 * C4B[1] * sc[1]) +
     s0[2] * (2.0 * C4B[1] * sc[2]) + s0[3] * (2.0 * C4B[2] * sc[3]) +
     s0[4] * (2.0 * C4B[2] * sc[4])) /
    3.0;
  grad_sc[1] =
    (s0[0] * (2.0 * C4B[1] * sc[1]) + s0[1] * (2.0 * C4B[1] * sc[0] - 2.0 * C4B[3] * sc[3]) +
     s0[2] * (C4B[4] * sc[4]) + s0[3] * (-2.0 * C4B[3] * sc[1]) + s0[4] * (C4B[4] * sc[2])) /
    3.0;
  grad_sc[2] =
    (s0[0] * (2.0 * C4B[1] * sc[2]) + s0[1] * (C4B[4] * sc[4]) +
     s0[2] * (2.0 * C4B[1] * sc[0] + 2.0 * C4B[3] * sc[3]) +
     s0[3] * (2.0 * C4B[3] * sc[2]) + s0[4] * (C4B[4] * sc[1])) /
    3.0;
  grad_sc[3] =
    (s0[0] * (2.0 * C4B[2] * sc[3]) + s0[1] * (-2.0 * C4B[3] * sc[1]) +
     s0[2] * (2.0 * C4B[3] * sc[2]) + s0[3] * (2.0 * C4B[2] * sc[0])) /
    3.0;
  grad_sc[4] =
    (s0[0] * (2.0 * C4B[2] * sc[4]) + s0[1] * (C4B[4] * sc[2]) + s0[2] * (C4B[4] * sc[1]) +
     s0[4] * (2.0 * C4B[2] * sc[0])) /
    3.0;
}

template <int L>
static __device__ __forceinline__ void fill_ylm_one(
  const float x12, const float y12, const float z12, float* ylm);

template <int L>
static __device__ __forceinline__ void accumulate_spin3body_one(
  const float xhat,
  const float yhat,
  const float zhat,
  const SpinReal w0,
  const SpinReal wc,
  const SpinReal wAx,
  const SpinReal wAy,
  const SpinReal wAz,
  const SpinReal wD,
  SpinReal* s0,
  SpinReal* sc,
  SpinReal* Ax,
  SpinReal* Ay,
  SpinReal* Az,
  SpinReal* D)
{
  float ylm[2 * L + 1];
  fill_ylm_one<L>(xhat, yhat, zhat, ylm);
  constexpr int start = L * L - 1;
  #pragma unroll
  for (int k = 0; k < 2 * L + 1; ++k) {
    const SpinReal y = ylm[k];
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
  const SpinReal w0,
  const SpinReal wc,
  const SpinReal wAx,
  const SpinReal wAy,
  const SpinReal wAz,
  const SpinReal wD,
  SpinReal* s0,
  SpinReal* sc,
  SpinReal* Ax,
  SpinReal* Ay,
  SpinReal* Az,
  SpinReal* D)
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
  const SpinReal w0,
  const SpinReal wc,
  const SpinReal wAx,
  const SpinReal wAy,
  const SpinReal wAz,
  const SpinReal wD,
  SpinReal* s0,
  SpinReal* sc,
  SpinReal* Ax,
  SpinReal* Ay,
  SpinReal* Az,
  SpinReal* D)
{
  switch (L_max) {
    case 4:
      accumulate_spin3body_all_lmax<4>(
        d12, x12, y12, z12, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
      break;
    case 3:
      accumulate_spin3body_all_lmax<3>(
        d12, x12, y12, z12, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
      break;
    case 2:
      accumulate_spin3body_all_lmax<2>(
        d12, x12, y12, z12, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
      break;
    default:
      accumulate_spin3body_all_lmax<1>(
        d12, x12, y12, z12, w0, wc, wAx, wAy, wAz, wD, s0, sc, Ax, Ay, Az, D);
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

template <int L, typename T>
static __device__ __forceinline__ SpinReal dot_packed_terms(const SpinReal* a, const T* b)
{
  SpinReal out = 0.0;
  #pragma unroll
  for (int k = 0; k < 2 * L + 1; ++k) {
    out += a[k] * b[k];
  }
  return out;
}

template <int L>
static __device__ __forceinline__ void accumulate_spin3body_force_one_L(
  const float d12inv,
  const SpinReal gn,
  const SpinReal gnp,
  const SpinReal si_dot_sj,
  const SpinReal phi_dmi,
  const SpinReal* sj,
  const float* rhat,
  const SpinReal* dEs0L,
  const SpinReal* dEscL,
  const SpinReal* dEAxL,
  const SpinReal* dEAyL,
  const SpinReal* dEAzL,
  const SpinReal* dEDL,
  float* f12,
  SpinReal& projc,
  SpinReal& projD,
  SpinReal& projAx,
  SpinReal& projAy,
  SpinReal& projAz)
{
  float ylm[2 * L + 1];
  float dEeff[2 * L + 1];
  fill_ylm_one<L>(rhat[0], rhat[1], rhat[2], ylm);
  #pragma unroll
  for (int k = 0; k < 2 * L + 1; ++k) {
    dEeff[k] = static_cast<float>(
      dEs0L[k] + si_dot_sj * dEscL[k] + sj[0] * dEAxL[k] + sj[1] * dEAyL[k] + sj[2] * dEAzL[k] +
      phi_dmi * dEDL[k]);
  }
  accumulate_f12_one<L>(d12inv, static_cast<float>(gn), static_cast<float>(gnp), dEeff, rhat, f12);
  projc += dot_packed_terms<L>(dEscL, ylm);
  projD += dot_packed_terms<L>(dEDL, ylm);
  projAx += dot_packed_terms<L>(dEAxL, ylm);
  projAy += dot_packed_terms<L>(dEAyL, ylm);
  projAz += dot_packed_terms<L>(dEAzL, ylm);
}

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
  float* g_fx,
  float* g_fy,
  float* g_fz)
{
  fi_x += f12[0];
  fi_y += f12[1];
  fi_z += f12[2];
  atomicAdd(&g_fx[n2], -f12[0]);
  atomicAdd(&g_fy[n2], -f12[1]);
  atomicAdd(&g_fz[n2], -f12[2]);
  v_xx -= r12[0] * f12[0];
  v_yy -= r12[1] * f12[1];
  v_zz -= r12[2] * f12[2];
  v_xy -= r12[0] * f12[1];
  v_yz -= r12[1] * f12[2];
  v_zx -= r12[2] * f12[0];
}

static __global__ void find_max_min(const int N, const SpinReal* g_q, float* g_q_scaler)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ SpinReal s_max[1024];
  __shared__ SpinReal s_min[1024];
  s_max[tid] = SpinReal(-1000000.0);
  s_min[tid] = SpinReal(+1000000.0);
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      const SpinReal q = g_q[m];
      if (q > s_max[tid]) s_max[tid] = q;
      if (q < s_min[tid]) s_min[tid] = q;
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) s_max[tid] = s_max[tid + offset];
      if (s_min[tid] > s_min[tid + offset]) s_min[tid] = s_min[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    SpinReal range = s_max[0] - s_min[0];
    if (!(range > SpinReal(1.0e-12))) range = SpinReal(1.0);
    g_q_scaler[bid] = min(g_q_scaler[bid], static_cast<float>(SpinReal(1.0) / range));
  }
}

// baseline radial descriptors
static __global__ void find_descriptors_radial_spinbase(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  SpinReal* g_descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
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
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
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
    g_descriptors[n1 + n * N] = (SpinReal)q[n];
  }
}

// baseline angular descriptors
static __global__ void find_descriptors_angular_spinbase(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  SpinReal* g_descriptors,
  float* g_sum_fxyz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
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
      const float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
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
      g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = (SpinReal)q[ln];
    }
  }
}

static __global__ void find_descriptors_spin_onsite(
  const int N,
  const NEP_Spin::ParaMB paramb,
  const int* __restrict__ g_type,
  const SpinReal* __restrict__ g_spin,
  SpinReal* __restrict__ g_descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N || paramb.spin_pmax <= 0) return;
  const int offset = nep_spin_block0_offset(paramb);
  const int t1 = g_type[n1];
  const SpinReal sx = __ldg(&g_spin[n1]);
  const SpinReal sy = __ldg(&g_spin[n1 + N]);
  const SpinReal sz = __ldg(&g_spin[n1 + N * 2]);
  const SpinReal si2 = sx * sx + sy * sy + sz * sz;
  if (si2 <= 1.0e-12) {
    for (int p = 0; p < paramb.spin_pmax; ++p) {
      g_descriptors[n1 + (offset + p) * N] = 0.0;
    }
    return;
  }
  if (paramb.spin_onsite_basis_mode == 0) {
    SpinReal m2p = si2;
    for (int p = 0; p < paramb.spin_pmax; ++p) {
      g_descriptors[n1 + (offset + p) * N] = m2p;
      m2p *= si2;
    }
    return;
  }
  SpinReal y = si2;
  SpinReal yref = paramb.spin_mref[t1] * paramb.spin_mref[t1];
  if (paramb.spin_onsite_basis_mode == 2) {
    y = sqrt(si2);
    yref = paramb.spin_mref[t1];
  }
  if (yref <= 0.0) yref = 1.0;
  SpinReal x = (y - yref) / (y + yref + 1.0e-12);
  if (x > SpinReal(1.0)) x = SpinReal(1.0);
  if (x < SpinReal(-1.0)) x = SpinReal(-1.0);
  SpinReal Tp[9] = {1.0};
  if (paramb.spin_pmax >= 1) Tp[1] = x;
  for (int p = 2; p <= paramb.spin_pmax; ++p) {
    Tp[p] = 2.0 * x * Tp[p - 1] - Tp[p - 2];
  }
  for (int p = 1; p <= paramb.spin_pmax; ++p) {
    g_descriptors[n1 + (offset + p - 1) * N] = Tp[p];
  }
}

static __global__ void find_descriptors_spin_2body(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const SpinReal* __restrict__ g_spin,
  SpinReal* __restrict__ g_descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  const int nspin = nep_spin_2body_count(paramb);
  const SpinReal si[3] = {
    __ldg(&g_spin[n1]),
    __ldg(&g_spin[n1 + N]),
    __ldg(&g_spin[n1 + N * 2])};
  SpinReal q_ex[MAX_NUM_N] = {0.0};
  SpinReal q_dmi[MAX_NUM_N] = {0.0};
  SpinReal q_ani[MAX_NUM_N] = {0.0};
  SpinReal q_sia[MAX_NUM_N] = {0.0};
  int bs = paramb.basis_size_spin_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = i1 * N + n1;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const StructReal r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    const SpinReal d12 = sqrt((double)r12[0] * r12[0] + (double)r12[1] * r12[1] + (double)r12[2] * r12[2]);
    if (!(d12 > 0.0f)) continue;
    const SpinReal inv_d12 = 1.0 / d12;
    const SpinReal rhat[3] = {r12[0] * inv_d12, r12[1] * inv_d12, r12[2] * inv_d12};
    const SpinReal sj[3] = {
      __ldg(&g_spin[n2]),
      __ldg(&g_spin[n2 + N]),
      __ldg(&g_spin[n2 + N * 2])};
    const SpinReal si_dot_sj = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
    const SpinReal si_dot_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const SpinReal sj_dot_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
    const SpinReal cross[3] = {
      si[1] * sj[2] - si[2] * sj[1],
      si[2] * sj[0] - si[0] * sj[2],
      si[0] * sj[1] - si[1] * sj[0]};
    const SpinReal phi_ex = si_dot_sj;
    const SpinReal phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
    const SpinReal phi_ani = si_dot_r * sj_dot_r;
    const SpinReal phi_sia = si_dot_r * si_dot_r;
    const StructReal rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const StructReal rcinv = 1.0f / rc;
    StructReal fc12;
    find_fc(rc, rcinv, (StructReal)d12, fc12);
    StructReal fn12[MAX_NUM_N];
    find_fn(bs, rcinv, (StructReal)d12, fc12, fn12);
    for (int n = 0; n < nspin; ++n) {
      SpinReal gn = 0.0;
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
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const SpinReal* __restrict__ g_spin,
  SpinReal* __restrict__ g_descriptors,
  SpinReal* __restrict__ g_sum_fxyz_0,
  SpinReal* __restrict__ g_sum_fxyz_c,
  SpinReal* __restrict__ g_sum_fxyz_Ax,
  SpinReal* __restrict__ g_sum_fxyz_Ay,
  SpinReal* __restrict__ g_sum_fxyz_Az,
  SpinReal* __restrict__ g_sum_fxyz_D)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  const int nspin3 = nep_spin_3body_count(paramb);
  const int abc_count = nep_spin_3body_abc_count(paramb);
  const SpinReal si[3] = {
    __ldg(&g_spin[n1]),
    __ldg(&g_spin[n1 + N]),
    __ldg(&g_spin[n1 + N * 2])};
  int bs = paramb.basis_size_spin_angular;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int n = 0; n < nspin3; ++n) {
    SpinReal s0[MAX_SPIN_ABC] = {0.0};
    SpinReal sc[MAX_SPIN_ABC] = {0.0};
    SpinReal Ax[MAX_SPIN_ABC] = {0.0};
    SpinReal Ay[MAX_SPIN_ABC] = {0.0};
    SpinReal Az[MAX_SPIN_ABC] = {0.0};
    SpinReal D[MAX_SPIN_ABC] = {0.0};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      const int index = i1 * N + n1;
      const int n2 = g_NL[index];
      const int t2 = g_type[n2];
      const float x12 = g_x12[index];
      const float y12 = g_y12[index];
      const float z12 = g_z12[index];
      const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
      if (!(d12 > 0.0f)) continue;
      const float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(bs, rcinv, d12, fc12, fn12);
      SpinReal gn = 0.0;
      for (int k = 0; k <= bs; ++k) {
        gn += fn12[k] * annmb.c[nep_spin_c_index_3body(paramb, n, k, t1, t2)];
      }
      const SpinReal sjx = __ldg(&g_spin[n2]);
      const SpinReal sjy = __ldg(&g_spin[n2 + N]);
      const SpinReal sjz = __ldg(&g_spin[n2 + N * 2]);
      const SpinReal gn_x_si_dot_sj = gn * (si[0] * sjx + si[1] * sjy + si[2] * sjz);
      const float d12inv = 1.0f / d12;
      const float rhat[3] = {x12 * d12inv, y12 * d12inv, z12 * d12inv};
      const SpinReal cross[3] = {
        si[1] * sjz - si[2] * sjy,
        si[2] * sjx - si[0] * sjz,
        si[0] * sjy - si[1] * sjx};
      const SpinReal phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
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
      SpinReal q2 = 0.0, q3 = 0.0, q4 = 0.0, qD0 = 0.0, qDc = 0.0;
      if (L == 1) {
        q2 = find_q_one_spin<1>(sc);
        q3 = find_q_one_spin<1>(Ax) + find_q_one_spin<1>(Ay) + find_q_one_spin<1>(Az);
        q4 = find_q_cross_one<1>(s0, sc);
        qD0 = find_q_cross_one<1>(s0, D);
        qDc = find_q_cross_one<1>(sc, D);
      } else if (L == 2) {
        q2 = find_q_one_spin<2>(sc);
        q3 = find_q_one_spin<2>(Ax) + find_q_one_spin<2>(Ay) + find_q_one_spin<2>(Az);
        q4 = find_q_cross_one<2>(s0, sc);
        qD0 = find_q_cross_one<2>(s0, D);
        qDc = find_q_cross_one<2>(sc, D);
      } else if (L == 3) {
        q2 = find_q_one_spin<3>(sc);
        q3 = find_q_one_spin<3>(Ax) + find_q_one_spin<3>(Ay) + find_q_one_spin<3>(Az);
        q4 = find_q_cross_one<3>(s0, sc);
        qD0 = find_q_cross_one<3>(s0, D);
        qDc = find_q_cross_one<3>(sc, D);
      } else if (L == 4) {
        q2 = find_q_one_spin<4>(sc);
        q3 = find_q_one_spin<4>(Ax) + find_q_one_spin<4>(Ay) + find_q_one_spin<4>(Az);
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
      SpinReal grad_s0_mix[5];
      SpinReal grad_sc_mix[5];
      const SpinReal q4b = compute_q4b_l2(sc + 3);
      SpinReal qmix = 0.0;
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
                     (g_sum_fxyz_Ax[idx1] * g_sum_fxyz_Ax[idx2] + g_sum_fxyz_Ay[idx1] * g_sum_fxyz_Ay[idx2] +
                      g_sum_fxyz_Az[idx1] * g_sum_fxyz_Az[idx2]);
        }
        g_descriptors[n1 + nep_spin_block2_across_index(paramb, n1_pair, n2_pair, L - 1) * N] = qAcross;
      }
    }
  }
}

static __global__ void zero_spin_descriptor_block(
  const int N, const int spin_dim, SpinReal* g_descriptors, const int spin_offset)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  for (int d = 0; d < spin_dim; ++d) {
    g_descriptors[n1 + (spin_offset + d) * N] = 0.0;
  }
}

static __global__ void zero_spin_sum_block(const int size, SpinReal* data)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) data[idx] = 0.0;
}

static __global__ void apply_ann_spin(
  const int N,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const SpinReal* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  SpinReal* g_pe,
  SpinReal* g_Fp)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  const int type = g_type[n1];
  SpinReal q[MAX_DIM];
  SpinReal Fp[MAX_DIM] = {0.0};
  for (int d = 0; d < annmb.dim; ++d) {
    q[d] = g_descriptors[n1 + d * N] * (SpinReal)g_q_scaler[d];
  }
  SpinReal F = 0.0;
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
    g_Fp[n1 + d * N] = Fp[d] * (SpinReal)g_q_scaler[d];
  }
}

static __global__ void zero_force_spin(
  const int N, float* g_fx, float* g_fy, float* g_fz, float* g_virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  g_fx[n1] = 0.0f;
  g_fy[n1] = 0.0f;
  g_fz[n1] = 0.0f;
  for (int d = 0; d < 6; ++d) {
    g_virial[n1 + d * N] = 0.0f;
  }
}

static __global__ void zero_mforce_spin(const int N, SpinReal* g_mx, SpinReal* g_my, SpinReal* g_mz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_mx[n1] = 0.0;
    g_my[n1] = 0.0;
    g_mz[n1] = 0.0;
  }
}

static __global__ void gpu_find_neighbor_list_spin(
  const NEP_Spin::ParaMB paramb,
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_type,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  const int cap_radial,
  const int cap_angular,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  float* x12_radial,
  float* y12_radial,
  float* z12_radial,
  float* x12_angular,
  float* y12_angular,
  float* z12_angular)
{
  const int N1 = Na_sum[blockIdx.x];
  const int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    const float x1 = x[n1];
    const float y1 = y[n1];
    const float z1 = z[n1];
    const int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) continue;
            const float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            const float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            const float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            dev_apply_mic(box, x12, y12, z12);
            const float d2 = x12 * x12 + y12 * y12 + z12 * z12;
            const int t2 = g_type[n2];
            const float rc_r = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            const float rc_a = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
            if (d2 < rc_r * rc_r && count_radial < cap_radial) {
              NL_radial[count_radial * N + n1] = n2;
              x12_radial[count_radial * N + n1] = x12;
              y12_radial[count_radial * N + n1] = y12;
              z12_radial[count_radial * N + n1] = z12;
              ++count_radial;
            }
            if (d2 < rc_a * rc_a && count_angular < cap_angular) {
              NL_angular[count_angular * N + n1] = n2;
              x12_angular[count_angular * N + n1] = x12;
              y12_angular[count_angular * N + n1] = y12;
              z12_angular[count_angular * N + n1] = z12;
              ++count_angular;
            }
          }
        }
      }
    }
    NN_radial[n1] = min(count_radial, cap_radial);
    NN_angular[n1] = min(count_angular, cap_angular);
  }
}

static __global__ void find_force_radial_spinbase(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const SpinReal* __restrict__ g_Fp,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  float fi_acc_x = 0.0f, fi_acc_y = 0.0f, fi_acc_z = 0.0f;
  float v_xx = 0.0f, v_yy = 0.0f, v_zz = 0.0f, v_xy = 0.0f, v_yz = 0.0f, v_zx = 0.0f;
  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = i1 * N + n1;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
    float f12[3] = {0.0f, 0.0f, 0.0f};
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gnp12 = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2;
        gnp12 += fnp12[k] * annmb.c[c_index];
      }
      const float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
    }
    add_force_and_virial(
      n2, r12, f12, fi_acc_x, fi_acc_y, fi_acc_z, v_xx, v_yy, v_zz, v_xy, v_yz, v_zx, g_fx, g_fy, g_fz);
  }
  if (neighbor_number > 0) {
    atomicAdd(&g_fx[n1], fi_acc_x);
    atomicAdd(&g_fy[n1], fi_acc_y);
    atomicAdd(&g_fz[n1], fi_acc_z);
  }
  g_virial[n1 + N * 0] += v_xx;
  g_virial[n1 + N * 1] += v_yy;
  g_virial[n1 + N * 2] += v_zz;
  g_virial[n1 + N * 3] += v_xy;
  g_virial[n1 + N * 4] += v_yz;
  g_virial[n1 + N * 5] += v_zx;
}

static __global__ void find_force_angular_spinbase(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const SpinReal* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  float fi_acc_x = 0.0f, fi_acc_y = 0.0f, fi_acc_z = 0.0f;
  float v_xx = 0.0f, v_yy = 0.0f, v_zz = 0.0f, v_xy = 0.0f, v_yz = 0.0f, v_zx = 0.0f;
  float Fp_loc[MAX_DIM_ANGULAR] = {0.0f};
  float sum_fxyz_loc[NUM_OF_ABC * MAX_NUM_N] = {0.0f};
  int dim_ang = paramb.dim_angular;
  if (dim_ang > MAX_DIM_ANGULAR) dim_ang = MAX_DIM_ANGULAR;
  for (int d = 0; d < dim_ang; ++d) {
    Fp_loc[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
  }
  const int abc_count = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  int nmax = paramb.n_max_angular + 1;
  if (nmax > MAX_NUM_N) nmax = MAX_NUM_N;
  for (int n = 0; n < nmax; ++n) {
    for (int abc = 0; abc < abc_count; ++abc) {
      sum_fxyz_loc[n * NUM_OF_ABC + abc] =
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
    }
  }
  int bs = paramb.basis_size_angular;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = i1 * N + n1;
    const int n2 = g_NL[index];
    const int t2 = g_type[n2];
    const float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
    float f12[3] = {0.0f, 0.0f, 0.0f};
    for (int n = 0; n < nmax; ++n) {
      float gn12 = 0.0f, gnp12 = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
        gnp12 += fnp12[k] * annmb.c[c_index];
      }
      accumulate_f12(
        paramb.L_max, paramb.num_L, n, nmax, d12, r12, gn12, gnp12, Fp_loc, sum_fxyz_loc, f12);
    }
    add_force_and_virial(
      n2, r12, f12, fi_acc_x, fi_acc_y, fi_acc_z, v_xx, v_yy, v_zz, v_xy, v_yz, v_zx, g_fx, g_fy, g_fz);
  }
  if (neighbor_number > 0) {
    atomicAdd(&g_fx[n1], fi_acc_x);
    atomicAdd(&g_fy[n1], fi_acc_y);
    atomicAdd(&g_fz[n1], fi_acc_z);
  }
  g_virial[n1 + N * 0] += v_xx;
  g_virial[n1 + N * 1] += v_yy;
  g_virial[n1 + N * 2] += v_zz;
  g_virial[n1 + N * 3] += v_xy;
  g_virial[n1 + N * 4] += v_yz;
  g_virial[n1 + N * 5] += v_zx;
}

static __global__ void find_mforce_spin_onsite(
  const int N,
  const NEP_Spin::ParaMB paramb,
  const int* __restrict__ g_type,
  const SpinReal* __restrict__ g_spin,
  const SpinReal* __restrict__ g_Fp,
  SpinReal* __restrict__ g_mx,
  SpinReal* __restrict__ g_my,
  SpinReal* __restrict__ g_mz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N || paramb.spin_pmax <= 0) return;
  const int t1 = g_type[n1];
  const SpinReal si[3] = {
    __ldg(&g_spin[n1]),
    __ldg(&g_spin[n1 + N]),
    __ldg(&g_spin[n1 + N * 2])};
  const SpinReal si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= 1.0e-12) return;
  const SpinReal msign = paramb.mforce_sign;
  const int offset = nep_spin_block0_offset(paramb);
  SpinReal mx = 0.0, my = 0.0, mz = 0.0;
  if (paramb.spin_onsite_basis_mode == 0) {
    SpinReal m2pow = 1.0;
    for (int p = 1; p <= paramb.spin_pmax; ++p) {
      const SpinReal Fp_p = __ldg(&g_Fp[n1 + (offset + p - 1) * N]);
      const SpinReal coeff = msign * Fp_p * (2.0 * p) * m2pow;
      mx += coeff * si[0];
      my += coeff * si[1];
      mz += coeff * si[2];
      m2pow *= si2;
    }
  } else {
    SpinReal y = si2;
    SpinReal yref = paramb.spin_mref[t1] * paramb.spin_mref[t1];
    const SpinReal si_norm = sqrt(si2);
    const SpinReal inv_si_norm = 1.0 / (si_norm + 1.0e-12);
    if (paramb.spin_onsite_basis_mode == 2) {
      y = si_norm;
      yref = paramb.spin_mref[t1];
    }
    if (yref <= 0.0) yref = 1.0;
    const SpinReal denom = y + yref;
    const SpinReal inv_denom = 1.0 / (denom + 1.0e-12);
    SpinReal x = (y - yref) * inv_denom;
    if (x > SpinReal(1.0)) x = SpinReal(1.0);
    if (x < SpinReal(-1.0)) x = SpinReal(-1.0);
    const SpinReal dx_dy = (2.0 * yref) * inv_denom * inv_denom;
    SpinReal dy_dsi[3] = {
      SpinReal(2.0) * si[0], SpinReal(2.0) * si[1], SpinReal(2.0) * si[2]};
    if (paramb.spin_onsite_basis_mode == 2) {
      dy_dsi[0] = si[0] * inv_si_norm;
      dy_dsi[1] = si[1] * inv_si_norm;
      dy_dsi[2] = si[2] * inv_si_norm;
    }
    SpinReal Tp[9] = {1.0};
    SpinReal dTp[9] = {0.0};
    if (paramb.spin_pmax >= 1) {
      Tp[1] = x;
      dTp[1] = 1.0;
    }
    for (int p = 2; p <= paramb.spin_pmax; ++p) {
      Tp[p] = 2.0 * x * Tp[p - 1] - Tp[p - 2];
      dTp[p] = 2.0 * Tp[p - 1] + 2.0 * x * dTp[p - 1] - dTp[p - 2];
    }
    for (int p = 1; p <= paramb.spin_pmax; ++p) {
      const SpinReal Fp_p = __ldg(&g_Fp[n1 + (offset + p - 1) * N]);
      const SpinReal coeff = msign * Fp_p * dTp[p] * dx_dy;
      mx += coeff * dy_dsi[0];
      my += coeff * dy_dsi[1];
      mz += coeff * dy_dsi[2];
    }
  }
  atomicAdd(&g_mx[n1], mx);
  atomicAdd(&g_my[n1], my);
  atomicAdd(&g_mz[n1], mz);
}

static __global__ void find_force_spin_2body(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const SpinReal* __restrict__ g_spin,
  const SpinReal* __restrict__ g_Fp,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  SpinReal* g_mx,
  SpinReal* g_my,
  SpinReal* g_mz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  const int nspin = nep_spin_2body_count(paramb);
  const SpinReal si[3] = {
    __ldg(&g_spin[n1]),
    __ldg(&g_spin[n1 + N]),
    __ldg(&g_spin[n1 + N * 2])};
  const SpinReal msign = paramb.mforce_sign;
  float fi_acc_x = 0.0f, fi_acc_y = 0.0f, fi_acc_z = 0.0f;
  float v_xx = 0.0f, v_yy = 0.0f, v_zz = 0.0f, v_xy = 0.0f, v_yz = 0.0f, v_zx = 0.0f;
  SpinReal mi_x = 0.0, mi_y = 0.0, mi_z = 0.0;
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
    const SpinReal sj[3] = {
      __ldg(&g_spin[n2]),
      __ldg(&g_spin[n2 + N]),
      __ldg(&g_spin[n2 + N * 2])};
    const SpinReal si_dot_sj = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
    const SpinReal si_dot_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const SpinReal sj_dot_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
    const SpinReal cross[3] = {
      si[1] * sj[2] - si[2] * sj[1],
      si[2] * sj[0] - si[0] * sj[2],
      si[0] * sj[1] - si[1] * sj[0]};
    const SpinReal phi_ex = si_dot_sj;
    const SpinReal phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
    const SpinReal phi_ani = si_dot_r * sj_dot_r;
    const SpinReal phi_sia = si_dot_r * si_dot_r;
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
    float f12[3] = {0.0f, 0.0f, 0.0f};
    SpinReal mj_x = 0.0, mj_y = 0.0, mj_z = 0.0;
    for (int n = 0; n < nspin; ++n) {
      SpinReal gn = 0.0, gnp = 0.0;
      for (int k = 0; k <= bs; ++k) {
        const float c = annmb.c[nep_spin_c_index_2body(paramb, n, k, t1, t2)];
        gn += fn12[k] * c;
        gnp += fnp12[k] * c;
      }
      const SpinReal Fex = __ldg(&g_Fp[n1 + nep_spin_block1_index(paramb, 0, n) * N]);
      const SpinReal Fdmi = __ldg(&g_Fp[n1 + nep_spin_block1_index(paramb, 1, n) * N]);
      const SpinReal Fani = __ldg(&g_Fp[n1 + nep_spin_block1_index(paramb, 2, n) * N]);
      const SpinReal Fsia = __ldg(&g_Fp[n1 + nep_spin_block1_index(paramb, 3, n) * N]);
      for (int d = 0; d < 3; ++d) {
        f12[d] += static_cast<float>(Fex * (phi_ex * gnp * rhat[d]));
      }
      if (Fdmi != 0.0f) {
        for (int d = 0; d < 3; ++d) {
          f12[d] += static_cast<float>(
            Fdmi * (gn * (cross[d] - phi_dmi * rhat[d]) * inv_d12 + phi_dmi * gnp * rhat[d]));
        }
      }
      if (Fani != 0.0f) {
        const SpinReal vec_ani[3] = {
          si_dot_r * (sj[0] - sj_dot_r * rhat[0]) + sj_dot_r * (si[0] - si_dot_r * rhat[0]),
          si_dot_r * (sj[1] - sj_dot_r * rhat[1]) + sj_dot_r * (si[1] - si_dot_r * rhat[1]),
          si_dot_r * (sj[2] - sj_dot_r * rhat[2]) + sj_dot_r * (si[2] - si_dot_r * rhat[2])};
        for (int d = 0; d < 3; ++d) {
          f12[d] += static_cast<float>(Fani * (gn * vec_ani[d] * inv_d12 + phi_ani * gnp * rhat[d]));
        }
      }
      if (Fsia != 0.0f) {
        const SpinReal vec_sia[3] = {
          2.0f * si_dot_r * (si[0] - si_dot_r * rhat[0]),
          2.0f * si_dot_r * (si[1] - si_dot_r * rhat[1]),
          2.0f * si_dot_r * (si[2] - si_dot_r * rhat[2])};
        for (int d = 0; d < 3; ++d) {
          f12[d] += static_cast<float>(Fsia * (gn * vec_sia[d] * inv_d12 + phi_sia * gnp * rhat[d]));
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
    atomicAdd(&g_mx[n2], mj_x);
    atomicAdd(&g_my[n2], mj_y);
    atomicAdd(&g_mz[n2], mj_z);
  }
  if (neighbor_number > 0) {
    atomicAdd(&g_fx[n1], fi_acc_x);
    atomicAdd(&g_fy[n1], fi_acc_y);
    atomicAdd(&g_fz[n1], fi_acc_z);
  }
  atomicAdd(&g_mx[n1], mi_x);
  atomicAdd(&g_my[n1], mi_y);
  atomicAdd(&g_mz[n1], mi_z);
  g_virial[n1 + N * 0] += v_xx;
  g_virial[n1 + N * 1] += v_yy;
  g_virial[n1 + N * 2] += v_zz;
  g_virial[n1 + N * 3] += v_xy;
  g_virial[n1 + N * 4] += v_yz;
  g_virial[n1 + N * 5] += v_zx;
}

static __global__ void find_force_spin_3body(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const SpinReal* __restrict__ g_spin,
  const SpinReal* __restrict__ g_Fp,
  const SpinReal* __restrict__ g_sum_fxyz_0,
  const SpinReal* __restrict__ g_sum_fxyz_c,
  const SpinReal* __restrict__ g_sum_fxyz_Ax,
  const SpinReal* __restrict__ g_sum_fxyz_Ay,
  const SpinReal* __restrict__ g_sum_fxyz_Az,
  const SpinReal* __restrict__ g_sum_fxyz_D,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  SpinReal* g_mx,
  SpinReal* g_my,
  SpinReal* g_mz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  const int t1 = g_type[n1];
  const int neighbor_number = g_NN[n1];
  const int nspin3 = nep_spin_3body_count(paramb);
  const SpinReal si[3] = {
    __ldg(&g_spin[n1]),
    __ldg(&g_spin[n1 + N]),
    __ldg(&g_spin[n1 + N * 2])};
  const SpinReal msign = paramb.mforce_sign;
  float fi_acc_x = 0.0f, fi_acc_y = 0.0f, fi_acc_z = 0.0f;
  float v_xx = 0.0f, v_yy = 0.0f, v_zz = 0.0f, v_xy = 0.0f, v_yz = 0.0f, v_zx = 0.0f;
  SpinReal mi_x = 0.0, mi_y = 0.0, mi_z = 0.0;
  const int sum_stride = nep_spin_3body_abc_count(paramb);
  const int g1_count = nep_spin_block2_g1_count(paramb);
  int bs = paramb.basis_size_spin_angular;
  if (bs >= MAX_NUM_N) bs = MAX_NUM_N - 1;
  for (int n = 0; n < nspin3; ++n) {
    SpinReal dEs0_1[3] = {0.0}, dEsc_1[3] = {0.0}, dEAx_1[3] = {0.0}, dEAy_1[3] = {0.0},
             dEAz_1[3] = {0.0}, dED_1[3] = {0.0};
    SpinReal dEs0_2[5] = {0.0}, dEsc_2[5] = {0.0}, dEAx_2[5] = {0.0}, dEAy_2[5] = {0.0},
             dEAz_2[5] = {0.0}, dED_2[5] = {0.0};
    SpinReal dEs0_3[7] = {0.0}, dEsc_3[7] = {0.0}, dEAx_3[7] = {0.0}, dEAy_3[7] = {0.0},
             dEAz_3[7] = {0.0}, dED_3[7] = {0.0};
    SpinReal dEs0_4[9] = {0.0}, dEsc_4[9] = {0.0}, dEAx_4[9] = {0.0}, dEAy_4[9] = {0.0},
             dEAz_4[9] = {0.0}, dED_4[9] = {0.0};

    auto fill_one_L = [&](const int L,
                          SpinReal* dEs0L,
                          SpinReal* dEscL,
                          SpinReal* dEAxL,
                          SpinReal* dEAyL,
                          SpinReal* dEAzL,
                          SpinReal* dEDL) {
      const SpinReal G2 = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 0, n, L - 1) * N]);
      const SpinReal G3 = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 1, n, L - 1) * N]);
      const SpinReal G4 = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 2, n, L - 1) * N]);
      const SpinReal GD0 = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 3, n, L - 1) * N]);
      const SpinReal GDc = __ldg(&g_Fp[n1 + nep_spin_block2_core_index(paramb, 4, n, L - 1) * N]);
      const int start = L * L - 1;
      const int terms = 2 * L + 1;
      for (int k = 0; k < terms; ++k) {
        const int abc = start + k;
        const SpinReal weight = (k == 0 ? 1.0 : 2.0) * C3B[abc];
        const int idx = (n * sum_stride + abc) * N + n1;
        const SpinReal s0 = g_sum_fxyz_0[idx];
        const SpinReal sc = g_sum_fxyz_c[idx];
        const SpinReal Ax = g_sum_fxyz_Ax[idx];
        const SpinReal Ay = g_sum_fxyz_Ay[idx];
        const SpinReal Az = g_sum_fxyz_Az[idx];
        const SpinReal D = g_sum_fxyz_D[idx];
        dEs0L[k] = G4 * weight * sc + GD0 * weight * D;
        dEscL[k] = 2.0 * G2 * weight * sc + G4 * weight * s0 + GDc * weight * D;
        dEAxL[k] = 2.0 * G3 * weight * Ax;
        dEAyL[k] = 2.0 * G3 * weight * Ay;
        dEAzL[k] = 2.0 * G3 * weight * Az;
        dEDL[k] = GD0 * weight * s0 + GDc * weight * sc;
      }
      if (L == 2 && g1_count > 0) {
        const SpinReal G4b = __ldg(&g_Fp[n1 + nep_spin_block2_g1_index(paramb, 0, n) * N]);
        const SpinReal Gmix = __ldg(&g_Fp[n1 + nep_spin_block2_g1_index(paramb, 1, n) * N]);
        if (G4b != 0.0f || Gmix != 0.0f) {
          float sc2[5];
          float s02[5];
          float grad_q4b[5];
          float grad_s0_mix[5];
          float grad_sc_mix[5];
          for (int k = 0; k < 5; ++k) {
            const int idx = (n * sum_stride + 3 + k) * N + n1;
            sc2[k] = static_cast<float>(g_sum_fxyz_c[idx]);
            s02[k] = static_cast<float>(g_sum_fxyz_0[idx]);
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
        const SpinReal GAcross =
          __ldg(&g_Fp[n1 + nep_spin_block2_across_index(paramb, low, high, L - 1) * N]);
        if (GAcross == 0.0f) continue;
        const int start = L * L - 1;
        const int terms = 2 * L + 1;
        for (int k = 0; k < terms; ++k) {
          const int abc = start + k;
          const SpinReal weight = (k == 0 ? 1.0 : 2.0) * C3B[abc];
          const int idx = (other * sum_stride + abc) * N + n1;
          const SpinReal Ax_other = g_sum_fxyz_Ax[idx];
          const SpinReal Ay_other = g_sum_fxyz_Ay[idx];
          const SpinReal Az_other = g_sum_fxyz_Az[idx];
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
      const float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12, fcp12;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
      SpinReal gn = 0.0, gnp = 0.0;
      for (int k = 0; k <= bs; ++k) {
        const float c = annmb.c[nep_spin_c_index_3body(paramb, n, k, t1, t2)];
        gn += fn12[k] * c;
        gnp += fnp12[k] * c;
      }
      const SpinReal sj[3] = {
        __ldg(&g_spin[n2]),
        __ldg(&g_spin[n2 + N]),
        __ldg(&g_spin[n2 + N * 2])};
      const SpinReal si_dot_sj = si[0] * sj[0] + si[1] * sj[1] + si[2] * sj[2];
      const SpinReal cross[3] = {
        si[1] * sj[2] - si[2] * sj[1],
        si[2] * sj[0] - si[0] * sj[2],
        si[0] * sj[1] - si[1] * sj[0]};
      const float d12inv = 1.0f / d12;
      const float rhat[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};
      const SpinReal phi_dmi = cross[0] * rhat[0] + cross[1] * rhat[1] + cross[2] * rhat[2];
      float f12[3] = {0.0f, 0.0f, 0.0f};
      SpinReal projc = 0.0, projD = 0.0, projAx = 0.0, projAy = 0.0, projAz = 0.0;
      if (paramb.l_max_spin_angular >= 1) {
        accumulate_spin3body_force_one_L<1>(
          d12inv,
          gn,
          gnp,
          si_dot_sj,
          phi_dmi,
          sj,
          rhat,
          dEs0_1,
          dEsc_1,
          dEAx_1,
          dEAy_1,
          dEAz_1,
          dED_1,
          f12,
          projc,
          projD,
          projAx,
          projAy,
          projAz);
      }
      if (paramb.l_max_spin_angular >= 2) {
        accumulate_spin3body_force_one_L<2>(
          d12inv,
          gn,
          gnp,
          si_dot_sj,
          phi_dmi,
          sj,
          rhat,
          dEs0_2,
          dEsc_2,
          dEAx_2,
          dEAy_2,
          dEAz_2,
          dED_2,
          f12,
          projc,
          projD,
          projAx,
          projAy,
          projAz);
      }
      if (paramb.l_max_spin_angular >= 3) {
        accumulate_spin3body_force_one_L<3>(
          d12inv,
          gn,
          gnp,
          si_dot_sj,
          phi_dmi,
          sj,
          rhat,
          dEs0_3,
          dEsc_3,
          dEAx_3,
          dEAy_3,
          dEAz_3,
          dED_3,
          f12,
          projc,
          projD,
          projAx,
          projAy,
          projAz);
      }
      if (paramb.l_max_spin_angular >= 4) {
        accumulate_spin3body_force_one_L<4>(
          d12inv,
          gn,
          gnp,
          si_dot_sj,
          phi_dmi,
          sj,
          rhat,
          dEs0_4,
          dEsc_4,
          dEAx_4,
          dEAy_4,
          dEAz_4,
          dED_4,
          f12,
          projc,
          projD,
          projAx,
          projAy,
          projAz);
      }
      if (projD != 0.0f) {
        for (int d = 0; d < 3; ++d) {
          f12[d] += static_cast<float>(gn * projD * (cross[d] - phi_dmi * rhat[d]) * d12inv);
        }
      }
      add_force_and_virial(
        n2, r12, f12, fi_acc_x, fi_acc_y, fi_acc_z, v_xx, v_yy, v_zz, v_xy, v_yz, v_zx, g_fx, g_fy, g_fz);
      const SpinReal cross_sj_r[3] = {
        sj[1] * rhat[2] - sj[2] * rhat[1],
        sj[2] * rhat[0] - sj[0] * rhat[2],
        sj[0] * rhat[1] - sj[1] * rhat[0]};
      const SpinReal cross_r_si[3] = {
        rhat[1] * si[2] - rhat[2] * si[1],
        rhat[2] * si[0] - rhat[0] * si[2],
        rhat[0] * si[1] - rhat[1] * si[0]};
      mi_x += msign * gn * projc * sj[0];
      mi_y += msign * gn * projc * sj[1];
      mi_z += msign * gn * projc * sj[2];
      mi_x += msign * gn * projD * cross_sj_r[0];
      mi_y += msign * gn * projD * cross_sj_r[1];
      mi_z += msign * gn * projD * cross_sj_r[2];
      atomicAdd(&g_mx[n2], msign * gn * (projc * si[0] + projAx + projD * cross_r_si[0]));
      atomicAdd(&g_my[n2], msign * gn * (projc * si[1] + projAy + projD * cross_r_si[1]));
      atomicAdd(&g_mz[n2], msign * gn * (projc * si[2] + projAz + projD * cross_r_si[2]));
    }
  }
  if (neighbor_number > 0) {
    atomicAdd(&g_fx[n1], fi_acc_x);
    atomicAdd(&g_fy[n1], fi_acc_y);
    atomicAdd(&g_fz[n1], fi_acc_z);
  }
  atomicAdd(&g_mx[n1], mi_x);
  atomicAdd(&g_my[n1], mi_y);
  atomicAdd(&g_mz[n1], mi_z);
  g_virial[n1 + N * 0] += v_xx;
  g_virial[n1 + N * 1] += v_yy;
  g_virial[n1 + N * 2] += v_zz;
  g_virial[n1 + N * 3] += v_xy;
  g_virial[n1 + N * 4] += v_yz;
  g_virial[n1 + N * 5] += v_zx;
}

} // namespace

static KernelTiming g_kernel_timing_spin[16];
static long long g_kernel_timing_spin_call = 0;

NEP_Spin::NEP_Spin(
  Parameters& para,
  int N,
  int N_times_max_NN_radial,
  int N_times_max_NN_angular,
  int version,
  int deviceCount)
{
  paramb.version = version;
  paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
  paramb.num_types = para.num_types;
  for (int t = 0; t < paramb.num_types; ++t) {
    paramb.rc_radial[t] = para.rc_radial[t];
    paramb.rc_angular[t] = para.rc_angular[t];
  }
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  paramb.spin_pmax = para.spin_pmax;
  paramb.spin_onsite_basis_mode = para.spin_onsite_basis_mode;
  for (int t = 0; t < NUM_ELEMENTS; ++t) {
    paramb.spin_mref[t] = para.spin_mref[t];
  }
  paramb.n_max_spin_radial = para.n_max_spin_radial;
  paramb.basis_size_spin_radial = para.basis_size_spin_radial;
  paramb.n_max_spin_angular = para.n_max_spin_angular;
  paramb.l_max_spin_angular = para.l_max_spin_angular;
  paramb.basis_size_spin_angular = para.basis_size_spin_angular;
  paramb.num_L = paramb.L_max;
  if (para.L_max_4body == 2) paramb.num_L += 1;
  if (para.L_max_5body == 1) paramb.num_L += 1;
  paramb.dim_angular = (para.n_max_angular + 1) * paramb.num_L;
  paramb.basis_size_radial = para.basis_size_radial;
  paramb.basis_size_angular = para.basis_size_angular;
  paramb.num_types_sq = para.num_types * para.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
  paramb.num_c_angular =
    paramb.num_types_sq * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
  paramb.num_c_spin_2body =
    paramb.num_types_sq * (para.n_max_spin_radial + 1) * (para.basis_size_spin_radial + 1);
  paramb.num_c_spin_3body =
    paramb.num_types_sq * (para.n_max_spin_angular + 1) * (para.basis_size_spin_angular + 1);
  paramb.c_spin_2body_offset = paramb.num_c_radial + paramb.num_c_angular;
  paramb.c_spin_3body_offset = paramb.c_spin_2body_offset + paramb.num_c_spin_2body;
  for (int n = 0; n < static_cast<int>(para.atomic_numbers.size()); ++n) {
    paramb.atomic_numbers[n] = para.atomic_numbers[n] - 1;
  }

  const int spin_sum_size = N * nep_spin_3body_count(paramb) * nep_spin_3body_abc_count(paramb);
  for (int device_id = 0; device_id < deviceCount; ++device_id) {
    gpuSetDevice(device_id);
    annmb[device_id].dim = para.dim;
    annmb[device_id].num_neurons1 = para.num_neurons1;
    annmb[device_id].num_para = para.number_of_variables;
    nep_data[device_id].NN_radial.resize(N);
    nep_data[device_id].NN_angular.resize(N);
    nep_data[device_id].NL_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].NL_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].x12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].y12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].z12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].x12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].y12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].z12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].descriptors.resize(N * annmb[device_id].dim);
    nep_data[device_id].Fp.resize(N * annmb[device_id].dim);
    nep_data[device_id].sum_fxyz.resize(
      N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
    nep_data[device_id].sum_fxyz_0.resize(spin_sum_size);
    nep_data[device_id].sum_fxyz_c.resize(spin_sum_size);
    nep_data[device_id].sum_fxyz_Ax.resize(spin_sum_size);
    nep_data[device_id].sum_fxyz_Ay.resize(spin_sum_size);
    nep_data[device_id].sum_fxyz_Az.resize(spin_sum_size);
    nep_data[device_id].sum_fxyz_D.resize(spin_sum_size);
    nep_data[device_id].parameters.resize(annmb[device_id].num_para);
  }
}

void NEP_Spin::update_potential(Parameters& para, float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) {
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

void NEP_Spin::find_force(
  Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  bool calculate_neighbor,
  int device_in_this_iter)
{
  const long long call_id = g_kernel_timing_spin_call++;
  const bool do_profile =
    (para.kernel_timing != 0) &&
    (call_id >= static_cast<long long>(para.kernel_timing_skip)) &&
    (((call_id - static_cast<long long>(para.kernel_timing_skip)) %
      static_cast<long long>(para.kernel_timing_every)) == 0);

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    nep_data[device_id].parameters.copy_from_host(parameters + device_id * para.number_of_variables);
    update_potential(para, nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    const int N = dataset[device_id].N;
    const int block_size = 32;
    const int grid_size = (N - 1) / block_size + 1;
    KernelTiming& kt = g_kernel_timing_spin[device_id];

    if (calculate_neighbor) {
      int cap_radial_per_atom = static_cast<int>(nep_data[device_id].NL_radial.size() / N);
      int cap_angular_per_atom = static_cast<int>(nep_data[device_id].NL_angular.size() / N);
      if (cap_radial_per_atom < 1) cap_radial_per_atom = 1;
      if (cap_angular_per_atom < 1) cap_angular_per_atom = 1;
      if (do_profile) {
        int tok = kt.begin("gpu_find_neighbor_list_spin");
        gpu_find_neighbor_list_spin<<<dataset[device_id].Nc, 256>>>(
          paramb,
          N,
          dataset[device_id].Na.data(),
          dataset[device_id].Na_sum.data(),
          dataset[device_id].type.data(),
          dataset[device_id].box.data(),
          dataset[device_id].box_original.data(),
          dataset[device_id].num_cell.data(),
          dataset[device_id].r.data(),
          dataset[device_id].r.data() + N,
          dataset[device_id].r.data() + N * 2,
          cap_radial_per_atom,
          cap_angular_per_atom,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          nep_data[device_id].NN_angular.data(),
          nep_data[device_id].NL_angular.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          nep_data[device_id].x12_angular.data(),
          nep_data[device_id].y12_angular.data(),
          nep_data[device_id].z12_angular.data());
        kt.end(tok);
      } else {
        gpu_find_neighbor_list_spin<<<dataset[device_id].Nc, 256>>>(
          paramb,
          N,
          dataset[device_id].Na.data(),
          dataset[device_id].Na_sum.data(),
          dataset[device_id].type.data(),
          dataset[device_id].box.data(),
          dataset[device_id].box_original.data(),
          dataset[device_id].num_cell.data(),
          dataset[device_id].r.data(),
          dataset[device_id].r.data() + N,
          dataset[device_id].r.data() + N * 2,
          cap_radial_per_atom,
          cap_angular_per_atom,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          nep_data[device_id].NN_angular.data(),
          nep_data[device_id].NL_angular.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          nep_data[device_id].x12_angular.data(),
          nep_data[device_id].y12_angular.data(),
          nep_data[device_id].z12_angular.data());
      }
      GPU_CHECK_KERNEL
    }

    if (do_profile) {
      int tok = kt.begin("find_descriptors_radial_spinbase");
      find_descriptors_radial_spinbase<<<grid_size, block_size>>>(
        N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].descriptors.data());
      kt.end(tok);
    } else {
      find_descriptors_radial_spinbase<<<grid_size, block_size>>>(
        N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].descriptors.data());
    }
    GPU_CHECK_KERNEL

    if (do_profile) {
      int tok = kt.begin("find_descriptors_angular_spinbase");
      find_descriptors_angular_spinbase<<<grid_size, block_size>>>(
        N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].descriptors.data(),
        nep_data[device_id].sum_fxyz.data());
      kt.end(tok);
    } else {
      find_descriptors_angular_spinbase<<<grid_size, block_size>>>(
        N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].descriptors.data(),
        nep_data[device_id].sum_fxyz.data());
    }
    GPU_CHECK_KERNEL

    const int spin_offset = nep_spin_base_offset(paramb);
    const int spin_dim = nep_spin_total_dim(paramb);
    const bool has_spin = (dataset[device_id].spin.size() == static_cast<size_t>(N) * 3);
    if (has_spin) {
      if (do_profile) {
        int tok = kt.begin("find_descriptors_spin_onsite");
        find_descriptors_spin_onsite<<<grid_size, block_size>>>(
          N,
          paramb,
          dataset[device_id].type.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].descriptors.data());
        kt.end(tok);
      } else {
        find_descriptors_spin_onsite<<<grid_size, block_size>>>(
          N,
          paramb,
          dataset[device_id].type.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].descriptors.data());
      }
      if (do_profile) {
        int tok = kt.begin("find_descriptors_spin_2body");
        find_descriptors_spin_2body<<<grid_size, block_size>>>(
          N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].descriptors.data());
        kt.end(tok);
      } else {
        find_descriptors_spin_2body<<<grid_size, block_size>>>(
          N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].descriptors.data());
      }
      if (do_profile) {
        int tok = kt.begin("find_descriptors_spin_3body");
        find_descriptors_spin_3body<<<grid_size, block_size>>>(
          N,
          nep_data[device_id].NN_angular.data(),
          nep_data[device_id].NL_angular.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_angular.data(),
          nep_data[device_id].y12_angular.data(),
          nep_data[device_id].z12_angular.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].descriptors.data(),
          nep_data[device_id].sum_fxyz_0.data(),
          nep_data[device_id].sum_fxyz_c.data(),
          nep_data[device_id].sum_fxyz_Ax.data(),
          nep_data[device_id].sum_fxyz_Ay.data(),
          nep_data[device_id].sum_fxyz_Az.data(),
          nep_data[device_id].sum_fxyz_D.data());
        kt.end(tok);
      } else {
        find_descriptors_spin_3body<<<grid_size, block_size>>>(
          N,
          nep_data[device_id].NN_angular.data(),
          nep_data[device_id].NL_angular.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_angular.data(),
          nep_data[device_id].y12_angular.data(),
          nep_data[device_id].z12_angular.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].descriptors.data(),
          nep_data[device_id].sum_fxyz_0.data(),
          nep_data[device_id].sum_fxyz_c.data(),
          nep_data[device_id].sum_fxyz_Ax.data(),
          nep_data[device_id].sum_fxyz_Ay.data(),
          nep_data[device_id].sum_fxyz_Az.data(),
          nep_data[device_id].sum_fxyz_D.data());
      }
      GPU_CHECK_KERNEL
    } else {
      zero_spin_descriptor_block<<<grid_size, block_size>>>(
        N, spin_dim, nep_data[device_id].descriptors.data(), spin_offset);
      const int sum_size = static_cast<int>(nep_data[device_id].sum_fxyz_0.size());
      const int grid_sum = (sum_size - 1) / 256 + 1;
      zero_spin_sum_block<<<grid_sum, 256>>>(sum_size, nep_data[device_id].sum_fxyz_0.data());
      zero_spin_sum_block<<<grid_sum, 256>>>(sum_size, nep_data[device_id].sum_fxyz_c.data());
      zero_spin_sum_block<<<grid_sum, 256>>>(sum_size, nep_data[device_id].sum_fxyz_Ax.data());
      zero_spin_sum_block<<<grid_sum, 256>>>(sum_size, nep_data[device_id].sum_fxyz_Ay.data());
      zero_spin_sum_block<<<grid_sum, 256>>>(sum_size, nep_data[device_id].sum_fxyz_Az.data());
      zero_spin_sum_block<<<grid_sum, 256>>>(sum_size, nep_data[device_id].sum_fxyz_D.data());
      GPU_CHECK_KERNEL
    }

    if (para.prediction == 1 && para.output_descriptor >= 1) {
      FILE* fid_descriptor = my_fopen("descriptor.out", "a");
      std::vector<SpinReal> descriptor_cpu(nep_data[device_id].descriptors.size());
      nep_data[device_id].descriptors.copy_to_host(descriptor_cpu.data());
      for (int nc = 0; nc < dataset[device_id].Nc; ++nc) {
        SpinReal q_structure[MAX_DIM] = {0.0};
        for (int na = 0; na < dataset[device_id].Na_cpu[nc]; ++na) {
          const int n = dataset[device_id].Na_sum_cpu[nc] + na;
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            const SpinReal q = descriptor_cpu[n + d * N] * para.q_scaler_cpu[d];
            q_structure[d] += q;
            if (para.output_descriptor == 2) fprintf(fid_descriptor, "%g ", q);
          }
          if (para.output_descriptor == 2) fprintf(fid_descriptor, "\n");
        }
        if (para.output_descriptor == 1) {
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            fprintf(fid_descriptor, "%g ", q_structure[d] / dataset[device_id].Na_cpu[nc]);
          }
          fprintf(fid_descriptor, "\n");
        }
      }
      fclose(fid_descriptor);
    }

    if (calculate_q_scaler) {
      find_max_min<<<annmb[device_id].dim, 1024>>>(
        N,
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data());
      GPU_CHECK_KERNEL
    }

    if (do_profile) {
      int tok = kt.begin("zero_force_spin");
      zero_force_spin<<<grid_size, block_size>>>(
        N,
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + N,
        dataset[device_id].force.data() + N * 2,
        dataset[device_id].virial.data());
      kt.end(tok);
    } else {
      zero_force_spin<<<grid_size, block_size>>>(
        N,
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + N,
        dataset[device_id].force.data() + N * 2,
        dataset[device_id].virial.data());
    }
    GPU_CHECK_KERNEL

    if (do_profile) {
      int tok = kt.begin("apply_ann_spin");
      apply_ann_spin<<<grid_size, block_size>>>(
        N,
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        dataset[device_id].spin_energy.data(),
        nep_data[device_id].Fp.data());
      kt.end(tok);
    } else {
      apply_ann_spin<<<grid_size, block_size>>>(
        N,
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        dataset[device_id].spin_energy.data(),
        nep_data[device_id].Fp.data());
    }
    GPU_CHECK_KERNEL

    if (do_profile) {
      int tok = kt.begin("find_force_radial_spinbase");
      find_force_radial_spinbase<<<grid_size, block_size>>>(
        N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].Fp.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + N,
        dataset[device_id].force.data() + N * 2,
        dataset[device_id].virial.data());
      kt.end(tok);
    } else {
      find_force_radial_spinbase<<<grid_size, block_size>>>(
        N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].Fp.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + N,
        dataset[device_id].force.data() + N * 2,
        dataset[device_id].virial.data());
    }
    if (do_profile) {
      int tok = kt.begin("find_force_angular_spinbase");
      find_force_angular_spinbase<<<grid_size, block_size>>>(
        N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].Fp.data(),
        nep_data[device_id].sum_fxyz.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + N,
        dataset[device_id].force.data() + N * 2,
        dataset[device_id].virial.data());
      kt.end(tok);
    } else {
      find_force_angular_spinbase<<<grid_size, block_size>>>(
        N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].Fp.data(),
        nep_data[device_id].sum_fxyz.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + N,
        dataset[device_id].force.data() + N * 2,
        dataset[device_id].virial.data());
    }
    GPU_CHECK_KERNEL

    const bool need_mforce =
      has_spin &&
      (dataset[device_id].mforce_ref_gpu.size() == static_cast<size_t>(N) * 3);
    if (need_mforce && dataset[device_id].mforce.size() != static_cast<size_t>(N) * 3) {
      dataset[device_id].mforce.resize(N * 3);
    }
    if (need_mforce) {
      if (do_profile) {
        int tok = kt.begin("zero_mforce_spin");
        zero_mforce_spin<<<grid_size, block_size>>>(
          N,
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + N,
          dataset[device_id].mforce.data() + N * 2);
        kt.end(tok);
      } else {
        zero_mforce_spin<<<grid_size, block_size>>>(
          N,
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + N,
          dataset[device_id].mforce.data() + N * 2);
      }
      GPU_CHECK_KERNEL
    }

    if (has_spin && need_mforce) {
      if (do_profile) {
        int tok = kt.begin("find_mforce_spin_onsite");
        find_mforce_spin_onsite<<<grid_size, block_size>>>(
          N,
          paramb,
          dataset[device_id].type.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + N,
          dataset[device_id].mforce.data() + N * 2);
        kt.end(tok);
      } else {
        find_mforce_spin_onsite<<<grid_size, block_size>>>(
          N,
          paramb,
          dataset[device_id].type.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + N,
          dataset[device_id].mforce.data() + N * 2);
      }
      if (do_profile) {
        int tok = kt.begin("find_force_spin_2body");
        find_force_spin_2body<<<grid_size, block_size>>>(
          N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + N,
          dataset[device_id].force.data() + N * 2,
          dataset[device_id].virial.data(),
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + N,
          dataset[device_id].mforce.data() + N * 2);
        kt.end(tok);
      } else {
        find_force_spin_2body<<<grid_size, block_size>>>(
          N,
          nep_data[device_id].NN_radial.data(),
          nep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_radial.data(),
          nep_data[device_id].y12_radial.data(),
          nep_data[device_id].z12_radial.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + N,
          dataset[device_id].force.data() + N * 2,
          dataset[device_id].virial.data(),
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + N,
          dataset[device_id].mforce.data() + N * 2);
      }
      if (do_profile) {
        int tok = kt.begin("find_force_spin_3body");
        find_force_spin_3body<<<grid_size, block_size>>>(
          N,
          nep_data[device_id].NN_angular.data(),
          nep_data[device_id].NL_angular.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_angular.data(),
          nep_data[device_id].y12_angular.data(),
          nep_data[device_id].z12_angular.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          nep_data[device_id].sum_fxyz_0.data(),
          nep_data[device_id].sum_fxyz_c.data(),
          nep_data[device_id].sum_fxyz_Ax.data(),
          nep_data[device_id].sum_fxyz_Ay.data(),
          nep_data[device_id].sum_fxyz_Az.data(),
          nep_data[device_id].sum_fxyz_D.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + N,
          dataset[device_id].force.data() + N * 2,
          dataset[device_id].virial.data(),
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + N,
          dataset[device_id].mforce.data() + N * 2);
        kt.end(tok);
      } else {
        find_force_spin_3body<<<grid_size, block_size>>>(
          N,
          nep_data[device_id].NN_angular.data(),
          nep_data[device_id].NL_angular.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].x12_angular.data(),
          nep_data[device_id].y12_angular.data(),
          nep_data[device_id].z12_angular.data(),
          dataset[device_id].spin.data(),
          nep_data[device_id].Fp.data(),
          nep_data[device_id].sum_fxyz_0.data(),
          nep_data[device_id].sum_fxyz_c.data(),
          nep_data[device_id].sum_fxyz_Ax.data(),
          nep_data[device_id].sum_fxyz_Ay.data(),
          nep_data[device_id].sum_fxyz_Az.data(),
          nep_data[device_id].sum_fxyz_D.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + N,
          dataset[device_id].force.data() + N * 2,
          dataset[device_id].virial.data(),
          dataset[device_id].mforce.data(),
          dataset[device_id].mforce.data() + N,
          dataset[device_id].mforce.data() + N * 2);
      }
      GPU_CHECK_KERNEL
    }

    if (do_profile) {
      kt.flush();
    }
  }

  if (do_profile) {
    KernelTiming merged;
    for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
      merged.merge_from(g_kernel_timing_spin[device_id]);
    }
    printf("[kernel_timing] NEP_Spin::find_force call=%lld devices=%d\n", call_id, device_in_this_iter);
    merged.print_top("NEP_Spin GPU kernels", para.kernel_timing_topk);
    for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
      g_kernel_timing_spin[device_id].reset();
    }
  }
}
