/*
    NEP_Spin: MD-side runtime for spin-enabled NEP on GPU.

    This file implements a spin-capable NEP evaluator for MD, using
    the same descriptor/ANN/force layout as the training-side
    implementation (src/main_nep/nep_spin*.cu) and the production
    MD NEP implementation (src/force/nep*.cu).
*/

#include "force/nep_spin.cuh"

#include "model/box.cuh"
#include "neighbor.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/kernel_timing.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/nep_spin_utilities.cuh"
#include "utilities/read_file.cuh"

#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <type_traits>

namespace {
thread_local KernelTiming* g_nep_spin_kernel_timing = nullptr;
}

#if defined(USE_HIP)
#define NEP_SPIN_LDG(ptr) (*(ptr))
#else
#define NEP_SPIN_LDG(ptr) __ldg(ptr)
#endif

static inline int nep_spin_pick_dim_bucket(int dim)
{
  if (dim <= 64) return 64;
  if (dim <= 96) return 96;
  if (dim <= 128) return 128;
  return MAX_DIM;
}

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

static __global__ void copy_double_to_float(int n, const double* in, float* out)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<float>(in[idx]);
  }
}

// Local periodic table symbols used to map header element names
// to atomic numbers (Z-1 index into COVALENT_RADIUS).
static const char* kElementSymbols[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

static __device__ void apply_mic_small_box(
  const Box& box, const NEP_Spin::ExpandedBox& ebox, double& x12, double& y12, double& z12)
{
  double sx12 = ebox.h[9] * x12 + ebox.h[10] * y12 + ebox.h[11] * z12;
  double sy12 = ebox.h[12] * x12 + ebox.h[13] * y12 + ebox.h[14] * z12;
  double sz12 = ebox.h[15] * x12 + ebox.h[16] * y12 + ebox.h[17] * z12;
  if (box.pbc_x == 1)
    sx12 -= nearbyint(sx12);
  if (box.pbc_y == 1)
    sy12 -= nearbyint(sy12);
  if (box.pbc_z == 1)
    sz12 -= nearbyint(sz12);
  x12 = ebox.h[0] * sx12 + ebox.h[1] * sy12 + ebox.h[2] * sz12;
  y12 = ebox.h[3] * sx12 + ebox.h[4] * sy12 + ebox.h[5] * sz12;
  z12 = ebox.h[6] * sx12 + ebox.h[7] * sy12 + ebox.h[8] * sz12;
}

static bool get_expanded_box_spin(const double rc, const Box& box, NEP_Spin::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;

  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.5 * rc) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.5 * rc) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.5 * rc) {
    is_small_box = true;
  }

  if (is_small_box) {
    if (thickness_x > 10 * rc || thickness_y > 10 * rc || thickness_z > 10 * rc) {
      std::cout << "Error:\n"
                << "    The box has\n"
                << "        a thickness < 2.5 radial cutoffs in a periodic direction.\n"
                << "        and a thickness > 10 radial cutoffs in another direction.\n"
                << "    Please increase the periodic direction(s).\n";
      exit(1);
    }

    ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
    ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
    ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];
    ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
    ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
    ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];
    ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
    ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
    ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

    ebox.h[9] = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
    ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
    ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
    ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
    ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
    ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
    ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
    ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
    ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];
    double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
                 ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
                 ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
    for (int n = 9; n < 18; n++) {
      ebox.h[n] /= det;
    }
  }

  return is_small_box;
}

// ----------------------------------------------------------------------
// Neighbor list: radial + angular (small-box style)
// ----------------------------------------------------------------------
static __global__ void find_neighbor_list_spin_small_box(
  NEP_Spin::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const NEP_Spin::ExpandedBox ebox,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int t1 = g_type[n1];

  int count_radial = 0;
  int count_angular = 0;

  for (int n2 = N1; n2 < N2; ++n2) {
    for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
      for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
        for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
          if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
            continue; // exclude self
          }

          double delta[3];
          delta[0] = box.cpu_h[0] * ia + box.cpu_h[1] * ib + box.cpu_h[2] * ic;
          delta[1] = box.cpu_h[3] * ia + box.cpu_h[4] * ib + box.cpu_h[5] * ic;
          delta[2] = box.cpu_h[6] * ia + box.cpu_h[7] * ib + box.cpu_h[8] * ic;

          double x12 = g_x[n2] + delta[0] - x1;
          double y12 = g_y[n2] + delta[1] - y1;
          double z12 = g_z[n2] + delta[2] - z1;

          apply_mic_small_box(box, ebox, x12, y12, z12);

          float distance_square = static_cast<float>(x12 * x12 + y12 * y12 + z12 * z12);

          int t2 = g_type[n2];
          float rc_radial = paramb.rc_radial;
          float rc_angular = paramb.rc_angular;
          if (paramb.use_typewise_cutoff) {
            int z1 = paramb.atomic_numbers[t1];
            int z2 = paramb.atomic_numbers[t2];
            rc_radial = min(
              (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_radial_factor,
              rc_radial);
            rc_angular = min(
              (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_angular_factor,
              rc_angular);
          }

          if (distance_square < rc_radial * rc_radial) {
            int idx = count_radial * N + n1;
            g_NL_radial[idx] = n2;
            g_x12_radial[idx] = static_cast<float>(x12);
            g_y12_radial[idx] = static_cast<float>(y12);
            g_z12_radial[idx] = static_cast<float>(z12);
            ++count_radial;
          }

          if (distance_square < rc_angular * rc_angular) {
            int idx = count_angular * N + n1;
            g_NL_angular[idx] = n2;
            g_x12_angular[idx] = static_cast<float>(x12);
            g_y12_angular[idx] = static_cast<float>(y12);
            g_z12_angular[idx] = static_cast<float>(z12);
            ++count_angular;
          }
        }
      }
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

// ----------------------------------------------------------------------
// Neighbor list: radial + angular (large-box / binning)
// ----------------------------------------------------------------------
static __global__ void find_neighbor_list_spin_large_box(
  NEP_Spin::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* g_type,
  const int* __restrict__ g_cell_count,
  const int* __restrict__ g_cell_count_sum,
  const int* __restrict__ g_cell_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int t1 = g_type[n1];

  int count_radial = 0;
  int count_angular = 0;

  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(
    box,
    x1,
    y1,
    z1,
    2.0f * paramb.rcinv_radial,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
        if (cell_id_x + xx < 0)
          neighbor_cell += nx;
        if (cell_id_x + xx >= nx)
          neighbor_cell -= nx;
        if (cell_id_y + yy < 0)
          neighbor_cell += ny * nx;
        if (cell_id_y + yy >= ny)
          neighbor_cell -= ny * nx;
        if (cell_id_z + zz < 0)
          neighbor_cell += nz * ny * nx;
        if (cell_id_z + zz >= nz)
          neighbor_cell -= nz * ny * nx;

        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];

          if (n2 < N1 || n2 >= N2 || n1 == n2) {
            continue;
          }

          double x12 = g_x[n2] - x1;
          double y12 = g_y[n2] - y1;
          double z12 = g_z[n2] - z1;
          apply_mic(box, x12, y12, z12);
          float x12f = static_cast<float>(x12);
          float y12f = static_cast<float>(y12);
          float z12f = static_cast<float>(z12);
          float d12_square = x12f * x12f + y12f * y12f + z12f * z12f;

          int t2 = g_type[n2];
          float rc_radial = paramb.rc_radial;
          float rc_angular = paramb.rc_angular;
          if (paramb.use_typewise_cutoff) {
            int z1 = paramb.atomic_numbers[t1];
            int z2 = paramb.atomic_numbers[t2];
            rc_radial = min(
              (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_radial_factor,
              rc_radial);
            rc_angular = min(
              (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_angular_factor,
              rc_angular);
          }

          if (d12_square < rc_radial * rc_radial) {
            int idx = count_radial * N + n1;
            g_NL_radial[idx] = n2;
            if (g_x12_radial != nullptr) {
              g_x12_radial[idx] = x12f;
              g_y12_radial[idx] = y12f;
              g_z12_radial[idx] = z12f;
            }
            ++count_radial;
          }

          if (d12_square < rc_angular * rc_angular) {
            int idx = count_angular * N + n1;
            g_NL_angular[idx] = n2;
            if (g_x12_angular != nullptr) {
              g_x12_angular[idx] = x12f;
              g_y12_angular[idx] = y12f;
              g_z12_angular[idx] = z12f;
            }
            ++count_angular;
          }
        }
      }
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_force_radial_spinbase_large_md(
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  int t1 = g_type[n1];
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;
  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int bs = paramb.basis_size_radial;
  if (bs >= MAX_NUM_N) {
    bs = MAX_NUM_N - 1;
  }

  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    int n2 = g_NL[n1 + N * i1];
    int t2 = g_type[n2];
    double dx = g_x[n2] - x1;
    double dy = g_y[n2] - y1;
    double dz = g_z[n2] - z1;
    apply_mic(box, dx, dy, dz);
    float r12[3] = {static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz)};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float d12inv = 1.0f / d12;
    float f12[3] = {0.0f};
    float f21[3] = {0.0f};

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
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gnp12 = 0.0f;
      float gnp21 = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
        gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];
      }
      float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
      float tmp21 = g_Fp[n2 + n * N] * gnp21 * d12inv;
      for (int d = 0; d < 3; ++d) {
        f12[d] += tmp12 * r12[d];
        f21[d] -= tmp21 * r12[d];
      }
    }

    s_fx += f12[0] - f21[0];
    s_fy += f12[1] - f21[1];
    s_fz += f12[2] - f21[2];
    s_sxx += r12[0] * f21[0];
    s_syy += r12[1] * f21[1];
    s_szz += r12[2] * f21[2];
    s_sxy += r12[0] * f21[1];
    s_sxz += r12[0] * f21[2];
    s_syx += r12[1] * f21[0];
    s_syz += r12[1] * f21[2];
    s_szx += r12[2] * f21[0];
    s_szy += r12[2] * f21[1];
  }

  g_fx[n1] += static_cast<double>(s_fx);
  g_fy[n1] += static_cast<double>(s_fy);
  g_fz[n1] += static_cast<double>(s_fz);
  g_virial[n1 + 0 * N] += static_cast<double>(s_sxx);
  g_virial[n1 + 1 * N] += static_cast<double>(s_syy);
  g_virial[n1 + 2 * N] += static_cast<double>(s_szz);
  g_virial[n1 + 3 * N] += static_cast<double>(s_sxy);
  g_virial[n1 + 4 * N] += static_cast<double>(s_sxz);
  g_virial[n1 + 5 * N] += static_cast<double>(s_syz);
  g_virial[n1 + 6 * N] += static_cast<double>(s_syx);
  g_virial[n1 + 7 * N] += static_cast<double>(s_szx);
  g_virial[n1 + 8 * N] += static_cast<double>(s_szy);
}

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

  g_pe[n1] += static_cast<double>(F);

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
  float* g_Fp)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) {
    return;
  }

  const int t1 = g_type[n1];

  float q[kMaxDim] = {0.0f};

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
    float d12inv = 1.0f / d12;

    float fc12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      rc = min(
        (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
         COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
          paramb.typewise_cutoff_radial_factor,
        rc);
    }
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);

    float fn12[MAX_NUM_N];
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
      float fn12[MAX_NUM_N];
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
      float yref = paramb.spin_mref * paramb.spin_mref;
      if (basis_mode == 2) {
        y = si_norm;
        yref = paramb.spin_mref;
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

  NepSpinApplyAnn<kDoAnn>::run(n1, N, paramb, annmb, g_type, q, g_q_scaler, g_pe, g_Fp);
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
    g_Fp);
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
    g_Fp);
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
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  int t1 = g_type[n1];
  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    int index = i1 * N + n1;
    int n2 = g_NL[index];
    int t2 = g_type[n2];
    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float d12inv = 1.0f / d12;
    float f12[3] = {0.0f, 0.0f, 0.0f};

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
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gnp12 = 0.0f;
      for (int k = 0; k <= bs; ++k) {
        int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2;
        gnp12 += fnp12[k] * annmb.c[c_index];
      }
      float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
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

    atomicAdd(&g_fx[n1], static_cast<double>(f12[0]));
    atomicAdd(&g_fy[n1], static_cast<double>(f12[1]));
    atomicAdd(&g_fz[n1], static_cast<double>(f12[2]));
    atomicAdd(&g_fx[n2], static_cast<double>(-f12[0]));
    atomicAdd(&g_fy[n2], static_cast<double>(-f12[1]));
    atomicAdd(&g_fz[n2], static_cast<double>(-f12[2]));

    // Save virial (9 components per atom, unsymmetrized as in NEP small box)
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
    atomicAdd(&g_virial[n2 + 1 * N], s_syy);
    atomicAdd(&g_virial[n2 + 2 * N], s_szz);
    atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
    atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
    atomicAdd(&g_virial[n2 + 5 * N], s_syz);
    atomicAdd(&g_virial[n2 + 6 * N], s_syx);
    atomicAdd(&g_virial[n2 + 7 * N], s_szx);
    atomicAdd(&g_virial[n2 + 8 * N], s_szy);
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
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  float Fp_loc[MAX_DIM_ANGULAR] = {0.0f};
  for (int d = 0; d < paramb.dim_angular; ++d) {
    Fp_loc[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
  }

  const int sum_stride = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  const float* sum_fxyz_base = g_sum_fxyz + n1;

  int t1 = g_type[n1];

  for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
    int index = i1 * N + n1;
    int n2 = g_NL_angular[n1 + N * i1];
    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float f12[3] = {0.0f, 0.0f, 0.0f};

    float fc12, fcp12;
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
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
        gnp12 += fnp12[k] * annmb.c[c_index];
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

    atomicAdd(&g_fx[n1], static_cast<double>(f12[0]));
    atomicAdd(&g_fy[n1], static_cast<double>(f12[1]));
    atomicAdd(&g_fz[n1], static_cast<double>(f12[2]));
    atomicAdd(&g_fx[n2], static_cast<double>(-f12[0]));
    atomicAdd(&g_fy[n2], static_cast<double>(-f12[1]));
    atomicAdd(&g_fz[n2], static_cast<double>(-f12[2]));

    atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
    atomicAdd(&g_virial[n2 + 1 * N], s_syy);
    atomicAdd(&g_virial[n2 + 2 * N], s_szz);
    atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
    atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
    atomicAdd(&g_virial[n2 + 5 * N], s_syz);
    atomicAdd(&g_virial[n2 + 6 * N], s_syx);
    atomicAdd(&g_virial[n2 + 7 * N], s_szx);
    atomicAdd(&g_virial[n2 + 8 * N], s_szy);
  }
}

static __global__ void find_partial_force_angular_spinbase_large_md(
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const NEP_Spin::ParaMB paramb,
  const NEP_Spin::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  float Fp[MAX_DIM_ANGULAR] = {0.0f};
  for (int d = 0; d < paramb.dim_angular; ++d) {
    Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
  }

  const int sum_stride = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  const float* sum_fxyz_base = g_sum_fxyz + n1;

  int t1 = g_type[n1];
  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];

  for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
    int index = i1 * N + n1;
    int n2 = g_NL_angular[n1 + N * i1];
    double dx = g_x[n2] - x1;
    double dy = g_y[n2] - y1;
    double dz = g_z[n2] - z1;
    apply_mic(box, dx, dy, dz);
    float r12[3] = {static_cast<float>(dx), static_cast<float>(dy), static_cast<float>(dz)};
    float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float f12[3] = {0.0f};

    float fc12, fcp12;
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
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
        c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
        gn12 += fn12[k] * annmb.c[c_index];
        gnp12 += fnp12[k] * annmb.c[c_index];
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
        Fp,
        sum_fxyz_base,
        sum_stride,
        N,
        f12);
    }

    g_f12x[index] = f12[0];
    g_f12y[index] = f12[1];
    g_f12z[index] = f12[2];
  }
}

static __global__ void find_force_many_body_spin_large_md(
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;

  int neighbor_number = g_NN[n1];
  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    int index = i1 * N + n1;
    int n2 = g_NL[index];

    double x12double = g_x[n2] - x1;
    double y12double = g_y[n2] - y1;
    double z12double = g_z[n2] - z1;
    apply_mic(box, x12double, y12double, z12double);
    float x12 = static_cast<float>(x12double);
    float y12 = static_cast<float>(y12double);
    float z12 = static_cast<float>(z12double);

    float f12x = g_f12x[index];
    float f12y = g_f12y[index];
    float f12z = g_f12z[index];

    int l = 0;
    int r = g_NN[n2];
    int m = 0;
    int tmp_value = 0;
    while (l < r) {
      m = (l + r) >> 1;
      tmp_value = g_NL[n2 + N * m];
      if (tmp_value < n1) {
        l = m + 1;
      } else if (tmp_value > n1) {
        r = m - 1;
      } else {
        break;
      }
    }
    int index_rev = ((l + r) >> 1) * N + n2;
    float f21x = g_f12x[index_rev];
    float f21y = g_f12y[index_rev];
    float f21z = g_f12z[index_rev];

    s_fx += f12x - f21x;
    s_fy += f12y - f21y;
    s_fz += f12z - f21z;

    s_sxx += x12 * f21x;
    s_syy += y12 * f21y;
    s_szz += z12 * f21z;
    s_sxy += x12 * f21y;
    s_sxz += x12 * f21z;
    s_syx += y12 * f21x;
    s_syz += y12 * f21z;
    s_szx += z12 * f21x;
    s_szy += z12 * f21y;
  }

  g_fx[n1] += static_cast<double>(s_fx);
  g_fy[n1] += static_cast<double>(s_fy);
  g_fz[n1] += static_cast<double>(s_fz);
  g_virial[n1 + 0 * N] += static_cast<double>(s_sxx);
  g_virial[n1 + 1 * N] += static_cast<double>(s_syy);
  g_virial[n1 + 2 * N] += static_cast<double>(s_szz);
  g_virial[n1 + 3 * N] += static_cast<double>(s_sxy);
  g_virial[n1 + 4 * N] += static_cast<double>(s_sxz);
  g_virial[n1 + 5 * N] += static_cast<double>(s_syz);
  g_virial[n1 + 6 * N] += static_cast<double>(s_syx);
  g_virial[n1 + 7 * N] += static_cast<double>(s_szx);
  g_virial[n1 + 8 * N] += static_cast<double>(s_szy);
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  constexpr int kmax_ex = KMAX_TERM;

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

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
  float s_virial_xy = 0.0f, s_virial_yz = 0.0f, s_virial_zx = 0.0f;

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

    float fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
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
    float fi_total[3] = {0.0f, 0.0f, 0.0f};

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
        const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
        const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
        const float inv = ex_invariant[kk];
        const float tmp_self = Fp1 * gnp12 * d12inv * inv;
        const float tmp_total = (Fp1 * gnp12 + Fp2 * gnp21) * d12inv * inv;
        fi_self[0] += tmp_self * r12[0];
        fi_self[1] += tmp_self * r12[1];
        fi_self[2] += tmp_self * r12[2];
        fi_total[0] += tmp_total * r12[0];
        fi_total[1] += tmp_total * r12[1];
        fi_total[2] += tmp_total * r12[2];
      }
    }

    s_fx += fi_total[0];
    s_fy += fi_total[1];
    s_fz += fi_total[2];

    s_virial_xx -= r12[0] * fi_self[0];
    s_virial_yy -= r12[1] * fi_self[1];
    s_virial_zz -= r12[2] * fi_self[2];
    s_virial_xy -= r12[0] * fi_self[1];
    s_virial_yz -= r12[1] * fi_self[2];
    s_virial_zx -= r12[2] * fi_self[0];
  }

  g_fx[n1] += s_fx;
  g_fy[n1] += s_fy;
  g_fz[n1] += s_fz;

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 6 * N] += s_virial_xy;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 8 * N] += s_virial_yz;
  g_virial[n1 + 4 * N] += s_virial_zx;
  g_virial[n1 + 7 * N] += s_virial_zx;
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_dmi = KMAX_TERM;

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int dmi_block0 = spin_blocks.dmi_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

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
  float s_virial_xy = 0.0f, s_virial_yz = 0.0f, s_virial_zx = 0.0f;

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

    float fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
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
    float fi_total[3] = {0.0f, 0.0f, 0.0f};

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
        const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
        const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
        const float Tk_k = Tk[kk];
        const float inv = dmi * Tk_k;
        const float tmp_self = Fp1 * gnp12 * d12inv * inv;
        const float tmp_total = (Fp1 * gnp12 + Fp2 * gnp21) * d12inv * inv;
        fi_self[0] += tmp_self * r12[0];
        fi_self[1] += tmp_self * r12[1];
        fi_self[2] += tmp_self * r12[2];
        fi_total[0] += tmp_total * r12[0];
        fi_total[1] += tmp_total * r12[1];
        fi_total[2] += tmp_total * r12[2];

        const float coeff_self = Fp1 * gn12 * Tk_k;
        const float coeff_total = (Fp1 * gn12 + Fp2 * gn21) * Tk_k;
        fi_self[0] += coeff_self * Jac_dmi[0];
        fi_self[1] += coeff_self * Jac_dmi[1];
        fi_self[2] += coeff_self * Jac_dmi[2];
        fi_total[0] += coeff_total * Jac_dmi[0];
        fi_total[1] += coeff_total * Jac_dmi[1];
        fi_total[2] += coeff_total * Jac_dmi[2];
      }
    }

    s_fx += fi_total[0];
    s_fy += fi_total[1];
    s_fz += fi_total[2];

    s_virial_xx -= r12[0] * fi_self[0];
    s_virial_yy -= r12[1] * fi_self[1];
    s_virial_zz -= r12[2] * fi_self[2];
    s_virial_xy -= r12[0] * fi_self[1];
    s_virial_yz -= r12[1] * fi_self[2];
    s_virial_zx -= r12[2] * fi_self[0];
  }

  g_fx[n1] += s_fx;
  g_fy[n1] += s_fy;
  g_fz[n1] += s_fz;

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 6 * N] += s_virial_xy;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 8 * N] += s_virial_yz;
  g_virial[n1 + 4 * N] += s_virial_zx;
  g_virial[n1 + 7 * N] += s_virial_zx;
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_ani = KMAX_TERM;

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int ani_block0 = spin_blocks.ani_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

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
  float s_virial_xy = 0.0f, s_virial_yz = 0.0f, s_virial_zx = 0.0f;

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

    float fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
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
    float fi_total[3] = {0.0f, 0.0f, 0.0f};

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
        const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
        const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
        const float Tk_k = Tk[kk];
        const float inv = ani_scalar * Tk_k;
        const float tmp_self = Fp1 * gnp12 * d12inv * inv;
        const float tmp_total = (Fp1 * gnp12 + Fp2 * gnp21) * d12inv * inv;
        fi_self[0] += tmp_self * r12[0];
        fi_self[1] += tmp_self * r12[1];
        fi_self[2] += tmp_self * r12[2];
        fi_total[0] += tmp_total * r12[0];
        fi_total[1] += tmp_total * r12[1];
        fi_total[2] += tmp_total * r12[2];

        const float coeff_self = Fp1 * gn12 * Tk_k;
        const float coeff_total = (Fp1 * gn12 + Fp2 * gn21) * Tk_k;
        fi_self[0] += coeff_self * Jac_ani[0];
        fi_self[1] += coeff_self * Jac_ani[1];
        fi_self[2] += coeff_self * Jac_ani[2];
        fi_total[0] += coeff_total * Jac_ani[0];
        fi_total[1] += coeff_total * Jac_ani[1];
        fi_total[2] += coeff_total * Jac_ani[2];
      }
    }

    s_fx += fi_total[0];
    s_fy += fi_total[1];
    s_fz += fi_total[2];

    s_virial_xx -= r12[0] * fi_self[0];
    s_virial_yy -= r12[1] * fi_self[1];
    s_virial_zz -= r12[2] * fi_self[2];
    s_virial_xy -= r12[0] * fi_self[1];
    s_virial_yz -= r12[1] * fi_self[2];
    s_virial_zx -= r12[2] * fi_self[0];
  }

  g_fx[n1] += s_fx;
  g_fy[n1] += s_fy;
  g_fz[n1] += s_fz;

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 6 * N] += s_virial_xy;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 8 * N] += s_virial_yz;
  g_virial[n1 + 4 * N] += s_virial_zx;
  g_virial[n1 + 7 * N] += s_virial_zx;
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_sia = KMAX_TERM;

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int sia_block0 = spin_blocks.sia_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

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
  float s_virial_xy = 0.0f, s_virial_yz = 0.0f, s_virial_zx = 0.0f;

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

    float fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
    find_fn_and_fnp(bs, rcinv, d12, fc12, fcp12, fn12, fnp12);

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);

    const float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const float sia_scalar_i = si_r * si_r;

    float sj_r = 0.0f;
    if (neighbor_has_spin) {
      sj_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
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
    float Jac_sia_j[3] = {0.0f, 0.0f, 0.0f};
    J_apply(si, Jac_sia_i, d12inv, rhat);
    if (neighbor_has_spin) {
      J_apply(sj, Jac_sia_j, d12inv, rhat);
    }

    float fi_self[3] = {0.0f, 0.0f, 0.0f};
    float fi_total[3] = {0.0f, 0.0f, 0.0f};

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
        const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
        const float Tk_k = Tk[kk];
        const float inv_i = sia_scalar_i * Tk_k;
        const float tmp_self = Fp1 * gnp12 * d12inv * inv_i;
        fi_self[0] += tmp_self * r12[0];
        fi_self[1] += tmp_self * r12[1];
        fi_self[2] += tmp_self * r12[2];
        fi_total[0] += tmp_self * r12[0];
        fi_total[1] += tmp_self * r12[1];
        fi_total[2] += tmp_self * r12[2];

        const float coeff_self = 2.0f * Fp1 * gn12 * Tk_k * si_r;
        fi_self[0] += coeff_self * Jac_sia_i[0];
        fi_self[1] += coeff_self * Jac_sia_i[1];
        fi_self[2] += coeff_self * Jac_sia_i[2];
        fi_total[0] += coeff_self * Jac_sia_i[0];
        fi_total[1] += coeff_self * Jac_sia_i[1];
        fi_total[2] += coeff_self * Jac_sia_i[2];

        if (neighbor_has_spin) {
          const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
          const float inv_j = (sj_r * sj_r) * Tk_k;
          const float tmp_neigh = Fp2 * gnp21 * d12inv * inv_j;
          fi_total[0] += tmp_neigh * r12[0];
          fi_total[1] += tmp_neigh * r12[1];
          fi_total[2] += tmp_neigh * r12[2];

          const float coeff_neigh = 2.0f * Fp2 * gn21 * Tk_k * sj_r;
          fi_total[0] += coeff_neigh * Jac_sia_j[0];
          fi_total[1] += coeff_neigh * Jac_sia_j[1];
          fi_total[2] += coeff_neigh * Jac_sia_j[2];
        }
      }
    }

    s_fx += fi_total[0];
    s_fy += fi_total[1];
    s_fz += fi_total[2];

    s_virial_xx -= r12[0] * fi_self[0];
    s_virial_yy -= r12[1] * fi_self[1];
    s_virial_zz -= r12[2] * fi_self[2];
    s_virial_xy -= r12[0] * fi_self[1];
    s_virial_yz -= r12[1] * fi_self[2];
    s_virial_zx -= r12[2] * fi_self[0];
  }

  g_fx[n1] += s_fx;
  g_fy[n1] += s_fy;
  g_fz[n1] += s_fz;

  g_virial[n1 + 0 * N] += s_virial_xx;
  g_virial[n1 + 1 * N] += s_virial_yy;
  g_virial[n1 + 2 * N] += s_virial_zz;
  g_virial[n1 + 3 * N] += s_virial_xy;
  g_virial[n1 + 6 * N] += s_virial_xy;
  g_virial[n1 + 5 * N] += s_virial_yz;
  g_virial[n1 + 8 * N] += s_virial_yz;
  g_virial[n1 + 4 * N] += s_virial_zx;
  g_virial[n1 + 7 * N] += s_virial_zx;
}

// Split kernels (MD/x12, no neighbor atomics): EX/DMI/ANI/SIA blocks.
// These avoid large-box MIC recomputation by consuming precomputed neighbor vectors (x12/y12/z12).
// On-site longitudinal terms only (p=1..spin_pmax).
static __global__ void find_mforce_onsite_spin_spherical_md(
  const int N,
  const int N1,
  const int N2,
  const NEP_Spin::ParaMB paramb,
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
    float yref = paramb.spin_mref * paramb.spin_mref;
    if (basis_mode == 2) {
      y = si_norm;
      yref = paramb.spin_mref;
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  constexpr int kmax_ex = KMAX_TERM;

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);
  float inv_si_norm = 1.0f / (si_norm + 1.0e-12f);

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
    float fn12[MAX_NUM_N];
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
        const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
        const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
        const float coeff = msign * (Fp1 * gn12 + Fp2 * gn21);
        const float grad_i[3] = {
          dphi_dsi[0] * Tk[kk] + phi * dTk[kk] * dc_dsi[0],
          dphi_dsi[1] * Tk[kk] + phi * dTk[kk] * dc_dsi[1],
          dphi_dsi[2] * Tk[kk] + phi * dTk[kk] * dc_dsi[2]};
        mfx_i += coeff * grad_i[0];
        mfy_i += coeff * grad_i[1];
        mfz_i += coeff * grad_i[2];
      }
    }
  }

  atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
  atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
  atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_dmi = KMAX_TERM;

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int dmi_block0 = spin_blocks.dmi_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

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
    float fn12[MAX_NUM_N];
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

    const float* c_base_pair12 = c_base_t1 + t2;
    const float* c_base_pair21 = c_base_init + t2 * paramb.num_types + t1;

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
        const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
        const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
        const float coeff = msign * (Fp1 * gn12 + Fp2 * gn21);
        const float grad_i[3] = {
          sjxr[0] * Tk[kk] + dmi * dTk[kk] * dc_dsi[0],
          sjxr[1] * Tk[kk] + dmi * dTk[kk] * dc_dsi[1],
          sjxr[2] * Tk[kk] + dmi * dTk[kk] * dc_dsi[2]};
        mfx_i += coeff * grad_i[0];
        mfy_i += coeff * grad_i[1];
        mfz_i += coeff * grad_i[2];
      }
    }
  }

  atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
  atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
  atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_ani = KMAX_TERM;

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int ani_block0 = spin_blocks.ani_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

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
    float fn12[MAX_NUM_N];
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
        const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
        const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
        const float coeff = msign * (Fp1 * gn12 + Fp2 * gn21);
        const float grad_i[3] = {
          (sj_r * rhat[0]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[0],
          (sj_r * rhat[1]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[1],
          (sj_r * rhat[2]) * Tk[kk] + ani_scalar * dTk[kk] * dc_dsi[2]};
        mfx_i += coeff * grad_i[0];
        mfy_i += coeff * grad_i[1];
        mfz_i += coeff * grad_i[2];
      }
    }
  }

  atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
  atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
  atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
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
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 >= N2) return;

  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb);
  constexpr int kmax_sia = KMAX_TERM;

  const int neighbor_number = g_NN[n1];
  if (neighbor_number <= 0) return;

  const int t1 = g_type[n1];
  const int nspin = nep_spin_nspin(paramb);
  const int sia_block0 = spin_blocks.sia_block0;

  float si[3] = {g_spin[n1], g_spin[n1 + N], g_spin[n1 + N * 2]};
  float si2 = si[0] * si[0] + si[1] * si[1] + si[2] * si[2];
  if (si2 <= kSpinZeroEpsSph) return;
  float si_norm = sqrtf(si2);

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
    float fn12[MAX_NUM_N];
    find_fn(bs, rcinv, d12, fc12, fn12);

    const float si_r = si[0] * rhat[0] + si[1] * rhat[1] + si[2] * rhat[2];
    const float sia_scalar = si_r * si_r;

    float sj[3] = {g_spin[n2], g_spin[n2 + N], g_spin[n2 + N * 2]};
    float sj2 = sj[0] * sj[0] + sj[1] * sj[1] + sj[2] * sj[2];
    const bool neighbor_has_spin = (sj2 > kSpinZeroEpsSph);

    float sj_r = 0.0f;
    float sj_r2 = 0.0f;
    if (neighbor_has_spin) {
      sj_r = sj[0] * rhat[0] + sj[1] * rhat[1] + sj[2] * rhat[2];
      sj_r2 = sj_r * sj_r;
    }

    float Tk[KMAX_TERM + 1] = {0.0f};
    float dTk[KMAX_TERM + 1] = {0.0f};
    Tk[0] = 1.0f;
    dTk[0] = 0.0f;
    float dc_dsi[3] = {0.0f, 0.0f, 0.0f};
    if (neighbor_has_spin) {
      const float sdot = nep_spin_dot3(si, sj);
      const float sj_norm = sqrtf(sj2);
      const float denom = si_norm * sj_norm;
      const float c = nep_spin_clamp_unit(sdot / (denom + 1.0e-12f));
      nep_spin_fill_Tk_and_dTk<KMAX_TERM>(c, kmax_sia, Tk, dTk);
      float dc_dsj[3] = {0.0f, 0.0f, 0.0f};
      nep_spin_fill_dc_dsi_dsj(si, sj, sdot, si_norm, sj_norm, dc_dsi, dc_dsj);
    }

    const float* c_base_pair12 = c_base_t1 + t2;
    const float* c_base_pair21 = c_base_init + t2 * paramb.num_types + t1;
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

        const int fp_idx = block * nspin + n;
        const float Fp1 = NEP_SPIN_LDG(&g_Fp[n1 + (spin_offset + fp_idx) * N]);
        const float coeff1 = msign * Fp1 * gn12;
        const float grad_i[3] = {
          (2.0f * si_r * rhat[0]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[0],
          (2.0f * si_r * rhat[1]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[1],
          (2.0f * si_r * rhat[2]) * Tk[kk] + sia_scalar * dTk[kk] * dc_dsi[2]};
        mfx_i += coeff1 * grad_i[0];
        mfy_i += coeff1 * grad_i[1];
        mfz_i += coeff1 * grad_i[2];

        if (neighbor_has_spin && kk > 0) {
          if (same_type) gn21 = gn12;
          const float Fp2 = NEP_SPIN_LDG(&g_Fp[n2 + (spin_offset + fp_idx) * N]);
          const float coeff2 = msign * Fp2 * gn21;
          const float grad_from_neighbor[3] = {
            sj_r2 * dTk[kk] * dc_dsi[0],
            sj_r2 * dTk[kk] * dc_dsi[1],
            sj_r2 * dTk[kk] * dc_dsi[2]};
          mfx_i += coeff2 * grad_from_neighbor[0];
          mfy_i += coeff2 * grad_from_neighbor[1];
          mfz_i += coeff2 * grad_from_neighbor[2];
        }
      }
    }
  }

  atomicAdd(&g_mx[n1], static_cast<double>(mfx_i));
  atomicAdd(&g_my[n1], static_cast<double>(mfy_i));
  atomicAdd(&g_mz[n1], static_cast<double>(mfz_i));
}

// Split kernels (NEP-style, no neighbor atomics): per-interaction mforce.
// Spin spherical magnetic forces (spin derivatives) - large-box MD variant, NEP-style (no neighbor atomics).
NEP_Spin::NEP_Spin(const char* file_potential, int max_atoms)
  : spin_mode_(0)
  , max_atoms_(max_atoms > 0 ? max_atoms : 0)
  , current_natoms_(0)
{
  read_potential_file(file_potential);
  // set Potential base metadata
  this->ilp_flag = 0;
  this->nep_model_type = -1;
  this->rc = paramb_.rc_radial;
  this->N1 = 0;
  this->N2 = max_atoms_;
}

NEP_Spin::~NEP_Spin() = default;

void NEP_Spin::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  PRINT_INPUT_ERROR("NEP_Spin requires spin degrees of freedom; use compute_with_spin instead.\n");
}

void NEP_Spin::read_potential_file(const char* file_potential)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  // 1st line: nep[3/4]_spin  num_types  elem...
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep*_spin must have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] != "nep3_spin" && tokens[0] != "nep4_spin") {
    std::cout << "Unsupported spin NEP tag: " << tokens[0] << std::endl;
    exit(1);
  }
  paramb_.version = (tokens[0] == "nep3_spin") ? 3 : 4;
  paramb_.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (paramb_.num_types <= 0 || paramb_.num_types > NUM_ELEMENTS) {
    PRINT_INPUT_ERROR("Invalid num_types in nep*_spin header.\n");
  }

  paramb_.num_types_sq = paramb_.num_types * paramb_.num_types;

  // Map element symbols to atomic numbers (Z-1 index into COVALENT_RADIUS)
  for (int n = 0; n < paramb_.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == kElementSymbols[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    paramb_.atomic_numbers[n] = atomic_number - 1;
  }

  // 2nd line: spin_mode <mode> [spin_header_lines]
  tokens = get_tokens(input);
  if ((tokens.size() != 2 && tokens.size() != 3) || tokens[0] != "spin_mode") {
    PRINT_INPUT_ERROR("Second line of nep*_spin must be 'spin_mode <mode> [spin_header_lines]'.\n");
  }
  spin_mode_ = get_int_from_token(tokens[1], __FILE__, __LINE__);
  int spin_header_lines = 1; // backward compatible default
  if (tokens.size() == 3) {
    spin_header_lines = get_int_from_token(tokens[2], __FILE__, __LINE__);
    if (spin_header_lines != 1 && spin_header_lines != 2) {
      PRINT_INPUT_ERROR("spin_header_lines must be 1 or 2.\n");
    }
  }

  // 3rd line: spin_feature <kmax_ex> <kmax_dmi> <kmax_ani> <kmax_sia> [pmax] [ex_phi_mode] [onsite_basis_mode] [mref]
  tokens = get_tokens(input);
  if (tokens.size() < 5 || tokens.size() > 9 || tokens[0] != "spin_feature") {
    PRINT_INPUT_ERROR(
      "Third line of nep*_spin must be 'spin_feature <kmax_ex> <kmax_dmi> <kmax_ani> <kmax_sia> [pmax] [ex_phi_mode] [onsite_basis_mode] [mref]'.\n");
  }

  paramb_.spin_kmax_ex = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.spin_kmax_dmi = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb_.spin_kmax_ani = get_int_from_token(tokens[3], __FILE__, __LINE__);
  paramb_.spin_kmax_sia = get_int_from_token(tokens[4], __FILE__, __LINE__);

  auto check_kmax = [&](int kmax, const char* name) {
    if (kmax < -1 || kmax > 8) {
      std::string msg = std::string("Invalid ") + name + " in spin_feature (must be in [-1,8]).\n";
      PRINT_INPUT_ERROR(msg.c_str());
    }
  };
  check_kmax(paramb_.spin_kmax_ex, "kmax_ex");
  check_kmax(paramb_.spin_kmax_dmi, "kmax_dmi");
  check_kmax(paramb_.spin_kmax_ani, "kmax_ani");
  check_kmax(paramb_.spin_kmax_sia, "kmax_sia");

  if (tokens.size() >= 6) {
    paramb_.spin_pmax = get_int_from_token(tokens[5], __FILE__, __LINE__);
  } else {
    paramb_.spin_pmax = 0;
  }
  if (paramb_.spin_pmax < 0 || paramb_.spin_pmax > 8) {
    PRINT_INPUT_ERROR("spin_feature pmax must be in [0,8].\n");
  }

  if (tokens.size() >= 7) {
    paramb_.spin_ex_phi_mode = get_int_from_token(tokens[6], __FILE__, __LINE__);
  } else {
    paramb_.spin_ex_phi_mode = 0;
  }
  if (paramb_.spin_ex_phi_mode < 0 || paramb_.spin_ex_phi_mode > 3) {
    PRINT_INPUT_ERROR("spin_feature ex_phi_mode must be in [0,3].\n");
  }

  if (tokens.size() >= 8) {
    paramb_.spin_onsite_basis_mode = get_int_from_token(tokens[7], __FILE__, __LINE__);
  } else {
    paramb_.spin_onsite_basis_mode = 0;
  }
  if (paramb_.spin_onsite_basis_mode < 0 || paramb_.spin_onsite_basis_mode > 2) {
    PRINT_INPUT_ERROR("spin_feature onsite_basis_mode must be in [0,2].\n");
  }

  if (tokens.size() >= 9) {
    double mref = get_double_from_token(tokens[8], __FILE__, __LINE__);
    paramb_.spin_mref = static_cast<float>(mref);
    if (!(paramb_.spin_mref > 0.0f)) {
      PRINT_INPUT_ERROR("spin_feature mref must be > 0.\n");
    }
  } else {
    paramb_.spin_mref = 1.0f;
  }

  paramb_.spin_n_max = -1; // default to n_max_radial after that line is read (or from spin_n_max line)

  paramb_.spin_blocks =
    nep_spin_blocks_from_kmax(paramb_.spin_kmax_ex) +
    nep_spin_blocks_from_kmax(paramb_.spin_kmax_dmi) +
    nep_spin_blocks_from_kmax(paramb_.spin_kmax_ani) +
    nep_spin_blocks_from_kmax(paramb_.spin_kmax_sia);

  if (spin_mode_ > 0 && paramb_.spin_blocks == 0 && paramb_.spin_pmax == 0) {
    PRINT_INPUT_ERROR("spin_mode>0 requires at least one enabled spin block (kmax>=0) or pmax>0.\n");
  }

  // Optional new-format line: spin_n_max <spin_n_max>
  // If spin_header_lines==2, this line must be present. We also accept it even if
  // spin_header_lines is omitted for robustness.
  tokens = get_tokens(input);
  if (!tokens.empty() && tokens[0] == "spin_n_max") {
    if (tokens.size() != 2) {
      PRINT_INPUT_ERROR("spin_n_max line must be 'spin_n_max <spin_n_max>'.\n");
    }
    int spin_n_max_from_line = get_int_from_token(tokens[1], __FILE__, __LINE__);
    if (spin_n_max_from_line < 0 || spin_n_max_from_line > 12) {
      PRINT_INPUT_ERROR("spin_n_max must be in [0,12].\n");
    }
    if (paramb_.spin_n_max >= 0 && paramb_.spin_n_max != spin_n_max_from_line) {
      PRINT_INPUT_ERROR("spin_n_max conflicts with the value provided in spin_feature.\n");
    }
    paramb_.spin_n_max = spin_n_max_from_line;
    // Next line should be cutoff
    tokens = get_tokens(input);
  } else {
    if (spin_header_lines == 2) {
      PRINT_INPUT_ERROR("spin_mode requests 2 spin header lines, but spin_n_max line is missing.\n");
    }
    // tokens already holds the cutoff line
  }

  // cutoff rc_radial rc_angular MN_radial MN_angular [radial_factor angular_factor zbl_factor]
  if (tokens.size() != 5 && tokens.size() != 8) {
    std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular "
                 "[radial_factor] [angular_factor] [zbl_factor].\n";
    exit(1);
  }
  if (tokens[0] != "cutoff") {
    PRINT_INPUT_ERROR("Expected 'cutoff' line in nep*_spin header.\n");
  }

  paramb_.rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);

  int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);

  // Use the same 1.25 enlargement factor as NEP MD.
  nep_data_.cap_radial_per_atom = static_cast<int>(std::ceil(MN_radial * 1.25f));
  nep_data_.cap_angular_per_atom = static_cast<int>(std::ceil(MN_angular * 1.25f));

  if (tokens.size() == 8) {
    paramb_.typewise_cutoff_radial_factor =
      get_double_from_token(tokens[5], __FILE__, __LINE__);
    paramb_.typewise_cutoff_angular_factor =
      get_double_from_token(tokens[6], __FILE__, __LINE__);
    paramb_.typewise_cutoff_zbl_factor =
      get_double_from_token(tokens[7], __FILE__, __LINE__);
    paramb_.use_typewise_cutoff = (paramb_.typewise_cutoff_radial_factor > 0.0f);
    paramb_.use_typewise_cutoff_zbl = (paramb_.typewise_cutoff_zbl_factor > 0.0f);
  }

  // n_max n_max_radial n_max_angular
  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "n_max") {
    PRINT_INPUT_ERROR("This line should be n_max n_max_radial n_max_angular.\n");
  }
  paramb_.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  if (paramb_.spin_n_max < 0) {
    paramb_.spin_n_max = paramb_.n_max_radial;
  }

  // basis_size basis_size_radial basis_size_angular
  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "basis_size") {
    PRINT_INPUT_ERROR("This line should be basis_size basis_size_radial basis_size_angular.\n");
  }
  paramb_.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb_.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // l_max L_max_3body L_max_4body L_max_5body
  tokens = get_tokens(input);
  if (tokens.size() != 4 || tokens[0] != "l_max") {
    PRINT_INPUT_ERROR("This line should be l_max L_max_3body L_max_4body L_max_5body.\n");
  }
  paramb_.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);

  paramb_.num_L = paramb_.L_max;
  if (L_max_4body == 2) {
    paramb_.num_L += 1;
  }
  if (L_max_5body == 1) {
    paramb_.num_L += 1;
  }
  paramb_.dim_angular = (paramb_.n_max_angular + 1) * paramb_.num_L;

  // ANN <num_neurons1> 0
  tokens = get_tokens(input);
  if (tokens.size() != 3 || tokens[0] != "ANN") {
    PRINT_INPUT_ERROR("This line should be ANN num_neurons1 0.\n");
  }
  annmb_.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);

  int dim_radial = paramb_.n_max_radial + 1;
  int dim_angular = paramb_.dim_angular;

  int dim_spin = 0;
  if (spin_mode_ > 0) {
    int nspin = paramb_.spin_n_max + 1;
    int pmax = paramb_.spin_pmax;
    if (pmax < 0) pmax = 0;
    if (pmax > 8) pmax = 8;
    dim_spin = nspin * paramb_.spin_blocks + pmax;
  }
  annmb_.dim = dim_radial + dim_angular + dim_spin;

  if (paramb_.basis_size_radial + 1 > MAX_NUM_N) {
    PRINT_INPUT_ERROR("basis_size_radial is too large for compiled MAX_NUM_N.\n");
  }
  if (paramb_.basis_size_angular + 1 > MAX_NUM_N) {
    PRINT_INPUT_ERROR("basis_size_angular is too large for compiled MAX_NUM_N.\n");
  }
  if (paramb_.n_max_radial + 1 > MAX_NUM_N) {
    PRINT_INPUT_ERROR("n_max_radial is too large for compiled MAX_NUM_N.\n");
  }
  if (spin_mode_ > 0 && (paramb_.spin_n_max + 1 > MAX_NUM_N)) {
    PRINT_INPUT_ERROR("spin_n_max is too large for compiled MAX_NUM_N.\n");
  }
  if (annmb_.dim > MAX_DIM) {
    PRINT_INPUT_ERROR("total number of descriptor components (dim) exceeds compiled MAX_DIM.\n");
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
  int num_c_angular =
    paramb_.num_types_sq * (paramb_.n_max_angular + 1) * (paramb_.basis_size_angular + 1);

  int nspin = paramb_.spin_n_max + 1;
  paramb_.c_spin_block_stride =
    paramb_.num_types_sq * nspin * (paramb_.basis_size_radial + 1);

  if (spin_mode_ == 2) {
    paramb_.num_c_spin = paramb_.c_spin_block_stride;
    paramb_.c_spin_offset = paramb_.num_c_radial + num_c_angular;
  } else if (spin_mode_ == 3) {
    paramb_.num_c_spin = paramb_.c_spin_block_stride * paramb_.spin_blocks;
    paramb_.c_spin_offset = paramb_.num_c_radial + num_c_angular;
  } else {
    paramb_.num_c_spin = 0;
    paramb_.c_spin_offset = 0;
  }

  int num_para_descriptor = paramb_.num_c_radial + num_c_angular + paramb_.num_c_spin;
  annmb_.num_para = annmb_.num_para_ann + num_para_descriptor;

  paramb_.rcinv_radial = 1.0f / paramb_.rc_radial;
  paramb_.rcinv_angular = 1.0f / paramb_.rc_angular;

  // Read ANN + descriptor parameters
  std::vector<float> parameters(annmb_.num_para);
  for (int n = 0; n < annmb_.num_para; ++n) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected EOF while reading nep*_spin parameters.\n");
    }
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_data_.parameters.resize(annmb_.num_para);
  nep_data_.parameters.copy_from_host(parameters.data());
  update_potential(nep_data_.parameters.data(), annmb_);

  // Read q_scaler
  nep_data_.q_scaler.resize(annmb_.dim);
  std::vector<float> q_scaler_host(annmb_.dim);
  for (int d = 0; d < annmb_.dim; ++d) {
    tokens = get_tokens(input);
    if (tokens.empty()) {
      PRINT_INPUT_ERROR("Unexpected EOF while reading nep*_spin q_scaler.\n");
    }
    q_scaler_host[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_data_.q_scaler.copy_from_host(q_scaler_host.data());

  input.close();

  // Optional parameter dump for debugging (removed debug output).
}

void NEP_Spin::update_potential(float* parameters, ANN& ann)
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

void NEP_Spin::ensure_capacity(int natoms)
{
  if (natoms == current_natoms_) {
    return;
  }

  // Align behavior with non-spin NEP: grow buffers on demand.
  current_natoms_ = natoms;
  max_atoms_ = natoms;
  this->N2 = natoms;
  const int N = current_natoms_;

  const int cap_r = nep_data_.cap_radial_per_atom;
  const int cap_a = nep_data_.cap_angular_per_atom;

  nep_data_.NN_radial.resize(N);
  nep_data_.NN_angular.resize(N);
  nep_data_.NL_radial.resize(static_cast<size_t>(N) * cap_r);
  nep_data_.NL_angular.resize(static_cast<size_t>(N) * cap_a);

  nep_data_.x12_radial.resize(static_cast<size_t>(N) * cap_r);
  nep_data_.y12_radial.resize(static_cast<size_t>(N) * cap_r);
  nep_data_.z12_radial.resize(static_cast<size_t>(N) * cap_r);
  nep_data_.x12_angular.resize(static_cast<size_t>(N) * cap_a);
  nep_data_.y12_angular.resize(static_cast<size_t>(N) * cap_a);
  nep_data_.z12_angular.resize(static_cast<size_t>(N) * cap_a);

  nep_data_.descriptors.resize(static_cast<size_t>(N) * annmb_.dim);
  nep_data_.Fp.resize(static_cast<size_t>(N) * annmb_.dim);

  size_t sum_fxyz_size = static_cast<size_t>(N) *
                         (paramb_.n_max_angular + 1) *
                         ((paramb_.L_max + 1) * (paramb_.L_max + 1) - 1);
  nep_data_.sum_fxyz.resize(sum_fxyz_size);

  size_t neighbor_storage = static_cast<size_t>(N) * cap_a;
  nep_data_.f12x.resize(neighbor_storage);
  nep_data_.f12y.resize(neighbor_storage);
  nep_data_.f12z.resize(neighbor_storage);
}

void NEP_Spin::compute_with_spin(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  GPU_Vector<double>& mforce)
{
  const int N = static_cast<int>(position.size() / 3);
  if (N <= 0) {
    current_natoms_ = 0;
    return;
  }

  if (static_cast<int>(type.size()) < N) {
    PRINT_INPUT_ERROR("NEP_Spin::compute_with_spin: type.size() < natoms.\n");
  }
  if (static_cast<int>(spin.size()) < 3 * N) {
    PRINT_INPUT_ERROR("NEP_Spin::compute_with_spin: spin.size() < 3*natoms.\n");
  }

  ensure_capacity(N);

  const int spin_size = 3 * N;
  if (nep_data_.spin_buffer.size() < static_cast<size_t>(spin_size)) {
    nep_data_.spin_buffer.resize(spin_size);
  }

  {
    KernelTiming* local_kt = g_nep_spin_kernel_timing;
    const int tok = local_kt ? local_kt->begin("copy_double_to_float(spin)") : -1;
    copy_double_to_float<<<(spin_size - 1) / 128 + 1, 128>>>(
      spin_size, spin.data(), nep_data_.spin_buffer.data());
    GPU_CHECK_KERNEL
    if (local_kt) local_kt->end(tok);
  }
  if (mforce.size() < static_cast<size_t>(spin_size)) {
    mforce.resize(static_cast<size_t>(spin_size));
  }

  const bool is_small_box = get_expanded_box_spin(paramb_.rc_radial, box, ebox_);
  if (is_small_box) {
    compute_small_box(box, type, position, nep_data_.spin_buffer, potential, force, virial, mforce);
  } else {
    compute_large_box(box, type, position, nep_data_.spin_buffer, potential, force, virial, mforce);
  }

  current_natoms_ = N;
}


void NEP_Spin::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<float>& spin,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  GPU_Vector<double>& mforce)
{
  KernelTiming* kt = g_nep_spin_kernel_timing;

  const int N = static_cast<int>(type.size());
  const int N1 = 0;
  const int N2 = N;
  const int BLOCK_SIZE = 64;
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;
  const int grid_size_full = (N - 1) / BLOCK_SIZE + 1;

  const int* g_type = type.data();
  const double* g_x = position.data();
  const double* g_y = position.data() + N;
  const double* g_z = position.data() + 2 * N;

  int* g_NN_r = nep_data_.NN_radial.data();
  int* g_NL_r = nep_data_.NL_radial.data();
  int* g_NN_a = nep_data_.NN_angular.data();
  int* g_NL_a = nep_data_.NL_angular.data();

  float* g_x12_r_all = nep_data_.x12_radial.data();
  float* g_y12_r_all = nep_data_.y12_radial.data();
  float* g_z12_r_all = nep_data_.z12_radial.data();
  float* g_x12_a_all = nep_data_.x12_angular.data();
  float* g_y12_a_all = nep_data_.y12_angular.data();
  float* g_z12_a_all = nep_data_.z12_angular.data();

  const double rc_cell_list = 0.5 * paramb_.rc_radial;
  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);
  const int num_bins_total = num_bins[0] * num_bins[1] * num_bins[2];

  if (nep_data_.cell_count.size() < static_cast<size_t>(num_bins_total)) {
    nep_data_.cell_count.resize(num_bins_total);
    nep_data_.cell_count_sum.resize(num_bins_total);
  }
  if (nep_data_.cell_contents.size() < static_cast<size_t>(N)) {
    nep_data_.cell_contents.resize(N);
  }

  {
    const int tok = kt ? kt->begin("find_cell_list") : -1;
    find_cell_list(
      rc_cell_list,
      num_bins,
      box,
      position,
      nep_data_.cell_count,
      nep_data_.cell_count_sum,
      nep_data_.cell_contents);
    if (kt) kt->end(tok);
  }

  {
    const int tok = kt ? kt->begin("find_neighbor_list_spin_large_box") : -1;
    find_neighbor_list_spin_large_box<<<grid_size, BLOCK_SIZE>>>(
      paramb_,
      N,
      N1,
      N2,
      num_bins[0],
      num_bins[1],
      num_bins[2],
      box,
      g_type,
      nep_data_.cell_count.data(),
      nep_data_.cell_count_sum.data(),
      nep_data_.cell_contents.data(),
      g_x,
      g_y,
      g_z,
      g_NN_r,
      g_NL_r,
      g_NN_a,
      g_NL_a,
      g_x12_r_all,
      g_y12_r_all,
      g_z12_r_all,
      g_x12_a_all,
      g_y12_a_all,
      g_z12_a_all);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  const int cap_r = nep_data_.cap_radial_per_atom;
  const int cap_a = nep_data_.cap_angular_per_atom;
  {
    const int tok = kt ? kt->begin("gpu_sort_neighbor_list(radial)") : -1;
    gpu_sort_neighbor_list_with_vectors<<<N, cap_r, cap_r * sizeof(int)>>>(
      N, g_NN_r, g_NL_r, g_x12_r_all, g_y12_r_all, g_z12_r_all);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }
  {
    const int tok = kt ? kt->begin("gpu_sort_neighbor_list(angular)") : -1;
    gpu_sort_neighbor_list_with_vectors<<<N, cap_a, cap_a * sizeof(int)>>>(
      N, g_NN_a, g_NL_a, g_x12_a_all, g_y12_a_all, g_z12_a_all);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  const float* g_spin = spin.data();
  float* g_sum_fxyz = nep_data_.sum_fxyz.data();
  double* g_pe = potential.data();
  float* g_Fp = nep_data_.Fp.data();
  const float* g_q_scaler = nep_data_.q_scaler.data();
  const int dim_bucket = nep_spin_pick_dim_bucket(annmb_.dim);
  const int spin_offset = (paramb_.n_max_radial + 1) + paramb_.dim_angular;
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb_);

  {
    const int tok = kt ? kt->begin("compute_all_q_and_ann_small_md") : -1;
    switch (dim_bucket) {
      case 64:
        compute_all_q_and_ann_small_md_tmpl<64><<<grid_size, BLOCK_SIZE>>>(
          N,
          N1,
          N2,
          g_NN_r,
          g_NL_r,
          g_NN_a,
          g_NL_a,
          paramb_,
          annmb_,
          g_type,
          g_x12_r_all,
          g_y12_r_all,
          g_z12_r_all,
          g_x12_a_all,
          g_y12_a_all,
          g_z12_a_all,
          g_spin,
          nullptr,
          g_sum_fxyz,
          g_q_scaler,
          g_pe,
          g_Fp);
        break;
      case 96:
        compute_all_q_and_ann_small_md_tmpl<96><<<grid_size, BLOCK_SIZE>>>(
          N,
          N1,
          N2,
          g_NN_r,
          g_NL_r,
          g_NN_a,
          g_NL_a,
          paramb_,
          annmb_,
          g_type,
          g_x12_r_all,
          g_y12_r_all,
          g_z12_r_all,
          g_x12_a_all,
          g_y12_a_all,
          g_z12_a_all,
          g_spin,
          nullptr,
          g_sum_fxyz,
          g_q_scaler,
          g_pe,
          g_Fp);
        break;
      case 128:
        compute_all_q_and_ann_small_md_tmpl<128><<<grid_size, BLOCK_SIZE>>>(
          N,
          N1,
          N2,
          g_NN_r,
          g_NL_r,
          g_NN_a,
          g_NL_a,
          paramb_,
          annmb_,
          g_type,
          g_x12_r_all,
          g_y12_r_all,
          g_z12_r_all,
          g_x12_a_all,
          g_y12_a_all,
          g_z12_a_all,
          g_spin,
          nullptr,
          g_sum_fxyz,
          g_q_scaler,
          g_pe,
          g_Fp);
        break;
      default:
        compute_all_q_and_ann_small_md<<<grid_size, BLOCK_SIZE>>>(
          N,
          N1,
          N2,
          g_NN_r,
          g_NL_r,
          g_NN_a,
          g_NL_a,
          paramb_,
          annmb_,
          g_type,
          g_x12_r_all,
          g_y12_r_all,
          g_z12_r_all,
          g_x12_a_all,
          g_y12_a_all,
          g_z12_a_all,
          g_spin,
          nullptr,
          g_sum_fxyz,
          g_q_scaler,
          g_pe,
          g_Fp);
        break;
    }
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  double* g_fx = force.data();
  double* g_fy = force.data() + N;
  double* g_fz = force.data() + 2 * N;
  double* g_virial = virial.data();

  {
    const int tok = kt ? kt->begin("find_force_radial_spinbase_large_md") : -1;
    find_force_radial_spinbase_large_md<<<grid_size, BLOCK_SIZE>>>(
      N,
      N1,
      N2,
      box,
      g_NN_r,
      g_NL_r,
      paramb_,
      annmb_,
      g_type,
      g_x,
      g_y,
      g_z,
      g_Fp,
      g_fx,
      g_fy,
      g_fz,
      g_virial);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  {
    const int tok = kt ? kt->begin("find_partial_force_angular_spinbase_large_md") : -1;
    find_partial_force_angular_spinbase_large_md<<<grid_size, BLOCK_SIZE>>>(
      N,
      N1,
      N2,
      box,
      g_NN_a,
      g_NL_a,
      paramb_,
      annmb_,
      g_type,
      g_x,
      g_y,
      g_z,
      g_Fp,
      g_sum_fxyz,
      nep_data_.f12x.data(),
      nep_data_.f12y.data(),
      nep_data_.f12z.data());
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  {
    const int tok = kt ? kt->begin("find_force_many_body_spin_large_md") : -1;
    find_force_many_body_spin_large_md<<<grid_size, BLOCK_SIZE>>>(
      N,
      N1,
      N2,
      box,
      g_NN_a,
      g_NL_a,
      g_x,
      g_y,
      g_z,
      nep_data_.f12x.data(),
      nep_data_.f12y.data(),
      nep_data_.f12z.data(),
      g_fx,
      g_fy,
      g_fz,
      g_virial);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  if (mforce.size() < static_cast<size_t>(3) * N) {
    PRINT_INPUT_ERROR("NEP_Spin::compute_large_box: mforce.size() < 3*natoms.\n");
  }
  double* g_m = mforce.data();
  double* g_mx = g_m;
  double* g_my = g_m + N;
  double* g_mz = g_m + 2 * N;

  {
    const int tok = kt ? kt->begin("zero_mforce_spin") : -1;
    zero_mforce_spin<<<grid_size_full, BLOCK_SIZE>>>(N, g_mx, g_my, g_mz);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  const int pair_blocks = spin_blocks.pair_blocks;
  const int nspin = nep_spin_nspin(paramb_);

  {
    constexpr int KMAX_PAIR = 8;
    const int onsite_offset = spin_offset + nspin * pair_blocks;
    const int tok_force = kt ? kt->begin("find_force_radial_spin_spherical_md_noatomic(split_terms)") : -1;
    if (spin_blocks.kmax_ex >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_ex,
        find_force_radial_spin_spherical_md_noatomic_ex_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r_all,
        g_y12_r_all,
        g_z12_r_all,
        g_spin,
        g_Fp,
        g_fx,
        g_fy,
        g_fz,
        g_virial,
        spin_offset);
    }
    if (spin_blocks.kmax_dmi >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_dmi,
        find_force_radial_spin_spherical_md_noatomic_dmi_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r_all,
        g_y12_r_all,
        g_z12_r_all,
        g_spin,
        g_Fp,
        g_fx,
        g_fy,
        g_fz,
        g_virial,
        spin_offset);
    }
    if (spin_blocks.kmax_ani >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_ani,
        find_force_radial_spin_spherical_md_noatomic_ani_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r_all,
        g_y12_r_all,
        g_z12_r_all,
        g_spin,
        g_Fp,
        g_fx,
        g_fy,
        g_fz,
        g_virial,
        spin_offset);
    }
    if (spin_blocks.kmax_sia >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_sia,
        find_force_radial_spin_spherical_md_noatomic_sia_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r_all,
        g_y12_r_all,
        g_z12_r_all,
        g_spin,
        g_Fp,
        g_fx,
        g_fy,
        g_fz,
        g_virial,
        spin_offset);
    }
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok_force);

    const int tok_mforce = kt ? kt->begin("find_mforce_radial_spin_spherical_md_noatomic(split_terms)") : -1;
    if (spin_blocks.kmax_ex >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_ex,
        find_mforce_radial_spin_spherical_md_noatomic_ex_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r_all,
        g_y12_r_all,
        g_z12_r_all,
        g_spin,
        g_Fp,
        g_mx,
        g_my,
        g_mz,
        spin_offset);
    }
    if (spin_blocks.kmax_dmi >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_dmi,
        find_mforce_radial_spin_spherical_md_noatomic_dmi_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r_all,
        g_y12_r_all,
        g_z12_r_all,
        g_spin,
        g_Fp,
        g_mx,
        g_my,
        g_mz,
        spin_offset);
    }
    if (spin_blocks.kmax_ani >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_ani,
        find_mforce_radial_spin_spherical_md_noatomic_ani_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r_all,
        g_y12_r_all,
        g_z12_r_all,
        g_spin,
        g_Fp,
        g_mx,
        g_my,
        g_mz,
        spin_offset);
    }
    if (spin_blocks.kmax_sia >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_sia,
        find_mforce_radial_spin_spherical_md_noatomic_sia_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r_all,
        g_y12_r_all,
        g_z12_r_all,
        g_spin,
        g_Fp,
        g_mx,
        g_my,
        g_mz,
        spin_offset);
    }
    if (paramb_.spin_pmax > 0) {
      find_mforce_onsite_spin_spherical_md<<<grid_size, BLOCK_SIZE>>>(
        N, N1, N2, paramb_, g_spin, g_Fp, g_mx, g_my, g_mz, onsite_offset);
    }
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok_mforce);
  }
}

void NEP_Spin::compute_small_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<float>& spin,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  GPU_Vector<double>& mforce)
{
  KernelTiming* kt = g_nep_spin_kernel_timing;

  const int N = static_cast<int>(type.size());
  const int N1 = 0;
  const int N2 = N;
  const int BLOCK_SIZE = 64;
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;
  const int grid_size_full = (N - 1) / BLOCK_SIZE + 1;

  const int* g_type = type.data();
  const double* g_x = position.data();
  const double* g_y = position.data() + N;
  const double* g_z = position.data() + 2 * N;

  int* g_NN_r = nep_data_.NN_radial.data();
  int* g_NL_r = nep_data_.NL_radial.data();
  int* g_NN_a = nep_data_.NN_angular.data();
  int* g_NL_a = nep_data_.NL_angular.data();

  float* g_x12_r = nep_data_.x12_radial.data();
  float* g_y12_r = nep_data_.y12_radial.data();
  float* g_z12_r = nep_data_.z12_radial.data();
  float* g_x12_a = nep_data_.x12_angular.data();
  float* g_y12_a = nep_data_.y12_angular.data();
  float* g_z12_a = nep_data_.z12_angular.data();

  {
    const int tok = kt ? kt->begin("find_neighbor_list_spin_small_box") : -1;
    find_neighbor_list_spin_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb_,
      N,
      N1,
      N2,
      box,
      ebox_,
      g_type,
      g_x,
      g_y,
      g_z,
      g_NN_r,
      g_NL_r,
      g_NN_a,
      g_NL_a,
      g_x12_r,
      g_y12_r,
      g_z12_r,
      g_x12_a,
      g_y12_a,
      g_z12_a);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  const float* g_spin = spin.data();
  float* g_sum_fxyz = nep_data_.sum_fxyz.data();
  double* g_pe = potential.data();
  float* g_Fp = nep_data_.Fp.data();
  const float* g_q_scaler = nep_data_.q_scaler.data();
  const int dim_bucket = nep_spin_pick_dim_bucket(annmb_.dim);
  const int spin_offset = (paramb_.n_max_radial + 1) + paramb_.dim_angular;
  const NepSpinPairBlocks spin_blocks = nep_spin_get_pair_blocks(paramb_);

  {
    const int tok = kt ? kt->begin("compute_all_q_and_ann_small_md") : -1;
    switch (dim_bucket) {
      case 64:
        compute_all_q_and_ann_small_md_tmpl<64><<<grid_size, BLOCK_SIZE>>>(
          N,
          N1,
          N2,
          g_NN_r,
          g_NL_r,
          g_NN_a,
          g_NL_a,
          paramb_,
          annmb_,
          g_type,
          g_x12_r,
          g_y12_r,
          g_z12_r,
          g_x12_a,
          g_y12_a,
          g_z12_a,
          g_spin,
          nullptr,
          g_sum_fxyz,
          g_q_scaler,
          g_pe,
          g_Fp);
        break;
      case 96:
        compute_all_q_and_ann_small_md_tmpl<96><<<grid_size, BLOCK_SIZE>>>(
          N,
          N1,
          N2,
          g_NN_r,
          g_NL_r,
          g_NN_a,
          g_NL_a,
          paramb_,
          annmb_,
          g_type,
          g_x12_r,
          g_y12_r,
          g_z12_r,
          g_x12_a,
          g_y12_a,
          g_z12_a,
          g_spin,
          nullptr,
          g_sum_fxyz,
          g_q_scaler,
          g_pe,
          g_Fp);
        break;
      case 128:
        compute_all_q_and_ann_small_md_tmpl<128><<<grid_size, BLOCK_SIZE>>>(
          N,
          N1,
          N2,
          g_NN_r,
          g_NL_r,
          g_NN_a,
          g_NL_a,
          paramb_,
          annmb_,
          g_type,
          g_x12_r,
          g_y12_r,
          g_z12_r,
          g_x12_a,
          g_y12_a,
          g_z12_a,
          g_spin,
          nullptr,
          g_sum_fxyz,
          g_q_scaler,
          g_pe,
          g_Fp);
        break;
      default:
        compute_all_q_and_ann_small_md<<<grid_size, BLOCK_SIZE>>>(
          N,
          N1,
          N2,
          g_NN_r,
          g_NL_r,
          g_NN_a,
          g_NL_a,
          paramb_,
          annmb_,
          g_type,
          g_x12_r,
          g_y12_r,
          g_z12_r,
          g_x12_a,
          g_y12_a,
          g_z12_a,
          g_spin,
          nullptr,
          g_sum_fxyz,
          g_q_scaler,
          g_pe,
          g_Fp);
        break;
    }
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  double* g_fx = force.data();
  double* g_fy = force.data() + N;
  double* g_fz = force.data() + 2 * N;
  double* g_virial = virial.data();

  {
    const int tok = kt ? kt->begin("find_force_radial_spinbase_md") : -1;
    find_force_radial_spinbase_md<<<grid_size, BLOCK_SIZE>>>(
      N,
      N1,
      N2,
      g_NN_r,
      g_NL_r,
      paramb_,
      annmb_,
      g_type,
      g_x12_r,
      g_y12_r,
      g_z12_r,
      g_Fp,
      g_fx,
      g_fy,
      g_fz,
      g_virial);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  {
    const int tok = kt ? kt->begin("find_force_angular_spinbase_md") : -1;
    find_force_angular_spinbase_md<<<grid_size, BLOCK_SIZE>>>(
      N,
      N1,
      N2,
      g_NN_a,
      g_NL_a,
      paramb_,
      annmb_,
      g_type,
      g_x12_a,
      g_y12_a,
      g_z12_a,
      g_Fp,
      g_sum_fxyz,
      g_fx,
      g_fy,
      g_fz,
      g_virial);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  if (mforce.size() < static_cast<size_t>(3) * N) {
    PRINT_INPUT_ERROR("NEP_Spin::compute_small_box: mforce.size() < 3*natoms.\n");
  }
  double* g_m = mforce.data();
  double* g_mx = g_m;
  double* g_my = g_m + N;
  double* g_mz = g_m + 2 * N;

  {
    const int tok = kt ? kt->begin("zero_mforce_spin") : -1;
    zero_mforce_spin<<<grid_size_full, BLOCK_SIZE>>>(N, g_mx, g_my, g_mz);
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok);
  }

  const int pair_blocks = spin_blocks.pair_blocks;
  const int nspin = nep_spin_nspin(paramb_);

  {
    constexpr int KMAX_PAIR = 8;
    const int onsite_offset = spin_offset + nspin * pair_blocks;
    const int tok_force = kt ? kt->begin("find_force_radial_spin_spherical_md_noatomic(split_terms)") : -1;
    if (spin_blocks.kmax_ex >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_ex,
        find_force_radial_spin_spherical_md_noatomic_ex_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r,
        g_y12_r,
        g_z12_r,
        g_spin,
        g_Fp,
        g_fx,
        g_fy,
        g_fz,
        g_virial,
        spin_offset);
    }
    if (spin_blocks.kmax_dmi >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_dmi,
        find_force_radial_spin_spherical_md_noatomic_dmi_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r,
        g_y12_r,
        g_z12_r,
        g_spin,
        g_Fp,
        g_fx,
        g_fy,
        g_fz,
        g_virial,
        spin_offset);
    }
    if (spin_blocks.kmax_ani >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_ani,
        find_force_radial_spin_spherical_md_noatomic_ani_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r,
        g_y12_r,
        g_z12_r,
        g_spin,
        g_Fp,
        g_fx,
        g_fy,
        g_fz,
        g_virial,
        spin_offset);
    }
    if (spin_blocks.kmax_sia >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_sia,
        find_force_radial_spin_spherical_md_noatomic_sia_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r,
        g_y12_r,
        g_z12_r,
        g_spin,
        g_Fp,
        g_fx,
        g_fy,
        g_fz,
        g_virial,
        spin_offset);
    }
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok_force);

    const int tok_mforce = kt ? kt->begin("find_mforce_radial_spin_spherical_md_noatomic(split_terms)") : -1;
    if (spin_blocks.kmax_ex >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_ex,
        find_mforce_radial_spin_spherical_md_noatomic_ex_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r,
        g_y12_r,
        g_z12_r,
        g_spin,
        g_Fp,
        g_mx,
        g_my,
        g_mz,
        spin_offset);
    }
    if (spin_blocks.kmax_dmi >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_dmi,
        find_mforce_radial_spin_spherical_md_noatomic_dmi_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r,
        g_y12_r,
        g_z12_r,
        g_spin,
        g_Fp,
        g_mx,
        g_my,
        g_mz,
        spin_offset);
    }
    if (spin_blocks.kmax_ani >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_ani,
        find_mforce_radial_spin_spherical_md_noatomic_ani_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r,
        g_y12_r,
        g_z12_r,
        g_spin,
        g_Fp,
        g_mx,
        g_my,
        g_mz,
        spin_offset);
    }
    if (spin_blocks.kmax_sia >= 0) {
      NEP_SPIN_DISPATCH_KMAX(
        spin_blocks.kmax_sia,
        find_mforce_radial_spin_spherical_md_noatomic_sia_k,
        N,
        N1,
        N2,
        g_NN_r,
        g_NL_r,
        paramb_,
        annmb_,
        g_type,
        g_x12_r,
        g_y12_r,
        g_z12_r,
        g_spin,
        g_Fp,
        g_mx,
        g_my,
        g_mz,
        spin_offset);
    }
    if (paramb_.spin_pmax > 0) {
      find_mforce_onsite_spin_spherical_md<<<grid_size, BLOCK_SIZE>>>(
        N, N1, N2, paramb_, g_spin, g_Fp, g_mx, g_my, g_mz, onsite_offset);
    }
    GPU_CHECK_KERNEL
    if (kt) kt->end(tok_mforce);
  }
}
