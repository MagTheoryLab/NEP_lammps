/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "nep_lmp_bridge.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include "utils/nep_lmp_utils.cuh"
#include "model/box.cuh"
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

static inline bool env_flag(const char* name)
{
  return std::getenv(name) != nullptr;
}

#ifdef USE_HIP
__device__ __constant__ float g_nep_rc_radial_by_type[NUM_ELEMENTS];
__device__ __constant__ float g_nep_rc_angular_by_type[NUM_ELEMENTS];
#else
__constant__ float g_nep_rc_radial_by_type[NUM_ELEMENTS];
__constant__ float g_nep_rc_angular_by_type[NUM_ELEMENTS];
#endif

static inline __device__ float nep_pair_rc_radial(const NEP::ParaMB& paramb, int t1, int t2)
{
  float rc = 0.5f * (g_nep_rc_radial_by_type[t1] + g_nep_rc_radial_by_type[t2]);
  if (paramb.use_typewise_cutoff) {
    rc = min(
      (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
        paramb.typewise_cutoff_radial_factor,
      rc);
  }
  return rc;
}

static inline __device__ float nep_pair_rc_angular(const NEP::ParaMB& paramb, int t1, int t2)
{
  float rc = 0.5f * (g_nep_rc_angular_by_type[t1] + g_nep_rc_angular_by_type[t2]);
  if (paramb.use_typewise_cutoff) {
    rc = min(
      (COVALENT_RADIUS[paramb.atomic_numbers[t1]] + COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
        paramb.typewise_cutoff_angular_factor,
      rc);
  }
  return rc;
}

} // namespace

static __global__ void validate_lmp_neighbor_lists(
  const int nlocal,
  const int natoms,
  const int mn_r,
  const int mn_a,
  const int* __restrict__ nn_r,
  const int* __restrict__ nl_r,
  const int* __restrict__ nn_a,
  const int* __restrict__ nl_a,
  int* __restrict__ flag)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nlocal) return;
  int bits = 0;
  const int nr = nn_r[i];
  const int na = nn_a[i];
  if (nr < 0 || nr > mn_r) bits |= 1;
  if (na < 0 || na > mn_a) bits |= 4;
  const int nr_c = (nr < 0 ? 0 : (nr > mn_r ? mn_r : nr));
  const int na_c = (na < 0 ? 0 : (na > mn_a ? mn_a : na));
  for (int s = 0; s < nr_c; ++s) {
    const int j = nl_r[i + nlocal * s];
    if (j < 0 || j >= natoms) { bits |= 2; break; }
  }
  for (int s = 0; s < na_c; ++s) {
    const int j = nl_a[i + nlocal * s];
    if (j < 0 || j >= natoms) { bits |= 8; break; }
  }
  if (bits) atomicOr(flag, bits);
}

static __global__ void reduce_ev_totals(
  const int nlocal,
  const double* __restrict__ pe,    // nlocal, may be nullptr
  const double* __restrict__ v_soa, // 9*nlocal, may be nullptr
  double* __restrict__ out7)        // 7 values: eng,vxx,vyy,vzz,vxy,vxz,vyz
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nlocal) {
    if (pe) atomic_add_force(&out7[0], pe[i]);
    if (v_soa) {
      atomic_add_force(&out7[1], v_soa[i + nlocal * 0]);
      atomic_add_force(&out7[2], v_soa[i + nlocal * 1]);
      atomic_add_force(&out7[3], v_soa[i + nlocal * 2]);
      atomic_add_force(&out7[4], v_soa[i + nlocal * 3]);
      atomic_add_force(&out7[5], v_soa[i + nlocal * 4]);
      atomic_add_force(&out7[6], v_soa[i + nlocal * 5]);
    }
  }
}

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

NEP::NEP(const char* file_potential, const int num_atoms)
{
  lmp_force_fp32_ = (std::getenv("NEP_GPU_LMP_FORCE_FP32") != nullptr);

  std::ifstream input(file_potential);
  if (!input.is_open()) {
    exit(1);
  }

  // nep3 1 C
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    exit(1);
  }
  if (tokens[0] == "nep3") {
    paramb.version = 3;
    zbl.enabled = false;
  } else if (tokens[0] == "nep3_zbl") {
    paramb.version = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4") {
    paramb.version = 4;
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    paramb.version = 4;
    zbl.enabled = true;
  } else if (tokens[0] == "nep5") {
    paramb.version = 5;
    zbl.enabled = false;
  } else if (tokens[0] == "nep5_zbl") {
    paramb.version = 5;
    zbl.enabled = true;
  } else if (tokens[0] == "nep3_temperature") {
    paramb.version = 3;
    paramb.model_type = 3;
  } else if (tokens[0] == "nep3_zbl_temperature") {
    paramb.version = 3;
    paramb.model_type = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_temperature") {
    paramb.version = 4;
    paramb.model_type = 3;
  } else if (tokens[0] == "nep4_zbl_temperature") {
    paramb.version = 4;
    paramb.model_type = 3;
    zbl.enabled = true;
  } else if (tokens[0] == "nep3_dipole") {
    paramb.version = 3;
    paramb.model_type = 1;
  } else if (tokens[0] == "nep4_dipole") {
    paramb.version = 4;
    paramb.model_type = 1;
  } else if (tokens[0] == "nep3_polarizability") {
    paramb.version = 3;
    paramb.model_type = 2;
  } else if (tokens[0] == "nep4_polarizability") {
    paramb.version = 4;
    paramb.model_type = 2;
  } else {
    exit(1);
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    exit(1);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    paramb.atomic_numbers[n] = atomic_number - 1;
    // suppress informational logging for each type
  }

  // zbl 0.7 1.4
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3) {
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
    }
  }

  // cutoff rc_r rc_a ... or per-type pairs from refactored main_nep
  tokens = get_tokens(input);
  const int per_type_tokens = 2 * paramb.num_types + 3;
  const int per_type_with_factors = per_type_tokens + 3;
  if (static_cast<int>(tokens.size()) != 5 &&
      static_cast<int>(tokens.size()) != 8 &&
      static_cast<int>(tokens.size()) != per_type_tokens &&
      static_cast<int>(tokens.size()) != per_type_with_factors) {
    exit(1);
  }
  const bool has_per_type_cutoff =
    (static_cast<int>(tokens.size()) == per_type_tokens ||
     static_cast<int>(tokens.size()) == per_type_with_factors);

  float rc_radial_by_type_host[NUM_ELEMENTS] = {0.0f};
  float rc_angular_by_type_host[NUM_ELEMENTS] = {0.0f};
  int MN_radial = 0;
  int MN_angular = 0;
  paramb.rc_radial = 0.0f;
  paramb.rc_angular = 0.0f;
  if (has_per_type_cutoff) {
    for (int i = 0; i < paramb.num_types; ++i) {
      rc_radial_by_type_host[i] =
        get_double_from_token(tokens[1 + 2 * i], __FILE__, __LINE__);
      rc_angular_by_type_host[i] =
        get_double_from_token(tokens[2 + 2 * i], __FILE__, __LINE__);
      if (rc_radial_by_type_host[i] > paramb.rc_radial) {
        paramb.rc_radial = rc_radial_by_type_host[i];
      }
      if (rc_angular_by_type_host[i] > paramb.rc_angular) {
        paramb.rc_angular = rc_angular_by_type_host[i];
      }
    }
    MN_radial = get_int_from_token(tokens[1 + 2 * paramb.num_types], __FILE__, __LINE__);
    MN_angular = get_int_from_token(tokens[2 + 2 * paramb.num_types], __FILE__, __LINE__);
  } else {
    paramb.rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb.rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (int i = 0; i < paramb.num_types; ++i) {
      rc_radial_by_type_host[i] = paramb.rc_radial;
      rc_angular_by_type_host[i] = paramb.rc_angular;
    }
    MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
    MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
  }
  CHECK(gpuMemcpyToSymbol(
    g_nep_rc_radial_by_type,
    rc_radial_by_type_host,
    sizeof(float) * NUM_ELEMENTS));
  CHECK(gpuMemcpyToSymbol(
    g_nep_rc_angular_by_type,
    rc_angular_by_type_host,
    sizeof(float) * NUM_ELEMENTS));
  if (MN_radial > 819) {
    exit(1);
  }
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));

  if (static_cast<int>(tokens.size()) == 8 || static_cast<int>(tokens.size()) == per_type_with_factors) {
    const int factor_offset = has_per_type_cutoff ? 1 + 2 * paramb.num_types : 3;
    paramb.typewise_cutoff_radial_factor =
      get_double_from_token(tokens[factor_offset + 2], __FILE__, __LINE__);
    paramb.typewise_cutoff_angular_factor =
      get_double_from_token(tokens[factor_offset + 3], __FILE__, __LINE__);
    paramb.typewise_cutoff_zbl_factor =
      get_double_from_token(tokens[factor_offset + 4], __FILE__, __LINE__);
    if (paramb.typewise_cutoff_radial_factor > 0.0f) {
      paramb.use_typewise_cutoff = true;
    }
    if (paramb.typewise_cutoff_zbl_factor > 0.0f) {
      paramb.use_typewise_cutoff_zbl = true;
    }
  }
#ifdef USE_TABLE
  if (paramb.use_typewise_cutoff || has_per_type_cutoff) {
    PRINT_INPUT_ERROR("Cannot use tabulated radial functions with typewise cutoff.");
  }
#endif

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() != 4) {
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.num_L = paramb.L_max;

  int L_max_4body = get_int_from_token(tokens[2], __FILE__, __LINE__);
  int L_max_5body = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (L_max_4body == 2) {
    paramb.num_L += 1;
  }
  if (L_max_5body == 1) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    exit(1);
  }
  annmb.num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  nep_model_type = paramb.model_type;
  if (paramb.model_type == 3) {
    annmb.dim += 1;
  }

  // calculated parameters:
  rc = paramb.rc_radial; // largest cutoff
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  if (paramb.version == 3) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 + 1;
  } else if (paramb.version == 4) {
    annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  } else {
    annmb.num_para_ann = ((annmb.dim + 2) * annmb.num_neurons1 + 1) * paramb.num_types + 1;
  }
  if (paramb.model_type == 2) {
    // Polarizability models have twice as many parameters
    annmb.num_para_ann *= 2;
  }
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para);
  for (int n = 0; n < annmb.num_para; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  nep_data.parameters.resize(annmb.num_para);
  nep_data.parameters.copy_from_host(parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  for (int d = 0; d < annmb.dim; ++d) {
    tokens = get_tokens(input);
    paramb.q_scaler[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;

    // Precompute a safe maximum ZBL outer cutoff for neighbor filtering.
    // The flexible ZBL format stores rc_inner, rc_outer as the first two values per type-pair.
    zbl_rc_outer_max_ = 0.0f;
    for (int idx = 0; idx < num_type_zbl; ++idx) {
      const float rc_outer = zbl.para[10 * idx + 1];
      if (rc_outer > zbl_rc_outer_max_) zbl_rc_outer_max_ = rc_outer;
    }
  } else if (zbl.enabled) {
    zbl_rc_outer_max_ = zbl.rc_outer;
  } else {
    zbl_rc_outer_max_ = 0.0f;
  }
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(num_atoms * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.Fp.resize(num_atoms * annmb.dim);
  nep_data.sum_fxyz.resize(
    num_atoms * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data.cell_count.resize(num_atoms);
  nep_data.cell_count_sum.resize(num_atoms);
  nep_data.cell_contents.resize(num_atoms);
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);

#ifdef USE_TABLE
  construct_table(parameters.data());
  // logging suppressed
#endif

  B_projection_size = annmb.num_neurons1 * (annmb.dim + 2);
}

NEP::~NEP(void)
{
  // nothing
}

void NEP::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
      pointer -= (ann.dim + 2) * ann.num_neurons1;
    }
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    if (paramb.version == 5) {
      pointer += 1; // one extra bias for NEP5 stored in ann.w1[t]
    }
  }
  ann.b1 = pointer;
  pointer += 1;

  // Possibly read polarizability parameters, which are placed after the regular nep parameters.
  if (paramb.model_type == 2) {
    for (int t = 0; t < paramb.num_types; ++t) {
      if (t > 0 && paramb.version == 3) { // Use the same set of NN parameters for NEP3
        pointer -= (ann.dim + 2) * ann.num_neurons1;
      }
      ann.w0_pol[t] = pointer;
      pointer += ann.num_neurons1 * ann.dim;
      ann.b0_pol[t] = pointer;
      pointer += ann.num_neurons1;
      ann.w1_pol[t] = pointer;
      pointer += ann.num_neurons1;
    }
    ann.b1_pol = pointer;
    pointer += 1;
  }

  ann.c = pointer;
}

#ifdef USE_TABLE
void NEP::construct_table(float* parameters)
{
  nep_data.gn_radial.resize(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  nep_data.gnp_radial.resize(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  nep_data.gn_angular.resize(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  nep_data.gnp_angular.resize(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  std::vector<float> gn_radial(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  std::vector<float> gnp_radial(table_length * paramb.num_types_sq * (paramb.n_max_radial + 1));
  std::vector<float> gn_angular(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  std::vector<float> gnp_angular(table_length * paramb.num_types_sq * (paramb.n_max_angular + 1));
  float* c_pointer = parameters + annmb.num_para_ann;
  construct_table_radial_or_angular(
    paramb.num_types,
    paramb.num_types_sq,
    paramb.n_max_radial,
    paramb.basis_size_radial,
    paramb.rc_radial,
    paramb.rcinv_radial,
    c_pointer,
    gn_radial.data(),
    gnp_radial.data());
  construct_table_radial_or_angular(
    paramb.num_types,
    paramb.num_types_sq,
    paramb.n_max_angular,
    paramb.basis_size_angular,
    paramb.rc_angular,
    paramb.rcinv_angular,
    c_pointer + paramb.num_c_radial,
    gn_angular.data(),
    gnp_angular.data());
  nep_data.gn_radial.copy_from_host(gn_radial.data());
  nep_data.gnp_radial.copy_from_host(gnp_radial.data());
  nep_data.gn_angular.copy_from_host(gn_angular.data());
  nep_data.gnp_angular.copy_from_host(gnp_angular.data());
}
#endif

static __global__ void find_descriptor(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int Nloc,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_owner,
  const bool is_polarizability,
#ifdef USE_TABLE
  const float* __restrict__ g_gn_radial,
  const float* __restrict__ g_gn_angular,
#endif
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz,
  bool need_B_projection,
  double* B_projection,
  int B_projection_size)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const bool need_mic = (N == Nloc) || (g_owner != nullptr);

    // Initialize per-atom outputs for this center atom (avoids global fills).
    g_pe[n1] = 0.0;
    g_virial[n1 + Nloc * 0] = 0.0;
    g_virial[n1 + Nloc * 1] = 0.0;
    g_virial[n1 + Nloc * 2] = 0.0;
    g_virial[n1 + Nloc * 3] = 0.0;
    g_virial[n1 + Nloc * 4] = 0.0;
    g_virial[n1 + Nloc * 5] = 0.0;
    g_virial[n1 + Nloc * 6] = 0.0;
    g_virial[n1 + Nloc * 7] = 0.0;
    g_virial[n1 + Nloc * 8] = 0.0;

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + Nloc * i1];
      const int t2 = g_type[n2];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      if (need_mic) apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
      const float r2 = x12 * x12 + y12 * y12 + z12 * z12;
      const float rc = nep_pair_rc_radial(paramb, t1, t2);
      if (r2 >= rc * rc) continue;

#ifdef USE_TABLE
      float d12 = sqrt(r2);
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        q[n] +=
          g_gn_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gn_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
      }
#else
      float fc12;
      const float d12 = sqrt(r2);
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
#endif
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + Nloc * i1];
        const int t2 = g_type[n2];
        double x12double = g_x[n2] - x1;
        double y12double = g_y[n2] - y1;
        double z12double = g_z[n2] - z1;
        if (need_mic) apply_mic(box, x12double, y12double, z12double);
        float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
        const float r2 = x12 * x12 + y12 * y12 + z12 * z12;
        const float rc = nep_pair_rc_angular(paramb, t1, t2);
        if (r2 >= rc * rc) continue;
#ifdef USE_TABLE
        float d12 = sqrt(r2);
        int index_left, index_right;
        float weight_left, weight_right;
        find_index_and_weight(
          d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
        int t12 = t1 * paramb.num_types + t2;
        float gn12 =
          g_gn_angular[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_left +
          g_gn_angular[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_right;
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
#else
        float fc12;
        const float d12 = sqrt(r2);
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
#endif
      }
      find_q(
        paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * Nloc + n1] = s[abc];
      }
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    if (is_polarizability) {
      apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0_pol[t1],
        annmb.b0_pol[t1],
        annmb.w1_pol[t1],
        annmb.b1_pol,
        q,
        F,
        Fp);
      // Add the potential F for this atom to the diagonal of the virial
      g_virial[n1] = F;
      g_virial[n1 + Nloc * 1] = F;
      g_virial[n1 + Nloc * 2] = F;

      // Reset the potential and forces such that they
      // are zero for the next call to the model. The next call
      // is not used in the case of is_pol = True, but it doesn't
      // hurt to clean up.
      F = 0.0f;
      for (int d = 0; d < annmb.dim; ++d) {
        Fp[d] = 0.0f;
      }
    }

    if (paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp);
    } else {
      if (!need_B_projection)
        apply_ann_one_layer(
          annmb.dim,
          annmb.num_neurons1,
          annmb.w0[t1],
          annmb.b0[t1],
          annmb.w1[t1],
          annmb.b1,
          q,
          F,
          Fp);
      else
        apply_ann_one_layer(
          annmb.dim,
          annmb.num_neurons1,
          annmb.w0[t1],
          annmb.b0[t1],
          annmb.w1[t1],
          annmb.b1,
          q,
          F,
          Fp,
          B_projection + n1 * B_projection_size);
    }
    g_pe[n1] = F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * Nloc + n1] = Fp[d] * paramb.q_scaler[d];
    }
  }
}

// Stub implementations to satisfy vtable; not used in LAMMPS-direct path.
inline void NEP::compute(Box&, const GPU_Vector<int>&, const GPU_Vector<double>&,
                  GPU_Vector<double>&, GPU_Vector<double>&, GPU_Vector<double>&) {}
inline void NEP::compute(const float, Box&, const GPU_Vector<int>&, const GPU_Vector<double>&,
                  GPU_Vector<double>&, GPU_Vector<double>&, GPU_Vector<double>&) {}
inline const GPU_Vector<int>& NEP::get_NN_radial_ptr() {
  return nep_data.NN_radial;
}
inline const GPU_Vector<int>& NEP::get_NL_radial_ptr() {
  return nep_data.NL_radial;
}

// LAMMPS-direct radial force kernel.
// Use a scatter formulation (atomics) for all cases. This avoids mixing atomic
// and non-atomic updates on the same force array, which can lead to data races
// and net-force bias when running with full neighbor lists.
template <typename ForceT>
static __global__ void find_force_radial_lmp(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int Nloc,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_owner,
  const float* __restrict__ g_Fp,
#ifdef USE_TABLE
  const float* __restrict__ g_gnp_radial,
#endif
  ForceT* g_fx,
  ForceT* g_fy,
  ForceT* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const bool need_mic = (N == Nloc) || (g_owner != nullptr);
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    // Accumulate central force in float (matches GPUMD's default NEP force path).
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

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + Nloc * i1];
      int t2 = g_type[n2];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      if (need_mic) apply_mic(box, x12double, y12double, z12double);
      float r12f[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12f[0] * r12f[0] + r12f[1] * r12f[1] + r12f[2] * r12f[2]);
      if (d12 == 0.0f) continue;
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f, 0.0f, 0.0f};
#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        float(d12 * paramb.rcinv_radial), index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 =
          g_gnp_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gnp_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
        float tmp12 = g_Fp[n1 + n * Nloc] * gnp12 * d12inv;
        f12[0] += tmp12 * r12f[0];
        f12[1] += tmp12 * r12f[1];
        f12[2] += tmp12 * r12f[2];
      }
#else
      float fc12, fcp12;
      const float rc = nep_pair_rc_radial(paramb, t1, t2);
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float tmp12 = g_Fp[n1 + n * Nloc] * gnp12 * d12inv;
        f12[0] += tmp12 * r12f[0];
        f12[1] += tmp12 * r12f[1];
        f12[2] += tmp12 * r12f[2];
      }
#endif
      s_fx += f12[0];
      s_fy += f12[1];
      s_fz += f12[2];

      const int n2_dst = map_owner_index(g_owner, n2, Nloc);
      atomic_add_force(g_fx + n2_dst, static_cast<ForceT>(-f12[0]));
      atomic_add_force(g_fy + n2_dst, static_cast<ForceT>(-f12[1]));
      atomic_add_force(g_fz + n2_dst, static_cast<ForceT>(-f12[2]));

      // Virial (assign to central atom)
      s_sxx -= float(x12double) * f12[0];
      s_syy -= float(y12double) * f12[1];
      s_szz -= float(z12double) * f12[2];
      s_sxy -= float(x12double) * f12[1];
      s_sxz -= float(x12double) * f12[2];
      s_syz -= float(y12double) * f12[2];
      s_syx -= float(y12double) * f12[0];
      s_szx -= float(z12double) * f12[0];
      s_szy -= float(z12double) * f12[1];
    }

    atomic_add_force(g_fx + n1, static_cast<ForceT>(s_fx));
    atomic_add_force(g_fy + n1, static_cast<ForceT>(s_fy));
    atomic_add_force(g_fz + n1, static_cast<ForceT>(s_fz));

    g_virial[n1 + 0 * Nloc] += double(s_sxx);
    g_virial[n1 + 1 * Nloc] += double(s_syy);
    g_virial[n1 + 2 * Nloc] += double(s_szz);
    g_virial[n1 + 3 * Nloc] += double(s_sxy);
    g_virial[n1 + 4 * Nloc] += double(s_sxz);
    g_virial[n1 + 5 * Nloc] += double(s_syz);
    g_virial[n1 + 6 * Nloc] += double(s_syx);
    g_virial[n1 + 7 * Nloc] += double(s_szx);
    g_virial[n1 + 8 * Nloc] += double(s_szy);
  }
} 
 
// LAMMPS-direct angular force kernel: only central descriptors; accumulates forces directly. 
template <typename ForceT>
static __global__ void find_force_angular_lmp(
  NEP::ParaMB paramb, 
  NEP::ANN annmb, 
  const int N,
  const int Nloc,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_owner,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
#ifdef USE_TABLE
  const float* __restrict__ g_gn_angular,
  const float* __restrict__ g_gnp_angular,
#endif
  ForceT* g_fx,
  ForceT* g_fy,
  ForceT* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const bool need_mic = (N == Nloc) || (g_owner != nullptr);
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * Nloc + n1];
    }

    const int sum_stride = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
    const float* __restrict__ sum_fxyz_base = g_sum_fxyz + n1;

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    double s_sxx = 0.0;
    double s_sxy = 0.0;
    double s_sxz = 0.0;
    double s_syx = 0.0;
    double s_syy = 0.0;
    double s_syz = 0.0;
    double s_szx = 0.0;
    double s_szy = 0.0;
    double s_szz = 0.0;

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * Nloc + n1;
      int n2 = g_NL_angular[index];
      int t2 = g_type[n2];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      if (need_mic) apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      if (d12 == 0.0f) continue;

      float f12[3] = {0.0f, 0.0f, 0.0f};
#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        int index_left_all =
          (index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        int index_right_all =
          (index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        float gn12 =
          g_gn_angular[index_left_all] * weight_left + g_gn_angular[index_right_all] * weight_right;
        float gnp12 = g_gnp_angular[index_left_all] * weight_left +
                      g_gnp_angular[index_right_all] * weight_right;
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
          Nloc,
          f12);
      }
#else
      float fc12, fcp12;
      const float rc = nep_pair_rc_angular(paramb, t1, t2);
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
          Nloc,
          f12);
      }
#endif

      s_fx += f12[0];
      s_fy += f12[1];
      s_fz += f12[2];

      const int n2_dst = map_owner_index(g_owner, n2, Nloc);
      atomic_add_force(g_fx + n2_dst, static_cast<ForceT>(-f12[0]));
      atomic_add_force(g_fy + n2_dst, static_cast<ForceT>(-f12[1]));
      atomic_add_force(g_fz + n2_dst, static_cast<ForceT>(-f12[2]));

      s_sxx -= float(x12double) * f12[0];
      s_syy -= float(y12double) * f12[1];
      s_szz -= float(z12double) * f12[2];
      s_sxy -= float(x12double) * f12[1];
      s_sxz -= float(x12double) * f12[2];
      s_syz -= float(y12double) * f12[2];
      s_syx -= float(y12double) * f12[0];
      s_szx -= float(z12double) * f12[0];
      s_szy -= float(z12double) * f12[1];
    }

    atomic_add_force(g_fx + n1, static_cast<ForceT>(s_fx));
    atomic_add_force(g_fy + n1, static_cast<ForceT>(s_fy));
    atomic_add_force(g_fz + n1, static_cast<ForceT>(s_fz));

    g_virial[n1 + 0 * Nloc] += double(s_sxx);
    g_virial[n1 + 1 * Nloc] += double(s_syy);
    g_virial[n1 + 2 * Nloc] += double(s_szz);
    g_virial[n1 + 3 * Nloc] += double(s_sxy);
    g_virial[n1 + 4 * Nloc] += double(s_sxz);
    g_virial[n1 + 5 * Nloc] += double(s_syz);
    g_virial[n1 + 6 * Nloc] += double(s_syx);
    g_virial[n1 + 7 * Nloc] += double(s_szx);
    g_virial[n1 + 8 * Nloc] += double(s_szy);
  }
}

template <typename ForceT>
static __global__ void find_force_ZBL(
  NEP::ParaMB paramb,
  const int N,
  const int Nloc,
  const NEP::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_owner,
  ForceT* g_fx,
  ForceT* g_fy,
  ForceT* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    const bool need_mic = (N == Nloc) || (g_owner != nullptr);
    float s_pe = 0.0f;
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
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + Nloc * i1];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      if (need_mic) apply_mic(box, x12double, y12double, z12double);
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = rc_outer * 0.5f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      s_fx += f12[0];
      s_fy += f12[1];
      s_fz += f12[2];

      // Pairwise ZBL: split force/energy across both directions in a full neighbor list.
      const int n2_dst = map_owner_index(g_owner, n2, Nloc);
      atomic_add_force(g_fx + n2_dst, static_cast<ForceT>(-f12[0]));
      atomic_add_force(g_fy + n2_dst, static_cast<ForceT>(-f12[1]));
      atomic_add_force(g_fz + n2_dst, static_cast<ForceT>(-f12[2]));

      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
    }
    // ZBL always uses atomics on neighbors; central updates must be atomic too to
    // avoid races when `n1` is a neighbor of another atom.
    atomic_add_force(g_fx + n1, static_cast<ForceT>(s_fx));
    atomic_add_force(g_fy + n1, static_cast<ForceT>(s_fy));
    atomic_add_force(g_fz + n1, static_cast<ForceT>(s_fz));
    g_virial[n1 + 0 * Nloc] += double(s_sxx);
    g_virial[n1 + 1 * Nloc] += double(s_syy);
    g_virial[n1 + 2 * Nloc] += double(s_szz);
    g_virial[n1 + 3 * Nloc] += double(s_sxy);
    g_virial[n1 + 4 * Nloc] += double(s_sxz);
    g_virial[n1 + 5 * Nloc] += double(s_syz);
    g_virial[n1 + 6 * Nloc] += double(s_syx);
    g_virial[n1 + 7 * Nloc] += double(s_szx);
    g_virial[n1 + 8 * Nloc] += double(s_szy);
    g_pe[n1] += s_pe;
  } 
} 

// LAMMPS-direct entry point (non-spin). Assumes neighbor lists are already built (compact stride=natoms).
void NEP::compute_with_neighbors(
  Box& box,
  int nlocal,
  int natoms,
  const int* type_host,
  const double* pos_soa_host,
  const int* NN_radial_host,
  const int* NL_radial_host,
  const int* NN_angular_host,
  const int* NL_angular_host,
  double* potential_host,
  double* force_host,
  double* virial_host)
{
  if (natoms <= 0) return;

  const bool need_potential = (potential_host != nullptr);
  const bool need_force = (force_host != nullptr);
  const bool need_virial = (virial_host != nullptr);

  // Configure work range: compute on all atoms (local+ghost) so symmetry is preserved,
  // but only scatter results for local atoms.
  this->N1 = 0;
  this->N2 = nlocal;

  // Neighbor lists and descriptor buffers are only needed for local atoms
  // [0,nlocal), so keep them compact with stride = nlocal.
  nep_data.NN_radial.resize(nlocal);
  nep_data.NL_radial.resize(static_cast<size_t>(nlocal) * this->paramb.MN_radial);
  nep_data.NN_angular.resize(nlocal);
  nep_data.NL_angular.resize(static_cast<size_t>(nlocal) * this->paramb.MN_angular);
  nep_data.Fp.resize(nlocal * this->annmb.dim);
  nep_data.sum_fxyz.resize(
    static_cast<size_t>(nlocal) * (this->paramb.n_max_angular + 1) * ((this->paramb.L_max + 1) * (this->paramb.L_max + 1) - 1));

  // Persistent device copies of type/position to avoid per-call cudaMalloc/cudaFree.
  lmp_type_dev.resize(natoms);
  lmp_type_dev.copy_from_host(type_host);

  lmp_pos_dev.resize(static_cast<size_t>(natoms) * 3);
  lmp_pos_dev.copy_from_host(pos_soa_host);

  // Copy neighbor lists
  nep_data.NN_radial.copy_from_host(NN_radial_host);
  nep_data.NL_radial.copy_from_host(NL_radial_host);
  nep_data.NN_angular.copy_from_host(NN_angular_host);
  nep_data.NL_angular.copy_from_host(NL_angular_host);

  // Output buffers on device
  lmp_potential.resize(nlocal);
  lmp_force.resize(static_cast<size_t>(natoms) * 3);
  lmp_virial.resize(static_cast<size_t>(nlocal) * 9);
  if (need_potential) lmp_potential.fill(0.0);
  if (need_force) lmp_force.fill(0.0);
  if (need_virial) lmp_virial.fill(0.0);
  nep_data.Fp.fill(0.0f);
  nep_data.sum_fxyz.fill(0.0f);

  const int BLOCK_SIZE = 64;
  const int grid_size_desc = (this->N2 - this->N1 - 1) / BLOCK_SIZE + 1;

  const bool is_polarizability = this->paramb.model_type == 2;
  find_descriptor<<<grid_size_desc, BLOCK_SIZE>>>(
    this->paramb,
    this->annmb,
    natoms,
    nlocal,
    this->N1,
    this->N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    lmp_type_dev.data(),
    lmp_pos_dev.data(),
    lmp_pos_dev.data() + natoms,
    lmp_pos_dev.data() + natoms * 2,
    nullptr,
    is_polarizability,
#ifdef USE_TABLE
    nep_data.gn_radial.data(),
    nep_data.gn_angular.data(),
#endif
    lmp_potential.data(),
    nep_data.Fp.data(),
    lmp_virial.data(),
    nep_data.sum_fxyz.data(),
    this->need_B_projection,
    this->B_projection,
    this->B_projection_size);
  GPU_CHECK_KERNEL

  find_force_radial_lmp<double><<<grid_size_desc, BLOCK_SIZE>>>(
    this->paramb,
    this->annmb,
    natoms,
    nlocal,
    this->N1,
    this->N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    lmp_type_dev.data(),
    lmp_pos_dev.data(),
    lmp_pos_dev.data() + natoms,
    lmp_pos_dev.data() + natoms * 2,
    nullptr,
    nep_data.Fp.data(),
#ifdef USE_TABLE
    nep_data.gnp_radial.data(),
#endif
    lmp_force.data(),
    lmp_force.data() + natoms,
    lmp_force.data() + natoms * 2,
    lmp_virial.data());
  GPU_CHECK_KERNEL

  find_force_angular_lmp<double><<<grid_size_desc, BLOCK_SIZE>>>(
    this->paramb,
    this->annmb,
    natoms,
    nlocal,
    this->N1,
    this->N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    lmp_type_dev.data(),
    lmp_pos_dev.data(),
    lmp_pos_dev.data() + natoms,
    lmp_pos_dev.data() + natoms * 2,
    nullptr,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
#ifdef USE_TABLE
    nep_data.gn_angular.data(),
    nep_data.gnp_angular.data(),
#endif
    lmp_force.data(),
    lmp_force.data() + natoms,
    lmp_force.data() + natoms * 2,
    lmp_virial.data());
  GPU_CHECK_KERNEL

  if (this->zbl.enabled) {
    find_force_ZBL<double><<<grid_size_desc, BLOCK_SIZE>>>(
      this->paramb,
      natoms,
      nlocal,
      this->zbl,
      this->N1,
      this->N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      lmp_type_dev.data(),
      lmp_pos_dev.data(),
      lmp_pos_dev.data() + natoms,
      lmp_pos_dev.data() + natoms * 2,
      nullptr,
      lmp_force.data(),
      lmp_force.data() + natoms,
      lmp_force.data() + natoms * 2,
      lmp_virial.data(),
      lmp_potential.data());
    GPU_CHECK_KERNEL
  }

  // Copy results back and scatter to local outputs.
  if (need_force) {
    lmp_force_tmp.resize(static_cast<size_t>(natoms) * 3);
    lmp_force.copy_to_host(lmp_force_tmp.data());

    // Scatter forces for all atoms (local + ghost) into AoS layout.
    for (int i = 0; i < natoms; ++i) {
      force_host[3 * i + 0] = lmp_force_tmp[i];
      force_host[3 * i + 1] = lmp_force_tmp[i + natoms];
      force_host[3 * i + 2] = lmp_force_tmp[i + natoms * 2];
    }
  }

  // Copy per-atom potential and virial only for local atoms.
  if (need_potential) {
    lmp_potential.copy_to_host(potential_host, nlocal);
  }

  if (need_virial) {
    lmp_virial_tmp.resize(static_cast<size_t>(nlocal) * 9);
    lmp_virial.copy_to_host(lmp_virial_tmp.data());

    for (int i = 0; i < nlocal; ++i) {
      virial_host[9 * i + 0] = lmp_virial_tmp[i + nlocal * 0]; // xx
      virial_host[9 * i + 1] = lmp_virial_tmp[i + nlocal * 1]; // yy
      virial_host[9 * i + 2] = lmp_virial_tmp[i + nlocal * 2]; // zz
      virial_host[9 * i + 3] = lmp_virial_tmp[i + nlocal * 3]; // xy
      virial_host[9 * i + 4] = lmp_virial_tmp[i + nlocal * 4]; // xz
      virial_host[9 * i + 5] = lmp_virial_tmp[i + nlocal * 5]; // yz
      virial_host[9 * i + 6] = lmp_virial_tmp[i + nlocal * 6]; // yx
      virial_host[9 * i + 7] = lmp_virial_tmp[i + nlocal * 7]; // zx
      virial_host[9 * i + 8] = lmp_virial_tmp[i + nlocal * 8]; // zy
    }
  }
}

void NEP::compute_with_neighbors_device(
  Box& box,
  int nlocal,
  int natoms,
  const int* type_dev,
  const double* xyz_aos_dev,
  const int* owner_dev,
  gpuStream_t stream,
  const int* NN_radial_dev,
  const int* NL_radial_dev,
  const int* NN_angular_dev,
  const int* NL_angular_dev,
  double* force_aos_dev,
  double* potential_dev,
  double* virial_aos_dev,
  bool need_energy,
  bool need_virial,
  double& eng_out,
  double virial_out[6])
{
  eng_out = 0.0;
  for (int k = 0; k < 6; ++k) virial_out[k] = 0.0;
  if (natoms <= 0 || nlocal <= 0) return;

  if (!type_dev || !xyz_aos_dev || !NN_radial_dev || !NL_radial_dev || !NN_angular_dev || !NL_angular_dev) {
    PRINT_INPUT_ERROR("NEP::compute_with_neighbors_device: null device pointer.\n");
  }

  const bool need_potential = (potential_dev != nullptr);
  const bool need_virial_atom = (virial_aos_dev != nullptr);

  this->N1 = 0;
  this->N2 = nlocal;

  // Scratch buffers needed for local atoms only
  nep_data.Fp.resize(nlocal * this->annmb.dim);
  nep_data.sum_fxyz.resize(
    static_cast<size_t>(nlocal) * (this->paramb.n_max_angular + 1) * ((this->paramb.L_max + 1) * (this->paramb.L_max + 1) - 1));

  // Pack AoS positions into persistent SoA buffer on device.
  lmp_pos_dev.resize(static_cast<size_t>(natoms) * 3);
  {
    const int block = 256;
    const int grid = (natoms + block - 1) / block;
    pack_xyz_aos_to_soa<<<grid, block, 0, stream>>>(natoms, xyz_aos_dev, lmp_pos_dev.data());
    GPU_CHECK_KERNEL
  }

  if (env_flag("NEP_GPU_LMP_VALIDATE_NL")) {
    lmp_validate.resize(1);
    fill_vector_async(lmp_validate, 0, stream);
    const int block = 256;
    const int grid = (nlocal + block - 1) / block;
    validate_lmp_neighbor_lists<<<grid, block, 0, stream>>>(
      nlocal,
      natoms,
      this->paramb.MN_radial,
      this->paramb.MN_angular,
      NN_radial_dev,
      NL_radial_dev,
      NN_angular_dev,
      NL_angular_dev,
      lmp_validate.data());
    GPU_CHECK_KERNEL
    int host_flag = 0;
    lmp_validate.copy_to_host(&host_flag);
    if (host_flag != 0) {
      std::fprintf(
        stderr,
        "NEP_GPU_LMP_VALIDATE_NL: invalid NN/NL detected (flag=%d). This indicates out-of-range neighbor indices or NN > MN_*.\n",
        host_flag);
      std::fflush(stderr);
      exit(1);
    }
  }

  // Output buffers on device (SoA for forces and virials, as required by kernels)
  const bool use_fp32_force = lmp_force_fp32_;
  const int force_n = natoms;

  if (use_fp32_force) {
    lmp_force_f.resize(static_cast<size_t>(force_n) * 3);
    fill_vector_async(lmp_force_f, 0.0f, stream);
  } else {
    lmp_force.resize(static_cast<size_t>(force_n) * 3);
    fill_vector_async(lmp_force, 0.0, stream);
  }

  // Potential and virial buffers are required for totals, and optionally exported.
  lmp_potential.resize(nlocal);
  lmp_virial.resize(static_cast<size_t>(nlocal) * 9);

  const int BLOCK_SIZE = 64;
  const int grid_size_desc = (this->N2 - this->N1 - 1) / BLOCK_SIZE + 1;

  const bool is_polarizability = this->paramb.model_type == 2;
  find_descriptor<<<grid_size_desc, BLOCK_SIZE, 0, stream>>>(
    this->paramb,
    this->annmb,
    natoms,
    nlocal,
    this->N1,
    this->N2,
    box,
    NN_radial_dev,
    NL_radial_dev,
    NN_angular_dev,
    NL_angular_dev,
    type_dev,
    lmp_pos_dev.data(),
    lmp_pos_dev.data() + natoms,
    lmp_pos_dev.data() + natoms * 2,
    owner_dev,
    is_polarizability,
#ifdef USE_TABLE
    nep_data.gn_radial.data(),
    nep_data.gn_angular.data(),
#endif
    lmp_potential.data(),
    nep_data.Fp.data(),
    lmp_virial.data(),
    nep_data.sum_fxyz.data(),
    this->need_B_projection,
    this->B_projection,
    this->B_projection_size);
  GPU_CHECK_KERNEL

  if (use_fp32_force) {
    find_force_radial_lmp<float><<<grid_size_desc, BLOCK_SIZE, 0, stream>>>(
      this->paramb,
      this->annmb,
      natoms,
      nlocal,
      this->N1,
      this->N2,
      box,
      NN_radial_dev,
      NL_radial_dev,
      type_dev,
      lmp_pos_dev.data(),
      lmp_pos_dev.data() + natoms,
      lmp_pos_dev.data() + natoms * 2,
      owner_dev,
      nep_data.Fp.data(),
#ifdef USE_TABLE
      nep_data.gnp_radial.data(),
#endif
      lmp_force_f.data(),
      lmp_force_f.data() + natoms,
      lmp_force_f.data() + natoms * 2,
      lmp_virial.data());
  } else {
    find_force_radial_lmp<double><<<grid_size_desc, BLOCK_SIZE, 0, stream>>>(
      this->paramb,
      this->annmb,
      natoms,
      nlocal,
      this->N1,
      this->N2,
      box,
      NN_radial_dev,
      NL_radial_dev,
      type_dev,
      lmp_pos_dev.data(),
      lmp_pos_dev.data() + natoms,
      lmp_pos_dev.data() + natoms * 2,
      owner_dev,
      nep_data.Fp.data(),
#ifdef USE_TABLE
      nep_data.gnp_radial.data(),
#endif
      lmp_force.data(),
      lmp_force.data() + natoms,
      lmp_force.data() + natoms * 2,
      lmp_virial.data());
  }
  GPU_CHECK_KERNEL

  if (use_fp32_force) {
    find_force_angular_lmp<float><<<grid_size_desc, BLOCK_SIZE, 0, stream>>>(
          this->paramb,
          this->annmb,
          natoms,
          nlocal,
          this->N1,
          this->N2,
          box,
          NN_angular_dev,
          NL_angular_dev,
          type_dev,
          lmp_pos_dev.data(),
          lmp_pos_dev.data() + natoms,
          lmp_pos_dev.data() + natoms * 2,
          owner_dev,
          nep_data.Fp.data(),
          nep_data.sum_fxyz.data(),
#ifdef USE_TABLE
          nep_data.gn_angular.data(),
          nep_data.gnp_angular.data(),
#endif
      lmp_force_f.data(),
      lmp_force_f.data() + natoms,
      lmp_force_f.data() + natoms * 2,
      lmp_virial.data());
  } else {
    find_force_angular_lmp<double><<<grid_size_desc, BLOCK_SIZE, 0, stream>>>(
          this->paramb,
          this->annmb,
          natoms,
          nlocal,
          this->N1,
          this->N2,
          box,
          NN_angular_dev,
          NL_angular_dev,
          type_dev,
          lmp_pos_dev.data(),
          lmp_pos_dev.data() + natoms,
          lmp_pos_dev.data() + natoms * 2,
          owner_dev,
          nep_data.Fp.data(),
          nep_data.sum_fxyz.data(),
#ifdef USE_TABLE
          nep_data.gn_angular.data(),
          nep_data.gnp_angular.data(),
#endif
      lmp_force.data(),
      lmp_force.data() + natoms,
      lmp_force.data() + natoms * 2,
      lmp_virial.data());
  }
  GPU_CHECK_KERNEL

  if (this->zbl.enabled) {
    if (use_fp32_force) {
      find_force_ZBL<float><<<grid_size_desc, BLOCK_SIZE, 0, stream>>>(
        this->paramb,
        natoms,
        nlocal,
        this->zbl,
        this->N1,
        this->N2,
        box,
        NN_angular_dev,
        NL_angular_dev,
        type_dev,
        lmp_pos_dev.data(),
        lmp_pos_dev.data() + natoms,
        lmp_pos_dev.data() + natoms * 2,
        owner_dev,
        lmp_force_f.data(),
        lmp_force_f.data() + natoms,
        lmp_force_f.data() + natoms * 2,
        lmp_virial.data(),
        lmp_potential.data());
    } else {
      find_force_ZBL<double><<<grid_size_desc, BLOCK_SIZE, 0, stream>>>(
        this->paramb,
        natoms,
        nlocal,
        this->zbl,
        this->N1,
        this->N2,
        box,
        NN_angular_dev,
        NL_angular_dev,
        type_dev,
        lmp_pos_dev.data(),
        lmp_pos_dev.data() + natoms,
        lmp_pos_dev.data() + natoms * 2,
        owner_dev,
        lmp_force.data(),
        lmp_force.data() + natoms,
        lmp_force.data() + natoms * 2,
        lmp_virial.data(),
        lmp_potential.data());
    }
    GPU_CHECK_KERNEL
  }

  // Scatter forces into AoS output if requested (accumulating).
  if (force_aos_dev) {
    const int block = 256;
    const int grid = (force_n + block - 1) / block;
    if (use_fp32_force) {
      scatter_force_soa_f_to_aos_add<<<grid, block, 0, stream>>>(
        force_n,
        lmp_force_f.data(),
        lmp_force_f.data() + force_n,
        lmp_force_f.data() + force_n * 2,
        force_aos_dev);
    } else {
      scatter_force_soa_to_aos_add<<<grid, block, 0, stream>>>(
        force_n,
        lmp_force.data(),
        lmp_force.data() + force_n,
        lmp_force.data() + force_n * 2,
        force_aos_dev);
    }
    GPU_CHECK_KERNEL
  }

  // Export per-atom outputs if requested.
  if (need_potential) {
    CHECK(gpuMemcpy(potential_dev, lmp_potential.data(), sizeof(double) * nlocal, gpuMemcpyDeviceToDevice));
  }
  if (need_virial_atom) {
    const int block = 256;
    const int grid = (nlocal + block - 1) / block;
    virial_soa_to_aos9<<<grid, block, 0, stream>>>(nlocal, lmp_virial.data(), virial_aos_dev);
    GPU_CHECK_KERNEL
  }

  // Reduce totals on device and copy just 7 scalars back to host (avoid sync on force-only steps).
  if (need_energy || need_virial) {
    lmp_totals.resize(7);
    fill_vector_async(lmp_totals, 0.0, stream);
    {
      const int block = 256;
      const int grid = (nlocal + block - 1) / block;
      reduce_ev_totals<<<grid, block, 0, stream>>>(nlocal, need_energy ? lmp_potential.data() : nullptr, need_virial ? lmp_virial.data() : nullptr, lmp_totals.data());
      GPU_CHECK_KERNEL
    }

    double host7[7];
    lmp_totals.copy_to_host(host7);
    eng_out = need_energy ? host7[0] : 0.0;
    virial_out[0] = need_virial ? host7[1] : 0.0;
    virial_out[1] = need_virial ? host7[2] : 0.0;
    virial_out[2] = need_virial ? host7[3] : 0.0;
    virial_out[3] = need_virial ? host7[4] : 0.0;
    virial_out[4] = need_virial ? host7[5] : 0.0;
    virial_out[5] = need_virial ? host7[6] : 0.0;
  }

  // For the LAMMPS/Kokkos integration this function is called from a Kokkos pair style.
  // Kokkos may use a non-default CUDA stream internally, while this backend launches
  // raw CUDA kernels. We must guarantee all device work is complete before returning,
  // otherwise subsequent Kokkos kernels may read partially-updated forces.
  //
  // We synchronize the whole device here for correctness. The cost is typically small
  // relative to NEP evaluation, and it also ensures illegal-address errors surface here.
  const bool force_sync = (std::getenv("NEP_GPU_LMP_SYNC") != nullptr);
  const bool disable_sync = (std::getenv("NEP_GPU_LMP_NO_SYNC") != nullptr);
  if (!disable_sync && (force_sync || stream == 0)) {
    CHECK(stream_synchronize(stream));
  }
}
