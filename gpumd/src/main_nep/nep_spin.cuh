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

#pragma once
#include <cuda_runtime.h>
#include "precision.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
class Parameters;
class Dataset;

struct NEP_Spin_Data {
  GPU_Vector<int> NN_radial;  // radial neighbor number
  GPU_Vector<int> NL_radial;  // radial neighbor list
  GPU_Vector<int> NN_angular; // angular neighbor number
  GPU_Vector<int> NL_angular; // angular neighbor list
  GPU_Vector<StructReal> x12_radial;
  GPU_Vector<StructReal> y12_radial;
  GPU_Vector<StructReal> z12_radial;
  GPU_Vector<StructReal> x12_angular;
  GPU_Vector<StructReal> y12_angular;
  GPU_Vector<StructReal> z12_angular;
  GPU_Vector<SpinReal> descriptors; // descriptors
  GPU_Vector<SpinReal> Fp;          // gradient of descriptors
  GPU_Vector<StructReal> sum_fxyz;  // baseline angular helper
  GPU_Vector<SpinReal> sum_fxyz_0;  // Block2 structural accumulators
  GPU_Vector<SpinReal> sum_fxyz_c;  // Block2 center-neighbor scalar accumulators
  GPU_Vector<SpinReal> sum_fxyz_Ax; // Block2 neighbor spin-x accumulators
  GPU_Vector<SpinReal> sum_fxyz_Ay; // Block2 neighbor spin-y accumulators
  GPU_Vector<SpinReal> sum_fxyz_Az; // Block2 neighbor spin-z accumulators
  GPU_Vector<SpinReal> sum_fxyz_D;  // Block2 DMI-weighted accumulators
  GPU_Vector<float> parameters;  // parameters to be optimized
};

class NEP_Spin : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.65f;
    float rc_radial[NUM_ELEMENTS] = {0.0f};  // radial cutoff
    float rc_angular[NUM_ELEMENTS] = {0.0f}; // angular cutoff
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;  // n = 0..n_max_radial
    int n_max_angular = 0; // n = 0..n_max_angular
    int L_max = 0;         // l = 1..L_max
    int dim_angular;
    int num_L;
    int num_types = 0;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_c_angular = 0;
    int num_c_spin_2body = 0;
    int num_c_spin_3body = 0;
    int c_spin_2body_offset = 0;
    int c_spin_3body_offset = 0;
    int version = 4;
    int atomic_numbers[NUM_ELEMENTS];
    float mforce_sign = -1.0f; // magnetic force sign: -dE/ds by default
    int spin_pmax = 2;              // on-site longitudinal order (p=1..spin_pmax)
    int spin_onsite_basis_mode = 0; // on-site basis mode (see Parameters::spin_onsite_basis_mode)
    float spin_mref[NUM_ELEMENTS] = {0.0f}; // per-type reference magnitude for on-site mapping
    int n_max_spin_radial = 1;
    int basis_size_spin_radial = 0;
    int n_max_spin_angular = 3;
    int l_max_spin_angular = 3;
    int basis_size_spin_angular = 0;
  };

  struct ANN {
    int dim = 0;             // descriptor dimension
    int num_neurons1 = 0;    // hidden neurons
    int num_para = 0;        // number of parameters
    const float* w0[NUM_ELEMENTS];
    const float* b0[NUM_ELEMENTS];
    const float* w1[NUM_ELEMENTS];
    const float* b1;  // bias for output
    const float* c;   // descriptor mixing coefficients
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    int num_types;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
  };

  NEP_Spin(
    Parameters& para,
    int N,
    int N_times_max_NN_radial,
    int N_times_max_NN_angular,
    int version,
    int deviceCount);

  void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    bool calculate_neighbor,
    int deviceCount) override;

private:
  ParaMB paramb;
  ANN annmb[16];
  NEP_Spin_Data nep_data[16];
  ZBL zbl;
  void update_potential(Parameters& para, float* parameters, ANN& ann);
};

