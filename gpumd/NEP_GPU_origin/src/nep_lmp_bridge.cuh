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
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

struct NEP_Data {
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
#ifdef USE_TABLE
  GPU_Vector<float> gn_radial;   // tabulated gn_radial functions
  GPU_Vector<float> gnp_radial;  // tabulated gnp_radial functions
  GPU_Vector<float> gn_angular;  // tabulated gn_angular functions
  GPU_Vector<float> gnp_angular; // tabulated gnp_angular functions
#endif
};

class NEP : public Potential
{
public:
  NEP_Data nep_data;
  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_radial_factor = 0.0f;
    float typewise_cutoff_angular_factor = 0.0f;
    float typewise_cutoff_zbl_factor = 0.0f;
    int version = 4; // NEP version, 3 for NEP3 and 4 for NEP4
    int model_type =
      0; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
    float rc_radial = 0.0f;     // max radial cutoff
    float rc_angular = 0.0f;    // max angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;  // for nep3
    int basis_size_angular = 8; // for nep3
    int num_types_sq = 0;       // for nep3
    int num_c_radial = 0;       // for nep3
    int num_types = 0;
    float q_scaler[140];
    int atomic_numbers[NUM_ELEMENTS];
  };

  struct ANN {
    int dim = 0;                   // dimension of the descriptor
    int num_neurons1 = 0;          // number of neurons in the 1st hidden layer
    int num_para = 0;              // number of parameters
    int num_para_ann = 0;          // number of parameters for the ANN part
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* b1;               // bias for the output layer
    const float* c;
    // for the scalar part of polarizability
    const float* w0_pol[10];
    const float* b0_pol[10];
    const float* w1_pol[10];
    const float* b1_pol;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
    int num_types;
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  struct Small_Box_Data {
        GPU_Vector<int> NN_radial;
        GPU_Vector<int> NL_radial;
        GPU_Vector<int> NN_angular;
        GPU_Vector<int> NL_angular;
        GPU_Vector<float> r12;
    } small_box_data;

  NEP(const char* file_potential, const int num_atoms);
  virtual ~NEP(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  virtual void compute(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  const GPU_Vector<int>& get_NN_radial_ptr();

  const GPU_Vector<int>& get_NL_radial_ptr();

  // Simple accessors for NEP_GPU wrapper
  float get_rc_radial() const { return paramb.rc_radial; }
  float get_rc_angular() const { return paramb.rc_angular; }
  float get_zbl_rc_outer_max() const { return zbl_rc_outer_max_; }
  int get_num_types() const { return paramb.num_types; }
  int get_MN_radial() const { return paramb.MN_radial; }
  int get_MN_angular() const { return paramb.MN_angular; }

  // LAMMPS-direct entry (non-spin): uses externally built neighbor lists
  // for local atoms with compact stride = nlocal.
  void compute_with_neighbors(
    Box& box,
    int nlocal,
    int natoms,
    const int* type_host,
    const double* pos_soa_host, // SoA: x(0..N-1), y(...), z(...)
    const int* NN_radial_host,
    const int* NL_radial_host,   // length nlocal*MN_radial, index = i + nlocal*slot
    const int* NN_angular_host,
    const int* NL_angular_host,  // length nlocal*MN_angular, index = i + nlocal*slot
    double* potential_host, // length nlocal (only local atoms are needed)
    double* force_host,     // AoS length 3*natoms (local + ghost)
    double* virial_host);   // length 9*nlocal (only local atoms are needed)

  // Kokkos/CUDA-friendly entry: all large arrays are already on the GPU.
  // - type_dev: mapped NEP types, length natoms
  // - xyz_aos_dev: positions in AoS layout (x0,y0,z0,x1,y1,z1, ...), length 3*natoms
  // - neighbor lists: device pointers with compact stride = nlocal (same as host API)
  // - force_aos_dev: if non-null, adds forces into AoS layout (length 3*natoms)
  // - potential_dev: if non-null, writes per-atom energies for local atoms (length nlocal)
  // - virial_aos_dev: if non-null, writes per-atom virial for local atoms (AoS 9* nlocal)
  //
  // Totals are returned on the host via eng_out and virial_out[6].
  void compute_with_neighbors_device(
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
    double virial_out[6]);


private:
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  ExpandedBox ebox;
  // DFTD3 support stripped in bridge build
  struct DFTD3Stub {} dftd3;

  // LAMMPS-direct persistent buffers to avoid per-timestep cudaMalloc/cudaFree.
  GPU_Vector<int>    lmp_type_dev;
  GPU_Vector<double> lmp_pos_dev;     // SoA: x,y,z
  GPU_Vector<double> lmp_potential;   // natoms
  GPU_Vector<double> lmp_force;       // 3*natoms (SoA)
  GPU_Vector<float>  lmp_force_f;     // 3*natoms (SoA, optional fp32 force accumulation)
  GPU_Vector<double> lmp_virial;      // 9*natoms (SoA)
  GPU_Vector<double> lmp_totals;      // 7 values: eng, vxx, vyy, vzz, vxy, vxz, vyz
  GPU_Vector<int>    lmp_validate;    // 1 int flag (optional runtime validation)
  std::vector<double> lmp_force_tmp;  // 3*natoms (SoA on host)
  std::vector<double> lmp_virial_tmp; // 9*natoms (SoA on host)
  bool lmp_force_fp32_ = false;       // enabled via env var NEP_GPU_LMP_FORCE_FP32=1
  float zbl_rc_outer_max_ = 0.0f;     // max possible ZBL rc_outer (used for safe neighbor filtering)

  void update_potential(float* parameters, ANN& ann);
#ifdef USE_TABLE
  void construct_table(float* parameters);
#endif

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_small_box(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  bool has_dftd3 = false;
};
