/*
  NEP_Spin_LMP: LAMMPS-direct GPU backend for spin-enabled NEP.

  This translation unit reuses the core NEP_Spin CUDA kernels (descriptors, ANN,
  forces, mforce), but provides a
  standalone entry point that consumes externally-built neighbor lists (from
  LAMMPS/Kokkos).

  NOTE: This header is intended to be included only by CUDA translation units
  (e.g. `nep_gpu_model_spin_lmp.cu`). Do not include it directly from LAMMPS
  sources compiled with a non-CUDA compiler.
*/

#pragma once

#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

// Minimal NEP_Spin type/ANN definitions required by the vendored kernels that are
// embedded into `NEP_GPU/src/nep_spin_lmp_bridge.cu`.
// Keep this header self-contained (no dependency on GPUMD's MD-side `src/force/nep_spin.*`).
class NEP_Spin
{
public:
  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_radial_factor = 0.0f;
    float typewise_cutoff_angular_factor = 0.0f;
    float typewise_cutoff_zbl_factor = 0.0f;
    float rc_radial = 0.0f;
    float rc_angular = 0.0f;
    float rcinv_radial = 0.0f;
    float rcinv_angular = 0.0f;
    float rc_radial_by_type[NUM_ELEMENTS] = {0.0f};
    float rc_angular_by_type[NUM_ELEMENTS] = {0.0f};

    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular = 0;
    int num_L = 0;

    int num_types = 0;
    int num_types_sq = 0;

    int num_c_radial = 0;
    int num_c_angular = 0;
    int num_c_spin_2body = 0;
    int num_c_spin_3body = 0;
    int c_spin_2body_offset = 0;
    int c_spin_3body_offset = 0;
    int num_c_spin = 0;
    int spin_blocks = 0;
    int c_spin_offset = 0;
    int c_spin_block_stride = 0;

    int version = 4;
    int atomic_numbers[NUM_ELEMENTS];

    float mforce_sign = -1.0f;
    int spin_kmax_ex = 2;
    int spin_kmax_dmi = 0;
    int spin_kmax_ani = 0;
    int spin_kmax_sia = 0;
    int spin_pmax = 2;
    int spin_n_max = -1;
    int spin_ex_phi_mode = 0;
    int spin_onsite_basis_mode = 0;
    int n_max_spin_radial = 1;
    int basis_size_spin_radial = 0;
    int n_max_spin_angular = 3;
    int l_max_spin_angular = 3;
    int basis_size_spin_angular = 0;
  };

  struct ANN {
    int dim = 0;
    int num_neurons1 = 0;
    int num_para = 0;
    int num_para_ann = 0;

    const float* w0[NUM_ELEMENTS];
    const float* b0[NUM_ELEMENTS];
    const float* w1[NUM_ELEMENTS];
    const float* b1;
    const float* c;
  };
};

class NEP_Spin_LMP
{
public:
  NEP_Spin_LMP(const char* file_potential, int max_atoms);
  ~NEP_Spin_LMP();

  float get_rc_radial() const;
  float get_rc_angular() const;
  int get_num_types() const;
  int get_MN_radial() const;
  int get_MN_angular() const;
  int get_descriptor_dim() const;
  int get_current_natoms() const;
  bool has_last_descriptors() const;
  void copy_last_descriptors_to_host(float* host_data, size_t size);
  void copy_last_Fp_to_host(float* host_data, size_t size);
  void copy_q_scaler_to_host(float* host_data, size_t size);

  // LAMMPS-direct CUDA entry:
  // - Neighbor lists are built externally for local atoms [0,nlocal) only.
  // - Incoming NN/NL use a stride = nlocal, i.e. index = i + nlocal*slot.
  //   The backend repacks them to its internal natoms-stride layout.
  // - `xyz_aos_dev`: device AoS positions (x0,y0,z0,x1,y1,z1,...) length 3*natoms.
  // - `sp4_aos_dev`: device AoS spins (spx,spy,spz,spmag) length 4*natoms.
  // - Accumulates into `force_aos_dev` and `fm_aos_dev` (AoS length 3*natoms) if non-null.
  void compute_with_neighbors_device(
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
    double* potential_dev,  // optional, length nlocal
    double* virial_aos_dev, // optional, AoS9 length 9*nlocal
    bool need_energy,
    bool need_virial,
    double& eng_out,
    double virial_out[6],
    double* virial_raw9_out); // optional, length 9, raw (non-symmetrized) tensor totals

private:
  using ParaMB = NEP_Spin::ParaMB;
  using ANN = NEP_Spin::ANN;

  struct NepData {
    GPU_Vector<float> q_scaler;
    GPU_Vector<float> parameters;
    GPU_Vector<float> descriptors; // length dim*natoms (SoA)
    GPU_Vector<float> Fp;          // length dim*natoms (SoA)
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> sum_fxyz_0;
    GPU_Vector<float> sum_fxyz_c;
    GPU_Vector<float> sum_fxyz_Ax;
    GPU_Vector<float> sum_fxyz_Ay;
    GPU_Vector<float> sum_fxyz_Az;
    GPU_Vector<float> sum_fxyz_D;
  };

  int spin_mode_{0};
  int max_atoms_{0};
  int current_natoms_{0};
  int MN_radial_{0};
  int MN_angular_{0};
  bool last_descriptors_valid_{false};

  ParaMB paramb_{};
  ANN annmb_{};
  NepData nep_data_{};
  float spin_mref_host_[NUM_ELEMENTS] = {0.0f};

  // Persistent device buffers (LAMMPS-direct path)
  GPU_Vector<double> pos_soa_;   // 3*natoms
  GPU_Vector<float>  spin_soa_;  // 3*natoms, physical spin vector S
  GPU_Vector<int>    nn_radial_;
  GPU_Vector<int>    nn_angular_;
  GPU_Vector<int>    nl_radial_;
  GPU_Vector<int>    nl_angular_;
  GPU_Vector<float>  x12_r_, y12_r_, z12_r_; // natoms*MN_radial
  GPU_Vector<float>  x12_a_, y12_a_, z12_a_; // natoms*MN_angular
  GPU_Vector<double> potential_; // natoms (only [0,nlocal) meaningful)
  GPU_Vector<double> force_soa_; // 3*natoms (SoA)
  GPU_Vector<double> virial_soa_; // 9*natoms (SoA)
  GPU_Vector<double> mforce_soa_; // 3*natoms (SoA)
  GPU_Vector<double> totals_;     // 7 scalars: eng, vxx, vyy, vzz, vxy, vxz, vyz
  GPU_Vector<double> totals_raw9_; // 9 scalars: raw tensor totals (xx..zy)

  void read_potential_file(const char* file_potential);
  void update_potential(float* parameters, ANN& ann);
  void ensure_capacity(int natoms);
};
