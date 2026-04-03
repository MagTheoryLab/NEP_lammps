/*
  Spin-enabled NEP runtime interface for GPUMD (GPU).
  This header declares NEP_Spin, a minimal MD-side wrapper
  around a spin-capable NEP model used by NEP_GPU.
*/

#pragma once

#include "model/box.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

class NEP_Spin : public Potential
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

    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;
    int spin_n_max = -1; // spin-block radial order: n = 0..spin_n_max (defaults to n_max_radial)
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular = 0;
    int num_L = 0;

    int num_types = 0;
    int num_types_sq = 0;

    int num_c_radial = 0;
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
    int spin_ex_phi_mode = 0;
    int spin_onsite_basis_mode = 0;
    float spin_mref = 1.0f;
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

  NEP_Spin(const char* file_potential, int max_atoms);
  ~NEP_Spin();

  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial) override;

  void compute_with_spin(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const GPU_Vector<double>& spin,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    GPU_Vector<double>& mforce);

  float get_rc_radial() const { return paramb_.rc_radial; }
  int get_num_types() const { return paramb_.num_types; }

private:
  struct NEP_Data {
    GPU_Vector<float> q_scaler;
    GPU_Vector<float> parameters;
    GPU_Vector<float> spin_buffer;

    GPU_Vector<int> NN_radial;
    GPU_Vector<int> NL_radial;
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<float> x12_radial;
    GPU_Vector<float> y12_radial;
    GPU_Vector<float> z12_radial;
    GPU_Vector<float> x12_angular;
    GPU_Vector<float> y12_angular;
    GPU_Vector<float> z12_angular;

    GPU_Vector<float> descriptors;
    GPU_Vector<float> Fp;
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> f12x;
    GPU_Vector<float> f12y;
    GPU_Vector<float> f12z;

    GPU_Vector<int> cell_count;
    GPU_Vector<int> cell_count_sum;
    GPU_Vector<int> cell_contents;

    int cap_radial_per_atom = 0;
    int cap_angular_per_atom = 0;
  };

  ParaMB paramb_;
  ANN annmb_;
  NEP_Data nep_data_;

  int spin_mode_ = 0;
  int max_atoms_ = 0;
  int current_natoms_ = 0;

  ExpandedBox ebox_;

  void read_potential_file(const char* file_potential);
  void update_potential(float* parameters, ANN& ann);
  void ensure_capacity(int natoms);

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const GPU_Vector<float>& spin,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    GPU_Vector<double>& mforce);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    const GPU_Vector<float>& spin,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    GPU_Vector<double>& mforce);
};
