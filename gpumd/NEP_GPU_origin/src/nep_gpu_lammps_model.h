#pragma once

#include <string>
#include <vector>

enum class NepGpuModelKind {
  potential,
  dipole,
  polarizability,
  temperature,
  spin,
};

struct NepGpuModelInfo {
  NepGpuModelKind kind = NepGpuModelKind::potential;
  int version = 0;
  int num_types = 0;
  int mn_radial = 0;
  int mn_angular = 0;
  bool has_zbl = false;
  bool needs_spin = false;
  double rc_radial_max = 0.0;
  double rc_angular_max = 0.0;
  double zbl_outer_max = 0.0;
  std::vector<std::string> elements;
  std::vector<double> rc_radial_by_type;
  std::vector<double> rc_angular_by_type;
};

struct NepGpuLammpsSystemHost {
  int natoms = 0;
  const int* type = nullptr;
  const double* xyz = nullptr; // AoS, length 3*natoms
  const double* sp4 = nullptr; // AoS, length 4*natoms
  double h[9] = {0.0};
  int pbc_x = 1;
  int pbc_y = 1;
  int pbc_z = 1;
};

struct NepGpuLammpsSystemDevice {
  int natoms = 0;
  const int* type = nullptr;
  const double* xyz = nullptr; // AoS, length 3*natoms
  const double* sp4 = nullptr; // AoS, length 4*natoms
  const int* owner = nullptr;  // optional owner map for non-spin backend
  void* stream = nullptr;
  double h[9] = {0.0};
  int pbc_x = 1;
  int pbc_y = 1;
  int pbc_z = 1;
};

struct NepGpuLammpsNeighborsHost {
  const int* NN_radial = nullptr;   // length nlocal
  const int* NL_radial = nullptr;   // length nlocal * mn_radial, index = i + nlocal * slot
  const int* NN_angular = nullptr;  // length nlocal
  const int* NL_angular = nullptr;  // length nlocal * mn_angular, index = i + nlocal * slot
};

struct NepGpuLammpsNeighborsDevice {
  const int* NN_radial = nullptr;   // length nlocal
  const int* NL_radial = nullptr;   // length nlocal * mn_radial, index = i + nlocal * slot
  const int* NN_angular = nullptr;  // length nlocal
  const int* NL_angular = nullptr;  // length nlocal * mn_angular, index = i + nlocal * slot
};

struct NepGpuLammpsResultHost {
  double eng = 0.0;
  double virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double virial_raw9[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bool want_virial_raw9 = false;
  double* f = nullptr;     // AoS, length 3*natoms
  double* fm = nullptr;    // AoS, length 3*natoms
  double inv_hbar = 0.0;
  double* eatom = nullptr; // length nlocal
  double* vatom = nullptr; // AoS9, length 9*nlocal
};

struct NepGpuLammpsResultDevice {
  double eng = 0.0;
  double virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double virial_raw9[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bool want_virial_raw9 = false;
  double* f = nullptr;     // AoS, length 3*natoms
  double* fm = nullptr;    // AoS, length 3*natoms
  double inv_hbar = 0.0;
  double* eatom = nullptr; // length nlocal
  double* vatom = nullptr; // AoS9, length 9*nlocal
};

class NepGpuLammpsModel {
public:
  NepGpuLammpsModel(const char* file_potential, int max_atoms);
  ~NepGpuLammpsModel();

  const NepGpuModelInfo& info() const { return info_; }

  void compute_host(
    const NepGpuLammpsSystemHost& sys,
    int nlocal,
    const NepGpuLammpsNeighborsHost& nb,
    NepGpuLammpsResultHost& res,
    bool need_energy,
    bool need_virial);

  void compute_device(
    const NepGpuLammpsSystemDevice& sys,
    int nlocal,
    const NepGpuLammpsNeighborsDevice& nb,
    NepGpuLammpsResultDevice& res,
    bool need_energy,
    bool need_virial);

  bool debug_copy_last_spin_descriptors_host(std::vector<float>& out);
  bool debug_copy_last_spin_fp_host(std::vector<float>& out);
  bool debug_copy_spin_q_scaler_host(std::vector<float>& out);

private:
  struct Impl;
  Impl* impl_ = nullptr;
  NepGpuModelInfo info_;
};

void nep_gpu_lammps_set_device(int dev_id);
