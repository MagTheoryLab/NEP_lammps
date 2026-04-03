/*
   LAMMPS neighbor-direct NEP Spin GPU wrapper.
   Exposes a minimal, CUDA-free C++ API for LAMMPS pair styles.
*/

#pragma once

// Keep this header free of CUDA/GPUMD includes so it can be included by LAMMPS
// translation units compiled with a non-CUDA compiler.

// Host-side inputs/outputs for a classic (non-Kokkos) LAMMPS pair style.
// These are staged to the GPU inside NepGpuModelSpinLmp.
struct NepGpuSpinSystem {
  int natoms = 0;
  const int* type = nullptr;     // host, length natoms (mapped NEP type indices)
  const double* xyz = nullptr;   // host, AoS length 3*natoms
  const double* sp4 = nullptr;   // host, AoS length 4*natoms: (spx, spy, spz, spmag)

  double h[9] = {0.0};
  int pbc_x = 1, pbc_y = 1, pbc_z = 1;
};

struct NepGpuSpinLmpNeighbors {
  // Neighbor lists are provided only for local atoms [0,nlocal) but use a
  // stride = natoms (LAMMPS atom indexing): index = i + natoms * slot.
  const int* NN_radial = nullptr;   // host, length natoms (only [0,nlocal) used)
  const int* NL_radial = nullptr;   // host, length natoms * MN_radial
  const int* NN_angular = nullptr;  // host, length natoms (only [0,nlocal) used)
  const int* NL_angular = nullptr;  // host, length natoms * MN_angular
};

struct NepGpuSpinResult {
  double eng = 0.0;
  double virial[6] = {0.0,0.0,0.0,0.0,0.0,0.0}; // xx,yy,zz,xy,xz,yz
  // Optional: raw (non-symmetrized) 3x3 virial tensor totals in SoA order:
  // [xx,yy,zz,xy,xz,yz,yx,zx,zy]. Only filled when want_virial_raw9=true.
  double virial_raw9[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  bool want_virial_raw9 = false;

  double* f = nullptr;   // host AoS, length 3*natoms (written)
  double* fm = nullptr;  // host AoS, length 3*natoms (written), units = 1/time

  double inv_hbar = 0.0; // scaling: fm += inv_hbar * mforce, where mforce = -dE/dS

  double* eatom = nullptr; // host, length nlocal (optional)
  double* vatom = nullptr; // host, AoS length 9*nlocal (optional)
};

struct NepGpuSpinDeviceSystem {
  int natoms = 0;
  const int* type = nullptr;     // device, length natoms (mapped NEP type indices)
  const double* xyz = nullptr;   // device, AoS length 3*natoms
  const double* sp4 = nullptr;   // device, AoS length 4*natoms: (spx, spy, spz, spmag)

  void* stream = nullptr;        // optional CUDA/HIP stream (nullptr -> default stream)

  double h[9] = {0.0};
  int pbc_x = 1, pbc_y = 1, pbc_z = 1;
};

struct NepGpuSpinLmpNeighborsDevice {
  // Neighbor lists are provided only for local atoms [0,nlocal) and use a
  // stride = natoms (LAMMPS atom indexing): index = i + natoms * slot.
  const int* NN_radial = nullptr;   // device, length natoms (only [0,nlocal) used)
  const int* NL_radial = nullptr;   // device, length natoms * MN_radial
  const int* NN_angular = nullptr;  // device, length natoms (only [0,nlocal) used)
  const int* NL_angular = nullptr;  // device, length natoms * MN_angular
};

struct NepGpuSpinDeviceResult {
  double eng = 0.0;
  double virial[6] = {0.0,0.0,0.0,0.0,0.0,0.0}; // xx,yy,zz,xy,xz,yz
  // Optional: raw (non-symmetrized) 3x3 virial tensor components in SoA order:
  // [xx,yy,zz,xy,xz,yz,yx,zx,zy]. Only filled when want_virial_raw9=true.
  double virial_raw9[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  bool want_virial_raw9 = false;

  double* f = nullptr;   // device AoS, length 3*natoms (accumulated into)
  double* fm = nullptr;  // device AoS, length 3*natoms (accumulated into), units = 1/time

  double inv_hbar = 0.0; // scaling: fm += inv_hbar * mforce, where mforce = -dE/dS

  double* eatom = nullptr; // device, length nlocal (optional)
  double* vatom = nullptr; // device, AoS length 9*nlocal (optional)
};

// Forward declaration (implemented in CUDA translation unit).
class NEP_Spin_LMP;

class NepGpuModelSpinLmp {
public:
  NepGpuModelSpinLmp(const char* file_potential, int max_atoms);
  ~NepGpuModelSpinLmp();

  void set_device(int dev_id);
  float cutoff() const;
  float cutoff_angular() const;
  int num_types() const;
  int mn_radial() const;
  int mn_angular() const;

  // Classic LAMMPS path: inputs/outputs are on the host and will be staged to the GPU.
  void compute_with_neighbors(
    const NepGpuSpinSystem& sys,
    int nlocal,
    const NepGpuSpinLmpNeighbors& nb,
    NepGpuSpinResult& res,
    bool need_energy,
    bool need_virial);

  void compute_with_neighbors_device(
    const NepGpuSpinDeviceSystem& sys,
    int nlocal,
    const NepGpuSpinLmpNeighborsDevice& nb,
    NepGpuSpinDeviceResult& res,
    bool need_energy,
    bool need_virial);

private:
  NEP_Spin_LMP* nep_ = nullptr;
  int cap_atoms_ = 0;

  // Host->device staging buffers for the classic (non-Kokkos) LAMMPS path.
  int staged_natoms_ = 0;
  int staged_nlocal_ = 0;
  int staged_mn_radial_ = 0;
  int staged_mn_angular_ = 0;
  bool staged_have_eatom_ = false;
  bool staged_have_vatom_ = false;
  int* d_type_ = nullptr;
  double* d_xyz_ = nullptr;
  double* d_sp4_ = nullptr;
  double* d_f_ = nullptr;
  double* d_fm_ = nullptr;
  int* d_nn_radial_ = nullptr;
  int* d_nn_angular_ = nullptr;
  int* d_nl_radial_ = nullptr;
  int* d_nl_angular_ = nullptr;
  double* d_eatom_ = nullptr;
  double* d_vatom_ = nullptr;

  void ensure_staging(int natoms, int nlocal, int mn_radial, int mn_angular, bool need_eatom, bool need_vatom);
};

// Set the active CUDA device for the NEP spin GPU backend.
// This is provided as a free function so it can be called from LAMMPS pair
// styles (compiled as C++), without relying on GPUMD's CHECK/gpuSetDevice macros.
void nep_spin_gpu_set_device(int dev_id);
