/*
   LAMMPS neighbor-direct NEP GPU wrapper (non-spin).
   Wraps NEP class from nep_lmp_bridge.cuh and exposes a NepGpuModel-like API.
*/

#pragma once

#include <vector>

// Minimal "public" C++ structs used by the LAMMPS bridge.
// Keep this header free of CUDA/GPUMD includes so it can be included by LAMMPS
// translation units compiled with a non-CUDA compiler.
struct NepGpuSystem {
  int natoms = 0;
  const int* type = nullptr;
  const double* xyz = nullptr; // AoS length 3*natoms: x0,y0,z0,...

  double h[9] = {0.0};
  int pbc_x = 1, pbc_y = 1, pbc_z = 1;
};

struct NepGpuResult {
  double eng = 0.0;
  double virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // xx,yy,zz,xy,xz,yz
  double* f = nullptr;     // host, AoS length 3*natoms
  double* eatom = nullptr; // host, length nlocal (optional)
  double* vatom = nullptr; // host, AoS length 9*nlocal (optional)
};

// Forward declaration to avoid pulling GPUMD headers into LAMMPS compile units.
class NEP;

struct NepGpuLmpNeighbors {
  // Neighbor lists are provided only for local atoms [0,nlocal) and use a
  // compact stride = nlocal (not natoms). Neighbor indices themselves still
  // refer to the full [0,natoms) array (local + ghost).
  const int* NN_radial = nullptr;   // length nlocal
  const int* NL_radial = nullptr;   // length nlocal * MN_radial, index = i + nlocal * slot
  const int* NN_angular = nullptr;  // length nlocal
  const int* NL_angular = nullptr;  // length nlocal * MN_angular, index = i + nlocal * slot
};

// Device-side equivalents for Kokkos/CUDA integration.
// Pointers are expected to refer to GPU memory (same device as the NEP kernels).
struct NepGpuDeviceSystem {
  int natoms = 0;
  const int*    type = nullptr; // device, length natoms (mapped NEP type indices)
  const double* xyz  = nullptr; // device, AoS length 3*natoms (x0,y0,z0,...)

  // Optional mapping from atom index [0,natoms) -> local "owner" index [0,nlocal).
  // This can be used to collapse periodic ghost copies (single-rank) and enable
  // atomic-free force evaluation paths. If null, the backend assumes no mapping.
  const int* owner = nullptr; // device, length natoms

  // Optional CUDA/HIP stream to use for all NEP kernels. When null, the backend
  // uses the default stream. For LAMMPS/Kokkos integration this should be set
  // to the current Kokkos device stream so Kokkos fences order correctly.
  void* stream = nullptr;

  double h[9] = {0.0};
  int pbc_x = 1, pbc_y = 1, pbc_z = 1;
};

struct NepGpuLmpNeighborsDevice {
  const int* NN_radial = nullptr;   // device, length nlocal
  const int* NL_radial = nullptr;   // device, length nlocal * MN_radial, index = i + nlocal * slot
  const int* NN_angular = nullptr;  // device, length nlocal
  const int* NL_angular = nullptr;  // device, length nlocal * MN_angular, index = i + nlocal * slot
};

struct NepGpuDeviceResult {
  double eng = 0.0;
  double virial[6] = {0.0,0.0,0.0,0.0,0.0,0.0}; // xx,yy,zz,xy,xz,yz
  double* f      = nullptr; // device, AoS length 3*natoms (accumulated into)
  double* eatom  = nullptr; // device, length nlocal (optional)
  double* vatom  = nullptr; // device, AoS length 9*nlocal (optional)
};

class NepGpuModelLmp {
public:
  NepGpuModelLmp(const char* file_potential, int max_atoms);
  ~NepGpuModelLmp();

  void set_device(int dev_id);
  float cutoff() const;
  float cutoff_angular() const;
  float cutoff_zbl_outer() const;
  int num_types() const;
  int mn_radial() const;
  int mn_angular() const;

  // sys.natoms must equal nlocal+nghost; neighbor lists use compact indices [0,natoms).
  void compute_with_neighbors(
    const NepGpuSystem& sys,
    int nlocal,
    const NepGpuLmpNeighbors& nb,
    NepGpuResult& res);

  // Kokkos path: all inputs are already on the GPU; writes forces to GPU directly.
  void compute_with_neighbors_device(
    const NepGpuDeviceSystem& sys,
    int nlocal,
    const NepGpuLmpNeighborsDevice& nb,
    NepGpuDeviceResult& res,
    bool need_energy,
    bool need_virial);

private:
  NEP* nep_ = nullptr;
  int cap_atoms_ = 0;

  // Host-side persistent buffers to avoid per-timestep allocations in LAMMPS.
  std::vector<double> pos_soa_;
  std::vector<double> potential_scratch_; // nlocal
  std::vector<double> force_scratch_;     // 3*natoms (AoS)
  std::vector<double> virial_scratch_;    // 9*nlocal (AoS)
};

// Set the active CUDA device for the NEP GPU backend.
// This is provided as a free function so it can be called from LAMMPS pair
// styles (compiled as C++), without relying on GPUMD's CHECK/gpuSetDevice macros.
void nep_gpu_set_device(int dev_id);
