/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This file is part of a user-contributed package (USER-NEP-GPU) and is
   distributed under the same GPL terms as LAMMPS.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   GPU NEP spin pair style (Kokkos-enabled).
   Implements pair_style nep/spin/gpu/kk and runs entirely on the device:
   - reads x/type/sp(4) from Kokkos atom arrays
   - builds compact NN/NL from LAMMPS full neighbor list
   - calls a CUDA backend (GPUMD/NEP spin kernels) that returns forces + fm
------------------------------------------------------------------------- */

#ifdef LMP_KOKKOS
#include "kokkos_type.h"

#if defined(LMP_KOKKOS_GPU) || defined(KOKKOS_ENABLE_CUDA)

#ifdef PAIR_CLASS
// clang-format off
PairStyle(nep/spin/gpu/kk,PairNEPSpinGPUKokkos<LMPDeviceType>);
PairStyle(nep/spin/gpu/kk/device,PairNEPSpinGPUKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_PAIR_NEP_SPIN_GPU_KOKKOS_H
#define LMP_PAIR_NEP_SPIN_GPU_KOKKOS_H

#include "pair.h"

#include <string>
#include <type_traits>

class NepGpuLammpsModel;

namespace LAMMPS_NS {

template<class DeviceType>
class PairNEPSpinGPUKokkos : public Pair {
 public:
  PairNEPSpinGPUKokkos(class LAMMPS *);
  ~PairNEPSpinGPUKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 private:
  typedef ArrayTypes<DeviceType> AT;
  typedef DeviceType device_type;

  class AtomKokkos *atomKK{nullptr};
  int neighflag{FULL};
  int eflag{0}, vflag{0};

  // The backend returns mforce = -dE/dM in eV/mu_B and we expose that directly.
  double cutoff{0.0};
  double cutoffsq{0.0};
  int* type_map{nullptr}; // LAMMPS type -> NEP type

  NepGpuLammpsModel* nep_model_spin_lmp{nullptr};
  std::string model_filename_;

  // Device buffers (resized as needed)
  Kokkos::View<int*, DeviceType> d_type_map;
  Kokkos::View<int*, DeviceType> d_type_mapped;

  // Compact NN/NL buffers use nlocal stride: i + nlocal*slot.
  Kokkos::View<int*, DeviceType> d_nn_radial;   // length nlocal
  Kokkos::View<int*, DeviceType> d_nn_angular;  // length nlocal
  Kokkos::View<int*, DeviceType> d_nl_radial;   // flattened: i + nlocal*slot
  Kokkos::View<int*, DeviceType> d_nl_angular;  // flattened: i + nlocal*slot
  Kokkos::View<int, DeviceType> d_overflow;

  // Optional per-atom outputs on device (only allocated when requested)
  Kokkos::View<double*, DeviceType> d_eatom;
  Kokkos::View<double*, DeviceType> d_vatom; // AoS 9*nlocal

  // Some LAMMPS/Kokkos builds use LayoutLeft for fixed-size arrays (e.g. when
  // LMP_KOKKOS_NO_LEGACY is enabled), while our CUDA backend expects AoS
  // buffers for xyz and sp(4). These optional packed buffers are used only
  // when needed (layout != LayoutRight).
  Kokkos::View<double*, DeviceType> d_xyz_aos; // length 3*nall
  Kokkos::View<double*, DeviceType> d_sp4_aos; // length 4*nall
  // Likewise, some builds may store forces in LayoutLeft; the backend writes AoS.
  // When layout is not AoS, compute into these temporary AoS buffers and scatter-add.
  Kokkos::View<double*, DeviceType> d_f_aos;  // length 3*nall (optional)
  Kokkos::View<double*, DeviceType> d_fm_aos; // length 3*nall (optional)

  int cached_ntypes{0};
  int cached_mn_r{0};
  int cached_mn_a{0};
  int cached_nlocal{0};
  int cached_nall{0};

  bool neighbors_packed_{false};
  void* packed_list_ptr_{nullptr};

  void allocate();
  void allocate_type_map();
  void ensure_device_maps();
  void ensure_device_buffers(int nlocal, int nall, int mn_r, int mn_a);
};

} // namespace LAMMPS_NS

#endif

#endif // PAIR_CLASS

#endif // LMP_KOKKOS_GPU || KOKKOS_ENABLE_CUDA
#endif // LMP_KOKKOS
