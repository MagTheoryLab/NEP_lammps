/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This file is part of a user-contributed package (USER-NEP-GPU) and is
   distributed under the same GPL terms as LAMMPS.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   GPU NEP pair style (Kokkos-enabled).
   This implements pair_style nep/gpu/kk which offloads NEP evaluation to a
   separate GPU library based on GPUMD's NEP implementation, while using
   LAMMPS/Kokkos data on the GPU (no host staging).
------------------------------------------------------------------------- */

#ifdef LMP_KOKKOS
// `LMP_KOKKOS_GPU` is defined in `kokkos_type.h` based on the enabled Kokkos backend.
#include "kokkos_type.h"

#if defined(LMP_KOKKOS_GPU) || defined(KOKKOS_ENABLE_CUDA)

#ifdef PAIR_CLASS
// clang-format off
PairStyle(nep/gpu/kk,PairNEPGPUKokkos<LMPDeviceType>);
PairStyle(nep/gpu/kk/device,PairNEPGPUKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_PAIR_NEP_GPU_KOKKOS_H
#define LMP_PAIR_NEP_GPU_KOKKOS_H

#include "pair_nep_gpu.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairNEPGPUKokkos : public PairNEPGPU {
 public:
  PairNEPGPUKokkos(class LAMMPS *);
  ~PairNEPGPUKokkos() override;

  void compute(int, int) override;
  void init_style() override;

 private:
  typedef ArrayTypes<DeviceType> AT;
  typedef DeviceType device_type;

  class AtomKokkos *atomKK{nullptr};
  int neighflag{FULL};
  int eflag{0}, vflag{0};

  // Cached device buffers (resized as needed)
  Kokkos::View<int*, DeviceType> d_type_map;
  Kokkos::View<int*, DeviceType> d_type_mapped;
  Kokkos::View<double*, DeviceType> d_rc_radial_by_type;
  Kokkos::View<double*, DeviceType> d_rc_angular_by_type;

  Kokkos::View<int*, DeviceType> d_nn_radial;
  Kokkos::View<int*, DeviceType> d_nn_angular;
  Kokkos::View<int*, DeviceType> d_nl_radial;   // flattened: i + nlocal*slot
  Kokkos::View<int*, DeviceType> d_nl_angular;  // flattened: i + nlocal*slot

  Kokkos::View<int, DeviceType> d_overflow;

  // Optional single-rank helper: map ghost indices -> owning local index so we
  // can run the backend in "no-ghost" (N==Nloc) mode for better symmetry.
  Kokkos::View<int*, DeviceType> d_owner;

  // Optional per-atom outputs on device (only allocated when requested)
  Kokkos::View<double*, DeviceType> d_eatom;
  Kokkos::View<double*, DeviceType> d_vatom; // AoS 9*nlocal

  // Some LAMMPS/Kokkos builds use LayoutLeft for fixed-size arrays, while the
  // CUDA backend expects AoS buffers for xyz and f. Use these packed buffers
  // only when needed (layout != LayoutRight).
  Kokkos::View<double*, DeviceType> d_xyz_aos; // length 3*nall (optional)
  Kokkos::View<double*, DeviceType> d_f_aos;   // length 3*nall (optional)

  int cached_ntypes{0};
  int cached_mn_r{0};
  int cached_mn_a{0};
  int cached_nlocal{0};
  int cached_nall{0};

  // Neighbor packing cache: we only rebuild compact NN/NL when the LAMMPS
  // neighbor list is rebuilt (or when buffer sizes change).
  bool neighbors_packed_{false};
  void* packed_list_ptr_{nullptr};

  void ensure_device_maps();
  void ensure_device_buffers(int nlocal, int nall, int mn_r, int mn_a);
};

} // namespace LAMMPS_NS

#endif
#endif

#endif // LMP_KOKKOS_GPU || KOKKOS_ENABLE_CUDA
#endif // LMP_KOKKOS
