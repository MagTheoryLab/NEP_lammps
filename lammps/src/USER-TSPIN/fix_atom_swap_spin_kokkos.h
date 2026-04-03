/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
#ifdef LMP_KOKKOS
FixStyle(atom/swap/spin/kk,FixAtomSwapSpinKokkos<LMPDeviceType>);
FixStyle(atom/swap/spin/kk/device,FixAtomSwapSpinKokkos<LMPDeviceType>);
FixStyle(atom/swap/spin/kk/host,FixAtomSwapSpinKokkos<LMPHostType>);
#endif
// clang-format on
#else

#ifndef LMP_FIX_ATOM_SWAP_SPIN_KOKKOS_H
#define LMP_FIX_ATOM_SWAP_SPIN_KOKKOS_H

#ifdef LMP_KOKKOS

#include "fix_atom_swap_spin.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

#include <Kokkos_Core.hpp>

namespace LAMMPS_NS {

class AtomKokkos;

template <class DeviceType>
class FixAtomSwapSpinKokkos : public FixAtomSwapSpin, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixAtomSwapSpinKokkos(class LAMMPS *, int, char **);
  ~FixAtomSwapSpinKokkos() override = default;

  void init() override;
  void pre_exchange() override;

  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_xfloat_1d &, int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_xfloat_1d &) override;

  // NOTE: These helpers must be public for NVCC "extended __host__ __device__ lambda" support.
  // NVCC rejects device lambdas inside private/protected member functions.
  void read_spin_local_device(int local_index, double sp_out[4]);
  void write_spin_local_device(int local_index, const double sp_in[4]);
  void write_type_local_device(int local_index, int new_type);
  void write_q_local_device(int local_index, double new_q);
  void write_rmass_local_device(int local_index, double new_rmass);
  void scale_v_local_device(int local_index, double factor);

  int attempt_swap();
  int attempt_semi_grand();

  double energy_full_kokkos();

 protected:
  AtomKokkos *atomKK;

  // Scratch buffers for reading 4 spin values from device without syncing full arrays.
  Kokkos::View<double[4], typename DeviceType::memory_space> d_sp4;
  Kokkos::View<double[4], Kokkos::HostSpace> h_sp4;
};

}    // namespace LAMMPS_NS

#endif

#endif
#endif
