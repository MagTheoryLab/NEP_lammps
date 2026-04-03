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
FixStyle(tspin/precession/spin/kk,FixTSPINPrecessionSpinKokkos<LMPDeviceType>);
FixStyle(tspin/precession/spin/kk/device,FixTSPINPrecessionSpinKokkos<LMPDeviceType>);
FixStyle(tspin/precession/spin/kk/host,FixTSPINPrecessionSpinKokkos<LMPHostType>);
#endif
// clang-format on
#else

#ifndef LMP_FIX_TSPIN_PRECESSION_SPIN_KOKKOS_H
#define LMP_FIX_TSPIN_PRECESSION_SPIN_KOKKOS_H

#ifdef LMP_KOKKOS

#include "fix_tspin_precession_spin.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomKokkos;

template <class DeviceType>
class FixTSPINPrecessionSpinKokkos : public FixTSPINPrecessionSpin, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixTSPINPrecessionSpinKokkos(class LAMMPS *, int, char **);
  ~FixTSPINPrecessionSpinKokkos() override = default;

  void init() override;
  void post_force(int) override;
  void min_post_force(int) override;

 protected:
  AtomKokkos *atomKK;
};

}    // namespace LAMMPS_NS

#endif

#endif
#endif
