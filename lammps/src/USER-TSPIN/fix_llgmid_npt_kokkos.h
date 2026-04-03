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
FixStyle(llgmid/npt/kk,FixLLGMidNPTKokkos<LMPDeviceType>);
FixStyle(llgmid/npt/kk/device,FixLLGMidNPTKokkos<LMPDeviceType>);
FixStyle(llgmid/npt/kk/host,FixLLGMidNPTKokkos<LMPHostType>);
#endif
// clang-format on
#else

#ifndef LMP_FIX_LLGMID_NPT_KOKKOS_H
#define LMP_FIX_LLGMID_NPT_KOKKOS_H

#ifdef LMP_KOKKOS

#include "fix_llgmid_nh_kokkos.h"

namespace LAMMPS_NS {

template <class DeviceType>
class FixLLGMidNPTKokkos : public FixLLGMidNHKokkos<DeviceType> {
 public:
  FixLLGMidNPTKokkos(class LAMMPS *, int, char **);
};

}    // namespace LAMMPS_NS

#endif

#endif
#endif

