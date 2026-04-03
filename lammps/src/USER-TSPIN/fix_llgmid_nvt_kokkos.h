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
FixStyle(llgmid/nvt/kk,FixLLGMidNVTKokkos<LMPDeviceType>);
FixStyle(llgmid/nvt/kk/device,FixLLGMidNVTKokkos<LMPDeviceType>);
FixStyle(llgmid/nvt/kk/host,FixLLGMidNVTKokkos<LMPHostType>);
#endif
// clang-format on
#else

#ifndef LMP_FIX_LLGMID_NVT_KOKKOS_H
#define LMP_FIX_LLGMID_NVT_KOKKOS_H

#ifdef LMP_KOKKOS

#include "fix_llgmid_nh_kokkos.h"

namespace LAMMPS_NS {

template <class DeviceType>
class FixLLGMidNVTKokkos : public FixLLGMidNHKokkos<DeviceType> {
 public:
  FixLLGMidNVTKokkos(class LAMMPS *, int, char **);
};

}    // namespace LAMMPS_NS

#endif

#endif
#endif

