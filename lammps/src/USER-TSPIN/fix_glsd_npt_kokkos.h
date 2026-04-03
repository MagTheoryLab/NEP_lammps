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
FixStyle(glsd/npt/kk,FixGLSDNPTKokkos<LMPDeviceType>);
FixStyle(glsd/npt/kk/device,FixGLSDNPTKokkos<LMPDeviceType>);
FixStyle(glsd/npt/kk/host,FixGLSDNPTKokkos<LMPHostType>);
#endif
// clang-format on
#else

#ifndef LMP_FIX_GLSD_NPT_KOKKOS_H
#define LMP_FIX_GLSD_NPT_KOKKOS_H

#ifdef LMP_KOKKOS

#include "fix_glsd_nh_kokkos.h"

namespace LAMMPS_NS {

template <class DeviceType>
class FixGLSDNPTKokkos : public FixGLSDNHKokkos<DeviceType> {
 public:
  FixGLSDNPTKokkos(class LAMMPS *, int, char **);
};

}    // namespace LAMMPS_NS

#endif

#endif
#endif

