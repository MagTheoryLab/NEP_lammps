/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_llggeom_nvt_kokkos.h"

#ifdef LMP_KOKKOS

#include "error.h"
#include "group.h"
#include "modify.h"
#include "utils.h"

using namespace LAMMPS_NS;

template <class DeviceType>
FixLLGGeomNVTKokkos<DeviceType>::FixLLGGeomNVTKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixLLGGeomNHKokkos<DeviceType>(lmp, narg, arg)
{
  this->kokkosable = 1;
  if (!this->tstat_flag) this->error->all(FLERR, "Temperature control must be used with fix {}", this->style);
  if (this->pstat_flag) this->error->all(FLERR, "Pressure control can not be used with fix {}", this->style);

  this->id_temp = utils::strdup(std::string(this->id) + "_temp");
  this->modify->add_compute(fmt::format("{} {} temp/kk", this->id_temp, this->group->names[this->igroup]));
  this->tcomputeflag = 1;
}

namespace LAMMPS_NS {
template class FixLLGGeomNVTKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixLLGGeomNVTKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

#endif
