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

#include "fix_tspin_precession_spin_kokkos.h"

#ifdef LMP_KOKKOS

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "input.h"
#include "modify.h"
#include "update.h"
#include "variable.h"

#include <cmath>

using namespace LAMMPS_NS;

template <class DeviceType>
FixTSPINPrecessionSpinKokkos<DeviceType>::FixTSPINPrecessionSpinKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixTSPINPrecessionSpin(lmp, narg, arg)
{
  if (lmp->kokkos == nullptr || lmp->atomKK == nullptr)
    error->all(
        FLERR,
        "Fix {} (Kokkos) requires Kokkos to be enabled at runtime (use '-k on ...' or 'package kokkos', and do not use '-sf kk' by itself)",
        style);

  kokkosable = 1;
  atomKK = dynamic_cast<AtomKokkos *>(atom);
  if (!atomKK) error->all(FLERR, "Fix {} requires atom_style spin/kk (or spin with Kokkos enabled)", style);
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
}

template <class DeviceType>
void FixTSPINPrecessionSpinKokkos<DeviceType>::init()
{
  FixTSPINPrecessionSpin::init();
}

template <class DeviceType>
void FixTSPINPrecessionSpinKokkos<DeviceType>::post_force(int /*vflag*/)
{
  update_variable_field_if_needed();

  atomKK->sync(execution_space, MASK_MASK | SP_MASK | FM_MASK);

  const int nlocal = atom->nlocal;
  const int groupbit_ = groupbit;
  const double hx_ = hx;
  const double hy_ = hy;
  const double hz_ = hz;

  auto mask_ = atomKK->k_mask.template view<DeviceType>();
  auto fm_ = atomKK->k_fm.template view<DeviceType>();
  auto sp_ = atomKK->k_sp.template view<DeviceType>();

  eflag = 0;
  double eprec_local = 0.0;

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<DeviceType>(0, nlocal),
      LAMMPS_LAMBDA(const int i, double &eacc) {
        if (!(mask_(i) & groupbit_)) return;

        fm_(i, 0) += hx_;
        fm_(i, 1) += hy_;
        fm_(i, 2) += hz_;

        // E = -H · M, with M = sp[3] * sp_dir and H in eV/muB
        const double mu = static_cast<double>(sp_(i, 3));
        const double sx = static_cast<double>(sp_(i, 0));
        const double sy = static_cast<double>(sp_(i, 1));
        const double sz = static_cast<double>(sp_(i, 2));
        eacc -= mu * (sx * hx_ + sy * hy_ + sz * hz_);
      },
      eprec_local);

  eprec = eprec_local;

  atomKK->modified(execution_space, FM_MASK);
}

template <class DeviceType>
void FixTSPINPrecessionSpinKokkos<DeviceType>::min_post_force(int vflag)
{
  post_force(vflag);
}

namespace LAMMPS_NS {
template class FixTSPINPrecessionSpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixTSPINPrecessionSpinKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

#endif
