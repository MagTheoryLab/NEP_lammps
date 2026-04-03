// clang-format off
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
// clang-format on

#include "fix_tspin_precession_spin.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "input.h"
#include "math_const.h"
#include "modify.h"
#include "respa.h"
#include "update.h"
#include "utils.h"
#include "variable.h"

#include "mpi.h"

#include <cmath>
#include <cstring>

#ifdef LMP_KOKKOS
#include "atom_kokkos.h"
#endif

using namespace LAMMPS_NS;
using namespace FixConst;

static constexpr double MUB_EV_PER_T = 5.78901e-5;

/* ---------------------------------------------------------------------- */

FixTSPINPrecessionSpin::FixTSPINPrecessionSpin(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  dynamic_group_allow = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  zeeman_flag = 0;

  magstr = nullptr;
  magfieldstyle = CONSTANT;
  magvar = -1;
  varflag = CONSTANT;
  tesla_value_logged = 0;

  H_input_raw = 0.0;
  H_field = 0.0;
  nhx = nhy = nhz = 0.0;
  hx = hy = hz = 0.0;

  eflag = 0;
  eprec = eprec_all = 0.0;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "zeeman") == 0) {
      if (iarg + 4 >= narg) error->all(FLERR, "Illegal fix {} command", style);
      zeeman_flag = 1;

      if (strncmp(arg[iarg + 1], "v_", 2) == 0) {
        magfieldstyle = EQUAL;
        delete[] magstr;
        magstr = utils::strdup(arg[iarg + 1] + 2);
      } else {
        H_input_raw = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        H_field = H_input_raw * MUB_EV_PER_T;
      }

      nhx = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      nhy = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      nhz = utils::numeric(FLERR, arg[iarg + 4], false, lmp);
      iarg += 5;
    } else {
      error->all(FLERR, "Illegal fix {} command", style);
    }
  }

  if (!zeeman_flag) error->all(FLERR, "Illegal fix {} command", style);

  const double norm2 = nhx * nhx + nhy * nhy + nhz * nhz;
  if (norm2 == 0.0) error->all(FLERR, "Illegal fix {} command", style);
  const double inorm = 1.0 / std::sqrt(norm2);
  nhx *= inorm;
  nhy *= inorm;
  nhz *= inorm;
}

FixTSPINPrecessionSpin::~FixTSPINPrecessionSpin()
{
  delete[] magstr;
}

/* ---------------------------------------------------------------------- */

int FixTSPINPrecessionSpin::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTSPINPrecessionSpin::init()
{
  if (!atom->sp_flag) error->all(FLERR, "Fix {} requires atom_style spin", style);
  if (!atom->fm) error->all(FLERR, "Fix {} requires atom_style spin with fm allocated", style);

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }

  if (magstr) {
    magvar = input->variable->find(magstr);
    if (magvar < 0) error->all(FLERR, "Illegal fix {} command", style);
    if (!input->variable->equalstyle(magvar)) error->all(FLERR, "Illegal fix {} command", style);
  }

  varflag = CONSTANT;
  if (magfieldstyle != CONSTANT) varflag = EQUAL;

  if (varflag == CONSTANT) set_field_components();

  if (comm->me == 0) {
    if (varflag == CONSTANT) {
      utils::logmesg(lmp,
                     "Fix {}: Zeeman input = {} T, converted to {} eV/muB (muB = {} eV/T)\n",
                     style, H_input_raw, H_field, MUB_EV_PER_T);
      tesla_value_logged = 1;
    } else {
      utils::logmesg(lmp,
                     "Fix {}: Zeeman input uses Tesla via v_{}, with H[eV/muB] = H[T] * {}\n",
                     style, magstr ? magstr : "", MUB_EV_PER_T);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixTSPINPrecessionSpin::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    (dynamic_cast<Respa *>(update->integrate))->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag, ilevel_respa, 0);
    (dynamic_cast<Respa *>(update->integrate))->copy_f_flevel(ilevel_respa);
  }
}

/* ---------------------------------------------------------------------- */

void FixTSPINPrecessionSpin::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixTSPINPrecessionSpin::set_field_components()
{
  if (!zeeman_flag) return;
  hx = H_field * nhx;
  hy = H_field * nhy;
  hz = H_field * nhz;
}

/* ---------------------------------------------------------------------- */

void FixTSPINPrecessionSpin::update_variable_field_if_needed()
{
  if (varflag == CONSTANT) return;

  modify->clearstep_compute();
  modify->addstep_compute(update->ntimestep + 1);
  H_input_raw = input->variable->compute_equal(magvar);
  H_field = H_input_raw * MUB_EV_PER_T;
  set_field_components();

  if (!tesla_value_logged && comm->me == 0) {
    utils::logmesg(lmp, "Fix {}: first evaluated Zeeman field = {} T -> {} eV/muB at step {}\n", style, H_input_raw,
                   H_field, update->ntimestep);
    tesla_value_logged = 1;
  }
}

/* ---------------------------------------------------------------------- */

void FixTSPINPrecessionSpin::post_force(int /*vflag*/)
{
  update_variable_field_if_needed();

#ifdef LMP_KOKKOS
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) atomKK->sync(Host, SP_MASK | FM_MASK);
#endif

  int *mask = atom->mask;
  double **fm = atom->fm;
  double **sp = atom->sp;
  const int nlocal = atom->nlocal;

  eflag = 0;
  eprec = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    fm[i][0] += hx;
    fm[i][1] += hy;
    fm[i][2] += hz;

    // E = -H · M, with M = sp[3] * sp_dir and H in eV/muB
    eprec -= sp[i][3] * (sp[i][0] * hx + sp[i][1] * hy + sp[i][2] * hz);
  }

#ifdef LMP_KOKKOS
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) atomKK->modified(Host, FM_MASK);
#endif
}

/* ---------------------------------------------------------------------- */

double FixTSPINPrecessionSpin::compute_scalar()
{
  if (eflag == 0) {
    MPI_Allreduce(&eprec, &eprec_all, 1, MPI_DOUBLE, MPI_SUM, world);
    eflag = 1;
  }
  return eprec_all;
}

/* ---------------------------------------------------------------------- */

void FixTSPINPrecessionSpin::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixTSPINPrecessionSpin::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}
