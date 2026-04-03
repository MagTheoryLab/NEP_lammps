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

#include "fix_atom_swap_spin.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "region.h"
#include "update.h"
#include "utils.h"

#ifdef LMP_KOKKOS
#include "fix_glsd_nh_kokkos.h"
#include "fix_tspin_nh_kokkos.h"
#endif

#include "mpi.h"

#include <cfloat>
#include <cctype>
#include <cmath>
#include <cstring>

#ifdef LMP_KOKKOS
#include "atom_kokkos.h"
#include "atom_masks.h"
#endif

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixAtomSwapSpin::FixAtomSwapSpin(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), region(nullptr), idregion(nullptr), type_list(nullptr), mu(nullptr), spmag_type(nullptr),
    spdir_type(nullptr), qtype(nullptr), mtype(nullptr), sqrt_mass_ratio(nullptr), local_swap_iatom_list(nullptr),
    local_swap_jatom_list(nullptr), local_swap_atom_list(nullptr), random_equal(nullptr), random_unequal(nullptr),
    c_pe(nullptr)
{
  if (narg < 10) utils::missing_cmd_args(FLERR, "fix atom/swap/spin", error);

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;
  time_depend = 1;

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  ncycles = utils::inumeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);
  const double temperature = utils::numeric(FLERR, arg[6], false, lmp);

  if (nevery <= 0) error->all(FLERR, "Illegal fix atom/swap/spin command");
  if (ncycles < 0) error->all(FLERR, "Illegal fix atom/swap/spin command");
  if (seed <= 0) error->all(FLERR, "Illegal fix atom/swap/spin command");
  if (temperature <= 0.0) error->all(FLERR, "Illegal fix atom/swap/spin command");

  beta = 1.0 / (force->boltz * temperature);

  ke_flag = 1;
  semi_grand_flag = 0;
  nswaptypes = 0;
  nswap = nswap_local = nswap_before = 0;

  memory->create(type_list, atom->ntypes, "atom/swap/spin:type_list");
  memory->create(mu, atom->ntypes + 1, "atom/swap/spin:mu");
  for (int i = 0; i <= atom->ntypes; i++) mu[i] = 0.0;

  options(narg - 7, &arg[7]);

  random_equal = new RanPark(lmp, seed);
  random_unequal = new RanPark(lmp, seed);

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  nswap_attempts = 0.0;
  nswap_successes = 0.0;

  atom_swap_nmax = 0;

  // include type (+q if present) and spin state in forward comm for ghost refresh during trials
  comm_forward = 1 + (atom->q_flag ? 1 : 0) + (atom->sp_flag ? 4 : 0);
}

/* ---------------------------------------------------------------------- */

FixAtomSwapSpin::~FixAtomSwapSpin()
{
  memory->destroy(type_list);
  memory->destroy(mu);
  memory->destroy(spmag_type);
  memory->destroy(spdir_type);
  memory->destroy(qtype);
  memory->destroy(mtype);
  memory->destroy(sqrt_mass_ratio);
  memory->destroy(local_swap_iatom_list);
  memory->destroy(local_swap_jatom_list);
  memory->destroy(local_swap_atom_list);
  delete[] idregion;
  delete random_equal;
  delete random_unequal;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR, "Illegal fix atom/swap/spin command");

  auto is_keyword = [](const char *s) {
    return (strcmp(s, "region") == 0) || (strcmp(s, "ke") == 0) || (strcmp(s, "semi-grand") == 0) ||
        (strcmp(s, "types") == 0) || (strcmp(s, "mu") == 0);
  };

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "region") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap/spin command");
      delete[] idregion;
      idregion = utils::strdup(arg[iarg + 1]);
      region = domain->get_region_by_id(idregion);
      if (!region) error->all(FLERR, "Region {} for fix atom/swap/spin does not exist", idregion);
      iarg += 2;
    } else if (strcmp(arg[iarg], "ke") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap/spin command");
      ke_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "semi-grand") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap/spin command");
      semi_grand_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "types") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal fix atom/swap/spin command");
      iarg++;
      while (iarg < narg) {
        if (is_keyword(arg[iarg])) break;
        if (nswaptypes >= atom->ntypes) error->all(FLERR, "Illegal fix atom/swap/spin command");
        type_list[nswaptypes] = utils::expand_type_int(FLERR, arg[iarg], Atom::ATOM, lmp);
        nswaptypes++;
        iarg++;
      }
    } else if (strcmp(arg[iarg], "mu") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap/spin command");
      iarg++;
      while (iarg < narg) {
        if (is_keyword(arg[iarg])) break;
        mu_values.push_back(utils::numeric(FLERR, arg[iarg], false, lmp));
        iarg++;
      }
    } else
      error->all(FLERR, "Illegal fix atom/swap/spin command");
  }
}

/* ---------------------------------------------------------------------- */

int FixAtomSwapSpin::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::init()
{
  if (!atom->sp_flag) error->all(FLERR, "Fix atom/swap/spin requires atom_style spin");

  c_pe = modify->get_compute_by_id("thermo_pe");

  // Optional: if a USER-TSPIN integrator is active, swap its per-atom state consistently
  // with atom->sp during MC trials/accept/reject. See FixTSPINNH::ensure_custom_peratom().
  {
    int flag = -1, cols = -1, ghost = 0;

    idx_tspin_vs = atom->find_custom_ghost("tspin_vs", flag, cols, ghost);
    if (idx_tspin_vs >= 0 && (flag != 1 || cols != 3))
      error->all(FLERR, "Custom property tspin_vs has incompatible type");

    flag = -1;
    cols = -1;
    ghost = 0;
    idx_tspin_sreal = atom->find_custom_ghost("tspin_sreal", flag, cols, ghost);
    if (idx_tspin_sreal >= 0 && (flag != 1 || cols != 3))
      error->all(FLERR, "Custom property tspin_sreal has incompatible type");

    flag = -1;
    cols = -1;
    ghost = 0;
    idx_tspin_smass = atom->find_custom_ghost("tspin_smass", flag, cols, ghost);
    if (idx_tspin_smass >= 0 && (flag != 1 || cols != 0))
      error->all(FLERR, "Custom property tspin_smass has incompatible type");

    flag = -1;
    cols = -1;
    ghost = 0;
    idx_tspin_isspin = atom->find_custom_ghost("tspin_isspin", flag, cols, ghost);
    if (idx_tspin_isspin >= 0 && (flag != 0 || cols != 0))
      error->all(FLERR, "Custom property tspin_isspin has incompatible type");
  }

  // Optional: same idea for GLSD caches (FixGLSDNH::ensure_custom_peratom()).
  {
    int flag = -1, cols = -1, ghost = 0;
    idx_glsd_fm_cache = atom->find_custom_ghost("glsd_fm_cache", flag, cols, ghost);
    if (idx_glsd_fm_cache >= 0 && (flag != 1 || cols != 3))
      error->all(FLERR, "Custom property glsd_fm_cache has incompatible type");

    flag = -1;
    cols = -1;
    ghost = 0;
    idx_glsd_s0_cache = atom->find_custom_ghost("glsd_s0_cache", flag, cols, ghost);
    if (idx_glsd_s0_cache >= 0 && (flag != 1 || cols != 3))
      error->all(FLERR, "Custom property glsd_s0_cache has incompatible type");
  }

  if (nswaptypes < 2) error->all(FLERR, "Must specify at least 2 types in fix atom/swap/spin command");
  if (!semi_grand_flag && nswaptypes != 2)
    error->all(FLERR, "Fix atom/swap/spin without semi-grand requires exactly 2 types");

  if (semi_grand_flag) {
    if (mu_values.empty()) error->all(FLERR, "Fix atom/swap/spin with semi-grand requires mu values");
    if (static_cast<int>(mu_values.size()) != nswaptypes)
      error->all(FLERR, "Fix atom/swap/spin requires one mu value per swap type");
    for (int k = 0; k < nswaptypes; k++) mu[type_list[k]] = mu_values[k];
  } else {
    if (!mu_values.empty()) error->all(FLERR, "Fix atom/swap/spin mu is only allowed with semi-grand yes");
  }

  // Precompute reference spin direction/magnitude for each swap type.
  // This is only used for semi-grand moves to set a spin consistent with the new type.
  memory->destroy(spmag_type);
  memory->destroy(spdir_type);
  memory->create(spmag_type, atom->ntypes + 1, "atom/swap/spin:spmag_type");
  memory->create(spdir_type, (atom->ntypes + 1) * 3, "atom/swap/spin:spdir_type");
  for (int t = 0; t <= atom->ntypes; t++) {
    spmag_type[t] = 0.0;
    spdir_type[3 * t + 0] = 0.0;
    spdir_type[3 * t + 1] = 0.0;
    spdir_type[3 * t + 2] = 0.0;
  }

  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
    const int t = type_list[iswaptype];
    double sum_mag_local = 0.0;
    double sum_mx_local = 0.0;
    double sum_my_local = 0.0;
    double sum_mz_local = 0.0;
    int count_local = 0;

    for (int i = 0; i < atom->nlocal; i++) {
      if (!(atom->mask[i] & groupbit)) continue;
      if (atom->type[i] != t) continue;
      const double mag = atom->sp[i][3];
      sum_mag_local += mag;
      sum_mx_local += atom->sp[i][0] * mag;
      sum_my_local += atom->sp[i][1] * mag;
      sum_mz_local += atom->sp[i][2] * mag;
      count_local++;
    }

    double sum_mag = 0.0, sum_mx = 0.0, sum_my = 0.0, sum_mz = 0.0;
    int count = 0;
    MPI_Allreduce(&sum_mag_local, &sum_mag, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&sum_mx_local, &sum_mx, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&sum_my_local, &sum_my, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&sum_mz_local, &sum_mz, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&count_local, &count, 1, MPI_INT, MPI_SUM, world);

    if (count == 0) continue;

    const double inv = 1.0 / static_cast<double>(count);
    spmag_type[t] = sum_mag * inv;
    const double mx = sum_mx * inv;
    const double my = sum_my * inv;
    const double mz = sum_mz * inv;
    const double norm = std::sqrt(mx * mx + my * my + mz * mz);
    if (norm > 0.0) {
      spdir_type[3 * t + 0] = mx / norm;
      spdir_type[3 * t + 1] = my / norm;
      spdir_type[3 * t + 2] = mz / norm;
    }
  }

  // cache per-type q and rmass (if present) so swapping preserves those properties
  if (atom->q_flag && !semi_grand_flag) {
    memory->create(qtype, nswaptypes, "atom/swap/spin:qtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) qtype[iswaptype] = 0.0;

    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      const int itype = type_list[iswaptype];
      int first = 1;
      int *type = atom->type;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == itype) {
            if (first > 0) {
              qtype[iswaptype] = atom->q[i];
              first = 0;
            } else if (qtype[iswaptype] != atom->q[i])
              first = -1;
          }
        }
      }
      int firstall;
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall < 0)
        error->all(FLERR, Error::NOLASTLINE, "All atoms of a swapped type must have the same charge.");
      if (firstall > 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "At least one atom of each swapped type must be present to define charges");
      if (first) qtype[iswaptype] = -DBL_MAX;
      double qmax, qmin;
      MPI_Allreduce(&qtype[iswaptype], &qmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first) qtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype], &qmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (qmax != qmin)
        error->all(FLERR, Error::NOLASTLINE, "All atoms of a swapped type must have same charge.");
      qtype[iswaptype] = qmax;
    }
  }

  if (atom->rmass != nullptr && !semi_grand_flag) {
    memory->create(mtype, nswaptypes, "atom/swap/spin:mtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) mtype[iswaptype] = 0.0;
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      const int itype = type_list[iswaptype];
      int first = 1;
      int *type = atom->type;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == itype) {
            if (first > 0) {
              mtype[iswaptype] = atom->rmass[i];
              first = 0;
            } else if (mtype[iswaptype] != atom->rmass[i])
              first = -1;
          }
        }
      }
      int firstall;
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall < 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "All atoms of a swapped type must have the same per-atom mass");
      if (firstall > 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "At least one atom of each swapped type must be present to define masses");
      if (first) mtype[iswaptype] = -DBL_MAX;
      double mmax, mmin;
      MPI_Allreduce(&mtype[iswaptype], &mmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first) mtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&mtype[iswaptype], &mmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (mmax != mmin)
        error->all(FLERR, Error::NOLASTLINE, "All atoms of a swapped type must have same mass.");
      mtype[iswaptype] = mmax;
    }
  }

  memory->create(sqrt_mass_ratio, atom->ntypes + 1, atom->ntypes + 1, "atom/swap/spin:sqrt_mass_ratio");
  if (atom->rmass != nullptr && mtype != nullptr) {
    for (int itype = 1; itype <= atom->ntypes; itype++)
      for (int jtype = 1; jtype <= atom->ntypes; jtype++) sqrt_mass_ratio[itype][jtype] = 1.0;
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      int itype = type_list[iswaptype];
      for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++) {
        int jtype = type_list[jswaptype];
        sqrt_mass_ratio[itype][jtype] = std::sqrt(mtype[iswaptype] / mtype[jswaptype]);
      }
    }
  } else {
    for (int itype = 1; itype <= atom->ntypes; itype++)
      for (int jtype = 1; jtype <= atom->ntypes; jtype++)
        sqrt_mass_ratio[itype][jtype] = std::sqrt(atom->mass[itype] / atom->mass[jtype]);
  }

  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++)
      for (int ktype = 1; ktype <= atom->ntypes; ktype++)
        if (cutsq[type_list[iswaptype]][ktype] != cutsq[type_list[jswaptype]][ktype])
          unequal_cutoffs = true;

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);

    if (flagall) error->all(FLERR, "Cannot do atom/swap/spin on atoms in atom_modify first group");
  }
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::pre_exchange()
{
  if (next_reneighbor != update->ntimestep) return;

#ifdef LMP_KOKKOS
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) {
    atomKK->sync(Host, X_MASK | V_MASK | TYPE_MASK | MASK_MASK | Q_MASK | RMASS_MASK);
    atomKK->sync(Host, SP_MASK);
  }
#endif

  // ensure current system is ready to compute energy
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);

  energy_stored = energy_full();

  int nsuccess = 0;
  if (semi_grand_flag) {
    update_semi_grand_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_semi_grand();
  } else {
    update_swap_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_swap();
  }

  nswap_attempts += ncycles;
  nswap_successes += nsuccess;

  next_reneighbor = update->ntimestep + nevery;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::gather_spin(int local_index, double sp_out[4]) const
{
  double sp_local[4] = {0.0, 0.0, 0.0, 0.0};
  if (local_index >= 0) {
    sp_local[0] = atom->sp[local_index][0];
    sp_local[1] = atom->sp[local_index][1];
    sp_local[2] = atom->sp[local_index][2];
    sp_local[3] = atom->sp[local_index][3];
  }
  MPI_Allreduce(sp_local, sp_out, 4, MPI_DOUBLE, MPI_SUM, world);
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::gather_custom_darray3(int idx, int local_index, double out[3]) const
{
  double tmp[3] = {0.0, 0.0, 0.0};
  if (idx >= 0 && local_index >= 0) {
    tmp[0] = atom->darray[idx][local_index][0];
    tmp[1] = atom->darray[idx][local_index][1];
    tmp[2] = atom->darray[idx][local_index][2];
  }
  MPI_Allreduce(tmp, out, 3, MPI_DOUBLE, MPI_SUM, world);
}

void FixAtomSwapSpin::gather_custom_dvector(int idx, int local_index, double &out) const
{
  double tmp = 0.0;
  if (idx >= 0 && local_index >= 0) tmp = atom->dvector[idx][local_index];
  MPI_Allreduce(&tmp, &out, 1, MPI_DOUBLE, MPI_SUM, world);
}

void FixAtomSwapSpin::gather_custom_ivector(int idx, int local_index, int &out) const
{
  int tmp = 0;
  if (idx >= 0 && local_index >= 0) tmp = atom->ivector[idx][local_index];
  MPI_Allreduce(&tmp, &out, 1, MPI_INT, MPI_SUM, world);
}

/* ---------------------------------------------------------------------- */

int FixAtomSwapSpin::attempt_semi_grand()
{
  if (nswap == 0) return 0;

  const double energy_before = energy_stored;

  int itype = 0;
  int jtype = 0;
  int jswaptype = 0;
  const int i = pick_semi_grand_atom();

  double sp_old[4] = {0.0, 0.0, 0.0, 0.0};
  double vs_old[3] = {0.0, 0.0, 0.0};
  double sreal_old[3] = {0.0, 0.0, 0.0};
  double smass_old = 0.0;
  int isspin_old = 0;
  double glsd_fm_old[3] = {0.0, 0.0, 0.0};
  double glsd_s0_old[3] = {0.0, 0.0, 0.0};

  if (idx_tspin_vs >= 0) gather_custom_darray3(idx_tspin_vs, i, vs_old);
  if (idx_tspin_sreal >= 0) gather_custom_darray3(idx_tspin_sreal, i, sreal_old);
  if (idx_tspin_smass >= 0) gather_custom_dvector(idx_tspin_smass, i, smass_old);
  if (idx_tspin_isspin >= 0) gather_custom_ivector(idx_tspin_isspin, i, isspin_old);
  if (idx_glsd_fm_cache >= 0) gather_custom_darray3(idx_glsd_fm_cache, i, glsd_fm_old);
  if (idx_glsd_s0_cache >= 0) gather_custom_darray3(idx_glsd_s0_cache, i, glsd_s0_old);

  if (i >= 0) {
    itype = atom->type[i];
    sp_old[0] = atom->sp[i][0];
    sp_old[1] = atom->sp[i][1];
    sp_old[2] = atom->sp[i][2];
    sp_old[3] = atom->sp[i][3];

    jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
    jtype = type_list[jswaptype];
    while (itype == jtype) {
      jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
      jtype = type_list[jswaptype];
    }

    atom->type[i] = jtype;

    // Update spin to match new type using the precomputed reference (direction may be undefined).
    const double dx = spdir_type[3 * jtype + 0];
    const double dy = spdir_type[3 * jtype + 1];
    const double dz = spdir_type[3 * jtype + 2];
    const double dnorm2 = dx * dx + dy * dy + dz * dz;
    const double mag_new = spmag_type[jtype];
    if (mag_new <= 0.0 && dnorm2 == 0.0) {
      // No reference info for this type: keep the atom's spin unchanged.
      atom->sp[i][0] = sp_old[0];
      atom->sp[i][1] = sp_old[1];
      atom->sp[i][2] = sp_old[2];
      atom->sp[i][3] = sp_old[3];
    } else {
      if (dnorm2 > 0.0) {
        atom->sp[i][0] = dx;
        atom->sp[i][1] = dy;
        atom->sp[i][2] = dz;
      } else {
        const double onorm = std::sqrt(sp_old[0] * sp_old[0] + sp_old[1] * sp_old[1] + sp_old[2] * sp_old[2]);
        if (onorm > 0.0) {
          atom->sp[i][0] = sp_old[0] / onorm;
          atom->sp[i][1] = sp_old[1] / onorm;
          atom->sp[i][2] = sp_old[2] / onorm;
        } else {
          atom->sp[i][0] = 1.0;
          atom->sp[i][1] = 0.0;
          atom->sp[i][2] = 0.0;
        }
      }
      atom->sp[i][3] = (mag_new > 0.0) ? mag_new : sp_old[3];
    }

    // Keep USER-TSPIN state consistent with atom->sp when present.
    if (idx_tspin_sreal >= 0) {
      atom->darray[idx_tspin_sreal][i][0] = atom->sp[i][0] * atom->sp[i][3];
      atom->darray[idx_tspin_sreal][i][1] = atom->sp[i][1] * atom->sp[i][3];
      atom->darray[idx_tspin_sreal][i][2] = atom->sp[i][2] * atom->sp[i][3];
    }
    if (idx_tspin_isspin >= 0) {
      atom->ivector[idx_tspin_isspin][i] = (atom->sp[i][3] > 1.0e-4) ? 1 : 0;
    }
    if (idx_tspin_smass >= 0 && atom->mass) {
      const double mi = atom->mass[itype];
      const double mj = atom->mass[jtype];
      if (mi > 0.0 && mj > 0.0 && smass_old > 0.0) atom->dvector[idx_tspin_smass][i] = smass_old * (mj / mi);
    }

    // Keep GLSD caches consistent with the updated spin state when present.
    if (idx_glsd_s0_cache >= 0) {
      atom->darray[idx_glsd_s0_cache][i][0] = atom->sp[i][0] * atom->sp[i][3];
      atom->darray[idx_glsd_s0_cache][i][1] = atom->sp[i][1] * atom->sp[i][3];
      atom->darray[idx_glsd_s0_cache][i][2] = atom->sp[i][2] * atom->sp[i][3];
    }
    if (idx_glsd_fm_cache >= 0) {
      // Conservative default: field cache unknown after a type/spin change; set to 0.
      atom->darray[idx_glsd_fm_cache][i][0] = 0.0;
      atom->darray[idx_glsd_fm_cache][i][1] = 0.0;
      atom->darray[idx_glsd_fm_cache][i][2] = 0.0;
    }
  }

#ifdef LMP_KOKKOS
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) {
    atomKK->modified(Host, TYPE_MASK | Q_MASK | RMASS_MASK);
    atomKK->modified(Host, SP_MASK);
  }
#endif

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  if (force->kspace) force->kspace->qsum_qsq();
  const double energy_after = energy_full();

  int accept_local = 0;
  if (i >= 0) {
    if (random_unequal->uniform() < std::exp(beta * (energy_before - energy_after + mu[jtype] - mu[itype])))
      accept_local = 1;
  }

  int accept_all = 0;
  MPI_Allreduce(&accept_local, &accept_all, 1, MPI_INT, MPI_MAX, world);

  if (accept_all) {
    update_semi_grand_atoms_list();
    energy_stored = energy_after;
    if (ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
      }
    }
    // If a Kokkos USER-TSPIN integrator is active, resync its internal per-atom state
    // to the now-accepted atom->sp/type for this local atom.
    if (i >= 0) mc_sync_user_tspin_kokkos_local(i);
    return 1;
  }

  if (i >= 0) {
    atom->type[i] = itype;
    atom->sp[i][0] = sp_old[0];
    atom->sp[i][1] = sp_old[1];
    atom->sp[i][2] = sp_old[2];
    atom->sp[i][3] = sp_old[3];
    if (idx_tspin_vs >= 0) {
      atom->darray[idx_tspin_vs][i][0] = vs_old[0];
      atom->darray[idx_tspin_vs][i][1] = vs_old[1];
      atom->darray[idx_tspin_vs][i][2] = vs_old[2];
    }
    if (idx_tspin_sreal >= 0) {
      atom->darray[idx_tspin_sreal][i][0] = sreal_old[0];
      atom->darray[idx_tspin_sreal][i][1] = sreal_old[1];
      atom->darray[idx_tspin_sreal][i][2] = sreal_old[2];
    }
    if (idx_tspin_smass >= 0) atom->dvector[idx_tspin_smass][i] = smass_old;
    if (idx_tspin_isspin >= 0) atom->ivector[idx_tspin_isspin][i] = isspin_old;
    if (idx_glsd_fm_cache >= 0) {
      atom->darray[idx_glsd_fm_cache][i][0] = glsd_fm_old[0];
      atom->darray[idx_glsd_fm_cache][i][1] = glsd_fm_old[1];
      atom->darray[idx_glsd_fm_cache][i][2] = glsd_fm_old[2];
    }
    if (idx_glsd_s0_cache >= 0) {
      atom->darray[idx_glsd_s0_cache][i][0] = glsd_s0_old[0];
      atom->darray[idx_glsd_s0_cache][i][1] = glsd_s0_old[1];
      atom->darray[idx_glsd_s0_cache][i][2] = glsd_s0_old[2];
    }
  }

#ifdef LMP_KOKKOS
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) {
    atomKK->modified(Host, TYPE_MASK | Q_MASK | RMASS_MASK);
    atomKK->modified(Host, SP_MASK);
  }
#endif

  if (force->kspace) force->kspace->qsum_qsq();
  return 0;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwapSpin::attempt_swap()
{
  if ((niswap == 0) || (njswap == 0)) return 0;

  const double energy_before = energy_stored;

  int i = pick_i_swap_atom();
  int j = pick_j_swap_atom();
  const int itype = type_list[0];
  const int jtype = type_list[1];

  // gather spin state (needed even when i/j are on different MPI ranks)
  double spi[4], spj[4];
  gather_spin(i, spi);
  gather_spin(j, spj);

  // gather USER-TSPIN per-atom state if present (needed even when i/j are on different MPI ranks)
  double vsi[3] = {0.0, 0.0, 0.0}, vsj[3] = {0.0, 0.0, 0.0};
  double sreal_i[3] = {0.0, 0.0, 0.0}, sreal_j[3] = {0.0, 0.0, 0.0};
  double smass_i = 0.0, smass_j = 0.0;
  int isspin_i = 0, isspin_j = 0;
  double glsd_fm_i[3] = {0.0, 0.0, 0.0}, glsd_fm_j[3] = {0.0, 0.0, 0.0};
  double glsd_s0_i[3] = {0.0, 0.0, 0.0}, glsd_s0_j[3] = {0.0, 0.0, 0.0};

  if (idx_tspin_vs >= 0) {
    gather_custom_darray3(idx_tspin_vs, i, vsi);
    gather_custom_darray3(idx_tspin_vs, j, vsj);
  }
  if (idx_tspin_sreal >= 0) {
    gather_custom_darray3(idx_tspin_sreal, i, sreal_i);
    gather_custom_darray3(idx_tspin_sreal, j, sreal_j);
  }
  if (idx_tspin_smass >= 0) {
    gather_custom_dvector(idx_tspin_smass, i, smass_i);
    gather_custom_dvector(idx_tspin_smass, j, smass_j);
  }
  if (idx_tspin_isspin >= 0) {
    gather_custom_ivector(idx_tspin_isspin, i, isspin_i);
    gather_custom_ivector(idx_tspin_isspin, j, isspin_j);
  }
  if (idx_glsd_fm_cache >= 0) {
    gather_custom_darray3(idx_glsd_fm_cache, i, glsd_fm_i);
    gather_custom_darray3(idx_glsd_fm_cache, j, glsd_fm_j);
  }
  if (idx_glsd_s0_cache >= 0) {
    gather_custom_darray3(idx_glsd_s0_cache, i, glsd_s0_i);
    gather_custom_darray3(idx_glsd_s0_cache, j, glsd_s0_j);
  }

  // apply proposed swap: types (and q/rmass if present) + exchange spin vectors
  if (i >= 0) {
    atom->type[i] = jtype;
    if (atom->q_flag) atom->q[i] = qtype[1];
    if (atom->rmass != nullptr) atom->rmass[i] = mtype[1];
    atom->sp[i][0] = spj[0];
    atom->sp[i][1] = spj[1];
    atom->sp[i][2] = spj[2];
    atom->sp[i][3] = spj[3];
    if (idx_tspin_vs >= 0) {
      atom->darray[idx_tspin_vs][i][0] = vsj[0];
      atom->darray[idx_tspin_vs][i][1] = vsj[1];
      atom->darray[idx_tspin_vs][i][2] = vsj[2];
    }
    if (idx_tspin_sreal >= 0) {
      atom->darray[idx_tspin_sreal][i][0] = sreal_j[0];
      atom->darray[idx_tspin_sreal][i][1] = sreal_j[1];
      atom->darray[idx_tspin_sreal][i][2] = sreal_j[2];
    }
    if (idx_tspin_smass >= 0) atom->dvector[idx_tspin_smass][i] = smass_j;
    if (idx_tspin_isspin >= 0) atom->ivector[idx_tspin_isspin][i] = isspin_j;
    if (idx_glsd_fm_cache >= 0) {
      atom->darray[idx_glsd_fm_cache][i][0] = glsd_fm_j[0];
      atom->darray[idx_glsd_fm_cache][i][1] = glsd_fm_j[1];
      atom->darray[idx_glsd_fm_cache][i][2] = glsd_fm_j[2];
    }
    if (idx_glsd_s0_cache >= 0) {
      atom->darray[idx_glsd_s0_cache][i][0] = glsd_s0_j[0];
      atom->darray[idx_glsd_s0_cache][i][1] = glsd_s0_j[1];
      atom->darray[idx_glsd_s0_cache][i][2] = glsd_s0_j[2];
    }
  }
  if (j >= 0) {
    atom->type[j] = itype;
    if (atom->q_flag) atom->q[j] = qtype[0];
    if (atom->rmass != nullptr) atom->rmass[j] = mtype[0];
    atom->sp[j][0] = spi[0];
    atom->sp[j][1] = spi[1];
    atom->sp[j][2] = spi[2];
    atom->sp[j][3] = spi[3];
    if (idx_tspin_vs >= 0) {
      atom->darray[idx_tspin_vs][j][0] = vsi[0];
      atom->darray[idx_tspin_vs][j][1] = vsi[1];
      atom->darray[idx_tspin_vs][j][2] = vsi[2];
    }
    if (idx_tspin_sreal >= 0) {
      atom->darray[idx_tspin_sreal][j][0] = sreal_i[0];
      atom->darray[idx_tspin_sreal][j][1] = sreal_i[1];
      atom->darray[idx_tspin_sreal][j][2] = sreal_i[2];
    }
    if (idx_tspin_smass >= 0) atom->dvector[idx_tspin_smass][j] = smass_i;
    if (idx_tspin_isspin >= 0) atom->ivector[idx_tspin_isspin][j] = isspin_i;
    if (idx_glsd_fm_cache >= 0) {
      atom->darray[idx_glsd_fm_cache][j][0] = glsd_fm_i[0];
      atom->darray[idx_glsd_fm_cache][j][1] = glsd_fm_i[1];
      atom->darray[idx_glsd_fm_cache][j][2] = glsd_fm_i[2];
    }
    if (idx_glsd_s0_cache >= 0) {
      atom->darray[idx_glsd_s0_cache][j][0] = glsd_s0_i[0];
      atom->darray[idx_glsd_s0_cache][j][1] = glsd_s0_i[1];
      atom->darray[idx_glsd_s0_cache][j][2] = glsd_s0_i[2];
    }
  }

#ifdef LMP_KOKKOS
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) {
    atomKK->modified(Host, TYPE_MASK | Q_MASK | RMASS_MASK);
    atomKK->modified(Host, SP_MASK);
  }
#endif

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  const double energy_after = energy_full();

  int accept_local = 0;
  if (random_equal->uniform() < std::exp(beta * (energy_before - energy_after))) accept_local = 1;

  int accept_all = 0;
  MPI_Allreduce(&accept_local, &accept_all, 1, MPI_INT, MPI_MAX, world);

  if (accept_all) {
    update_swap_atoms_list();
    if (ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
      }
      if (j >= 0) {
        atom->v[j][0] *= sqrt_mass_ratio[jtype][itype];
        atom->v[j][1] *= sqrt_mass_ratio[jtype][itype];
        atom->v[j][2] *= sqrt_mass_ratio[jtype][itype];
      }
    }
    energy_stored = energy_after;
    // Resync USER-TSPIN Kokkos integrator per-atom state after acceptance.
    if (i >= 0) mc_sync_user_tspin_kokkos_local(i);
    if (j >= 0) mc_sync_user_tspin_kokkos_local(j);
    return 1;
  }

  // reject: restore local atoms (ghosts will be refreshed on next trial via forward_comm)
  if (i >= 0) {
    atom->type[i] = type_list[0];
    if (atom->q_flag) atom->q[i] = qtype[0];
    if (atom->rmass != nullptr) atom->rmass[i] = mtype[0];
    atom->sp[i][0] = spi[0];
    atom->sp[i][1] = spi[1];
    atom->sp[i][2] = spi[2];
    atom->sp[i][3] = spi[3];
    if (idx_tspin_vs >= 0) {
      atom->darray[idx_tspin_vs][i][0] = vsi[0];
      atom->darray[idx_tspin_vs][i][1] = vsi[1];
      atom->darray[idx_tspin_vs][i][2] = vsi[2];
    }
    if (idx_tspin_sreal >= 0) {
      atom->darray[idx_tspin_sreal][i][0] = sreal_i[0];
      atom->darray[idx_tspin_sreal][i][1] = sreal_i[1];
      atom->darray[idx_tspin_sreal][i][2] = sreal_i[2];
    }
    if (idx_tspin_smass >= 0) atom->dvector[idx_tspin_smass][i] = smass_i;
    if (idx_tspin_isspin >= 0) atom->ivector[idx_tspin_isspin][i] = isspin_i;
    if (idx_glsd_fm_cache >= 0) {
      atom->darray[idx_glsd_fm_cache][i][0] = glsd_fm_i[0];
      atom->darray[idx_glsd_fm_cache][i][1] = glsd_fm_i[1];
      atom->darray[idx_glsd_fm_cache][i][2] = glsd_fm_i[2];
    }
    if (idx_glsd_s0_cache >= 0) {
      atom->darray[idx_glsd_s0_cache][i][0] = glsd_s0_i[0];
      atom->darray[idx_glsd_s0_cache][i][1] = glsd_s0_i[1];
      atom->darray[idx_glsd_s0_cache][i][2] = glsd_s0_i[2];
    }
  }
  if (j >= 0) {
    atom->type[j] = type_list[1];
    if (atom->q_flag) atom->q[j] = qtype[1];
    if (atom->rmass != nullptr) atom->rmass[j] = mtype[1];
    atom->sp[j][0] = spj[0];
    atom->sp[j][1] = spj[1];
    atom->sp[j][2] = spj[2];
    atom->sp[j][3] = spj[3];
    if (idx_tspin_vs >= 0) {
      atom->darray[idx_tspin_vs][j][0] = vsj[0];
      atom->darray[idx_tspin_vs][j][1] = vsj[1];
      atom->darray[idx_tspin_vs][j][2] = vsj[2];
    }
    if (idx_tspin_sreal >= 0) {
      atom->darray[idx_tspin_sreal][j][0] = sreal_j[0];
      atom->darray[idx_tspin_sreal][j][1] = sreal_j[1];
      atom->darray[idx_tspin_sreal][j][2] = sreal_j[2];
    }
    if (idx_tspin_smass >= 0) atom->dvector[idx_tspin_smass][j] = smass_j;
    if (idx_tspin_isspin >= 0) atom->ivector[idx_tspin_isspin][j] = isspin_j;
    if (idx_glsd_fm_cache >= 0) {
      atom->darray[idx_glsd_fm_cache][j][0] = glsd_fm_j[0];
      atom->darray[idx_glsd_fm_cache][j][1] = glsd_fm_j[1];
      atom->darray[idx_glsd_fm_cache][j][2] = glsd_fm_j[2];
    }
    if (idx_glsd_s0_cache >= 0) {
      atom->darray[idx_glsd_s0_cache][j][0] = glsd_s0_j[0];
      atom->darray[idx_glsd_s0_cache][j][1] = glsd_s0_j[1];
      atom->darray[idx_glsd_s0_cache][j][2] = glsd_s0_j[2];
    }
  }

#ifdef LMP_KOKKOS
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) {
    atomKK->modified(Host, TYPE_MASK | Q_MASK | RMASS_MASK);
    atomKK->modified(Host, SP_MASK);
  }
#endif

  return 0;
}

/* ---------------------------------------------------------------------- */

double FixAtomSwapSpin::energy_full()
{
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag, vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (force->kspace) force->kspace->compute(eflag, vflag);

  if (modify->n_post_force_any) {
    // MC energy evaluation must not advance spin/lattice integrators.
    // In particular, `fix glsd/*` has POST_FORCE and would evolve spins during the trial.
    // So we replay post_force fixes except for USER-TSPIN integrators.
    for (int ifix = 0; ifix < modify->nfix; ifix++) {
      Fix *f = modify->fix[ifix];
      if (!f) continue;
      if (!(modify->fmask[ifix] & POST_FORCE)) continue;
      if (utils::strmatch(f->style, "^glsd/")) continue;
      if (utils::strmatch(f->style, "^tspin/")) continue;
      f->post_force(vflag);
    }
  }

  update->eflag_global = update->ntimestep;
  return c_pe->compute_scalar();
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::update_swap_atoms_list()
{
  const int nlocal = atom->nlocal;
  int *type = atom->type;
  double **x = atom->x;

  if (atom->nmax > atom_swap_nmax) {
    memory->destroy(local_swap_iatom_list);
    memory->destroy(local_swap_jatom_list);
    atom_swap_nmax = atom->nmax;
    memory->create(local_swap_iatom_list, atom_swap_nmax, "atom/swap/spin:local_swap_iatom_list");
    memory->create(local_swap_jatom_list, atom_swap_nmax, "atom/swap/spin:local_swap_jatom_list");
  }

  niswap_local = 0;
  njswap_local = 0;

  if (region) {
    for (int i = 0; i < nlocal; i++) {
      if (region->match(x[i][0], x[i][1], x[i][2]) != 1) continue;
      if (!(atom->mask[i] & groupbit)) continue;
      if (type[i] == type_list[0]) {
        local_swap_iatom_list[niswap_local++] = i;
      } else if (type[i] == type_list[1]) {
        local_swap_jatom_list[njswap_local++] = i;
      }
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (!(atom->mask[i] & groupbit)) continue;
      if (type[i] == type_list[0]) {
        local_swap_iatom_list[niswap_local++] = i;
      } else if (type[i] == type_list[1]) {
        local_swap_jatom_list[njswap_local++] = i;
      }
    }
  }

  MPI_Allreduce(&niswap_local, &niswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&niswap_local, &niswap_before, 1, MPI_INT, MPI_SUM, world);
  niswap_before -= niswap_local;

  MPI_Allreduce(&njswap_local, &njswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&njswap_local, &njswap_before, 1, MPI_INT, MPI_SUM, world);
  njswap_before -= njswap_local;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::update_semi_grand_atoms_list()
{
  const int nlocal = atom->nlocal;
  double **x = atom->x;
  int *type = atom->type;

  if (atom->nmax > atom_swap_nmax) {
    memory->destroy(local_swap_atom_list);
    atom_swap_nmax = atom->nmax;
    memory->create(local_swap_atom_list, atom_swap_nmax, "atom/swap/spin:local_swap_atom_list");
  }

  nswap_local = 0;
  if (region) {
    for (int i = 0; i < nlocal; i++) {
      if (region->match(x[i][0], x[i][1], x[i][2]) != 1) continue;
      if (!(atom->mask[i] & groupbit)) continue;
      int itype = type[i];
      int iswaptype;
      for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
        if (itype == type_list[iswaptype]) break;
      if (iswaptype == nswaptypes) continue;
      local_swap_atom_list[nswap_local++] = i;
    }
  } else {
    for (int i = 0; i < nlocal; i++) {
      if (!(atom->mask[i] & groupbit)) continue;
      int itype = type[i];
      int iswaptype;
      for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
        if (itype == type_list[iswaptype]) break;
      if (iswaptype == nswaptypes) continue;
      local_swap_atom_list[nswap_local++] = i;
    }
  }

  MPI_Allreduce(&nswap_local, &nswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&nswap_local, &nswap_before, 1, MPI_INT, MPI_SUM, world);
  nswap_before -= nswap_local;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwapSpin::pick_semi_grand_atom()
{
  int i = -1;
  const int iwhichglobal = static_cast<int>(nswap * random_equal->uniform());
  if ((iwhichglobal >= nswap_before) && (iwhichglobal < nswap_before + nswap_local)) {
    const int iwhichlocal = iwhichglobal - nswap_before;
    i = local_swap_atom_list[iwhichlocal];
  }
  return i;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwapSpin::pick_i_swap_atom()
{
  int i = -1;
  const int iwhichglobal = static_cast<int>(niswap * random_equal->uniform());
  if ((iwhichglobal >= niswap_before) && (iwhichglobal < niswap_before + niswap_local)) {
    const int iwhichlocal = iwhichglobal - niswap_before;
    i = local_swap_iatom_list[iwhichlocal];
  }
  return i;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwapSpin::pick_j_swap_atom()
{
  int j = -1;
  const int jwhichglobal = static_cast<int>(njswap * random_equal->uniform());
  if ((jwhichglobal >= njswap_before) && (jwhichglobal < njswap_before + njswap_local)) {
    const int jwhichlocal = jwhichglobal - njswap_before;
    j = local_swap_jatom_list[jwhichlocal];
  }
  return j;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwapSpin::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int m = 0;
  int *type = atom->type;
  double *q = atom->q;
  double **sp = atom->sp;

  const int pack_q = atom->q_flag ? 1 : 0;
  const int pack_sp = atom->sp_flag ? 1 : 0;

  for (int ii = 0; ii < n; ii++) {
    const int j = list[ii];
    buf[m++] = type[j];
    if (pack_q) buf[m++] = q[j];
    if (pack_sp) {
      buf[m++] = sp[j][0];
      buf[m++] = sp[j][1];
      buf[m++] = sp[j][2];
      buf[m++] = sp[j][3];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;

  int *type = atom->type;
  double *q = atom->q;
  double **sp = atom->sp;

  const int unpack_q = atom->q_flag ? 1 : 0;
  const int unpack_sp = atom->sp_flag ? 1 : 0;

  for (int i = first; i < last; i++) {
    type[i] = static_cast<int>(buf[m++]);
    if (unpack_q) q[i] = buf[m++];
    if (unpack_sp) {
      sp[i][0] = buf[m++];
      sp[i][1] = buf[m++];
      sp[i][2] = buf[m++];
      sp[i][3] = buf[m++];
    }
  }

#ifdef LMP_KOKKOS
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) {
    atomKK->modified(Host, TYPE_MASK | Q_MASK);
    atomKK->modified(Host, SP_MASK);
  }
#endif
}

/* ---------------------------------------------------------------------- */

double FixAtomSwapSpin::compute_vector(int n)
{
  if (n == 0) return nswap_attempts;
  if (n == 1) return nswap_successes;
  return 0.0;
}

/* ---------------------------------------------------------------------- */

double FixAtomSwapSpin::memory_usage()
{
  double bytes = (double) atom_swap_nmax * sizeof(int);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwapSpin::mc_sync_user_tspin_kokkos_local(int local_index)
{
#ifdef LMP_KOKKOS
  if (local_index < 0) return;

  // Only care about fixes in this package that maintain their own per-atom state.
  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    Fix *f = modify->fix[ifix];
    if (!f) continue;
    if (!utils::strmatch(f->style, "^tspin/") && !utils::strmatch(f->style, "^glsd/")) continue;

    if (auto *tspin = dynamic_cast<FixTSPINNHKokkos<LMPDeviceType> *>(f)) {
      tspin->mc_sync_from_atom(local_index);
      continue;
    }
    if (auto *glsd = dynamic_cast<FixGLSDNHKokkos<LMPDeviceType> *>(f)) {
      glsd->mc_sync_from_atom(local_index);
      continue;
    }

#ifdef LMP_KOKKOS_GPU
    if (auto *tspin = dynamic_cast<FixTSPINNHKokkos<LMPHostType> *>(f)) {
      tspin->mc_sync_from_atom(local_index);
      continue;
    }
    if (auto *glsd = dynamic_cast<FixGLSDNHKokkos<LMPHostType> *>(f)) {
      glsd->mc_sync_from_atom(local_index);
      continue;
    }
#endif
  }
#else
  (void) local_index;
#endif
}

void FixAtomSwapSpin::mc_sync_atom_custom_after_accept_local(int local_index, int old_type, int new_type)
{
  if (local_index < 0) return;

#ifdef LMP_KOKKOS
  // If atom data is Kokkos-managed, ensure host sp/type is up-to-date before we use it
  // to update Atom custom properties (which are host-resident).
  if (auto *atomKK = dynamic_cast<AtomKokkos *>(atom)) {
    atomKK->sync(Host, TYPE_MASK | SP_MASK);
  }
#endif

  // tspin custom state (host implementation)
  if (idx_tspin_sreal >= 0) {
    const double sx = atom->sp[local_index][0];
    const double sy = atom->sp[local_index][1];
    const double sz = atom->sp[local_index][2];
    const double smag = atom->sp[local_index][3];
    atom->darray[idx_tspin_sreal][local_index][0] = sx * smag;
    atom->darray[idx_tspin_sreal][local_index][1] = sy * smag;
    atom->darray[idx_tspin_sreal][local_index][2] = sz * smag;
  }
  if (idx_tspin_vs >= 0) {
    atom->darray[idx_tspin_vs][local_index][0] = 0.0;
    atom->darray[idx_tspin_vs][local_index][1] = 0.0;
    atom->darray[idx_tspin_vs][local_index][2] = 0.0;
  }
  if (idx_tspin_isspin >= 0) {
    atom->ivector[idx_tspin_isspin][local_index] = (atom->sp[local_index][3] > 1.0e-4) ? 1 : 0;
  }
  if (idx_tspin_smass >= 0 && atom->mass && old_type > 0 && new_type > 0) {
    const double mo = atom->mass[old_type];
    const double mn = atom->mass[new_type];
    if (mo > 0.0 && mn > 0.0) atom->dvector[idx_tspin_smass][local_index] *= (mn / mo);
  }

  // glsd custom caches (host implementation)
  if (idx_glsd_s0_cache >= 0) {
    const double sx = atom->sp[local_index][0];
    const double sy = atom->sp[local_index][1];
    const double sz = atom->sp[local_index][2];
    const double smag = atom->sp[local_index][3];
    atom->darray[idx_glsd_s0_cache][local_index][0] = sx * smag;
    atom->darray[idx_glsd_s0_cache][local_index][1] = sy * smag;
    atom->darray[idx_glsd_s0_cache][local_index][2] = sz * smag;
  }
  if (idx_glsd_fm_cache >= 0) {
    atom->darray[idx_glsd_fm_cache][local_index][0] = 0.0;
    atom->darray[idx_glsd_fm_cache][local_index][1] = 0.0;
    atom->darray[idx_glsd_fm_cache][local_index][2] = 0.0;
  }
}
