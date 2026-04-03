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

#include "fix_tspin_nh.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "random_park.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

// Keep integer values consistent with FixNH's internal enums in src/fix_nh.cpp.
enum {NOBIAS, BIAS};
enum {ISO, ANISO, TRICLINIC};

// Legacy threshold from the old FixNH-based implementation:
// treat atoms with |S| <= 1e-4 as "no spin".
static constexpr double SPIN_MAG_EPS = 1.0e-4;

namespace {
static inline std::uint64_t splitmix64(std::uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static inline double u01_from_u64(std::uint64_t &state)
{
  state = splitmix64(state);
  // 53 random bits -> double in [0,1)
  return (state >> 11) * (1.0 / 9007199254740992.0);
}

static inline double gaussian_tag(std::uint64_t seed, tagint tag, int component)
{
  std::uint64_t state = seed;
  // Mix in tag and component using large odd constants (deterministic, decomposition-independent).
  state ^= static_cast<std::uint64_t>(tag) * 0xd1b54a32d192ed03ULL;
  state ^= static_cast<std::uint64_t>(component) * 0x9e3779b97f4a7c15ULL;

  double u1 = 0.0;
  do {
    u1 = u01_from_u64(state);
  } while (u1 <= 0.0);
  const double u2 = u01_from_u64(state);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(MathConst::MY_2PI * u2);
}
}    // namespace

struct FixTSPINNH::ParsedArgs {
  std::vector<std::string> fixnh_strings;
  std::vector<char *> fixnh_argv;
  int fixnh_narg = 0;

  // tspin-specific options (parsed from a trailing "tspin ..." block)
  int spin_flag = 1;
  int lattice_flag = 1;
  int ghost_custom = 0;
  int seed = 12345;
  int reinit_spin_vel = 0;
  int fm_is_frequency = 0;
  std::vector<double> mass_factor;    // size 0 => default 1.0 for all types
};

FixTSPINNH::ParsedArgs FixTSPINNH::parse_tspin_trailing_block(LAMMPS *lmp, int narg, char **arg)
{
  ParsedArgs out;
  out.fixnh_strings.reserve(narg);

  // Find optional trailing "tspin" marker.
  // If present, everything after it is parsed as tspin-specific options and is not passed to FixNH.
  int tspin_pos = -1;
  for (int i = 3; i < narg; i++) {
    if (strcmp(arg[i], "tspin") == 0) {
      tspin_pos = i;
      break;
    }
  }

  const int pass_end = (tspin_pos >= 0) ? tspin_pos : narg;
  for (int i = 0; i < pass_end; i++) out.fixnh_strings.emplace_back(arg[i]);

  // Parse tspin-specific block.
  if (tspin_pos >= 0) {
    int iarg = tspin_pos + 1;
    while (iarg < narg) {
      if ((strcmp(arg[iarg], "on") == 0) || (strcmp(arg[iarg], "yes") == 0)) {
        out.spin_flag = 1;
        iarg += 1;
      } else if ((strcmp(arg[iarg], "off") == 0) || (strcmp(arg[iarg], "no") == 0)) {
        out.spin_flag = 0;
        iarg += 1;
      } else if (strcmp(arg[iarg], "seed") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin seed", lmp->error);
        out.seed = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.seed <= 0) lmp->error->all(FLERR, "fix tspin ... tspin seed must be > 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "lattice") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin lattice", lmp->error);
        if ((strcmp(arg[iarg + 1], "yes") == 0) || (strcmp(arg[iarg + 1], "on") == 0) || (strcmp(arg[iarg + 1], "1") == 0)) {
          out.lattice_flag = 1;
        } else if ((strcmp(arg[iarg + 1], "no") == 0) || (strcmp(arg[iarg + 1], "off") == 0) || (strcmp(arg[iarg + 1], "0") == 0)) {
          out.lattice_flag = 0;
        } else {
          lmp->error->all(FLERR, "fix tspin ... tspin lattice must be 'on' or 'off'");
        }
        iarg += 2;
      } else if (strcmp(arg[iarg], "reinit") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin reinit", lmp->error);
        if ((strcmp(arg[iarg + 1], "yes") == 0) || (strcmp(arg[iarg + 1], "on") == 0) || (strcmp(arg[iarg + 1], "1") == 0)) {
          out.reinit_spin_vel = 1;
        } else if ((strcmp(arg[iarg + 1], "no") == 0) || (strcmp(arg[iarg + 1], "off") == 0) || (strcmp(arg[iarg + 1], "0") == 0)) {
          out.reinit_spin_vel = 0;
        } else {
          lmp->error->all(FLERR, "fix tspin ... tspin reinit must be 'on' or 'off'");
        }
        iarg += 2;
      } else if (strcmp(arg[iarg], "mu") == 0) {
        lmp->error->all(FLERR, "fix tspin ... tspin mu is no longer supported (use tspin mass)");
      } else if (strcmp(arg[iarg], "dtf") == 0) {
        lmp->error->all(FLERR, "fix tspin ... tspin dtf is no longer supported (spin updates always use dtf)");
      } else if (strcmp(arg[iarg], "mass") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin mass", lmp->error);
        const int ntypes = lmp->atom->ntypes;
        int count = 0;
        while ((iarg + 1 + count) < narg && utils::is_double(arg[iarg + 1 + count])) count++;

        if (count == 1) {
          // allow even if ntypes==0 (we will expand later in init()).
          out.mass_factor.assign(2, utils::numeric(FLERR, arg[iarg + 1], false, lmp));
        } else {
          if (ntypes <= 0)
            lmp->error->all(FLERR, "fix tspin ... tspin mass with multiple values requires atom types to be defined");
          if (count != ntypes)
            lmp->error->all(FLERR, "Illegal fix tspin ... tspin mass values (need 1 or ntypes values)");
          out.mass_factor.assign(ntypes + 1, 1.0);
          for (int t = 1; t <= ntypes; t++) out.mass_factor[t] = utils::numeric(FLERR, arg[iarg + t], false, lmp);
        }
        iarg += 1 + count;
      } else if (strcmp(arg[iarg], "ghost") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin ghost", lmp->error);
        out.ghost_custom = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.ghost_custom != 0 && out.ghost_custom != 1)
          lmp->error->all(FLERR, "fix tspin ... tspin ghost must be 0 or 1");
        iarg += 2;
      } else if (strcmp(arg[iarg], "fm_units") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin fm_units", lmp->error);
        if (strcmp(arg[iarg + 1], "field") == 0) out.fm_is_frequency = 0;
        else if (strcmp(arg[iarg + 1], "energy") == 0) {
          lmp->error->warning(FLERR, "fix tspin ... tspin fm_units 'energy' is deprecated; use 'field' (eV/μB)");
          out.fm_is_frequency = 0;
        } else if (strcmp(arg[iarg + 1], "frequency") == 0) out.fm_is_frequency = 1;
        else lmp->error->all(FLERR, "fix tspin ... tspin fm_units must be 'field' or 'frequency'");
        iarg += 2;
      } else {
        lmp->error->all(FLERR, "Illegal fix tspin ... tspin option: {}", arg[iarg]);
      }
    }
  }

  // finalize argv pointers (must be stable during FixNH construction)
  out.fixnh_argv.reserve(out.fixnh_strings.size());
  for (auto &s : out.fixnh_strings) out.fixnh_argv.push_back(s.data());
  out.fixnh_narg = static_cast<int>(out.fixnh_argv.size());

  return out;
}

FixTSPINNH::FixTSPINNH(LAMMPS *lmp, int narg, char **arg) :
    FixTSPINNH(lmp, parse_tspin_trailing_block(lmp, narg, arg))
{
}

FixTSPINNH::FixTSPINNH(LAMMPS *lmp, ParsedArgs parsed) :
    FixNH(lmp, parsed.fixnh_narg, parsed.fixnh_argv.data()),
    size_vector_nh(size_vector),
    lattice_flag(parsed.lattice_flag),
    spin_flag(parsed.spin_flag),
    spin_dof(0),
    ghost_custom(parsed.ghost_custom),
    idx_vs(-1),
    idx_sreal(-1),
    idx_smass(-1),
    idx_isspin(-1),
    vs(nullptr),
    sreal(nullptr),
    smass(nullptr),
    isspin(nullptr),
    nmax_old(0),
    grow_callback_added(0),
    restart_callback_added(0),
    restart_from_legacy(0),
    mass_factor(std::move(parsed.mass_factor)),
    seed(parsed.seed),
    reinit_spin_vel(parsed.reinit_spin_vel),
    spin_state_initialized(0),
    fm_is_frequency(parsed.fm_is_frequency),
    twoKs_global(0.0),
    Ks_global(0.0),
    etas(),
    etas_dot(),
    etas_dotdot(),
    etas_mass()
{
  // If atom style doesn't support spins, force-disable.
  if (!atom->sp_flag) spin_flag = 0;

  // Append two vector components for spin outputs without changing FixNH indexing.
  // f_ID[size_vector_nh]     = Ks_global
  // f_ID[size_vector_nh + 1] = twoKs_global
  size_vector += 2;

  // Prepare per-atom state storage so it migrates with atoms across MPI ranks.
  // We store state in Atom custom properties but must register exchange callbacks
  // (like fix property/atom) to pack/unpack them during Comm::exchange().
  if (atom->sp_flag) {
    ensure_custom_peratom();
    maxexchange = 8;
    restart_peratom = 1;
    atom->add_callback(Atom::GROW);
    atom->add_callback(Atom::RESTART);
    grow_callback_added = 1;
    restart_callback_added = 1;
    grow_arrays(atom->nmax);
  }
}

FixTSPINNH::~FixTSPINNH()
{
  if (copymode) return;
  if (grow_callback_added && modify && (modify->find_fix(id) >= 0)) atom->delete_callback(id, Atom::GROW);
  if (restart_callback_added && modify && (modify->find_fix(id) >= 0)) atom->delete_callback(id, Atom::RESTART);
}

void FixTSPINNH::ensure_custom_peratom()
{
  // ghost_custom default 0 (this fix uses owned atoms only); set `... tspin ghost 1` only if a future pair/compute needs tspin_* on ghost atoms.

  int flag = -1, cols = -1, ghost = 0;

  idx_vs = atom->find_custom_ghost("tspin_vs", flag, cols, ghost);
  if (idx_vs < 0) idx_vs = atom->add_custom("tspin_vs", 1, 3, ghost_custom);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property tspin_vs has incompatible type");

  idx_sreal = atom->find_custom_ghost("tspin_sreal", flag, cols, ghost);
  if (idx_sreal < 0) idx_sreal = atom->add_custom("tspin_sreal", 1, 3, ghost_custom);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property tspin_sreal has incompatible type");

  idx_smass = atom->find_custom_ghost("tspin_smass", flag, cols, ghost);
  if (idx_smass < 0) idx_smass = atom->add_custom("tspin_smass", 1, 0, ghost_custom);
  else if (flag != 1 || cols != 0) error->all(FLERR, "Custom property tspin_smass has incompatible type");

  idx_isspin = atom->find_custom_ghost("tspin_isspin", flag, cols, ghost);
  if (idx_isspin < 0) idx_isspin = atom->add_custom("tspin_isspin", 0, 0, ghost_custom);
  else if (flag != 0 || cols != 0) error->all(FLERR, "Custom property tspin_isspin has incompatible type");

  // cache pointers for fast access (may be refreshed on grow)
  vs = atom->darray[idx_vs];
  sreal = atom->darray[idx_sreal];
  smass = atom->dvector[idx_smass];
  isspin = atom->ivector[idx_isspin];
}

void FixTSPINNH::grow_arrays(int nmax)
{
  // AtomVec::grow() does not grow Atom::add_custom() arrays; the creating fix
  // must do so via its grow_arrays() callback.
  if (idx_vs < 0 || idx_sreal < 0 || idx_smass < 0 || idx_isspin < 0) return;

  if (nmax > nmax_old) {
    memory->grow(atom->darray[idx_vs], nmax, 3, "tspin/nh:vs");
    memory->grow(atom->darray[idx_sreal], nmax, 3, "tspin/nh:sreal");
    memory->grow(atom->dvector[idx_smass], nmax, "tspin/nh:smass");
    memory->grow(atom->ivector[idx_isspin], nmax, "tspin/nh:isspin");
  }

  vs = atom->darray[idx_vs];
  sreal = atom->darray[idx_sreal];
  smass = atom->dvector[idx_smass];
  isspin = atom->ivector[idx_isspin];

  // initialize new entries (growth does not guarantee initialization)
  for (int i = nmax_old; i < nmax; i++) {
    if (vs) vs[i][0] = vs[i][1] = vs[i][2] = 0.0;
    if (sreal) sreal[i][0] = sreal[i][1] = sreal[i][2] = 0.0;
    if (smass) smass[i] = 0.0;
    if (isspin) isspin[i] = 0;
  }
  nmax_old = nmax;
}

void FixTSPINNH::copy_arrays(int i, int j, int /*delflag*/)
{
  if (idx_vs < 0 || idx_sreal < 0 || idx_smass < 0 || idx_isspin < 0) return;
  if (!vs || !sreal || !smass || !isspin) ensure_custom_peratom();

  for (int k = 0; k < 3; k++) {
    vs[j][k] = vs[i][k];
    sreal[j][k] = sreal[i][k];
  }
  smass[j] = smass[i];
  isspin[j] = isspin[i];
}

int FixTSPINNH::pack_exchange(int i, double *buf)
{
  if (idx_vs < 0 || idx_sreal < 0 || idx_smass < 0 || idx_isspin < 0) return 0;
  if (!vs || !sreal || !smass || !isspin) ensure_custom_peratom();

  buf[0] = vs[i][0];
  buf[1] = vs[i][1];
  buf[2] = vs[i][2];
  buf[3] = sreal[i][0];
  buf[4] = sreal[i][1];
  buf[5] = sreal[i][2];
  buf[6] = smass[i];
  buf[7] = static_cast<double>(isspin[i]);
  return 8;
}

int FixTSPINNH::unpack_exchange(int nlocal, double *buf)
{
  if (idx_vs < 0 || idx_sreal < 0 || idx_smass < 0 || idx_isspin < 0) return 0;
  if (!vs || !sreal || !smass || !isspin) ensure_custom_peratom();

  vs[nlocal][0] = buf[0];
  vs[nlocal][1] = buf[1];
  vs[nlocal][2] = buf[2];
  sreal[nlocal][0] = buf[3];
  sreal[nlocal][1] = buf[4];
  sreal[nlocal][2] = buf[5];
  smass[nlocal] = buf[6];
  isspin[nlocal] = static_cast<int>(buf[7]);
  return 8;
}

int FixTSPINNH::pack_restart(int i, double *buf)
{
  if (!atom->sp_flag || idx_vs < 0 || idx_sreal < 0 || idx_smass < 0 || idx_isspin < 0) {
    buf[0] = 1.0;
    return 1;
  }
  if (!vs || !sreal || !smass || !isspin) ensure_custom_peratom();

  buf[0] = 9.0;
  buf[1] = vs[i][0];
  buf[2] = vs[i][1];
  buf[3] = vs[i][2];
  buf[4] = sreal[i][0];
  buf[5] = sreal[i][1];
  buf[6] = sreal[i][2];
  buf[7] = smass[i];
  buf[8] = static_cast<double>(isspin[i]);
  return 9;
}

void FixTSPINNH::unpack_restart(int nlocal, int nth)
{
  if (!atom->sp_flag || idx_vs < 0 || idx_sreal < 0 || idx_smass < 0 || idx_isspin < 0) return;
  if (!vs || !sreal || !smass || !isspin) ensure_custom_peratom();

  double **extra = atom->extra;
  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int>(extra[nlocal][m]);

  const int nvals = static_cast<int>(extra[nlocal][m++]);
  if (nvals <= 1) {
    vs[nlocal][0] = vs[nlocal][1] = vs[nlocal][2] = 0.0;
    sreal[nlocal][0] = sreal[nlocal][1] = sreal[nlocal][2] = 0.0;
    smass[nlocal] = 0.0;
    isspin[nlocal] = 0;
    return;
  }

  if (nvals < 9) {
    error->warning(FLERR,
                   "Fix {} style {} encountered truncated per-atom restart payload; using safe fallback for atom {}",
                   id, style, nlocal);
    vs[nlocal][0] = vs[nlocal][1] = vs[nlocal][2] = 0.0;
    sreal[nlocal][0] = sreal[nlocal][1] = sreal[nlocal][2] = 0.0;
    smass[nlocal] = 0.0;
    isspin[nlocal] = 0;
    return;
  }

  vs[nlocal][0] = extra[nlocal][m++];
  vs[nlocal][1] = extra[nlocal][m++];
  vs[nlocal][2] = extra[nlocal][m++];
  sreal[nlocal][0] = extra[nlocal][m++];
  sreal[nlocal][1] = extra[nlocal][m++];
  sreal[nlocal][2] = extra[nlocal][m++];
  smass[nlocal] = extra[nlocal][m++];
  isspin[nlocal] = static_cast<int>(extra[nlocal][m++]);
}

int FixTSPINNH::maxsize_restart()
{
  if (!atom->sp_flag) return 1;
  return 9;
}

int FixTSPINNH::size_restart(int /*nlocal*/)
{
  if (!atom->sp_flag) return 1;
  return 9;
}

void FixTSPINNH::init()
{
  FixNH::init();

  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    if (utils::strmatch(modify->fix[ifix]->style, "^precession/spin")) {
      error->all(FLERR,
                 "Fix {} (USER-TSPIN) does not support fix precession/spin; use fix tspin/precession/spin or setforce/spin instead",
                 style);
    }
  }

  if (!lattice_flag && pstat_flag)
    error->all(FLERR, "Fix {} cannot disable lattice integration with pressure control enabled", style);

  if (!spin_flag) return;

  ensure_mass_factor();

  // Keep chain runtime state restored from new-format restart; only initialize
  // defaults for non-restart, legacy fallback, or incompatible chain lengths.
  const int mchain = mtchain;
  const bool chain_from_restart = (!restart_from_legacy) && (static_cast<int>(etas.size()) == mchain) &&
      (static_cast<int>(etas_dot.size()) == (mchain + 1)) && (static_cast<int>(etas_dotdot.size()) == mchain) &&
      (static_cast<int>(etas_mass.size()) == mchain);
  if (!chain_from_restart) {
    etas.assign(mchain, 0.0);
    etas_dot.assign(mchain + 1, 0.0);
    etas_dotdot.assign(mchain, 0.0);
    etas_mass.assign(mchain, 0.0);
  }

  ensure_custom_peratom();
}

void FixTSPINNH::setup(int vflag)
{
  FixNH::setup(vflag);

  if (!spin_flag) return;
  if (!atom->sp_flag) return;

  // Defensive: some workflows (e.g., `run ... pre no`) may bypass init().
  // Ensure custom per-atom storage and spin mass factor are available before use.
  ensure_custom_peratom();
  ensure_mass_factor();

  update_spin_dof_and_flags();
  refresh_spin_state_from_atom();
  if (!spin_state_initialized || reinit_spin_vel) init_spin_velocities_legacy();
  spin_state_initialized = 1;
  compute_twoKs_global();
}

void FixTSPINNH::ensure_mass_factor()
{
  const int ntypes = atom->ntypes;
  if (ntypes <= 0) return;

  if (mass_factor.empty()) {
    mass_factor.assign(ntypes + 1, 1.0);
  } else if (static_cast<int>(mass_factor.size()) == 2) {
    // special case: scalar provided before ntypes was known
    const double v = mass_factor[1];
    mass_factor.assign(ntypes + 1, v);
  } else if (static_cast<int>(mass_factor.size()) != ntypes + 1) {
    mass_factor.assign(ntypes + 1, 1.0);
  }
}

void FixTSPINNH::update_spin_dof_and_flags()
{
  auto **sp = atom->sp;
  auto *mask = atom->mask;
  const int nlocal = atom->nlocal;
  auto *isspin = atom->ivector[idx_isspin];

  int local_dof = 0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) {
      isspin[i] = 0;
      continue;
    }
    const double smag = sp[i][3];
    if (smag > SPIN_MAG_EPS) {
      isspin[i] = 1;
      local_dof += 3;
    } else {
      isspin[i] = 0;
    }
  }

  MPI_Allreduce(&local_dof, &spin_dof, 1, MPI_INT, MPI_SUM, world);
}

void FixTSPINNH::refresh_spin_state_from_atom()
{
  if (atom->ntypes <= 0) error->all(FLERR, "Fix {} requires atom types to be defined", style);
  if (atom->mass == nullptr)
    error->all(FLERR, "Fix {} requires per-type masses (use 'mass' command or Masses section in data file)", style);
  ensure_mass_factor();

  const int nlocal = atom->nlocal;
  auto **sp = atom->sp;
  auto **sreal = atom->darray[idx_sreal];
  auto *smass = atom->dvector[idx_smass];
  auto *mask = atom->mask;
  auto *type = atom->type;
  auto *amass = atom->mass;
  auto *mass_setflag = atom->mass_setflag;

  // mu_i ("mass") = atom_mass[type] * mass_factor[type]
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const int itype = type[i];
    if (itype < 1 || itype > atom->ntypes) error->all(FLERR, "Fix {} encountered invalid atom type {}", style, itype);
    if (mass_setflag && !mass_setflag[itype])
      error->all(FLERR, "Fix {} requires mass for atom type {} (use 'mass' command or Masses section)", style, itype);
    smass[i] = amass[itype] * mass_factor[itype];
  }

  // S = sp(unit) * sp(mag)
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const double smag = sp[i][3];
    sreal[i][0] = sp[i][0] * smag;
    sreal[i][1] = sp[i][1] * smag;
    sreal[i][2] = sp[i][2] * smag;
  }
}

void FixTSPINNH::init_spin_velocities_legacy()
{
  if (seed <= 0) error->all(FLERR, "Fix {} requires tspin seed > 0", style);

  refresh_spin_state_from_atom();

  // Fixed policy: seed stream is always keyed by atom ID (tag), so it is invariant
  // under MPI decomposition changes.
  if (atom->tag_enable == 0)
    error->all(FLERR, "Fix {} requires atoms to have IDs (tag_enable) for tspin initialization", style);

  const int nlocal = atom->nlocal;
  auto **vs = atom->darray[idx_vs];
  auto *smass = atom->dvector[idx_smass];
  auto *isspin = atom->ivector[idx_isspin];
  auto *mask = atom->mask;
  auto *type = atom->type;
  auto *tag = atom->tag;

  constexpr double vscale0 = 0.5;

  // Decomposition-independent initialization similar to "velocity create ... loop all":
  // vs is keyed by atom ID, so it is invariant under MPI decomposition changes.
  const std::uint64_t base_seed = static_cast<std::uint64_t>(seed);
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (!isspin[i]) {
      vs[i][0] = vs[i][1] = vs[i][2] = 0.0;
      continue;
    }
    const tagint tid = tag[i];
    vs[i][0] = vscale0 * gaussian_tag(base_seed, tid, 0);
    vs[i][1] = vscale0 * gaussian_tag(base_seed, tid, 1);
    vs[i][2] = vscale0 * gaussian_tag(base_seed, tid, 2);
  }

  const int ntypes = atom->ntypes;
  for (int itype = 1; itype <= ntypes; itype++) {
    double twoK_local = 0.0;
    int dof_local = 0;

    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (!isspin[i]) continue;
      if (type[i] != itype) continue;

      const double mu_i = smass[i];
      if (mu_i <= 0.0) error->all(FLERR, "Fix {} requires positive mu_i for spin atoms", style);

      const double v2 = vs[i][0] * vs[i][0] + vs[i][1] * vs[i][1] + vs[i][2] * vs[i][2];
      twoK_local += mu_i * v2;
      dof_local += 3;
    }

    double twoK_global_units = 0.0;
    int dof_global = 0;
    MPI_Allreduce(&twoK_local, &twoK_global_units, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&dof_local, &dof_global, 1, MPI_INT, MPI_SUM, world);

    if (dof_global == 0) continue;
    const double twoK_energy = twoK_global_units * force->mvv2e;
    const double t_type = twoK_energy / (static_cast<double>(dof_global) * force->boltz);
    if (t_type <= 0.0) continue;

    const double rescale = std::sqrt(t_start / t_type);
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (!isspin[i]) continue;
      if (type[i] != itype) continue;
      vs[i][0] *= rescale;
      vs[i][1] *= rescale;
      vs[i][2] *= rescale;
    }
  }
}

void FixTSPINNH::compute_twoKs_global()
{
  const int nlocal = atom->nlocal;
  auto **vs = atom->darray[idx_vs];
  auto *smass = atom->dvector[idx_smass];
  auto *isspin = atom->ivector[idx_isspin];
  auto *mask = atom->mask;

  double twoKs_local = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (!isspin[i]) continue;
    const double v2 = vs[i][0] * vs[i][0] + vs[i][1] * vs[i][1] + vs[i][2] * vs[i][2];
    twoKs_local += smass[i] * v2;
  }
  twoKs_local *= force->mvv2e;

  MPI_Allreduce(&twoKs_local, &twoKs_global, 1, MPI_DOUBLE, MPI_SUM, world);
  Ks_global = 0.5 * twoKs_global;
}

void FixTSPINNH::nhc_spin_integrate()
{
  if (!spin_flag) return;
  if (spin_dof <= 0) return;

  compute_twoKs_global();
  const double twoK_target_spin = static_cast<double>(spin_dof) * force->boltz * t_target;

  // mimic FixNH eta_mass update pattern
  etas_mass[0] = twoK_target_spin / (t_freq * t_freq);
  for (int ich = 1; ich < mtchain; ich++) etas_mass[ich] = force->boltz * t_target / (t_freq * t_freq);

  etas_dotdot[0] = (etas_mass[0] > 0.0) ? ((twoKs_global - twoK_target_spin) / etas_mass[0]) : 0.0;

  const double ncfac = 1.0 / nc_tchain;
  for (int iloop = 0; iloop < nc_tchain; iloop++) {
    for (int ich = mtchain - 1; ich > 0; ich--) {
      const double expfac = std::exp(-ncfac * dt8 * etas_dot[ich + 1]);
      etas_dot[ich] *= expfac;
      etas_dot[ich] += etas_dotdot[ich] * ncfac * dt4;
      etas_dot[ich] *= tdrag_factor;
      etas_dot[ich] *= expfac;
    }

    double expfac = std::exp(-ncfac * dt8 * etas_dot[1]);
    etas_dot[0] *= expfac;
    etas_dot[0] += etas_dotdot[0] * ncfac * dt4;
    etas_dot[0] *= tdrag_factor;
    etas_dot[0] *= expfac;

    const double factor = std::exp(-ncfac * dthalf * etas_dot[0]);
    auto **vs = atom->darray[idx_vs];
    auto *isspin = atom->ivector[idx_isspin];
    auto *mask = atom->mask;
    const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      if (!isspin[i]) continue;
      vs[i][0] *= factor;
      vs[i][1] *= factor;
      vs[i][2] *= factor;
    }

    twoKs_global *= factor * factor;
    Ks_global = 0.5 * twoKs_global;

    etas_dotdot[0] = (etas_mass[0] > 0.0) ? ((twoKs_global - twoK_target_spin) / etas_mass[0]) : 0.0;
    for (int ich = 0; ich < mtchain; ich++) etas[ich] += ncfac * dthalf * etas_dot[ich];

    etas_dot[0] *= expfac;
    etas_dot[0] += etas_dotdot[0] * ncfac * dt4;
    etas_dot[0] *= expfac;

    for (int ich = 1; ich < mtchain; ich++) {
      expfac = std::exp(-ncfac * dt8 * etas_dot[ich + 1]);
      etas_dot[ich] *= expfac;
      etas_dotdot[ich] = (etas_mass[ich] > 0.0) ?
          ((etas_mass[ich - 1] * etas_dot[ich - 1] * etas_dot[ich - 1] - force->boltz * t_target) / etas_mass[ich]) :
          0.0;
      etas_dot[ich] += etas_dotdot[ich] * ncfac * dt4;
      etas_dot[ich] *= expfac;
    }
  }
}

void FixTSPINNH::nve_v_spin()
{
  auto **vs = atom->darray[idx_vs];
  auto **fm = atom->fm;
  auto *smass = atom->dvector[idx_smass];
  auto *isspin = atom->ivector[idx_isspin];
  auto *mask = atom->mask;
  int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;

  constexpr double g = 2.0;
  double fm_scale = 1.0;
  double dt_spin = dtf;
  if (fm_is_frequency) {
    const double hbar = force->hplanck / MathConst::MY_2PI;
    if (hbar == 0.0) error->all(FLERR, "Fix {} tspin fm_units frequency requires nonzero hbar (use physical units)", style);
    dt_spin = dthalf;
    fm_scale = hbar / g;
  }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (!isspin[i]) continue;
    const double mu_i = smass[i];
    if (mu_i <= 0.0) continue;
    const double dtmu = dt_spin / mu_i;
    vs[i][0] += dtmu * fm_scale * fm[i][0];
    vs[i][1] += dtmu * fm_scale * fm[i][1];
    vs[i][2] += dtmu * fm_scale * fm[i][2];
  }
}

void FixTSPINNH::nve_s_spin()
{
  auto **sreal = atom->darray[idx_sreal];
  auto **sp = atom->sp;
  auto **vs = atom->darray[idx_vs];
  auto *isspin = atom->ivector[idx_isspin];
  auto *mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (!isspin[i]) continue;
    sreal[i][0] += dtv * vs[i][0];
    sreal[i][1] += dtv * vs[i][1];
    sreal[i][2] += dtv * vs[i][2];

    const double smag = std::sqrt(sreal[i][0] * sreal[i][0] + sreal[i][1] * sreal[i][1] + sreal[i][2] * sreal[i][2]);
    sp[i][3] = smag;
    if (smag > SPIN_MAG_EPS) {
      sp[i][0] = sreal[i][0] / smag;
      sp[i][1] = sreal[i][1] / smag;
      sp[i][2] = sreal[i][2] / smag;
    }
  }
}

void FixTSPINNH::initial_integrate(int vflag)
{
  // Copy FixNH::initial_integrate() order and insert spin steps where legacy code did.

  if (pstat_flag && mpchain) nhc_press_integrate();

  // Compute this step's target temperature once so both lattice and spin thermostats
  // use the same value (important when ramping T).
  if (tstat_flag) compute_temp_target();

  if (spin_flag) nhc_spin_integrate();
  if (tstat_flag && lattice_flag) nhc_temp_integrate();

  if (pstat_flag) {
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep + 1);
  }

  if (pstat_flag) {
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
  }

  if (spin_flag) nve_v_spin();
  if (lattice_flag) nve_v();

  if (pstat_flag) remap();

  if (spin_flag) nve_s_spin();
  if (lattice_flag) nve_x();

  if (pstat_flag) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

void FixTSPINNH::final_integrate()
{
  if (lattice_flag) nve_v();
  if (spin_flag) nve_v_spin();

  if (which == BIAS && neighbor->ago == 0) t_current = temperature->compute_scalar();

  if (pstat_flag) nh_v_press();

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  if (pstat_flag) {
    if (pstyle == ISO) pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep + 1);
  }

  if (pstat_flag) nh_omega_dot();

  if (tstat_flag && lattice_flag) nhc_temp_integrate();
  if (spin_flag) nhc_spin_integrate();
  if (pstat_flag && mpchain) nhc_press_integrate();
}

double FixTSPINNH::compute_scalar()
{
  // Base extended energy (lattice chain + barostat, if enabled)
  double energy = FixNH::compute_scalar();

  if (!spin_flag) return energy;

  compute_twoKs_global();

  // add spin chain extended energy + spin kinetic energy (Ks_global)
  const double kt = boltz * t_target;
  const double twoK_target_spin = static_cast<double>(spin_dof) * boltz * t_target;

  energy += Ks_global;

  if (mtchain > 0 && !etas.empty()) {
    energy += twoK_target_spin * etas[0] + 0.5 * etas_mass[0] * etas_dot[0] * etas_dot[0];
    for (int ich = 1; ich < mtchain; ich++)
      energy += kt * etas[ich] + 0.5 * etas_mass[ich] * etas_dot[ich] * etas_dot[ich];
  }

  return energy;
}

double FixTSPINNH::compute_vector(int n)
{
  // Preserve FixNH vector layout and append two extra components.
  if (n < size_vector_nh) return FixNH::compute_vector(n);

  if (!spin_flag) return 0.0;
  compute_twoKs_global();

  if (n == size_vector_nh) return Ks_global;
  if (n == size_vector_nh + 1) return twoKs_global;
  return 0.0;
}

int FixTSPINNH::modify_param(int narg, char **arg)
{
  // enable configuring spin-specific parameters without modifying FixNH parsing.
  // Example:
  //   fix_modify ID tspin seed 12345
  //   fix_modify ID tspin mass 1.0          (all types)
  //   fix_modify ID tspin mass 1.0 2.0 ...  (ntypes values)
  //   fix_modify ID tspin on|off

  if (narg < 1) return FixNH::modify_param(narg, arg);

  if (strcmp(arg[0], "tspin") != 0) return FixNH::modify_param(narg, arg);
  if (narg < 2) error->all(FLERR, "Illegal fix_modify tspin command");

  if ((strcmp(arg[1], "on") == 0) || (strcmp(arg[1], "yes") == 0)) {
    spin_flag = 1;
    return 2;
  }
  if ((strcmp(arg[1], "off") == 0) || (strcmp(arg[1], "no") == 0)) {
    spin_flag = 0;
    return 2;
  }

  if (strcmp(arg[1], "lattice") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify tspin lattice command");
    if ((strcmp(arg[2], "yes") == 0) || (strcmp(arg[2], "on") == 0) || (strcmp(arg[2], "1") == 0)) {
      lattice_flag = 1;
    } else if ((strcmp(arg[2], "no") == 0) || (strcmp(arg[2], "off") == 0) || (strcmp(arg[2], "0") == 0)) {
      lattice_flag = 0;
    } else {
      error->all(FLERR, "Illegal fix_modify tspin lattice command (must be 'on' or 'off')");
    }
    if (!lattice_flag && pstat_flag)
      error->all(FLERR, "Fix {} cannot disable lattice integration with pressure control enabled", style);
    return 3;
  }

  if (strcmp(arg[1], "reinit") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify tspin reinit command");
    if ((strcmp(arg[2], "yes") == 0) || (strcmp(arg[2], "on") == 0) || (strcmp(arg[2], "1") == 0)) {
      reinit_spin_vel = 1;
    } else if ((strcmp(arg[2], "no") == 0) || (strcmp(arg[2], "off") == 0) || (strcmp(arg[2], "0") == 0)) {
      reinit_spin_vel = 0;
    } else {
      error->all(FLERR, "Illegal fix_modify tspin reinit command (must be 'on' or 'off')");
    }
    return 3;
  }

  if (strcmp(arg[1], "seed") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify tspin seed command");
    seed = utils::inumeric(FLERR, arg[2], false, lmp);
    if (seed <= 0) error->all(FLERR, "Illegal fix_modify tspin seed command (seed must be > 0)");
    return 3;
  }

  if (strcmp(arg[1], "mu") == 0) {
    error->all(FLERR, "Illegal fix_modify tspin mu command (use 'mass')");
  }

  if (strcmp(arg[1], "dtf") == 0) {
    error->all(FLERR, "Illegal fix_modify tspin dtf command (spin updates always use dtf)");
  }

  if (strcmp(arg[1], "mass") == 0) {
    const int ntypes = atom->ntypes;
    if (narg < 3) error->all(FLERR, "Illegal fix_modify tspin mass command");

    int count = 0;
    while (count < ntypes && (2 + count) < narg && utils::is_double(arg[2 + count])) count++;
    if (count != 1 && count != ntypes) error->all(FLERR, "Illegal fix_modify tspin mass values");

    mass_factor.assign(ntypes + 1, 1.0);
    if (count == 1) {
      const double v = utils::numeric(FLERR, arg[2], false, lmp);
      for (int t = 1; t <= ntypes; t++) mass_factor[t] = v;
    } else {
      for (int t = 1; t <= ntypes; t++) mass_factor[t] = utils::numeric(FLERR, arg[1 + t], false, lmp);
    }
    return 2 + count;
  }

  if (strcmp(arg[1], "fm_units") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify tspin fm_units command");
    if (strcmp(arg[2], "frequency") == 0) {
      fm_is_frequency = 1;
    } else if (strcmp(arg[2], "field") == 0) {
      fm_is_frequency = 0;
    } else if (strcmp(arg[2], "energy") == 0) {
      error->warning(FLERR, "Fix {} tspin fm_units 'energy' is deprecated; use 'field' (eV/μB)", style);
      fm_is_frequency = 0;
    } else {
      error->all(FLERR, "Illegal fix_modify tspin fm_units command (must be 'field' or 'frequency')");
    }
    return 3;
  }

  error->all(FLERR, "Illegal fix_modify tspin command");
  return 0;
}

int FixTSPINNH::nh_payload_size_from_list(const double *list, int max_n)
{
  int n = 0;
  auto pull_int = [&](int &v) -> bool {
    if (max_n >= 0 && n >= max_n) return false;
    v = static_cast<int>(list[n++]);
    return true;
  };
  auto skip = [&](int count) -> bool {
    if (count < 0) return false;
    if (max_n >= 0 && (n + count > max_n)) return false;
    n += count;
    return true;
  };

  int flag = 0;
  if (!pull_int(flag)) return -1;
  if (flag) {
    int m = 0;
    if (!pull_int(m)) return -1;
    if (!skip(2 * m)) return -1;
  }

  if (!pull_int(flag)) return -1;
  if (flag) {
    if (!skip(14)) return -1;    // omega[6], omega_dot[6], vol0, t0
    int m = 0;
    if (!pull_int(m)) return -1;
    if (!skip(2 * m)) return -1;    // etap, etap_dot
    if (!pull_int(flag)) return -1;
    if (flag && !skip(6)) return -1;    // h0_inv[6]
  }

  return n;
}

int FixTSPINNH::pack_restart_payload_v1(double *list) const
{
  int n = 0;
  const int mchain = static_cast<int>(etas.size());

  if (list) list[n] = static_cast<double>(spin_flag);
  n++;
  if (list) list[n] = static_cast<double>(lattice_flag);
  n++;
  if (list) list[n] = static_cast<double>(seed);
  n++;
  if (list) list[n] = static_cast<double>(reinit_spin_vel);
  n++;
  if (list) list[n] = static_cast<double>(spin_state_initialized);
  n++;
  if (list) list[n] = static_cast<double>(fm_is_frequency);
  n++;
  if (list) list[n] = static_cast<double>(spin_dof);
  n++;
  if (list) list[n] = twoKs_global;
  n++;
  if (list) list[n] = Ks_global;
  n++;
  if (list) list[n] = static_cast<double>(mchain);
  n++;

  for (int i = 0; i < mchain; i++) {
    if (list) list[n] = etas[i];
    n++;
  }
  for (int i = 0; i < mchain + 1; i++) {
    if (list) list[n] = etas_dot[i];
    n++;
  }
  for (int i = 0; i < mchain; i++) {
    if (list) list[n] = etas_dotdot[i];
    n++;
  }
  for (int i = 0; i < mchain; i++) {
    if (list) list[n] = etas_mass[i];
    n++;
  }

  return n;
}

void FixTSPINNH::unpack_restart_payload_v1(const double *list)
{
  int n = 0;
  spin_flag = static_cast<int>(list[n++]);
  lattice_flag = static_cast<int>(list[n++]);
  seed = static_cast<int>(list[n++]);
  reinit_spin_vel = static_cast<int>(list[n++]);
  spin_state_initialized = static_cast<int>(list[n++]);
  fm_is_frequency = static_cast<int>(list[n++]);
  spin_dof = static_cast<int>(list[n++]);
  twoKs_global = list[n++];
  Ks_global = list[n++];

  const int mchain = static_cast<int>(list[n++]);
  if (mchain < 0) error->all(FLERR, "Fix {} style {} restart payload has invalid tspin chain length {}", id, style,
                             mchain);

  etas.assign(mchain, 0.0);
  etas_dot.assign(mchain + 1, 0.0);
  etas_dotdot.assign(mchain, 0.0);
  etas_mass.assign(mchain, 0.0);

  for (int i = 0; i < mchain; i++) etas[i] = list[n++];
  for (int i = 0; i < mchain + 1; i++) etas_dot[i] = list[n++];
  for (int i = 0; i < mchain; i++) etas_dotdot[i] = list[n++];
  for (int i = 0; i < mchain; i++) etas_mass[i] = list[n++];
}

int FixTSPINNH::size_restart_global()
{
  return 2 + 1 + FixNH::size_restart_global() + 1 + pack_restart_payload_v1(nullptr);
}

int FixTSPINNH::pack_restart_data(double *list)
{
  int n = 0;
  list[n++] = RESTART_MAGIC;
  list[n++] = static_cast<double>(RESTART_VERSION);

  const int nh_n = FixNH::pack_restart_data(list + n + 1);
  list[n] = static_cast<double>(nh_n);
  n += nh_n + 1;

  const int tspin_n = pack_restart_payload_v1(list + n + 1);
  list[n] = static_cast<double>(tspin_n);
  n += tspin_n + 1;

  return n;
}

void FixTSPINNH::restart(char *buf)
{
  auto *list = reinterpret_cast<double *>(buf);
  restart_from_legacy = 0;

  if (list[0] != RESTART_MAGIC) {
    FixNH::restart(buf);
    restart_from_legacy = 1;
    spin_state_initialized = 0;
    error->warning(
        FLERR,
        "Fix {} style {} read legacy restart payload without USER-TSPIN tspin state; compatibility fallback initialization will be used",
        id, style);
    return;
  }

  const int version = static_cast<int>(list[1]);
  if (version != RESTART_VERSION)
    error->all(FLERR, "Fix {} style {} restart payload version {} is not supported", id, style, version);

  int n = 2;
  const int nh_n = static_cast<int>(list[n++]);
  if (nh_n <= 0) error->all(FLERR, "Fix {} style {} restart payload has invalid NH size {}", id, style, nh_n);

  const int nh_parsed = nh_payload_size_from_list(list + n, nh_n);
  if (nh_parsed != nh_n)
    error->all(FLERR, "Fix {} style {} restart payload NH size mismatch (stored {}, parsed {})", id, style, nh_n,
               nh_parsed);

  FixNH::restart(reinterpret_cast<char *>(list + n));
  n += nh_n;

  const int tspin_n = static_cast<int>(list[n++]);
  const double *payload = list + n;
  if (tspin_n < 10)
    error->all(FLERR, "Fix {} style {} restart payload has invalid tspin size {}", id, style, tspin_n);
  const int mchain = static_cast<int>(payload[9]);
  if (mchain < 0)
    error->all(FLERR, "Fix {} style {} restart payload has invalid tspin chain length {}", id, style, mchain);
  const int parsed_payload = 10 + mchain + (mchain + 1) + mchain + mchain;
  if (tspin_n != parsed_payload)
    error->all(FLERR, "Fix {} style {} restart payload tspin size mismatch (stored {}, parsed {})", id, style,
               tspin_n, parsed_payload);

  unpack_restart_payload_v1(payload);
  restart_from_legacy = 0;
}
