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

#include "fix_llgmid_nh.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "dihedral.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

// Keep integer values consistent with FixNH's internal enums in src/fix_nh.cpp.
enum { NOBIAS, BIAS };
enum { ISO, ANISO, TRICLINIC };

static constexpr double SPIN_EPS = 1.0e-12;

namespace {
static inline int parse_on_off(const char *s, LAMMPS *lmp, const char *what)
{
  if ((strcmp(s, "on") == 0) || (strcmp(s, "yes") == 0) || (strcmp(s, "true") == 0) || (strcmp(s, "1") == 0))
    return 1;
  if ((strcmp(s, "off") == 0) || (strcmp(s, "no") == 0) || (strcmp(s, "false") == 0) || (strcmp(s, "0") == 0))
    return 0;
  lmp->error->all(FLERR, "{} must be 'on' or 'off'", what);
  return 0;
}
}    // namespace

struct FixLLGMidNH::ParsedArgs {
  std::vector<std::string> fixnh_strings;
  std::vector<char *> fixnh_argv;
  int fixnh_narg = 0;

  int lattice_flag = 1;
  int midpoint_iter = 2;
  double midpoint_tol = 0.0;
  double alpha = 0.0;
  double gamma = -1.0;

  int debug_flag = 0;
  int debug_every = 1;
  int debug_rank = 0;
  int debug_flush = 0;
  bigint debug_start = 0;
  std::string debug_file;
};

FixLLGMidNH::ParsedArgs FixLLGMidNH::parse_llgmid_trailing_block(LAMMPS *lmp, int narg, char **arg)
{
  ParsedArgs out;
  out.fixnh_strings.reserve(narg);

  int llgmid_pos = -1;
  for (int i = 3; i < narg; i++) {
    if (strcmp(arg[i], "llgmid") == 0) {
      llgmid_pos = i;
      break;
    }
  }

  const int pass_end = (llgmid_pos >= 0) ? llgmid_pos : narg;
  for (int i = 0; i < pass_end; i++) out.fixnh_strings.emplace_back(arg[i]);

  if (llgmid_pos >= 0) {
    int iarg = llgmid_pos + 1;
    while (iarg < narg) {
      if (strcmp(arg[iarg], "lattice") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid lattice", lmp->error);
        out.lattice_flag = parse_on_off(arg[iarg + 1], lmp, "fix llgmid ... llgmid lattice");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_iter") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid midpoint_iter", lmp->error);
        out.midpoint_iter = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.midpoint_iter < 2) lmp->error->all(FLERR, "fix llgmid ... llgmid midpoint_iter must be >= 2");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_tol") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid midpoint_tol", lmp->error);
        out.midpoint_tol = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.midpoint_tol < 0.0) lmp->error->all(FLERR, "fix llgmid ... llgmid midpoint_tol must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "alpha") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid alpha", lmp->error);
        out.alpha = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.alpha < 0.0) lmp->error->all(FLERR, "fix llgmid ... llgmid alpha must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "gamma") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid gamma", lmp->error);
        out.gamma = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.gamma < 0.0) lmp->error->all(FLERR, "fix llgmid ... llgmid gamma must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "fm_units") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid fm_units", lmp->error);
        if (strcmp(arg[iarg + 1], "field") != 0)
          lmp->error->all(FLERR, "fix llgmid ... llgmid fm_units must be 'field' (H = -dE/dM in eV/muB)");
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid debug", lmp->error);
        out.debug_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_every") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid debug_every", lmp->error);
        out.debug_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.debug_every < 1) lmp->error->all(FLERR, "fix llgmid ... llgmid debug_every must be >= 1");
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_rank") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid debug_rank", lmp->error);
        out.debug_rank = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_flush") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid debug_flush", lmp->error);
        out.debug_flush = utils::logical(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_start") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid debug_start", lmp->error);
        out.debug_start = utils::bnumeric(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_file") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llgmid ... llgmid debug_file", lmp->error);
        out.debug_file = arg[iarg + 1];
        iarg += 2;
      } else {
        lmp->error->all(FLERR, "Illegal fix llgmid ... llgmid option: {}", arg[iarg]);
      }
    }
  }

  out.fixnh_argv.reserve(out.fixnh_strings.size());
  for (auto &s : out.fixnh_strings) out.fixnh_argv.push_back(s.data());
  out.fixnh_narg = static_cast<int>(out.fixnh_argv.size());
  return out;
}

FixLLGMidNH::FixLLGMidNH(LAMMPS *lmp, int narg, char **arg) :
    FixLLGMidNH(lmp, parse_llgmid_trailing_block(lmp, narg, arg))
{
}

FixLLGMidNH::FixLLGMidNH(LAMMPS *lmp, ParsedArgs parsed) :
    FixNH(lmp, parsed.fixnh_narg, parsed.fixnh_argv.data()),
    size_vector_nh(size_vector),
    lattice_flag(parsed.lattice_flag),
    midpoint_iter(parsed.midpoint_iter),
    midpoint_tol(parsed.midpoint_tol),
    alpha(parsed.alpha),
    gamma(parsed.gamma),
    hbar(0.0),
    g_over_hbar(0.0),
    debug_flag(parsed.debug_flag),
    debug_every(parsed.debug_every),
    debug_rank(parsed.debug_rank),
    debug_flush(parsed.debug_flush),
    debug_start(parsed.debug_start),
    debug_header_printed(0),
    debug_file(std::move(parsed.debug_file)),
    debug_fp(nullptr),
    pe_prev_end(0.0),
    idx_fm_cache(-1),
    fm_cache(nullptr),
    idx_s0_cache(-1),
    s0_cache(nullptr),
    idx_x0_cache(-1),
    x0_cache(nullptr),
    idx_v0_cache(-1),
    v0_cache(nullptr),
    idx_f0_cache(-1),
    f0_cache(nullptr),
    nmax_old(0),
    grow_callback_added(0),
    restart_callback_added(0),
    restart_from_legacy(0),
    nmax_s_guess(0),
    s_guess(nullptr),
    nmax_x_end(0),
    x_end(nullptr),
    replay_fixes(),
    replay_fix_indices()
{
  if (atom->sp_flag) {
    ensure_custom_peratom();
    maxexchange = 15;
    restart_peratom = 1;
    atom->add_callback(Atom::GROW);
    atom->add_callback(Atom::RESTART);
    grow_callback_added = 1;
    restart_callback_added = 1;
    grow_arrays(atom->nmax);
  }
}

FixLLGMidNH::~FixLLGMidNH()
{
  if (copymode) return;
  debug_close();
  memory->destroy(s_guess);
  memory->destroy(x_end);
  if (grow_callback_added && modify && (modify->find_fix(id) >= 0)) atom->delete_callback(id, Atom::GROW);
  if (restart_callback_added && modify && (modify->find_fix(id) >= 0)) atom->delete_callback(id, Atom::RESTART);
}

int FixLLGMidNH::setmask()
{
  int mask = FixNH::setmask();
  mask |= POST_FORCE;
  if (!lattice_flag) mask &= ~PRE_EXCHANGE;
  return mask;
}

void FixLLGMidNH::ensure_custom_peratom()
{
  int flag = -1, cols = -1, ghost = 0;

  idx_fm_cache = atom->find_custom_ghost("llgmid_fm_cache", flag, cols, ghost);
  if (idx_fm_cache < 0) idx_fm_cache = atom->add_custom("llgmid_fm_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llgmid_fm_cache has incompatible type");

  idx_s0_cache = atom->find_custom_ghost("llgmid_s0_cache", flag, cols, ghost);
  if (idx_s0_cache < 0) idx_s0_cache = atom->add_custom("llgmid_s0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llgmid_s0_cache has incompatible type");

  idx_x0_cache = atom->find_custom_ghost("llgmid_x0_cache", flag, cols, ghost);
  if (idx_x0_cache < 0) idx_x0_cache = atom->add_custom("llgmid_x0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llgmid_x0_cache has incompatible type");

  idx_v0_cache = atom->find_custom_ghost("llgmid_v0_cache", flag, cols, ghost);
  if (idx_v0_cache < 0) idx_v0_cache = atom->add_custom("llgmid_v0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llgmid_v0_cache has incompatible type");

  idx_f0_cache = atom->find_custom_ghost("llgmid_f0_cache", flag, cols, ghost);
  if (idx_f0_cache < 0) idx_f0_cache = atom->add_custom("llgmid_f0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llgmid_f0_cache has incompatible type");

  fm_cache = atom->darray[idx_fm_cache];
  s0_cache = atom->darray[idx_s0_cache];
  x0_cache = atom->darray[idx_x0_cache];
  v0_cache = atom->darray[idx_v0_cache];
  f0_cache = atom->darray[idx_f0_cache];
}

void FixLLGMidNH::grow_arrays(int nmax)
{
  if (idx_fm_cache < 0) return;

  if (nmax > nmax_old) {
    memory->grow(atom->darray[idx_fm_cache], nmax, 3, "llgmid/nh:fm_cache");
    memory->grow(atom->darray[idx_s0_cache], nmax, 3, "llgmid/nh:s0_cache");
    memory->grow(atom->darray[idx_x0_cache], nmax, 3, "llgmid/nh:x0_cache");
    memory->grow(atom->darray[idx_v0_cache], nmax, 3, "llgmid/nh:v0_cache");
    memory->grow(atom->darray[idx_f0_cache], nmax, 3, "llgmid/nh:f0_cache");
  }

  fm_cache = atom->darray[idx_fm_cache];
  s0_cache = atom->darray[idx_s0_cache];
  x0_cache = atom->darray[idx_x0_cache];
  v0_cache = atom->darray[idx_v0_cache];
  f0_cache = atom->darray[idx_f0_cache];

  for (int i = nmax_old; i < nmax; i++) {
    for (int k = 0; k < 3; k++) {
      fm_cache[i][k] = 0.0;
      s0_cache[i][k] = 0.0;
      x0_cache[i][k] = 0.0;
      v0_cache[i][k] = 0.0;
      f0_cache[i][k] = 0.0;
    }
  }
  nmax_old = nmax;
}

void FixLLGMidNH::copy_arrays(int i, int j, int /*delflag*/)
{
  if (idx_fm_cache < 0) return;
  if (!fm_cache) ensure_custom_peratom();
  for (int k = 0; k < 3; k++) {
    fm_cache[j][k] = fm_cache[i][k];
    s0_cache[j][k] = s0_cache[i][k];
    x0_cache[j][k] = x0_cache[i][k];
    v0_cache[j][k] = v0_cache[i][k];
    f0_cache[j][k] = f0_cache[i][k];
  }
}

int FixLLGMidNH::pack_exchange(int i, double *buf)
{
  if (idx_fm_cache < 0) return 0;
  if (!fm_cache) ensure_custom_peratom();
  int m = 0;
  for (int k = 0; k < 3; k++) buf[m++] = fm_cache[i][k];
  for (int k = 0; k < 3; k++) buf[m++] = s0_cache[i][k];
  for (int k = 0; k < 3; k++) buf[m++] = x0_cache[i][k];
  for (int k = 0; k < 3; k++) buf[m++] = v0_cache[i][k];
  for (int k = 0; k < 3; k++) buf[m++] = f0_cache[i][k];
  return m;
}

int FixLLGMidNH::unpack_exchange(int nlocal, double *buf)
{
  if (idx_fm_cache < 0) return 0;
  if (!fm_cache) ensure_custom_peratom();
  int m = 0;
  for (int k = 0; k < 3; k++) fm_cache[nlocal][k] = buf[m++];
  for (int k = 0; k < 3; k++) s0_cache[nlocal][k] = buf[m++];
  for (int k = 0; k < 3; k++) x0_cache[nlocal][k] = buf[m++];
  for (int k = 0; k < 3; k++) v0_cache[nlocal][k] = buf[m++];
  for (int k = 0; k < 3; k++) f0_cache[nlocal][k] = buf[m++];
  return m;
}

int FixLLGMidNH::pack_restart(int i, double *buf)
{
  if (idx_fm_cache < 0) {
    buf[0] = 1.0;
    return 1;
  }
  if (!fm_cache) ensure_custom_peratom();
  buf[0] = 4.0;
  buf[1] = fm_cache[i][0];
  buf[2] = fm_cache[i][1];
  buf[3] = fm_cache[i][2];
  return 4;
}

void FixLLGMidNH::unpack_restart(int nlocal, int nth)
{
  if (idx_fm_cache < 0) return;
  if (!fm_cache) ensure_custom_peratom();

  double **extra = atom->extra;
  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int>(extra[nlocal][m]);

  const int nvals = static_cast<int>(extra[nlocal][m++]);
  fm_cache[nlocal][0] = fm_cache[nlocal][1] = fm_cache[nlocal][2] = 0.0;
  s0_cache[nlocal][0] = s0_cache[nlocal][1] = s0_cache[nlocal][2] = 0.0;
  x0_cache[nlocal][0] = x0_cache[nlocal][1] = x0_cache[nlocal][2] = 0.0;
  v0_cache[nlocal][0] = v0_cache[nlocal][1] = v0_cache[nlocal][2] = 0.0;
  f0_cache[nlocal][0] = f0_cache[nlocal][1] = f0_cache[nlocal][2] = 0.0;

  if (nvals >= 4) {
    fm_cache[nlocal][0] = extra[nlocal][m++];
    fm_cache[nlocal][1] = extra[nlocal][m++];
    fm_cache[nlocal][2] = extra[nlocal][m++];
  }
}

int FixLLGMidNH::maxsize_restart()
{
  return 4;
}

int FixLLGMidNH::size_restart(int /*nlocal*/)
{
  return 4;
}

void FixLLGMidNH::init()
{
  FixNH::init();

  if (!atom->sp_flag) error->all(FLERR, "Fix {} requires atom_style spin", style);
  if (!atom->fm) error->all(FLERR, "Fix {} requires atom_style spin with fm allocated", style);

  const int my_index = modify->find_fix(id);
  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    Fix *f = modify->fix[ifix];
    if (f == this) continue;
    if (!f->time_integrate) continue;
    if (f->igroup == igroup) {
      error->all(FLERR, "Fix {} cannot be used together with time integration fix {} on the same group", style,
                 f->id);
    }
  }

  if (utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "Fix {} is not supported with rRESPA", style);

  if (midpoint_iter < 2) error->all(FLERR, "Fix {} llgmid midpoint_iter must be >= 2", style);
  if (midpoint_tol < 0.0) error->all(FLERR, "Fix {} llgmid midpoint_tol must be >= 0", style);
  if (alpha < 0.0) error->all(FLERR, "Fix {} llgmid alpha must be >= 0", style);
  if (gamma < 0.0 && gamma != -1.0) error->all(FLERR, "Fix {} llgmid gamma must be >= 0", style);

  hbar = force->hplanck / MathConst::MY_2PI;
  if (gamma < 0.0 && hbar == 0.0) error->all(FLERR, "Fix {} requires nonzero hbar (use physical units)", style);
  refresh_g_over_hbar();

  if (debug_every < 1) error->all(FLERR, "Fix {} llgmid debug_every must be >= 1", style);
  if (debug_rank < 0 || debug_rank >= comm->nprocs)
    error->all(FLERR, "Fix {} llgmid debug_rank must be between 0 and nprocs-1", style);

  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    if (utils::strmatch(modify->fix[ifix]->style, "^langevin/spin"))
      error->all(FLERR, "Fix {} cannot be combined with fix langevin/spin", style);
  }

  if (!lattice_flag && pstat_flag)
    error->warning(FLERR, "Fix {} with llgmid lattice off disables pressure control (barostat is inactive)", style);

  if (my_index >= 0) {
    for (int ifix = my_index + 1; ifix < modify->nfix; ifix++) {
      Fix *f = modify->fix[ifix];
      if (f && f->time_integrate)
        error->all(FLERR, "Fix {} with llgmid midpoint_iter must be the last time integration fix (found {} after it)",
                   style, f->id);
    }
  }

  replay_fixes.clear();
  replay_fix_indices.clear();
  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    const char *s = modify->fix[ifix]->style;
    if (utils::strmatch(s, "^precession/spin")) {
      error->all(FLERR,
                 "Fix {} (USER-TSPIN) does not support fix precession/spin; use fix tspin/precession/spin or setforce/spin instead",
                 style);
    }
    if (utils::strmatch(s, "^setforce/spin") || utils::strmatch(s, "^tspin/precession/spin")) {
      replay_fixes.push_back(modify->fix[ifix]);
      replay_fix_indices.push_back(ifix);
      if (my_index >= 0 && ifix > my_index) {
        error->all(FLERR, "Fix {} must be defined after {} to preserve external fields in llgmid recompute", style,
                   modify->fix[ifix]->id);
      }
    }
  }

  if (my_index >= 0) {
    auto is_replayed = [this](Fix *f) {
      return std::find(replay_fixes.begin(), replay_fixes.end(), f) != replay_fixes.end();
    };
    for (int ifix = 0; ifix < my_index; ifix++) {
      Fix *f = modify->fix[ifix];
      if (!f || f == this) continue;
      if (!(modify->fmask[ifix] & POST_FORCE)) continue;
      if (is_replayed(f)) continue;
      error->all(FLERR,
                 "Fix {} performs internal force/field recomputes for midpoint iterations and will overwrite contributions from earlier post_force fix {} (style {}). Define that fix after {}.",
                 style, f->id, f->style, id);
    }
  }

  ensure_custom_peratom();
  debug_open();
}

void FixLLGMidNH::setup(int vflag)
{
  FixNH::setup(vflag);
  ensure_custom_peratom();
  cache_current_fm();
  pe_prev_end = current_pe_total();
}

void FixLLGMidNH::debug_open()
{
  if (!debug_flag) return;
  if (debug_fp) return;
  if (comm->me != debug_rank) return;

  std::string fname = debug_file;
  if (fname.empty()) {
    fname = "llgmid_nh_debug.";
    fname += id;
    fname += ".log";
  }

  debug_fp = fopen(fname.c_str(), "w");
  if (!debug_fp)
    error->one(FLERR, "Fix {} could not open debug_file {}: {}", style, fname, utils::getsyserror());

  debug_header_printed = 0;
  if (debug_flush) setvbuf(debug_fp, nullptr, _IOLBF, 0);
}

void FixLLGMidNH::debug_close()
{
  if (!debug_fp) return;
  fclose(debug_fp);
  debug_fp = nullptr;
}

double FixLLGMidNH::current_pe_total() const
{
  // Mirror ComputePE::compute_scalar() tally logic.
  double one = 0.0;
  if (force->pair) one += force->pair->eng_vdwl + force->pair->eng_coul;

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) one += force->bond->energy;
    if (force->angle) one += force->angle->energy;
    if (force->dihedral) one += force->dihedral->energy;
    if (force->improper) one += force->improper->energy;
  }

  double scalar = 0.0;
  MPI_Allreduce(&one, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);

  if (force->kspace) scalar += force->kspace->energy;

  if (force->pair && force->pair->tail_flag) {
    const double volume = domain->xprd * domain->yprd * domain->zprd;
    scalar += force->pair->etail / volume;
  }

  if (modify->n_energy_global) scalar += modify->energy_global();

  return scalar;
}

void FixLLGMidNH::debug_log_energy(double pe_mid, double pe_end)
{
  if (!debug_flag) return;
  if (update->ntimestep < debug_start) return;
  if ((debug_every > 1) && ((update->ntimestep % debug_every) != 0)) return;
  if (comm->me != debug_rank) return;
  debug_open();
  if (!debug_fp) return;

  if (!debug_header_printed) {
    fprintf(debug_fp,
            "# fix %s energy diagnostics\n"
            "# columns:\n"
            "# step time dt pe_prev_end pe_mid pe_end dE_step (pe_end-pe_prev_end) dE_mid_end (pe_end-pe_mid)\n",
            style);
    debug_header_printed = 1;
  }

  const double dE_step = pe_end - pe_prev_end;
  const double dE_mid_end = pe_end - pe_mid;

  fprintf(debug_fp, "%lld %.16g %.16g %.16g %.16g %.16g %.16g %.16g\n",
          static_cast<long long>(update->ntimestep), update->atime, update->dt,
          pe_prev_end, pe_mid, pe_end, dE_step, dE_mid_end);

  if (debug_flush) fflush(debug_fp);
}

double FixLLGMidNH::get_mass(int i) const
{
  if (atom->rmass) return atom->rmass[i];
  return atom->mass[atom->type[i]];
}

void FixLLGMidNH::refresh_g_over_hbar()
{
  if (gamma >= 0.0) {
    g_over_hbar = gamma;
    return;
  }

  if (hbar == 0.0) {
    g_over_hbar = 0.0;
    return;
  }

  constexpr double g_default = 2.0;
  g_over_hbar = g_default / hbar;
}

void FixLLGMidNH::exact_flow_direction(const double *e0, const double *h, double dt, double *e1) const
{
  const double e0n = std::sqrt(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]);
  if (e0n <= SPIN_EPS) {
    e1[0] = e1[1] = e1[2] = 0.0;
    return;
  }

  const double hmag = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
  if (hmag <= SPIN_EPS) {
    const double inv = 1.0 / e0n;
    e1[0] = e0[0] * inv;
    e1[1] = e0[1] * inv;
    e1[2] = e0[2] * inv;
    return;
  }

  const double inv_e = 1.0 / e0n;
  const double ex = e0[0] * inv_e;
  const double ey = e0[1] * inv_e;
  const double ez = e0[2] * inv_e;
  const double hx = h[0] / hmag;
  const double hy = h[1] / hmag;
  const double hz = h[2] / hmag;

  const double u0 = std::max(-1.0, std::min(1.0, ex * hx + ey * hy + ez * hz));
  double px = ex - u0 * hx;
  double py = ey - u0 * hy;
  double pz = ez - u0 * hz;
  const double pnorm = std::sqrt(px * px + py * py + pz * pz);
  if (pnorm <= SPIN_EPS) {
    e1[0] = ex;
    e1[1] = ey;
    e1[2] = ez;
    return;
  }

  const double inv_p = 1.0 / pnorm;
  px *= inv_p;
  py *= inv_p;
  pz *= inv_p;

  const double cx = hy * pz - hz * py;
  const double cy = hz * px - hx * pz;
  const double cz = hx * py - hy * px;

  const double a = g_over_hbar / (1.0 + alpha * alpha);
  const double b = alpha * g_over_hbar / (1.0 + alpha * alpha);
  const double arg0 = std::atanh(std::max(-1.0 + 1.0e-15, std::min(1.0 - 1.0e-15, u0)));
  const double u1 = std::tanh(arg0 + b * hmag * dt);
  const double dphi = -a * hmag * dt;
  const double sint = std::sqrt(std::max(0.0, 1.0 - u1 * u1));
  const double cphi = std::cos(dphi);
  const double sphi = std::sin(dphi);

  e1[0] = u1 * hx + sint * (cphi * px + sphi * cx);
  e1[1] = u1 * hy + sint * (cphi * py + sphi * cy);
  e1[2] = u1 * hz + sint * (cphi * pz + sphi * cz);

  const double n = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  if (n > SPIN_EPS) {
    const double inv = 1.0 / n;
    e1[0] *= inv;
    e1[1] *= inv;
    e1[2] *= inv;
  }
}

void FixLLGMidNH::write_spin_from_vector(int i, const double *vec)
{
  const double mag = atom->sp[i][3];
  if (mag <= SPIN_EPS) {
    atom->sp[i][0] = atom->sp[i][1] = atom->sp[i][2] = 0.0;
    atom->sp[i][3] = 0.0;
    return;
  }

  const double n = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
  if (n <= SPIN_EPS) return;
  const double inv = 1.0 / n;
  atom->sp[i][0] = vec[0] * inv;
  atom->sp[i][1] = vec[1] * inv;
  atom->sp[i][2] = vec[2] * inv;
  atom->sp[i][3] = mag;
}

void FixLLGMidNH::cache_lattice_moving_step_start_state()
{
  if (!fm_cache) ensure_custom_peratom();

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **sp = atom->sp;
  int *mask = atom->mask;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    for (int k = 0; k < 3; k++) {
      x0_cache[i][k] = x[i][k];
      v0_cache[i][k] = v[i][k];
      f0_cache[i][k] = f[i][k];
    }

    const double mag = sp[i][3];
    if (mag > SPIN_EPS) {
      const double dirn = std::sqrt(sp[i][0] * sp[i][0] + sp[i][1] * sp[i][1] + sp[i][2] * sp[i][2]);
      if (dirn > SPIN_EPS) {
        const double inv = mag / dirn;
        s0_cache[i][0] = sp[i][0] * inv;
        s0_cache[i][1] = sp[i][1] * inv;
        s0_cache[i][2] = sp[i][2] * inv;
      } else {
        s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
      }
    } else {
      s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
    }
  }
}

void FixLLGMidNH::build_predictor_midpoint_state()
{
  if (!fm_cache) ensure_custom_peratom();

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const double dt = update->dt;
  const double ftm2v = force->ftm2v;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **sp = atom->sp;
  int *mask = atom->mask;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    for (int k = 0; k < 3; k++) {
      x0_cache[i][k] = x[i][k];
      v0_cache[i][k] = v[i][k];
      f0_cache[i][k] = f[i][k];
    }

    const double mag = sp[i][3];
    if (mag > SPIN_EPS) {
      const double dirn = std::sqrt(sp[i][0] * sp[i][0] + sp[i][1] * sp[i][1] + sp[i][2] * sp[i][2]);
      if (dirn > SPIN_EPS) {
        const double inv = mag / dirn;
        s0_cache[i][0] = sp[i][0] * inv;
        s0_cache[i][1] = sp[i][1] * inv;
        s0_cache[i][2] = sp[i][2] * inv;
      } else {
        s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
      }
    } else {
      s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
    }

    if (lattice_flag) {
      const double mass = get_mass(i);
      const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
      for (int k = 0; k < 3; k++)
        x[i][k] = x0_cache[i][k] + 0.5 * dt * v0_cache[i][k] + 0.125 * dt * dt * ftm2v * f0_cache[i][k] * inv_mass;
    }

    if (mag > SPIN_EPS) {
      double e0[3] = {s0_cache[i][0] / mag, s0_cache[i][1] / mag, s0_cache[i][2] / mag};
      double ehalf[3];
      exact_flow_direction(e0, fm_cache[i], 0.5 * dt, ehalf);
      double emid[3] = {e0[0] + ehalf[0], e0[1] + ehalf[1], e0[2] + ehalf[2]};
      const double nn = std::sqrt(emid[0] * emid[0] + emid[1] * emid[1] + emid[2] * emid[2]);
      if (nn > SPIN_EPS) {
        const double inv = 1.0 / nn;
        sp[i][0] = emid[0] * inv;
        sp[i][1] = emid[1] * inv;
        sp[i][2] = emid[2] * inv;
      }
      sp[i][3] = mag;
    }
  }
}

void FixLLGMidNH::apply_corrector_from_midpoint_field(double **x_mid, double **e_mid, int update_lattice)
{
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const double dt = update->dt;
  const double ftm2v = force->ftm2v;
  double **sp = atom->sp;
  double **v = atom->v;
  int *mask = atom->mask;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    const double mag = sp[i][3];
    if (update_lattice) {
      const double mass = get_mass(i);
      const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
      for (int k = 0; k < 3; k++) {
        x_end[i][k] = x0_cache[i][k] + dt * v0_cache[i][k] + 0.5 * dt * dt * ftm2v * atom->f[i][k] * inv_mass;
        v[i][k] = v0_cache[i][k] + dt * ftm2v * atom->f[i][k] * inv_mass;
        x_mid[i][k] = 0.5 * (x0_cache[i][k] + x_end[i][k]);
      }
    } else {
      for (int k = 0; k < 3; k++) {
        x_end[i][k] = x0_cache[i][k];
        x_mid[i][k] = x0_cache[i][k];
      }
    }

    if (mag > SPIN_EPS) {
      double e0[3] = {s0_cache[i][0] / mag, s0_cache[i][1] / mag, s0_cache[i][2] / mag};
      double e1[3];
      exact_flow_direction(e0, atom->fm[i], dt, e1);
      for (int k = 0; k < 3; k++) s_guess[i][k] = mag * e1[k];
      double emid_vec[3] = {e0[0] + e1[0], e0[1] + e1[1], e0[2] + e1[2]};
      const double nn = std::sqrt(emid_vec[0] * emid_vec[0] + emid_vec[1] * emid_vec[1] + emid_vec[2] * emid_vec[2]);
      if (nn > SPIN_EPS) {
        const double scale = mag / nn;
        e_mid[i][0] = emid_vec[0] * scale;
        e_mid[i][1] = emid_vec[1] * scale;
        e_mid[i][2] = emid_vec[2] * scale;
      } else {
        e_mid[i][0] = e_mid[i][1] = e_mid[i][2] = 0.0;
      }
    } else {
      s_guess[i][0] = s_guess[i][1] = s_guess[i][2] = 0.0;
      e_mid[i][0] = e_mid[i][1] = e_mid[i][2] = 0.0;
      sp[i][3] = 0.0;
    }
  }
}

void FixLLGMidNH::cache_current_fm()
{
  if (idx_fm_cache < 0) return;
  if (!fm_cache) ensure_custom_peratom();
  double **fm = atom->fm;
  for (int i = 0; i < atom->nlocal; i++) {
    fm_cache[i][0] = fm[i][0];
    fm_cache[i][1] = fm[i][1];
    fm_cache[i][2] = fm[i][2];
  }
}

void FixLLGMidNH::clear_force_arrays()
{
  const int nclear = force->newton ? (atom->nlocal + atom->nghost) : atom->nlocal;
  double **f = atom->f;
  double **fm = atom->fm;
  double **fm_long = atom->fm_long;
  for (int i = 0; i < nclear; i++) {
    f[i][0] = f[i][1] = f[i][2] = 0.0;
    fm[i][0] = fm[i][1] = fm[i][2] = 0.0;
    if (fm_long) fm_long[i][0] = fm_long[i][1] = fm_long[i][2] = 0.0;
  }
}

void FixLLGMidNH::rebuild_neighbors_for_current_positions()
{
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  if (atom->sortfreq > 0 && update->ntimestep >= atom->nextsort) atom->sort();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);
}

void FixLLGMidNH::replay_external_spin_fields(int vflag)
{
  if (replay_fixes.empty()) return;
  for (auto *f : replay_fixes) f->post_force(vflag);
}

void FixLLGMidNH::recompute_force_and_field(int eflag, int vflag)
{
  comm->forward_comm();
  clear_force_arrays();
  if (force->pair) force->pair->compute(eflag, vflag);
  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }
  if (force->kspace) force->kspace->compute(eflag, vflag);
  if (force->newton) comm->reverse_comm();
  replay_external_spin_fields(vflag);
}

bool FixLLGMidNH::solve_spin_midpoint(bool lattice_mode, int vflag, double pe_mid)
{
  if (midpoint_iter <= 1) return false;
  if (lattice_mode != (lattice_flag != 0)) return false;

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  if (atom->nmax > nmax_s_guess) {
    nmax_s_guess = atom->nmax;
    memory->grow(s_guess, nmax_s_guess, 3, "llgmid/nh:s_guess");
  }
  if (atom->nmax > nmax_x_end) {
    nmax_x_end = atom->nmax;
    memory->grow(x_end, nmax_x_end, 3, "llgmid/nh:x_end");
  }

  double **sp = atom->sp;
  double **x = atom->x;
  int *mask = atom->mask;
  double **x_mid = f0_cache;
  double **e_mid = fm_cache;
  const int update_lattice = 0;

  for (int iter = 0; iter < midpoint_iter; iter++) {
    apply_corrector_from_midpoint_field(x_mid, e_mid, update_lattice);

    double max_dev_local = 0.0;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      const double mag = sp[i][3];
      const double curx = mag * sp[i][0];
      const double cury = mag * sp[i][1];
      const double curz = mag * sp[i][2];
      const double dsx = e_mid[i][0] - curx;
      const double dsy = e_mid[i][1] - cury;
      const double dsz = e_mid[i][2] - curz;
      max_dev_local = std::max(max_dev_local, std::sqrt(dsx * dsx + dsy * dsy + dsz * dsz));
    }

    double max_dev_all = max_dev_local;
    if (midpoint_tol > 0.0) MPI_Allreduce(&max_dev_local, &max_dev_all, 1, MPI_DOUBLE, MPI_MAX, world);
    if ((iter == midpoint_iter - 1) || (midpoint_tol > 0.0 && max_dev_all <= midpoint_tol)) break;

    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      write_spin_from_vector(i, e_mid[i]);
    }

    recompute_force_and_field(1, 0);
  }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    write_spin_from_vector(i, s_guess[i]);
  }

  recompute_force_and_field(1, vflag);
  cache_current_fm();
  const double pe_end = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
  debug_log_energy(pe_mid, pe_end);
  pe_prev_end = pe_end;
  return true;
}

void FixLLGMidNH::initial_integrate(int /*vflag*/)
{
  const int do_pstat = pstat_flag && lattice_flag;

  if (do_pstat && mpchain) nhc_press_integrate();
  if (tstat_flag) compute_temp_target();
  if (tstat_flag && lattice_flag) nhc_temp_integrate();

  if (do_pstat) {
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep + 1);
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
  }

  if (lattice_flag) {
    if (midpoint_iter > 1) cache_lattice_moving_step_start_state();
    nve_v();
    if (do_pstat) remap();
    nve_x();
  } else {
    build_predictor_midpoint_state();
  }

  if (do_pstat) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

void FixLLGMidNH::post_force(int vflag)
{
  const double pe_mid = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
  solve_spin_midpoint(lattice_flag != 0, vflag, pe_mid);
}

void FixLLGMidNH::final_integrate()
{
  const int do_pstat = pstat_flag && lattice_flag;

  if (lattice_flag) nve_v();
  if (which == BIAS && neighbor->ago == 0) t_current = temperature->compute_scalar();
  if (do_pstat) nh_v_press();

  t_current = temperature->compute_scalar();
  tdof = temperature->dof;

  if (do_pstat) {
    if (pstyle == ISO) pressure->compute_scalar();
    else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    couple();
    pressure->addstep(update->ntimestep + 1);
    nh_omega_dot();
  }

  if (tstat_flag && lattice_flag) nhc_temp_integrate();
  if (do_pstat && mpchain) nhc_press_integrate();
}

int FixLLGMidNH::modify_param(int narg, char **arg)
{
  if (narg < 1) return FixNH::modify_param(narg, arg);
  if (strcmp(arg[0], "llgmid") != 0) return FixNH::modify_param(narg, arg);
  if (narg < 2) error->all(FLERR, "Illegal fix_modify llgmid command");

  if (strcmp(arg[1], "lattice") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid lattice command");
    lattice_flag = parse_on_off(arg[2], lmp, "fix_modify llgmid lattice");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_iter") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid midpoint_iter command");
    midpoint_iter = utils::inumeric(FLERR, arg[2], false, lmp);
    if (midpoint_iter < 2) error->all(FLERR, "Illegal fix_modify llgmid midpoint_iter command (must be >= 2)");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_tol") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid midpoint_tol command");
    midpoint_tol = utils::numeric(FLERR, arg[2], false, lmp);
    if (midpoint_tol < 0.0) error->all(FLERR, "Illegal fix_modify llgmid midpoint_tol command (must be >= 0)");
    return 3;
  }
  if (strcmp(arg[1], "alpha") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid alpha command");
    alpha = utils::numeric(FLERR, arg[2], false, lmp);
    if (alpha < 0.0) error->all(FLERR, "Illegal fix_modify llgmid alpha command (must be >= 0)");
    return 3;
  }
  if (strcmp(arg[1], "gamma") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid gamma command");
    gamma = utils::numeric(FLERR, arg[2], false, lmp);
    if (gamma < 0.0) error->all(FLERR, "Illegal fix_modify llgmid gamma command (must be >= 0)");
    refresh_g_over_hbar();
    return 3;
  }
  if (strcmp(arg[1], "fm_units") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid fm_units command");
    if (strcmp(arg[2], "field") != 0)
      error->all(FLERR, "Illegal fix_modify llgmid fm_units command (must be 'field')");
    return 3;
  }
  if (strcmp(arg[1], "debug") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid debug command");
    debug_flag = utils::logical(FLERR, arg[2], false, lmp);
    if (!debug_flag) debug_close();
    else debug_open();
    return 3;
  }
  if (strcmp(arg[1], "debug_every") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid debug_every command");
    debug_every = utils::inumeric(FLERR, arg[2], false, lmp);
    if (debug_every < 1) error->all(FLERR, "Illegal fix_modify llgmid debug_every command (must be >= 1)");
    return 3;
  }
  if (strcmp(arg[1], "debug_rank") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid debug_rank command");
    debug_rank = utils::inumeric(FLERR, arg[2], false, lmp);
    if (debug_rank < 0 || debug_rank >= comm->nprocs)
      error->all(FLERR, "Illegal fix_modify llgmid debug_rank command (must be between 0 and nprocs-1)");
    debug_close();
    debug_open();
    return 3;
  }
  if (strcmp(arg[1], "debug_flush") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid debug_flush command");
    debug_flush = utils::logical(FLERR, arg[2], false, lmp);
    if (debug_fp) setvbuf(debug_fp, nullptr, debug_flush ? _IOLBF : _IOFBF, 0);
    return 3;
  }
  if (strcmp(arg[1], "debug_start") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid debug_start command");
    debug_start = utils::bnumeric(FLERR, arg[2], false, lmp);
    return 3;
  }
  if (strcmp(arg[1], "debug_file") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llgmid debug_file command");
    debug_file = arg[2];
    debug_close();
    debug_open();
    return 3;
  }

  error->all(FLERR, "Illegal fix_modify llgmid command");
  return 0;
}

int FixLLGMidNH::nh_payload_size_from_list(const double *list, int max_n)
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
    if (!skip(14)) return -1;
    int m = 0;
    if (!pull_int(m)) return -1;
    if (!skip(2 * m)) return -1;
    if (!pull_int(flag)) return -1;
    if (flag && !skip(6)) return -1;
  }

  return n;
}

int FixLLGMidNH::pack_restart_payload_v1(double *list) const
{
  int n = 0;
  if (list) list[n] = static_cast<double>(lattice_flag);
  n++;
  if (list) list[n] = static_cast<double>(midpoint_iter);
  n++;
  if (list) list[n] = midpoint_tol;
  n++;
  if (list) list[n] = alpha;
  n++;
  if (list) list[n] = pe_prev_end;
  n++;
  if (list) list[n] = static_cast<double>(debug_flag);
  n++;
  if (list) list[n] = static_cast<double>(debug_every);
  n++;
  if (list) list[n] = static_cast<double>(debug_rank);
  n++;
  if (list) list[n] = static_cast<double>(debug_flush);
  n++;
  if (list) list[n] = static_cast<double>(debug_start);
  n++;
  if (list) list[n] = static_cast<double>(debug_header_printed);
  n++;
  return n;
}

void FixLLGMidNH::unpack_restart_payload_v1(const double *list)
{
  int n = 0;
  lattice_flag = static_cast<int>(list[n++]);
  midpoint_iter = static_cast<int>(list[n++]);
  midpoint_tol = list[n++];
  alpha = list[n++];
  pe_prev_end = list[n++];
  debug_flag = static_cast<int>(list[n++]);
  debug_every = static_cast<int>(list[n++]);
  debug_rank = static_cast<int>(list[n++]);
  debug_flush = static_cast<int>(list[n++]);
  debug_start = static_cast<bigint>(list[n++]);
  debug_header_printed = static_cast<int>(list[n++]);
  gamma = -1.0;
  refresh_g_over_hbar();
}

int FixLLGMidNH::pack_restart_payload_v2(double *list) const
{
  const int n = pack_restart_payload_v1(list);
  if (list) list[n] = gamma;
  return n + 1;
}

void FixLLGMidNH::unpack_restart_payload_v2(const double *list)
{
  unpack_restart_payload_v1(list);
  gamma = list[pack_restart_payload_v1(nullptr)];
  refresh_g_over_hbar();
}

int FixLLGMidNH::size_restart_global()
{
  return 2 + 1 + FixNH::size_restart_global() + 1 + pack_restart_payload_v2(nullptr);
}

int FixLLGMidNH::pack_restart_data(double *list)
{
  int n = 0;
  list[n++] = RESTART_MAGIC;
  list[n++] = static_cast<double>(RESTART_VERSION);

  const int nh_n = FixNH::pack_restart_data(list + n + 1);
  list[n] = static_cast<double>(nh_n);
  n += nh_n + 1;

  const int llgmid_n = pack_restart_payload_v2(list + n + 1);
  list[n] = static_cast<double>(llgmid_n);
  n += llgmid_n + 1;
  return n;
}

void FixLLGMidNH::restart(char *buf)
{
  auto *list = reinterpret_cast<double *>(buf);
  restart_from_legacy = 0;

  if (list[0] != RESTART_MAGIC) {
    FixNH::restart(buf);
    restart_from_legacy = 1;
    error->warning(
        FLERR,
        "Fix {} style {} read legacy restart payload without USER-TSPIN llgmid state; compatibility fallback reconstruction will be used",
        id, style);
    return;
  }

  const int version = static_cast<int>(list[1]);
  if (version != 1 && version != RESTART_VERSION)
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

  const int llgmid_n = static_cast<int>(list[n++]);
  const int expected = (version == RESTART_VERSION) ? pack_restart_payload_v2(nullptr) : pack_restart_payload_v1(nullptr);
  if (llgmid_n != expected)
    error->all(FLERR, "Fix {} style {} restart payload llgmid size mismatch (stored {}, expected {})", id, style,
               llgmid_n, expected);

  if (version == RESTART_VERSION)
    unpack_restart_payload_v2(list + n);
  else
    unpack_restart_payload_v1(list + n);
  restart_from_legacy = 0;
}
