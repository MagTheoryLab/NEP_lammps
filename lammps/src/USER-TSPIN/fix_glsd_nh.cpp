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

#include "fix_glsd_nh.h"

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

struct FixGLSDNH::ParsedArgs {
  std::vector<std::string> fixnh_strings;
  std::vector<char *> fixnh_argv;
  int fixnh_narg = 0;

  int lattice_flag = 1;
  int midpoint_iter = 3;
  double midpoint_tol = 0.0;
  double gammas = 0.0;    // interpreted as λ (direct); keyword aliases: gammas, lambda
  double alpha = -1.0;    // dimensionless: λ = alpha*(g/ħ); alpha<0 means disabled
  double spin_temperature = 0.0;
  int seed = 12345;

  int debug_flag = 0;
  int debug_every = 1;
  int debug_rank = 0;
  int debug_flush = 0;
  bigint debug_start = 0;
  std::string debug_file;
};

FixGLSDNH::ParsedArgs FixGLSDNH::parse_glsd_trailing_block(LAMMPS *lmp, int narg, char **arg)
{
  ParsedArgs out;
  out.fixnh_strings.reserve(narg);

  int glsd_pos = -1;
  for (int i = 3; i < narg; i++) {
    if (strcmp(arg[i], "glsd") == 0) {
      glsd_pos = i;
      break;
    }
  }

  const int pass_end = (glsd_pos >= 0) ? glsd_pos : narg;
  for (int i = 0; i < pass_end; i++) out.fixnh_strings.emplace_back(arg[i]);

  if (glsd_pos >= 0) {
    int iarg = glsd_pos + 1;
    while (iarg < narg) {
      if (strcmp(arg[iarg], "lattice") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd lattice", lmp->error);
        out.lattice_flag = parse_on_off(arg[iarg + 1], lmp, "fix glsd ... glsd lattice");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_iter") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd midpoint_iter", lmp->error);
        out.midpoint_iter = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.midpoint_iter < 2) lmp->error->all(FLERR, "fix glsd ... glsd midpoint_iter must be >= 2");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_tol") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd midpoint_tol", lmp->error);
        out.midpoint_tol = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.midpoint_tol < 0.0) lmp->error->all(FLERR, "fix glsd ... glsd midpoint_tol must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_anderson") == 0 || strcmp(arg[iarg], "midpoint_anderson_reg") == 0) {
        lmp->error->all(FLERR,
                        "fix glsd ... glsd {} has been removed; use midpoint_iter and midpoint_tol only",
                        arg[iarg]);
      } else if (strcmp(arg[iarg], "gammas") == 0 || strcmp(arg[iarg], "lambda") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd gammas", lmp->error);
        out.gammas = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (strcmp(arg[iarg], "gammas") == 0)
          lmp->error->warning(FLERR, "fix glsd ... glsd gammas is deprecated; use glsd alpha or glsd lambda");
        iarg += 2;
      } else if (strcmp(arg[iarg], "alpha") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd alpha", lmp->error);
        out.alpha = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.alpha < 0.0) lmp->error->all(FLERR, "fix glsd ... glsd alpha must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "stemp") == 0 || strcmp(arg[iarg], "temp") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd stemp", lmp->error);
        if (strcmp(arg[iarg], "temp") == 0)
          lmp->error->warning(FLERR, "fix glsd ... glsd temp is deprecated; use stemp");
        out.spin_temperature = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "seed") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd seed", lmp->error);
        out.seed = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.seed <= 0) lmp->error->all(FLERR, "fix glsd ... glsd seed must be > 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "fm_units") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd fm_units", lmp->error);
        if (strcmp(arg[iarg + 1], "field") != 0)
          lmp->error->all(FLERR, "fix glsd ... glsd fm_units must be 'field' (H = -dE/dM in eV/μB)");
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd debug", lmp->error);
        out.debug_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_every") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd debug_every", lmp->error);
        out.debug_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.debug_every < 1) lmp->error->all(FLERR, "fix glsd ... glsd debug_every must be >= 1");
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_rank") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd debug_rank", lmp->error);
        out.debug_rank = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_flush") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd debug_flush", lmp->error);
        out.debug_flush = utils::logical(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_start") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd debug_start", lmp->error);
        out.debug_start = utils::bnumeric(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_file") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix glsd ... glsd debug_file", lmp->error);
        out.debug_file = arg[iarg + 1];
        iarg += 2;
      } else {
        lmp->error->all(FLERR, "Illegal fix glsd ... glsd option: {}", arg[iarg]);
      }
    }
  }

  out.fixnh_argv.reserve(out.fixnh_strings.size());
  for (auto &s : out.fixnh_strings) out.fixnh_argv.push_back(s.data());
  out.fixnh_narg = static_cast<int>(out.fixnh_argv.size());

  return out;
}

FixGLSDNH::FixGLSDNH(LAMMPS *lmp, int narg, char **arg) : FixGLSDNH(lmp, parse_glsd_trailing_block(lmp, narg, arg))
{
}

FixGLSDNH::FixGLSDNH(LAMMPS *lmp, ParsedArgs parsed) :
    FixNH(lmp, parsed.fixnh_narg, parsed.fixnh_argv.data()),
    size_vector_nh(size_vector),
    lattice_flag(parsed.lattice_flag),
    midpoint_iter(parsed.midpoint_iter),
    midpoint_tol(parsed.midpoint_tol),
    lambda(parsed.gammas),
    alpha(parsed.alpha),
    spin_temperature(parsed.spin_temperature),
    seed(parsed.seed),
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
    nmax_old(0),
    grow_callback_added(0),
    restart_callback_added(0),
    restart_from_legacy(0),
    nmax_s0(0),
    s0(nullptr),
    nmax_s_guess(0),
    s_guess(nullptr),
    nmax_s_map(0),
    s_map(nullptr),
    replay_fixes(),
    replay_fix_indices()
{
  if (atom->sp_flag) {
    ensure_custom_peratom();
    maxexchange = 6;
    restart_peratom = 1;
    atom->add_callback(Atom::GROW);
    atom->add_callback(Atom::RESTART);
    grow_callback_added = 1;
    restart_callback_added = 1;
    grow_arrays(atom->nmax);
  }
}

FixGLSDNH::~FixGLSDNH()
{
  if (copymode) return;
  debug_close();
  memory->destroy(s0);
  memory->destroy(s_guess);
  memory->destroy(s_map);
  // When a fix constructor throws (e.g., illegal command), the fix may not be
  // registered in Modify yet, so Atom::delete_callback() would not find it and
  // would hard-error. Guard against that failure mode.
  if (grow_callback_added && modify && (modify->find_fix(id) >= 0)) atom->delete_callback(id, Atom::GROW);
  if (restart_callback_added && modify && (modify->find_fix(id) >= 0)) atom->delete_callback(id, Atom::RESTART);
}

int FixGLSDNH::setmask()
{
  int mask = FixNH::setmask();
  mask |= POST_FORCE;
  if (!lattice_flag) mask &= ~PRE_EXCHANGE;
  return mask;
}

void FixGLSDNH::ensure_custom_peratom()
{
  int flag = -1, cols = -1, ghost = 0;
  idx_fm_cache = atom->find_custom_ghost("glsd_fm_cache", flag, cols, ghost);
  if (idx_fm_cache < 0) idx_fm_cache = atom->add_custom("glsd_fm_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property glsd_fm_cache has incompatible type");

  fm_cache = atom->darray[idx_fm_cache];

  flag = -1;
  cols = -1;
  ghost = 0;
  idx_s0_cache = atom->find_custom_ghost("glsd_s0_cache", flag, cols, ghost);
  if (idx_s0_cache < 0) idx_s0_cache = atom->add_custom("glsd_s0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property glsd_s0_cache has incompatible type");

  s0_cache = atom->darray[idx_s0_cache];
}

void FixGLSDNH::grow_arrays(int nmax)
{
  if (idx_fm_cache < 0) return;

  if (nmax > nmax_old) {
    memory->grow(atom->darray[idx_fm_cache], nmax, 3, "glsd/nh:fm_cache");
    if (idx_s0_cache >= 0) memory->grow(atom->darray[idx_s0_cache], nmax, 3, "glsd/nh:s0_cache");
  }

  fm_cache = atom->darray[idx_fm_cache];
  if (idx_s0_cache >= 0) s0_cache = atom->darray[idx_s0_cache];
  for (int i = nmax_old; i < nmax; i++) {
    if (fm_cache) fm_cache[i][0] = fm_cache[i][1] = fm_cache[i][2] = 0.0;
    if (s0_cache) s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
  }
  nmax_old = nmax;
}

void FixGLSDNH::copy_arrays(int i, int j, int /*delflag*/)
{
  if (idx_fm_cache < 0) return;
  if (!fm_cache) ensure_custom_peratom();
  fm_cache[j][0] = fm_cache[i][0];
  fm_cache[j][1] = fm_cache[i][1];
  fm_cache[j][2] = fm_cache[i][2];
  if (s0_cache) {
    s0_cache[j][0] = s0_cache[i][0];
    s0_cache[j][1] = s0_cache[i][1];
    s0_cache[j][2] = s0_cache[i][2];
  }
}

int FixGLSDNH::pack_exchange(int i, double *buf)
{
  if (idx_fm_cache < 0) return 0;
  if (!fm_cache) ensure_custom_peratom();
  buf[0] = fm_cache[i][0];
  buf[1] = fm_cache[i][1];
  buf[2] = fm_cache[i][2];
  if (s0_cache) {
    buf[3] = s0_cache[i][0];
    buf[4] = s0_cache[i][1];
    buf[5] = s0_cache[i][2];
    return 6;
  }
  return 3;
}

int FixGLSDNH::unpack_exchange(int nlocal, double *buf)
{
  if (idx_fm_cache < 0) return 0;
  if (!fm_cache) ensure_custom_peratom();
  fm_cache[nlocal][0] = buf[0];
  fm_cache[nlocal][1] = buf[1];
  fm_cache[nlocal][2] = buf[2];
  if (s0_cache) {
    s0_cache[nlocal][0] = buf[3];
    s0_cache[nlocal][1] = buf[4];
    s0_cache[nlocal][2] = buf[5];
    return 6;
  }
  return 3;
}

int FixGLSDNH::pack_restart(int i, double *buf)
{
  if (idx_fm_cache < 0) {
    buf[0] = 1.0;
    return 1;
  }
  if (!fm_cache) ensure_custom_peratom();

  buf[0] = 7.0;
  buf[1] = fm_cache[i][0];
  buf[2] = fm_cache[i][1];
  buf[3] = fm_cache[i][2];
  if (s0_cache) {
    buf[4] = s0_cache[i][0];
    buf[5] = s0_cache[i][1];
    buf[6] = s0_cache[i][2];
  } else {
    buf[4] = buf[5] = buf[6] = 0.0;
  }
  return 7;
}

void FixGLSDNH::unpack_restart(int nlocal, int nth)
{
  if (idx_fm_cache < 0) return;
  if (!fm_cache) ensure_custom_peratom();

  double **extra = atom->extra;
  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int>(extra[nlocal][m]);

  const int nvals = static_cast<int>(extra[nlocal][m++]);
  if (nvals <= 1) {
    fm_cache[nlocal][0] = fm_cache[nlocal][1] = fm_cache[nlocal][2] = 0.0;
    if (s0_cache) s0_cache[nlocal][0] = s0_cache[nlocal][1] = s0_cache[nlocal][2] = 0.0;
    return;
  }

  if (nvals < 4) {
    error->warning(FLERR,
                   "Fix {} style {} encountered truncated per-atom restart payload; using safe fallback for atom {}",
                   id, style, nlocal);
    fm_cache[nlocal][0] = fm_cache[nlocal][1] = fm_cache[nlocal][2] = 0.0;
    if (s0_cache) s0_cache[nlocal][0] = s0_cache[nlocal][1] = s0_cache[nlocal][2] = 0.0;
    return;
  }

  fm_cache[nlocal][0] = extra[nlocal][m++];
  fm_cache[nlocal][1] = extra[nlocal][m++];
  fm_cache[nlocal][2] = extra[nlocal][m++];

  if (s0_cache) {
    if (nvals >= 7) {
      s0_cache[nlocal][0] = extra[nlocal][m++];
      s0_cache[nlocal][1] = extra[nlocal][m++];
      s0_cache[nlocal][2] = extra[nlocal][m++];
    } else {
      s0_cache[nlocal][0] = s0_cache[nlocal][1] = s0_cache[nlocal][2] = 0.0;
    }
  }
}

int FixGLSDNH::maxsize_restart()
{
  if (idx_fm_cache < 0) return 1;
  return 7;
}

int FixGLSDNH::size_restart(int /*nlocal*/)
{
  if (idx_fm_cache < 0) return 1;
  return 7;
}

void FixGLSDNH::init()
{
  FixNH::init();

  if (!atom->sp_flag) error->all(FLERR, "Fix {} requires atom_style spin", style);
  if (!atom->fm) error->all(FLERR, "Fix {} requires atom_style spin with fm allocated", style);

  const int my_index = modify->find_fix(id);

  // Avoid subtle/invalid setups where the same atoms are integrated by multiple time_integrate fixes.
  // LAMMPS will warn, but we make it a hard error for this integrator on the same group.
  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    Fix *f = modify->fix[ifix];
    if (f == this) continue;
    if (!f->time_integrate) continue;
    if (f->igroup == igroup) {
      error->all(FLERR, "Fix {} cannot be used together with time integration fix {} on the same group",
                 style, f->id);
    }
  }

  if (utils::strmatch(update->integrate_style, "^respa")) {
    error->all(FLERR, "Fix {} is not supported with rRESPA", style);
  }

  if (lambda < 0.0) error->all(FLERR, "Fix {} glsd lambda must be >= 0", style);
  if (spin_temperature < 0.0 && spin_temperature != -1.0)
    error->all(FLERR, "Fix {} glsd stemp must be -1 (no noise), 0 (follow lattice temperature), or > 0", style);
  if (seed <= 0) error->all(FLERR, "Fix {} glsd seed must be > 0", style);

  hbar = force->hplanck / MathConst::MY_2PI;
  g_over_hbar = 0.0;
  if (hbar != 0.0) {
    constexpr double g = 2.0;
    g_over_hbar = g / hbar;
  }

  if (debug_every < 1) error->all(FLERR, "Fix {} glsd debug_every must be >= 1", style);
  if (debug_rank < 0 || debug_rank >= comm->nprocs)
    error->all(FLERR, "Fix {} glsd debug_rank must be between 0 and nprocs-1", style);

  if (hbar == 0.0) error->all(FLERR, "Fix {} requires nonzero hbar (use physical units)", style);

  // Convenience: allow users to provide a dimensionless damping ratio alpha and convert internally.
  // In our μB-absorbed convention (H in eV/μB, M in μB), lambda and (g/ħ) share the same 1/(eV*time) scaling.
  // Thus lambda = alpha * (g/ħ) makes alpha comparable to a Gilbert-like damping strength.
  if (alpha >= 0.0) {
    const double lambda_from_alpha = alpha * g_over_hbar;
    if (lambda != 0.0) {
      const double scale = std::max(1.0, std::max(std::fabs(lambda), std::fabs(lambda_from_alpha)));
      const double tol = 1.0e-12 * scale;
      if (std::fabs(lambda - lambda_from_alpha) > tol)
        error->all(FLERR,
                   "Fix {} has inconsistent glsd lambda ({}) and alpha ({}) after unit conversion "
                   "(expected lambda={} from alpha*g/hbar)",
                   style, lambda, alpha, lambda_from_alpha);
    }
    lambda = lambda_from_alpha;
  }

  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    if (utils::strmatch(modify->fix[ifix]->style, "^langevin/spin"))
      error->all(FLERR, "Fix {} cannot be combined with fix langevin/spin", style);
  }

  if (!lattice_flag && pstat_flag) {
    error->warning(FLERR, "Fix {} with glsd lattice off disables pressure control (barostat is inactive)", style);
  }

  if (midpoint_iter < 2) error->all(FLERR, "Fix {} glsd midpoint_iter must be >= 2", style);
  if (midpoint_tol < 0.0) error->all(FLERR, "Fix {} glsd midpoint_tol must be >= 0", style);

  // Midpoint iteration does internal force/field recomputes. Ensure no later time integrators observe
  // intermediate cleared/temporary force arrays.
  if (my_index >= 0) {
    for (int ifix = my_index + 1; ifix < modify->nfix; ifix++) {
      Fix *f = modify->fix[ifix];
      if (f && f->time_integrate)
        error->all(FLERR, "Fix {} with glsd midpoint_iter must be the last time integration fix (found {} after it)",
                   style, f->id);
    }
  }

  debug_open();

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
        error->all(FLERR, "Fix {} must be defined after {} to preserve external fields in glsd recompute",
                   style, modify->fix[ifix]->id);
      }
    }
  }

  // Midpoint iterations recompute forces/fields internally by clearing and re-running pair/bond/kspace.
  // Any earlier POST_FORCE fix that modifies forces/fields would have its contributions overwritten,
  // except for the explicitly replayed spin-field fixes above.
  if (lattice_flag && my_index >= 0) {
    auto is_replayed = [this](Fix *f) {
      return std::find(replay_fixes.begin(), replay_fixes.end(), f) != replay_fixes.end();
    };
    for (int ifix = 0; ifix < my_index; ifix++) {
      Fix *f = modify->fix[ifix];
      if (!f || f == this) continue;
      if (!(modify->fmask[ifix] & POST_FORCE)) continue;
      if (is_replayed(f)) continue;
      error->all(FLERR,
                 "Fix {} performs internal force/field recomputes for glsd midpoint iterations and will overwrite "
                 "contributions from earlier post_force fix {} (style {}). Define that fix after {}.",
                 style, f->id, f->style, id);
    }
  }

  ensure_custom_peratom();
}

void FixGLSDNH::setup(int vflag)
{
  FixNH::setup(vflag);
  ensure_custom_peratom();
  cache_current_fm();

  // Baseline end-of-step potential energy for drift diagnostics.
  // This is valid in typical workflows because Verlet::setup() has just performed a force evaluation.
  pe_prev_end = current_pe_total();
}

void FixGLSDNH::debug_open()
{
  if (!debug_flag) return;
  if (debug_fp) return;
  if (comm->me != debug_rank) return;

  std::string fname = debug_file;
  if (fname.empty()) {
    fname = "glsd_nh_debug.";
    fname += id;
    fname += ".log";
  }

  debug_fp = fopen(fname.c_str(), "w");
  if (!debug_fp)
    error->one(FLERR, "Fix {} could not open debug_file {}: {}", style, fname, utils::getsyserror());

  debug_header_printed = 0;
  if (debug_flush) setvbuf(debug_fp, nullptr, _IOLBF, 0);
}

void FixGLSDNH::debug_close()
{
  if (!debug_fp) return;
  fclose(debug_fp);
  debug_fp = nullptr;
}

double FixGLSDNH::current_pe_total() const
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

void FixGLSDNH::debug_log_energy(double pe_mid, double pe_end)
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

std::uint64_t FixGLSDNH::splitmix64(std::uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

double FixGLSDNH::gaussian_u64(std::uint64_t seed64, tagint tag, std::uint64_t step, int phase, int component)
{
  std::uint64_t state = seed64;
  state ^= static_cast<std::uint64_t>(tag) * 0xd1b54a32d192ed03ULL;
  state ^= step * 0x9e3779b97f4a7c15ULL;
  state ^= static_cast<std::uint64_t>(phase) * 0xbf58476d1ce4e5b9ULL;
  state ^= static_cast<std::uint64_t>(component) * 0x94d049bb133111ebULL;

  double u1 = 0.0;
  do {
    state = splitmix64(state);
    u1 = (state >> 11) * (1.0 / 9007199254740992.0);
  } while (u1 <= 0.0);
  state = splitmix64(state);
  const double u2 = (state >> 11) * (1.0 / 9007199254740992.0);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(MathConst::MY_2PI * u2);
}

double FixGLSDNH::fm_to_frequency(double fm_component) const
{
  // Field H = -dE/dM in eV/μB -> frequency: ω = (g/ħ) * H
  return g_over_hbar * fm_component;
}

void FixGLSDNH::glsd_map(double dt, double **s_in, double **fm_use, int noise_phase, double **s_out)
{
  auto *mask = atom->mask;
  auto *tag = atom->tag;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;

  double temp_use = spin_temperature;
  if (spin_temperature == 0.0) {
    if (!temperature) error->all(FLERR, "Fix {} glsd stemp=0 requires a valid lattice temperature compute", style);
    temp_use = temperature->compute_scalar();
  }
  if (spin_temperature < 0.0) temp_use = 0.0;

  const double kbt = force->boltz * temp_use;

  const std::uint64_t seed64 = static_cast<std::uint64_t>(seed);
  const std::uint64_t step64 = static_cast<std::uint64_t>(update->ntimestep);

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double Sx = s_in[i][0];
    double Sy = s_in[i][1];
    double Sz = s_in[i][2];
    const double gamma_eff = lambda;
    const double mu_s = (gamma_eff > 0.0 && temp_use > 0.0) ? (2.0 * gamma_eff * kbt) : 0.0;
    const double sigma_half = (mu_s > 0.0) ? std::sqrt(0.5 * mu_s * dt) : 0.0;

    const double fm_x = fm_use[i][0];
    const double fm_y = fm_use[i][1];
    const double fm_z = fm_use[i][2];

    const double Hx_w = fm_to_frequency(fm_x);
    const double Hy_w = fm_to_frequency(fm_y);
    const double Hz_w = fm_to_frequency(fm_z);

    const double Hx = fm_x;
    const double Hy = fm_y;
    const double Hz = fm_z;

    if (gamma_eff != 0.0) {
      Sx += 0.5 * dt * gamma_eff * Hx;
      Sy += 0.5 * dt * gamma_eff * Hy;
      Sz += 0.5 * dt * gamma_eff * Hz;
    }

    if (sigma_half > 0.0) {
      const tagint ti = tag ? tag[i] : static_cast<tagint>(i + 1);
      const int phase0 = noise_phase * 2 + 0;
      Sx += sigma_half * gaussian_u64(seed64, ti, step64, phase0, 0);
      Sy += sigma_half * gaussian_u64(seed64, ti, step64, phase0, 1);
      Sz += sigma_half * gaussian_u64(seed64, ti, step64, phase0, 2);
    }

    const double Ox = -Hx_w;
    const double Oy = -Hy_w;
    const double Oz = -Hz_w;
    const double Om2 = Ox * Ox + Oy * Oy + Oz * Oz;
    if (Om2 > 0.0) {
      const double tx = 0.5 * dt * Ox;
      const double ty = 0.5 * dt * Oy;
      const double tz = 0.5 * dt * Oz;
      const double t2 = tx * tx + ty * ty + tz * tz;

      const double vpx = Sx + (ty * Sz - tz * Sy);
      const double vpy = Sy + (tz * Sx - tx * Sz);
      const double vpz = Sz + (tx * Sy - ty * Sx);

      const double sx = 2.0 * tx / (1.0 + t2);
      const double sy = 2.0 * ty / (1.0 + t2);
      const double sz = 2.0 * tz / (1.0 + t2);

      Sx += sy * vpz - sz * vpy;
      Sy += sz * vpx - sx * vpz;
      Sz += sx * vpy - sy * vpx;
    }

    if (gamma_eff != 0.0) {
      Sx += 0.5 * dt * gamma_eff * Hx;
      Sy += 0.5 * dt * gamma_eff * Hy;
      Sz += 0.5 * dt * gamma_eff * Hz;
    }

    if (sigma_half > 0.0) {
      const tagint ti = tag ? tag[i] : static_cast<tagint>(i + 1);
      const int phase1 = noise_phase * 2 + 1;
      Sx += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 0);
      Sy += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 1);
      Sz += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 2);
    }

    s_out[i][0] = Sx;
    s_out[i][1] = Sy;
    s_out[i][2] = Sz;
  }
}

void FixGLSDNH::glsd_step(double dt, double **fm_use, int noise_phase)
{
  auto **sp = atom->sp;
  auto *mask = atom->mask;
  auto *tag = atom->tag;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;

  double temp_use = spin_temperature;
  if (spin_temperature == 0.0) {
    if (!temperature) error->all(FLERR, "Fix {} glsd stemp=0 requires a valid lattice temperature compute", style);
    temp_use = temperature->compute_scalar();
  }
  if (spin_temperature < 0.0) temp_use = 0.0;

  const double kbt = force->boltz * temp_use;

  const std::uint64_t seed64 = static_cast<std::uint64_t>(seed);
  const std::uint64_t step64 = static_cast<std::uint64_t>(update->ntimestep);

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    const double Smag = sp[i][3];
    if (Smag == 0.0) continue;

    const double sx_dir = sp[i][0];
    const double sy_dir = sp[i][1];
    const double sz_dir = sp[i][2];
    const double snorm = std::sqrt(sx_dir * sx_dir + sy_dir * sy_dir + sz_dir * sz_dir);
    if (snorm < SPIN_EPS) continue;
    const double inv_s = 1.0 / snorm;

    double Sx = Smag * sx_dir * inv_s;
    double Sy = Smag * sy_dir * inv_s;
    double Sz = Smag * sz_dir * inv_s;

    const double fm_x = fm_use[i][0];
    const double fm_y = fm_use[i][1];
    const double fm_z = fm_use[i][2];

    const double Hx_w = fm_to_frequency(fm_x);
    const double Hy_w = fm_to_frequency(fm_y);
    const double Hz_w = fm_to_frequency(fm_z);

    const double Hx = fm_x;
    const double Hy = fm_y;
    const double Hz = fm_z;

    const double gamma_eff = lambda;
    const double mu_s = (gamma_eff > 0.0 && temp_use > 0.0) ? (2.0 * gamma_eff * kbt) : 0.0;
    const double sigma_half = (mu_s > 0.0) ? std::sqrt(0.5 * mu_s * dt) : 0.0;

    // --- half-kick 1: dissipative drift + noise (B + C) ---
    if (gamma_eff != 0.0) {
      Sx += 0.5 * dt * gamma_eff * Hx;
      Sy += 0.5 * dt * gamma_eff * Hy;
      Sz += 0.5 * dt * gamma_eff * Hz;
    }

    if (sigma_half > 0.0) {
      const tagint ti = tag ? tag[i] : static_cast<tagint>(i + 1);
      const int phase0 = noise_phase * 2 + 0;
      Sx += sigma_half * gaussian_u64(seed64, ti, step64, phase0, 0);
      Sy += sigma_half * gaussian_u64(seed64, ti, step64, phase0, 1);
      Sz += sigma_half * gaussian_u64(seed64, ti, step64, phase0, 2);
    }

    // --- rotation: precession only (A) via Boris t/s form ---
    const double Ox = -Hx_w;
    const double Oy = -Hy_w;
    const double Oz = -Hz_w;
    const double Om2 = Ox * Ox + Oy * Oy + Oz * Oz;
    if (Om2 > 0.0) {
      const double tx = 0.5 * dt * Ox;
      const double ty = 0.5 * dt * Oy;
      const double tz = 0.5 * dt * Oz;
      const double t2 = tx * tx + ty * ty + tz * tz;

      const double vpx = Sx + (ty * Sz - tz * Sy);
      const double vpy = Sy + (tz * Sx - tx * Sz);
      const double vpz = Sz + (tx * Sy - ty * Sx);

      const double sx = 2.0 * tx / (1.0 + t2);
      const double sy = 2.0 * ty / (1.0 + t2);
      const double sz = 2.0 * tz / (1.0 + t2);

      Sx += sy * vpz - sz * vpy;
      Sy += sz * vpx - sx * vpz;
      Sz += sx * vpy - sy * vpx;
    }

    // --- half-kick 2: dissipative drift + noise (B + C) ---
    if (gamma_eff != 0.0) {
      Sx += 0.5 * dt * gamma_eff * Hx;
      Sy += 0.5 * dt * gamma_eff * Hy;
      Sz += 0.5 * dt * gamma_eff * Hz;
    }

    if (sigma_half > 0.0) {
      const tagint ti = tag ? tag[i] : static_cast<tagint>(i + 1);
      const int phase1 = noise_phase * 2 + 1;
      Sx += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 0);
      Sy += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 1);
      Sz += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 2);
    }

    const double Snew = std::sqrt(Sx * Sx + Sy * Sy + Sz * Sz);
    if (Snew > SPIN_EPS) {
      const double inv = 1.0 / Snew;
      sp[i][3] = Snew;
      sp[i][0] = Sx * inv;
      sp[i][1] = Sy * inv;
      sp[i][2] = Sz * inv;
    } else {
      sp[i][3] = 0.0;
    }
  }
}

void FixGLSDNH::cache_current_fm()
{
  if (idx_fm_cache < 0) return;
  if (!fm_cache) ensure_custom_peratom();

  double **fm = atom->fm;
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    fm_cache[i][0] = fm[i][0];
    fm_cache[i][1] = fm[i][1];
    fm_cache[i][2] = fm[i][2];
  }
}

void FixGLSDNH::clear_force_arrays()
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

void FixGLSDNH::replay_external_spin_fields(int vflag)
{
  if (replay_fixes.empty()) return;

  for (size_t k = 0; k < replay_fixes.size(); k++) {
    replay_fixes[k]->post_force(vflag);
  }
}

void FixGLSDNH::recompute_force_and_field(int eflag, int vflag)
{
  comm->forward_comm();
  clear_force_arrays();

  if (force->pair) {
    force->pair->compute(eflag, vflag);
  }

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

bool FixGLSDNH::solve_spin_midpoint(bool lattice_mode, int vflag, double pe_mid)
{
  if (midpoint_iter <= 1) return false;
  if (lattice_mode != (lattice_flag != 0)) return false;

  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double **sp = atom->sp;
  int *mask = atom->mask;
  const double dt = update->dt;
  const double eps = 1.0e-12;

  // Use per-atom cached S0 for lattice-moving midpoint; for spin-only midpoint store S0 in scratch.
  double **s0_src = nullptr;
  if (lattice_mode) {
    if (!s0_cache) ensure_custom_peratom();
    s0_src = s0_cache;
  } else {
    if (atom->nmax > nmax_s0) {
      nmax_s0 = atom->nmax;
      memory->grow(s0, nmax_s0, 3, "glsd/nh:s0");
    }
    // Store starting spins S0 (full vector).
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      const double Smag = sp[i][3];
      if (Smag == 0.0) {
        s0[i][0] = s0[i][1] = s0[i][2] = 0.0;
        continue;
      }
      const double sx_dir = sp[i][0];
      const double sy_dir = sp[i][1];
      const double sz_dir = sp[i][2];
      const double snorm = std::sqrt(sx_dir * sx_dir + sy_dir * sy_dir + sz_dir * sz_dir);
      if (snorm < eps) {
        s0[i][0] = s0[i][1] = s0[i][2] = 0.0;
        continue;
      }
      const double inv = 1.0 / snorm;
      s0[i][0] = Smag * sx_dir * inv;
      s0[i][1] = Smag * sy_dir * inv;
      s0[i][2] = Smag * sz_dir * inv;
    }
    s0_src = s0;
  }

  if (atom->nmax > nmax_s_guess) {
    nmax_s_guess = atom->nmax;
    memory->grow(s_guess, nmax_s_guess, 3, "glsd/nh:s_guess");
  }
  if (atom->nmax > nmax_s_map) {
    nmax_s_map = atom->nmax;
    memory->grow(s_map, nmax_s_map, 3, "glsd/nh:s_map");
  }

  // Initial guess: one explicit full-step mapping using the best available field estimate.
  double **fm_pred = lattice_mode ? atom->fm : fm_cache;
  glsd_map(dt, s0_src, fm_pred, 0, s_guess);

  for (int iter = 0; iter < midpoint_iter; iter++) {
    // Set spins to midpoint vector S^{n+1/2} = 0.5*(S0 + S_guess) for field evaluation.
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;

      const double S0x = s0_src[i][0];
      const double S0y = s0_src[i][1];
      const double S0z = s0_src[i][2];

      const double midx = 0.5 * (S0x + s_guess[i][0]);
      const double midy = 0.5 * (S0y + s_guess[i][1]);
      const double midz = 0.5 * (S0z + s_guess[i][2]);
      const double midn = std::sqrt(midx * midx + midy * midy + midz * midz);
      if (midn > eps) {
        const double inv = 1.0 / midn;
        sp[i][0] = midx * inv;
        sp[i][1] = midy * inv;
        sp[i][2] = midz * inv;
        sp[i][3] = midn;
      } else {
        sp[i][0] = sp[i][1] = sp[i][2] = 0.0;
        sp[i][3] = 0.0;
      }
    }

    // Compute field at midpoint spins.
    // Use eflag=1 to avoid pair-style differences between "energy off" and "energy on" paths.
    recompute_force_and_field(1, 0);

    double max_dev = 0.0;
    constexpr double alpha = 1.0;

    // Picard update: x_{k+1} = mix(f(x_k), x_k).
    glsd_map(dt, s0_src, atom->fm, 0, s_map);
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;

      const double Gx = s_guess[i][0];
      const double Gy = s_guess[i][1];
      const double Gz = s_guess[i][2];

      const double Sx = s_map[i][0];
      const double Sy = s_map[i][1];
      const double Sz = s_map[i][2];

      const double dx = Sx - Gx;
      const double dy = Sy - Gy;
      const double dz = Sz - Gz;
      const double d2 = dx * dx + dy * dy + dz * dz;
      const double s2 = Sx * Sx + Sy * Sy + Sz * Sz;
      const double g2 = Gx * Gx + Gy * Gy + Gz * Gz;
      const double denom = std::sqrt(std::max(std::max(s2, g2), eps * eps));
      const double dev = (denom > 0.0) ? (std::sqrt(d2) / denom) : 0.0;
      if (dev > max_dev) max_dev = dev;

      s_guess[i][0] = alpha * Sx + (1.0 - alpha) * Gx;
      s_guess[i][1] = alpha * Sy + (1.0 - alpha) * Gy;
      s_guess[i][2] = alpha * Sz + (1.0 - alpha) * Gz;
    }

    if (midpoint_tol > 0.0) {
      double max_dev_all = 0.0;
      MPI_Allreduce(&max_dev, &max_dev_all, 1, MPI_DOUBLE, MPI_MAX, world);
      if (max_dev_all <= midpoint_tol) break;
    }
  }

  // Commit the final guess as S^{n+1}.
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const double Gx = s_guess[i][0];
    const double Gy = s_guess[i][1];
    const double Gz = s_guess[i][2];
    const double Gmag = std::sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
    if (Gmag > eps) {
      const double inv = 1.0 / Gmag;
      sp[i][3] = Gmag;
      sp[i][0] = Gx * inv;
      sp[i][1] = Gy * inv;
      sp[i][2] = Gz * inv;
    } else {
      sp[i][3] = 0.0;
    }
  }

  if (lattice_mode) {
    // f^{n+1}, fm^{n+1} at the final spin state.
    recompute_force_and_field(1, vflag);
    cache_current_fm();

    const double pe_end = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
    debug_log_energy(pe_mid, pe_end);
    pe_prev_end = pe_end;
  } else {
    // Update ghost spins before the regular pair->compute() in Verlet::run().
    comm->forward_comm();
    clear_force_arrays();
  }

  return true;
}
void FixGLSDNH::initial_integrate(int vflag)
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
  }

  if (do_pstat) {
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
  }

  // Save the starting spin vectors S0 for implicit-midpoint iterations performed later (post_force)
  // or (in lattice-frozen mode) inside this method.
  if (midpoint_iter > 1 && lattice_flag) {
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup) nlocal = atom->nfirst;
    if (!s0_cache) ensure_custom_peratom();

    double **sp = atom->sp;
    int *mask = atom->mask;
    const double eps = 1.0e-12;

    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      const double Smag = sp[i][3];
      if (Smag == 0.0) {
        s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
        continue;
      }
      const double sx_dir = sp[i][0];
      const double sy_dir = sp[i][1];
      const double sz_dir = sp[i][2];
      const double snorm = std::sqrt(sx_dir * sx_dir + sy_dir * sy_dir + sz_dir * sz_dir);
      if (snorm < eps) {
        s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
        continue;
      }
      const double inv = 1.0 / snorm;
      s0_cache[i][0] = Smag * sx_dir * inv;
      s0_cache[i][1] = Smag * sy_dir * inv;
      s0_cache[i][2] = Smag * sz_dir * inv;
    }
  }

  if (!lattice_flag && midpoint_iter > 1) {
    solve_spin_midpoint(false, vflag, 0.0);
    return;
  }

  if (lattice_flag) nve_v();

  if (do_pstat) remap();

  if (lattice_flag) nve_x();

  if (do_pstat) {
    remap();
    if (kspace_flag) force->kspace->setup();
  }
}

void FixGLSDNH::post_force(int vflag)
{
  // Lattice-moving implicit-midpoint iterations (after the regular force evaluation and neighbor build).
  // This computes S^{n+1} from the saved S0 using a self-consistent midpoint field and then recomputes
  // f^{n+1}, fm^{n+1} at (x^{n+1}, S^{n+1}) for FixNH::final_integrate().
  if (lattice_flag && midpoint_iter > 1) {
    const double pe_mid = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
    solve_spin_midpoint(true, vflag, pe_mid);
    return;
  }

  // In spin-only implicit-midpoint mode, spins were already advanced to S^{n+1} inside initial_integrate().
  // The regular Verlet force evaluation on this timestep now provides fm_total^{n+1} for caching.
  if (!lattice_flag && midpoint_iter > 1) {
    cache_current_fm();
    const double pe_end = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
    debug_log_energy(pe_end, pe_end);
    pe_prev_end = pe_end;
    return;
  }
}

void FixGLSDNH::final_integrate()
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
  }

  if (do_pstat) nh_omega_dot();

  if (tstat_flag && lattice_flag) nhc_temp_integrate();
  if (do_pstat && mpchain) nhc_press_integrate();
}

int FixGLSDNH::modify_param(int narg, char **arg)
{
  if (narg < 1) return FixNH::modify_param(narg, arg);
  if (strcmp(arg[0], "glsd") != 0) return FixNH::modify_param(narg, arg);
  if (narg < 2) error->all(FLERR, "Illegal fix_modify glsd command");

  if (strcmp(arg[1], "lattice") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd lattice command");
    lattice_flag = parse_on_off(arg[2], lmp, "fix_modify glsd lattice");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_iter") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd midpoint_iter command");
    midpoint_iter = utils::inumeric(FLERR, arg[2], false, lmp);
    if (midpoint_iter < 2) error->all(FLERR, "Illegal fix_modify glsd midpoint_iter command (must be >= 2)");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_tol") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd midpoint_tol command");
    midpoint_tol = utils::numeric(FLERR, arg[2], false, lmp);
    if (midpoint_tol < 0.0) error->all(FLERR, "Illegal fix_modify glsd midpoint_tol command (must be >= 0)");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_anderson") == 0 || strcmp(arg[1], "midpoint_anderson_reg") == 0) {
    error->all(FLERR, "Illegal fix_modify glsd {} command (option removed; use midpoint_iter and midpoint_tol)",
               arg[1]);
  }
  if (strcmp(arg[1], "gammas") == 0 || strcmp(arg[1], "lambda") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd lambda command");
    lambda = utils::numeric(FLERR, arg[2], false, lmp);
    if (lambda < 0.0) error->all(FLERR, "Illegal fix_modify glsd lambda command (must be >= 0)");
    if (strcmp(arg[1], "gammas") == 0)
      error->warning(FLERR, "Fix {}: fix_modify glsd gammas is deprecated; use fix_modify glsd lambda", style);
    alpha = -1.0;
    return 3;
  }
  if (strcmp(arg[1], "alpha") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd alpha command");
    alpha = utils::numeric(FLERR, arg[2], false, lmp);
    if (alpha < 0.0) error->all(FLERR, "Illegal fix_modify glsd alpha command (must be >= 0)");
    if (g_over_hbar == 0.0) error->all(FLERR, "Illegal fix_modify glsd alpha command (requires nonzero hbar)");
    lambda = alpha * g_over_hbar;
    return 3;
  }
  if (strcmp(arg[1], "stemp") == 0 || strcmp(arg[1], "temp") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd stemp command");
    if (strcmp(arg[1], "temp") == 0)
      error->warning(FLERR, "Fix {}: fix_modify glsd temp is deprecated; use stemp", style);
    spin_temperature = utils::numeric(FLERR, arg[2], false, lmp);
    if (spin_temperature < 0.0 && spin_temperature != -1.0)
      error->all(FLERR, "Illegal fix_modify glsd stemp command (must be -1, 0, or > 0)");
    return 3;
  }
  if (strcmp(arg[1], "seed") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd seed command");
    seed = utils::inumeric(FLERR, arg[2], false, lmp);
    if (seed <= 0) error->all(FLERR, "Illegal fix_modify glsd seed command (seed must be > 0)");
    return 3;
  }
  if (strcmp(arg[1], "fm_units") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd fm_units command");
    if (strcmp(arg[2], "field") != 0)
      error->all(FLERR, "Illegal fix_modify glsd fm_units command (must be 'field': H = -dE/dM in eV/μB)");
    return 3;
  }
  if (strcmp(arg[1], "debug") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd debug command");
    debug_flag = utils::logical(FLERR, arg[2], false, lmp);
    if (!debug_flag) debug_close();
    else debug_open();
    return 3;
  }
  if (strcmp(arg[1], "debug_every") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd debug_every command");
    debug_every = utils::inumeric(FLERR, arg[2], false, lmp);
    if (debug_every < 1) error->all(FLERR, "Illegal fix_modify glsd debug_every command (must be >= 1)");
    return 3;
  }
  if (strcmp(arg[1], "debug_rank") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd debug_rank command");
    debug_rank = utils::inumeric(FLERR, arg[2], false, lmp);
    if (debug_rank < 0 || debug_rank >= comm->nprocs)
      error->all(FLERR, "Illegal fix_modify glsd debug_rank command (must be between 0 and nprocs-1)");
    debug_close();
    debug_open();
    return 3;
  }
  if (strcmp(arg[1], "debug_flush") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd debug_flush command");
    debug_flush = utils::logical(FLERR, arg[2], false, lmp);
    if (debug_fp) setvbuf(debug_fp, nullptr, debug_flush ? _IOLBF : _IOFBF, 0);
    return 3;
  }
  if (strcmp(arg[1], "debug_start") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd debug_start command");
    debug_start = utils::bnumeric(FLERR, arg[2], false, lmp);
    return 3;
  }
  if (strcmp(arg[1], "debug_file") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify glsd debug_file command");
    debug_file = arg[2];
    debug_close();
    debug_open();
    return 3;
  }

  error->all(FLERR, "Illegal fix_modify glsd command");
  return 0;
}

int FixGLSDNH::nh_payload_size_from_list(const double *list, int max_n)
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

int FixGLSDNH::pack_restart_payload_v1(double *list) const
{
  int n = 0;
  if (list) list[n] = static_cast<double>(lattice_flag);
  n++;
  if (list) list[n] = static_cast<double>(midpoint_iter);
  n++;
  if (list) list[n] = midpoint_tol;
  n++;
  if (list) list[n] = lambda;
  n++;
  if (list) list[n] = alpha;
  n++;
  if (list) list[n] = spin_temperature;
  n++;
  if (list) list[n] = static_cast<double>(seed);
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

void FixGLSDNH::unpack_restart_payload_v1(const double *list)
{
  int n = 0;
  lattice_flag = static_cast<int>(list[n++]);
  midpoint_iter = static_cast<int>(list[n++]);
  midpoint_tol = list[n++];
  lambda = list[n++];
  alpha = list[n++];
  spin_temperature = list[n++];
  seed = static_cast<int>(list[n++]);
  pe_prev_end = list[n++];
  debug_flag = static_cast<int>(list[n++]);
  debug_every = static_cast<int>(list[n++]);
  debug_rank = static_cast<int>(list[n++]);
  debug_flush = static_cast<int>(list[n++]);
  debug_start = static_cast<bigint>(list[n++]);
  debug_header_printed = static_cast<int>(list[n++]);
}

int FixGLSDNH::size_restart_global()
{
  return 2 + 1 + FixNH::size_restart_global() + 1 + pack_restart_payload_v1(nullptr);
}

int FixGLSDNH::pack_restart_data(double *list)
{
  int n = 0;
  list[n++] = RESTART_MAGIC;
  list[n++] = static_cast<double>(RESTART_VERSION);

  const int nh_n = FixNH::pack_restart_data(list + n + 1);
  list[n] = static_cast<double>(nh_n);
  n += nh_n + 1;

  const int glsd_n = pack_restart_payload_v1(list + n + 1);
  list[n] = static_cast<double>(glsd_n);
  n += glsd_n + 1;

  return n;
}

void FixGLSDNH::restart(char *buf)
{
  auto *list = reinterpret_cast<double *>(buf);
  restart_from_legacy = 0;

  if (list[0] != RESTART_MAGIC) {
    FixNH::restart(buf);
    restart_from_legacy = 1;
    error->warning(
        FLERR,
        "Fix {} style {} read legacy restart payload without USER-TSPIN glsd state; compatibility fallback reconstruction will be used",
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

  const int glsd_n = static_cast<int>(list[n++]);
  const int expected = pack_restart_payload_v1(nullptr);
  if (glsd_n != expected)
    error->all(FLERR, "Fix {} style {} restart payload glsd size mismatch (stored {}, expected {})", id, style,
               glsd_n, expected);

  unpack_restart_payload_v1(list + n);
  restart_from_legacy = 0;
}
