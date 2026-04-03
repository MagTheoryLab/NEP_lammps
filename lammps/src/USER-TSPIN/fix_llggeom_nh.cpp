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

#include "fix_llggeom_nh.h"

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

enum { NOBIAS, BIAS };
enum { ISO, ANISO, TRICLINIC };

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

struct FixLLGGeomNH::ParsedArgs {
  std::vector<std::string> fixnh_strings;
  std::vector<char *> fixnh_argv;
  int fixnh_narg = 0;

  int lattice_flag = 1;
  int midpoint_iter = 2;
  double midpoint_tol_r = 1.0e-4;
  double midpoint_tol_e = 1.0e-4;
  double midpoint_relax = 1.0;
  double alpha = 0.0;
  double gamma = -1.0;
  double spin_temperature = -1.0;
  int seed = 12345;

  int debug_flag = 0;
  int debug_every = 1;
  int debug_rank = 0;
  int debug_flush = 0;
  bigint debug_start = 0;
  std::string debug_file;
};

FixLLGGeomNH::ParsedArgs FixLLGGeomNH::parse_llggeom_trailing_block(LAMMPS *lmp, int narg, char **arg)
{
  ParsedArgs out;
  out.fixnh_strings.reserve(narg);

  int llggeom_pos = -1;
  for (int i = 3; i < narg; i++) {
    if (strcmp(arg[i], "llggeom") == 0) {
      llggeom_pos = i;
      break;
    }
  }

  const int pass_end = (llggeom_pos >= 0) ? llggeom_pos : narg;
  for (int i = 0; i < pass_end; i++) out.fixnh_strings.emplace_back(arg[i]);

  if (llggeom_pos >= 0) {
    int iarg = llggeom_pos + 1;
    while (iarg < narg) {
      if (strcmp(arg[iarg], "lattice") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom lattice", lmp->error);
        out.lattice_flag = parse_on_off(arg[iarg + 1], lmp, "fix llggeom ... llggeom lattice");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_iter") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom midpoint_iter", lmp->error);
        out.midpoint_iter = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.midpoint_iter < 2) lmp->error->all(FLERR, "fix llggeom ... llggeom midpoint_iter must be >= 2");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_tol_r") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom midpoint_tol_r", lmp->error);
        out.midpoint_tol_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.midpoint_tol_r < 0.0) lmp->error->all(FLERR, "fix llggeom ... llggeom midpoint_tol_r must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_tol_e") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom midpoint_tol_e", lmp->error);
        out.midpoint_tol_e = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.midpoint_tol_e < 0.0) lmp->error->all(FLERR, "fix llggeom ... llggeom midpoint_tol_e must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "midpoint_relax") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom midpoint_relax", lmp->error);
        out.midpoint_relax = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.midpoint_relax <= 0.0 || out.midpoint_relax > 1.0)
          lmp->error->all(FLERR, "fix llggeom ... llggeom midpoint_relax must satisfy 0 < midpoint_relax <= 1");
        iarg += 2;
      } else if (strcmp(arg[iarg], "alpha") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom alpha", lmp->error);
        out.alpha = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.alpha < 0.0) lmp->error->all(FLERR, "fix llggeom ... llggeom alpha must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "gamma") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom gamma", lmp->error);
        out.gamma = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.gamma < 0.0) lmp->error->all(FLERR, "fix llggeom ... llggeom gamma must be >= 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "stemp") == 0 || strcmp(arg[iarg], "temp") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom stemp", lmp->error);
        if (strcmp(arg[iarg], "temp") == 0)
          lmp->error->warning(FLERR, "fix llggeom ... llggeom temp is deprecated; use stemp");
        out.spin_temperature = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "seed") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom seed", lmp->error);
        out.seed = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.seed <= 0) lmp->error->all(FLERR, "fix llggeom ... llggeom seed must be > 0");
        iarg += 2;
      } else if (strcmp(arg[iarg], "fm_units") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom fm_units", lmp->error);
        if (strcmp(arg[iarg + 1], "field") != 0)
          lmp->error->all(FLERR, "fix llggeom ... llggeom fm_units must be 'field' (H = -dE/dM in eV/muB)");
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom debug", lmp->error);
        out.debug_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_every") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom debug_every", lmp->error);
        out.debug_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (out.debug_every < 1) lmp->error->all(FLERR, "fix llggeom ... llggeom debug_every must be >= 1");
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_rank") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom debug_rank", lmp->error);
        out.debug_rank = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_flush") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom debug_flush", lmp->error);
        out.debug_flush = utils::logical(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_start") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom debug_start", lmp->error);
        out.debug_start = utils::bnumeric(FLERR, arg[iarg + 1], false, lmp);
        iarg += 2;
      } else if (strcmp(arg[iarg], "debug_file") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix llggeom ... llggeom debug_file", lmp->error);
        out.debug_file = arg[iarg + 1];
        iarg += 2;
      } else {
        lmp->error->all(FLERR, "Illegal fix llggeom ... llggeom option: {}", arg[iarg]);
      }
    }
  }

  out.fixnh_argv.reserve(out.fixnh_strings.size());
  for (auto &s : out.fixnh_strings) out.fixnh_argv.push_back(s.data());
  out.fixnh_narg = static_cast<int>(out.fixnh_argv.size());
  return out;
}

FixLLGGeomNH::FixLLGGeomNH(LAMMPS *lmp, int narg, char **arg) :
    FixLLGGeomNH(lmp, parse_llggeom_trailing_block(lmp, narg, arg))
{
}

FixLLGGeomNH::FixLLGGeomNH(LAMMPS *lmp, ParsedArgs parsed) :
    FixNH(lmp, parsed.fixnh_narg, parsed.fixnh_argv.data()),
    size_vector_nh(size_vector),
    lattice_flag(parsed.lattice_flag),
    midpoint_iter(parsed.midpoint_iter),
    midpoint_tol_r(parsed.midpoint_tol_r),
    midpoint_tol_e(parsed.midpoint_tol_e),
    midpoint_relax(parsed.midpoint_relax),
    alpha(parsed.alpha),
    gamma(parsed.gamma),
    spin_temperature(parsed.spin_temperature),
    spin_temperature_cached(0.0),
    spin_temperature_cached_step(-1),
    spin_temperature_cache_valid(0),
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
    nmax_r_mid_guess(0),
    r_mid_guess(nullptr),
    nmax_e_mid_guess(0),
    e_mid_guess(nullptr),
    nmax_e_pred(0),
    e_pred(nullptr),
    nmax_f_mid(0),
    f_mid(nullptr),
    nmax_h_mid(0),
    h_mid(nullptr),
    nmax_r_new(0),
    r_new(nullptr),
    nmax_v_new(0),
    v_new(nullptr),
    nmax_e_new(0),
    e_new(nullptr),
    nmax_h_th(0),
    h_th(nullptr),
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

FixLLGGeomNH::~FixLLGGeomNH()
{
  if (copymode) return;
  debug_close();
  memory->destroy(r_mid_guess);
  memory->destroy(e_mid_guess);
  memory->destroy(e_pred);
  memory->destroy(f_mid);
  memory->destroy(h_mid);
  memory->destroy(r_new);
  memory->destroy(v_new);
  memory->destroy(e_new);
  memory->destroy(h_th);
  if (grow_callback_added && modify && (modify->find_fix(id) >= 0)) atom->delete_callback(id, Atom::GROW);
  if (restart_callback_added && modify && (modify->find_fix(id) >= 0)) atom->delete_callback(id, Atom::RESTART);
}

int FixLLGGeomNH::setmask()
{
  int mask = FixNH::setmask();
  mask |= POST_FORCE;
  if (!lattice_flag) mask &= ~PRE_EXCHANGE;
  return mask;
}

void FixLLGGeomNH::ensure_custom_peratom()
{
  int flag = -1, cols = -1, ghost = 0;

  idx_fm_cache = atom->find_custom_ghost("llggeom_fm_cache", flag, cols, ghost);
  if (idx_fm_cache < 0) idx_fm_cache = atom->add_custom("llggeom_fm_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llggeom_fm_cache has incompatible type");

  idx_s0_cache = atom->find_custom_ghost("llggeom_s0_cache", flag, cols, ghost);
  if (idx_s0_cache < 0) idx_s0_cache = atom->add_custom("llggeom_s0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llggeom_s0_cache has incompatible type");

  idx_x0_cache = atom->find_custom_ghost("llggeom_x0_cache", flag, cols, ghost);
  if (idx_x0_cache < 0) idx_x0_cache = atom->add_custom("llggeom_x0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llggeom_x0_cache has incompatible type");

  idx_v0_cache = atom->find_custom_ghost("llggeom_v0_cache", flag, cols, ghost);
  if (idx_v0_cache < 0) idx_v0_cache = atom->add_custom("llggeom_v0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llggeom_v0_cache has incompatible type");

  idx_f0_cache = atom->find_custom_ghost("llggeom_f0_cache", flag, cols, ghost);
  if (idx_f0_cache < 0) idx_f0_cache = atom->add_custom("llggeom_f0_cache", 1, 3, 0);
  else if (flag != 1 || cols != 3) error->all(FLERR, "Custom property llggeom_f0_cache has incompatible type");

  fm_cache = atom->darray[idx_fm_cache];
  s0_cache = atom->darray[idx_s0_cache];
  x0_cache = atom->darray[idx_x0_cache];
  v0_cache = atom->darray[idx_v0_cache];
  f0_cache = atom->darray[idx_f0_cache];
}

void FixLLGGeomNH::ensure_solver_arrays(int nmax)
{
  if (nmax > nmax_r_mid_guess) {
    nmax_r_mid_guess = nmax;
    memory->grow(r_mid_guess, nmax_r_mid_guess, 3, "llggeom/nh:r_mid_guess");
  }
  if (nmax > nmax_e_mid_guess) {
    nmax_e_mid_guess = nmax;
    memory->grow(e_mid_guess, nmax_e_mid_guess, 3, "llggeom/nh:e_mid_guess");
  }
  if (nmax > nmax_e_pred) {
    nmax_e_pred = nmax;
    memory->grow(e_pred, nmax_e_pred, 3, "llggeom/nh:e_pred");
  }
  if (nmax > nmax_f_mid) {
    nmax_f_mid = nmax;
    memory->grow(f_mid, nmax_f_mid, 3, "llggeom/nh:f_mid");
  }
  if (nmax > nmax_h_mid) {
    nmax_h_mid = nmax;
    memory->grow(h_mid, nmax_h_mid, 3, "llggeom/nh:h_mid");
  }
  if (nmax > nmax_r_new) {
    nmax_r_new = nmax;
    memory->grow(r_new, nmax_r_new, 3, "llggeom/nh:r_new");
  }
  if (nmax > nmax_v_new) {
    nmax_v_new = nmax;
    memory->grow(v_new, nmax_v_new, 3, "llggeom/nh:v_new");
  }
  if (nmax > nmax_e_new) {
    nmax_e_new = nmax;
    memory->grow(e_new, nmax_e_new, 3, "llggeom/nh:e_new");
  }
  if (nmax > nmax_h_th) {
    nmax_h_th = nmax;
    memory->grow(h_th, nmax_h_th, 3, "llggeom/nh:h_th");
  }
}

void FixLLGGeomNH::grow_arrays(int nmax)
{
  if (idx_fm_cache < 0) return;

  if (nmax > nmax_old) {
    memory->grow(atom->darray[idx_fm_cache], nmax, 3, "llggeom/nh:fm_cache");
    memory->grow(atom->darray[idx_s0_cache], nmax, 3, "llggeom/nh:s0_cache");
    memory->grow(atom->darray[idx_x0_cache], nmax, 3, "llggeom/nh:x0_cache");
    memory->grow(atom->darray[idx_v0_cache], nmax, 3, "llggeom/nh:v0_cache");
    memory->grow(atom->darray[idx_f0_cache], nmax, 3, "llggeom/nh:f0_cache");
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

void FixLLGGeomNH::copy_arrays(int i, int j, int /*delflag*/)
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

int FixLLGGeomNH::pack_exchange(int i, double *buf)
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

int FixLLGGeomNH::unpack_exchange(int nlocal, double *buf)
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

int FixLLGGeomNH::pack_restart(int i, double *buf)
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

void FixLLGGeomNH::unpack_restart(int nlocal, int nth)
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

int FixLLGGeomNH::maxsize_restart()
{
  return 4;
}

int FixLLGGeomNH::size_restart(int /*nlocal*/)
{
  return 4;
}

void FixLLGGeomNH::init()
{
  FixNH::init();

  if (!atom->sp_flag) error->all(FLERR, "Fix {} requires atom_style spin", style);
  if (!atom->fm) error->all(FLERR, "Fix {} requires atom_style spin with fm allocated", style);

  const int my_index = modify->find_fix(id);
  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    Fix *fix = modify->fix[ifix];
    if (fix == this) continue;
    if (!fix->time_integrate) continue;
    if (fix->igroup == igroup)
      error->all(FLERR, "Fix {} cannot be used together with time integration fix {} on the same group", style,
                 fix->id);
  }

  if (utils::strmatch(update->integrate_style, "^respa"))
    error->all(FLERR, "Fix {} is not supported with rRESPA", style);

  if (midpoint_iter < 2) error->all(FLERR, "Fix {} llggeom midpoint_iter must be >= 2", style);
  if (midpoint_tol_r < 0.0) error->all(FLERR, "Fix {} llggeom midpoint_tol_r must be >= 0", style);
  if (midpoint_tol_e < 0.0) error->all(FLERR, "Fix {} llggeom midpoint_tol_e must be >= 0", style);
  if (midpoint_relax <= 0.0 || midpoint_relax > 1.0)
    error->all(FLERR, "Fix {} llggeom midpoint_relax must satisfy 0 < midpoint_relax <= 1", style);
  if (alpha < 0.0) error->all(FLERR, "Fix {} llggeom alpha must be >= 0", style);
  if (gamma < 0.0 && gamma != -1.0) error->all(FLERR, "Fix {} llggeom gamma must be >= 0", style);
  if (spin_temperature < 0.0 && spin_temperature != -1.0)
    error->all(FLERR, "Fix {} llggeom stemp must be -1 (no noise), 0 (follow lattice temperature), or > 0", style);
  if (seed <= 0) error->all(FLERR, "Fix {} llggeom seed must be > 0", style);

  hbar = force->hplanck / MathConst::MY_2PI;
  if (gamma < 0.0 && hbar == 0.0) error->all(FLERR, "Fix {} requires nonzero hbar (use physical units)", style);
  refresh_g_over_hbar();

  if (debug_every < 1) error->all(FLERR, "Fix {} llggeom debug_every must be >= 1", style);
  if (debug_rank < 0 || debug_rank >= comm->nprocs)
    error->all(FLERR, "Fix {} llggeom debug_rank must be between 0 and nprocs-1", style);

  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    if (utils::strmatch(modify->fix[ifix]->style, "^langevin/spin"))
      error->all(FLERR, "Fix {} cannot be combined with fix langevin/spin", style);
  }

  if (!lattice_flag && pstat_flag)
    error->warning(FLERR, "Fix {} with llggeom lattice off disables pressure control (barostat is inactive)", style);

  if (my_index >= 0) {
    for (int ifix = my_index + 1; ifix < modify->nfix; ifix++) {
      Fix *fix = modify->fix[ifix];
      if (fix && fix->time_integrate)
        error->all(FLERR,
                   "Fix {} with llggeom midpoint_iter must be the last time integration fix (found {} after it)",
                   style, fix->id);
    }
  }

  replay_fixes.clear();
  replay_fix_indices.clear();
  for (int ifix = 0; ifix < modify->nfix; ifix++) {
    const char *s = modify->fix[ifix]->style;
    if (utils::strmatch(s, "^precession/spin")) {
      error->all(
          FLERR,
          "Fix {} (USER-TSPIN) does not support fix precession/spin; use fix tspin/precession/spin or setforce/spin instead",
          style);
    }
    if (utils::strmatch(s, "^setforce/spin") || utils::strmatch(s, "^tspin/precession/spin")) {
      replay_fixes.push_back(modify->fix[ifix]);
      replay_fix_indices.push_back(ifix);
      if (my_index >= 0 && ifix > my_index)
        error->all(FLERR, "Fix {} must be defined after {} to preserve external fields in llggeom recompute", style,
                   modify->fix[ifix]->id);
    }
  }

  if (my_index >= 0) {
    auto is_replayed = [this](Fix *fix) {
      return std::find(replay_fixes.begin(), replay_fixes.end(), fix) != replay_fixes.end();
    };
    for (int ifix = 0; ifix < my_index; ifix++) {
      Fix *fix = modify->fix[ifix];
      if (!fix || fix == this) continue;
      if (!(modify->fmask[ifix] & POST_FORCE)) continue;
      if (is_replayed(fix)) continue;
      error->all(FLERR,
                 "Fix {} performs internal force/field recomputes for llggeom midpoint iterations and will overwrite contributions from earlier post_force fix {} (style {}). Define that fix after {}.",
                 style, fix->id, fix->style, id);
    }
  }

  ensure_custom_peratom();
  ensure_solver_arrays(atom->nmax);
  debug_open();
}

void FixLLGGeomNH::setup(int vflag)
{
  FixNH::setup(vflag);
  ensure_custom_peratom();
  ensure_solver_arrays(atom->nmax);
  cache_current_fm();
  pe_prev_end = current_pe_total();
}

void FixLLGGeomNH::debug_open()
{
  if (!debug_flag) return;
  if (debug_fp) return;
  if (comm->me != debug_rank) return;

  std::string fname = debug_file;
  if (fname.empty()) {
    fname = "llggeom_nh_debug.";
    fname += id;
    fname += ".log";
  }

  debug_fp = fopen(fname.c_str(), "w");
  if (!debug_fp) error->one(FLERR, "Fix {} could not open debug_file {}: {}", style, fname, utils::getsyserror());

  debug_header_printed = 0;
  if (debug_flush) setvbuf(debug_fp, nullptr, _IOLBF, 0);
}

void FixLLGGeomNH::debug_close()
{
  if (!debug_fp) return;
  fclose(debug_fp);
  debug_fp = nullptr;
}

double FixLLGGeomNH::current_pe_total() const
{
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

std::uint64_t FixLLGGeomNH::splitmix64(std::uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

double FixLLGGeomNH::gaussian_u64(std::uint64_t seed64, tagint tag, std::uint64_t step, int component)
{
  std::uint64_t state = seed64;
  state ^= static_cast<std::uint64_t>(tag) * 0xd1b54a32d192ed03ULL;
  state ^= step * 0x9e3779b97f4a7c15ULL;
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

double FixLLGGeomNH::compute_spin_temperature()
{
  if (spin_temperature < 0.0) return 0.0;
  if (spin_temperature > 0.0) return spin_temperature;
  if (!temperature) error->all(FLERR, "Fix {} llggeom stemp=0 requires a valid lattice temperature compute", style);

  const bigint step = update->ntimestep;
  if (spin_temperature_cache_valid && spin_temperature_cached_step == step) return spin_temperature_cached;

  spin_temperature_cached = temperature->compute_scalar();
  spin_temperature_cached_step = step;
  spin_temperature_cache_valid = 1;
  return spin_temperature_cached;
}

void FixLLGGeomNH::fill_frozen_thermal_field()
{
  ensure_solver_arrays(atom->nmax);

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const double dt = update->dt;
  const double temp_use = compute_spin_temperature();
  const double prefactor_base =
      (alpha > 0.0 && temp_use > 0.0 && g_over_hbar > 0.0 && dt > 0.0) ? (2.0 * alpha * force->boltz * temp_use) : 0.0;
  const std::uint64_t seed64 = static_cast<std::uint64_t>(seed);
  const std::uint64_t step64 = static_cast<std::uint64_t>(update->ntimestep);
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  double **sp = atom->sp;

  for (int i = 0; i < nlocal; i++) {
    h_th[i][0] = h_th[i][1] = h_th[i][2] = 0.0;
    if (!(mask[i] & groupbit)) continue;

    const double mu_s = sp[i][3];
    if (prefactor_base <= 0.0 || mu_s <= THERMAL_EPS) continue;

    const double sigma = std::sqrt(prefactor_base / (g_over_hbar * mu_s * dt));
    const tagint ti = tag ? tag[i] : static_cast<tagint>(i + 1);
    h_th[i][0] = sigma * gaussian_u64(seed64, ti, step64, 0);
    h_th[i][1] = sigma * gaussian_u64(seed64, ti, step64, 1);
    h_th[i][2] = sigma * gaussian_u64(seed64, ti, step64, 2);
  }
}

void FixLLGGeomNH::debug_log_energy(double pe_mid, double pe_end)
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
          static_cast<long long>(update->ntimestep), update->atime, update->dt, pe_prev_end, pe_mid, pe_end,
          dE_step, dE_mid_end);

  if (debug_flush) fflush(debug_fp);
}

double FixLLGGeomNH::get_mass(int i) const
{
  if (atom->rmass) return atom->rmass[i];
  return atom->mass[atom->type[i]];
}

void FixLLGGeomNH::refresh_g_over_hbar()
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

void FixLLGGeomNH::normalize_direction(const double *vin, double *vout) const
{
  const double n = std::sqrt(vin[0] * vin[0] + vin[1] * vin[1] + vin[2] * vin[2]);
  if (n <= SPIN_EPS) {
    vout[0] = vout[1] = vout[2] = 0.0;
    return;
  }
  const double inv = 1.0 / n;
  vout[0] = vin[0] * inv;
  vout[1] = vin[1] * inv;
  vout[2] = vin[2] * inv;
}

void FixLLGGeomNH::damping_half_step_direction(const double *ein, const double *h, double tau, double *eout) const
{
  double e[3];
  normalize_direction(ein, e);
  const double hmag = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
  if (hmag < FIELD_EPS || tau == 0.0 || alpha == 0.0) {
    eout[0] = e[0];
    eout[1] = e[1];
    eout[2] = e[2];
    return;
  }

  const double hh[3] = {h[0] / hmag, h[1] / hmag, h[2] / hmag};
  const double u = std::max(-1.0, std::min(1.0, e[0] * hh[0] + e[1] * hh[1] + e[2] * hh[2]));
  const double wx = e[0] - u * hh[0];
  const double wy = e[1] - u * hh[1];
  const double wz = e[2] - u * hh[2];
  const double rho = std::sqrt(wx * wx + wy * wy + wz * wz);

  const double b = alpha * g_over_hbar / (1.0 + alpha * alpha);

  if (rho < PERP_EPS) {
    eout[0] = (u >= 0.0 ? 1.0 : -1.0) * hh[0];
    eout[1] = (u >= 0.0 ? 1.0 : -1.0) * hh[1];
    eout[2] = (u >= 0.0 ? 1.0 : -1.0) * hh[2];
    return;
  }

  double q;
  if (1.0 + u > PERP_EPS)
    q = rho / (1.0 + u);
  else
    q = (rho > PERP_EPS) ? (1.0 - u) / rho : 1.0e300;

  const double qp = q * std::exp(-b * hmag * tau);
  const double qp2 = qp * qp;
  const double up = (1.0 - qp2) / (1.0 + qp2);
  const double rhop = 2.0 * qp / (1.0 + qp2);
  const double scale = rhop / rho;

  eout[0] = up * hh[0] + scale * wx;
  eout[1] = up * hh[1] + scale * wy;
  eout[2] = up * hh[2] + scale * wz;
  normalize_direction(eout, eout);
}

void FixLLGGeomNH::boris_precession_step_direction(const double *ein, const double *h, double dt, double *eout) const
{
  double e[3];
  normalize_direction(ein, e);
  const double a = g_over_hbar / (1.0 + alpha * alpha);
  const double tx = -0.5 * dt * a * h[0];
  const double ty = -0.5 * dt * a * h[1];
  const double tz = -0.5 * dt * a * h[2];
  const double t2 = tx * tx + ty * ty + tz * tz;

  const double epx = e[0] + (e[1] * tz - e[2] * ty);
  const double epy = e[1] + (e[2] * tx - e[0] * tz);
  const double epz = e[2] + (e[0] * ty - e[1] * tx);

  const double sx = 2.0 * tx / (1.0 + t2);
  const double sy = 2.0 * ty / (1.0 + t2);
  const double sz = 2.0 * tz / (1.0 + t2);

  eout[0] = e[0] + (epy * sz - epz * sy);
  eout[1] = e[1] + (epz * sx - epx * sz);
  eout[2] = e[2] + (epx * sy - epy * sx);
  normalize_direction(eout, eout);
}

void FixLLGGeomNH::spin_map_geom_direction(const double *e0, const double *h, double dt, double *e1) const
{
  double tmp1[3], tmp2[3];
  damping_half_step_direction(e0, h, 0.5 * dt, tmp1);
  boris_precession_step_direction(tmp1, h, dt, tmp2);
  damping_half_step_direction(tmp2, h, 0.5 * dt, e1);
  normalize_direction(e1, e1);
}

void FixLLGGeomNH::write_spin_from_direction(int i, const double *dir)
{
  const double mag = atom->sp[i][3];
  if (mag <= SPIN_EPS) {
    atom->sp[i][0] = atom->sp[i][1] = atom->sp[i][2] = 0.0;
    atom->sp[i][3] = 0.0;
    return;
  }

  double unit[3];
  normalize_direction(dir, unit);
  atom->sp[i][0] = unit[0];
  atom->sp[i][1] = unit[1];
  atom->sp[i][2] = unit[2];
  atom->sp[i][3] = mag;
}

void FixLLGGeomNH::cache_lattice_moving_step_start_state()
{
  if (!fm_cache) ensure_custom_peratom();
  ensure_solver_arrays(atom->nmax);
  spin_temperature_cache_valid = 0;
  fill_frozen_thermal_field();

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
      r_mid_guess[i][k] = x[i][k];
      r_new[i][k] = x[i][k];
      v_new[i][k] = v[i][k];
      e_mid_guess[i][k] = 0.0;
      e_pred[i][k] = 0.0;
      f_mid[i][k] = 0.0;
      h_mid[i][k] = 0.0;
      e_new[i][k] = 0.0;
    }

    const double mag = sp[i][3];
    if (mag > SPIN_EPS) {
      const double dirn = std::sqrt(sp[i][0] * sp[i][0] + sp[i][1] * sp[i][1] + sp[i][2] * sp[i][2]);
      if (dirn > SPIN_EPS) {
        const double scale = mag / dirn;
        s0_cache[i][0] = sp[i][0] * scale;
        s0_cache[i][1] = sp[i][1] * scale;
        s0_cache[i][2] = sp[i][2] * scale;
      } else {
        s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
      }
    } else {
      s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
    }
  }
}

void FixLLGGeomNH::build_predictor_midpoint_state()
{
  if (!fm_cache) ensure_custom_peratom();
  ensure_solver_arrays(atom->nmax);
  spin_temperature_cache_valid = 0;
  fill_frozen_thermal_field();

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
      r_mid_guess[i][k] = x[i][k];
      e_mid_guess[i][k] = 0.0;
      e_pred[i][k] = 0.0;
      f_mid[i][k] = 0.0;
      h_mid[i][k] = 0.0;
      r_new[i][k] = x[i][k];
      v_new[i][k] = v[i][k];
      e_new[i][k] = 0.0;
    }

    const double mag = sp[i][3];
    if (mag > SPIN_EPS) {
      const double dirn = std::sqrt(sp[i][0] * sp[i][0] + sp[i][1] * sp[i][1] + sp[i][2] * sp[i][2]);
      if (dirn > SPIN_EPS) {
        const double scale = mag / dirn;
        s0_cache[i][0] = sp[i][0] * scale;
        s0_cache[i][1] = sp[i][1] * scale;
        s0_cache[i][2] = sp[i][2] * scale;
      } else {
        s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
      }
    } else {
      s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
    }

    if (lattice_flag) {
      const double mass = get_mass(i);
      const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
      for (int k = 0; k < 3; k++) {
        r_mid_guess[i][k] =
            x0_cache[i][k] + 0.5 * dt * v0_cache[i][k] + 0.125 * dt * dt * ftm2v * f0_cache[i][k] * inv_mass;
        x[i][k] = r_mid_guess[i][k];
      }
    }

    if (mag > SPIN_EPS) {
      const double e0[3] = {s0_cache[i][0] / mag, s0_cache[i][1] / mag, s0_cache[i][2] / mag};
      const double h_use[3] = {fm_cache[i][0] + h_th[i][0], fm_cache[i][1] + h_th[i][1], fm_cache[i][2] + h_th[i][2]};
      spin_map_geom_direction(e0, h_use, 0.5 * dt, e_pred[i]);
      const double emid_sum[3] = {e0[0] + e_pred[i][0], e0[1] + e_pred[i][1], e0[2] + e_pred[i][2]};
      normalize_direction(emid_sum, e_mid_guess[i]);
      sp[i][0] = e_mid_guess[i][0];
      sp[i][1] = e_mid_guess[i][1];
      sp[i][2] = e_mid_guess[i][2];
      sp[i][3] = mag;
    } else {
      sp[i][0] = sp[i][1] = sp[i][2] = 0.0;
      sp[i][3] = 0.0;
    }
  }
}

void FixLLGGeomNH::apply_midpoint_corrector(double **r_mid_corr, double **e_mid_corr, int update_lattice)
{
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const double dt = update->dt;
  const double ftm2v = force->ftm2v;
  int *mask = atom->mask;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    for (int k = 0; k < 3; k++) {
      f_mid[i][k] = atom->f[i][k];
      h_mid[i][k] = atom->fm[i][k] + h_th[i][k];
    }

    if (update_lattice) {
      const double mass = get_mass(i);
      const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
      for (int k = 0; k < 3; k++) {
        v_new[i][k] = v0_cache[i][k] + dt * ftm2v * f_mid[i][k] * inv_mass;
        r_new[i][k] = x0_cache[i][k] + 0.5 * dt * (v0_cache[i][k] + v_new[i][k]);
        r_mid_corr[i][k] = 0.5 * (x0_cache[i][k] + r_new[i][k]);
      }
    } else {
      for (int k = 0; k < 3; k++) {
        v_new[i][k] = v0_cache[i][k];
        r_new[i][k] = x0_cache[i][k];
        r_mid_corr[i][k] = x0_cache[i][k];
      }
    }

    const double mag = std::sqrt(s0_cache[i][0] * s0_cache[i][0] + s0_cache[i][1] * s0_cache[i][1] +
                                 s0_cache[i][2] * s0_cache[i][2]);
    if (mag > SPIN_EPS) {
      const double e0[3] = {s0_cache[i][0] / mag, s0_cache[i][1] / mag, s0_cache[i][2] / mag};
      spin_map_geom_direction(e0, h_mid[i], dt, e_new[i]);
      const double emid_sum[3] = {e0[0] + e_new[i][0], e0[1] + e_new[i][1], e0[2] + e_new[i][2]};
      normalize_direction(emid_sum, e_mid_corr[i]);
    } else {
      e_new[i][0] = e_new[i][1] = e_new[i][2] = 0.0;
      e_mid_corr[i][0] = e_mid_corr[i][1] = e_mid_corr[i][2] = 0.0;
    }
  }
}

void FixLLGGeomNH::cache_current_fm()
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

void FixLLGGeomNH::clear_force_arrays()
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

void FixLLGGeomNH::rebuild_neighbors_for_current_positions()
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

void FixLLGGeomNH::replay_external_spin_fields(int vflag)
{
  if (replay_fixes.empty()) return;
  for (auto *fix : replay_fixes) fix->post_force(vflag);
}

void FixLLGGeomNH::recompute_force_and_field(int eflag, int vflag)
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

bool FixLLGGeomNH::solve_spin_midpoint(bool lattice_mode, int vflag, double pe_mid)
{
  if (midpoint_iter <= 1) return false;
  if (lattice_mode != (lattice_flag != 0)) return false;
  ensure_solver_arrays(atom->nmax);

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const double dt = update->dt;
  const double tol_e = midpoint_tol_e;
  const int update_lattice = 0;
  double **x = atom->x;
  double **v = atom->v;
  double **sp = atom->sp;
  int *mask = atom->mask;

  for (int iter = 0; iter < midpoint_iter; iter++) {
    apply_midpoint_corrector(r_mid_guess, e_mid_guess, update_lattice);

    double re_local = 0.0;
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;

      const double en = std::sqrt(e_mid_guess[i][0] * e_mid_guess[i][0] + e_mid_guess[i][1] * e_mid_guess[i][1] +
                                  e_mid_guess[i][2] * e_mid_guess[i][2]);
      const double sn = std::sqrt(sp[i][0] * sp[i][0] + sp[i][1] * sp[i][1] + sp[i][2] * sp[i][2]);
      double de = 0.0;
      if (en > SPIN_EPS && sn > SPIN_EPS) {
        double dot = (e_mid_guess[i][0] * sp[i][0] + e_mid_guess[i][1] * sp[i][1] + e_mid_guess[i][2] * sp[i][2]) /
            (en * sn);
        dot = std::max(-1.0, std::min(1.0, dot));
        de = std::sqrt(std::max(0.0, 0.5 * (1.0 - dot)));
      } else {
        const double dsx = e_mid_guess[i][0] - sp[i][0];
        const double dsy = e_mid_guess[i][1] - sp[i][1];
        const double dsz = e_mid_guess[i][2] - sp[i][2];
        de = std::sqrt(dsx * dsx + dsy * dsy + dsz * dsz);
      }
      re_local = std::max(re_local, de);
    }

    double re_all = re_local;
    MPI_Allreduce(&re_local, &re_all, 1, MPI_DOUBLE, MPI_MAX, world);
    if ((iter == midpoint_iter - 1) || (re_all <= tol_e)) break;

    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;

      double mixed[3] = {midpoint_relax * e_mid_guess[i][0] + (1.0 - midpoint_relax) * sp[i][0],
                         midpoint_relax * e_mid_guess[i][1] + (1.0 - midpoint_relax) * sp[i][1],
                         midpoint_relax * e_mid_guess[i][2] + (1.0 - midpoint_relax) * sp[i][2]};
      normalize_direction(mixed, mixed);
      write_spin_from_direction(i, mixed);
    }

    recompute_force_and_field(1, 0);
  }

  // The iteration updates e_mid_guess/r_mid_guess to the latest corrected midpoint,
  // but atom->x/sp and atom->fm still correspond to the previous iterate because the
  // loop exits before another recompute. Refresh once so the endpoint map uses the
  // field evaluated at the final corrected midpoint, not a one-iteration-stale field.
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    write_spin_from_direction(i, e_mid_guess[i]);
  }

  recompute_force_and_field(1, 0);
  apply_midpoint_corrector(r_mid_guess, e_mid_guess, update_lattice);

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    write_spin_from_direction(i, e_new[i]);
  }

  recompute_force_and_field(1, vflag);
  cache_current_fm();
  const double pe_end = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
  debug_log_energy(pe_mid, pe_end);
  pe_prev_end = pe_end;
  return true;
}

void FixLLGGeomNH::initial_integrate(int /*vflag*/)
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

void FixLLGGeomNH::post_force(int vflag)
{
  const double pe_mid = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
  solve_spin_midpoint(lattice_flag != 0, vflag, pe_mid);
}

void FixLLGGeomNH::final_integrate()
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

int FixLLGGeomNH::modify_param(int narg, char **arg)
{
  if (narg < 1) return FixNH::modify_param(narg, arg);
  if (strcmp(arg[0], "llggeom") != 0) return FixNH::modify_param(narg, arg);
  if (narg < 2) error->all(FLERR, "Illegal fix_modify llggeom command");

  if (strcmp(arg[1], "lattice") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom lattice command");
    lattice_flag = parse_on_off(arg[2], lmp, "fix_modify llggeom lattice");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_iter") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom midpoint_iter command");
    midpoint_iter = utils::inumeric(FLERR, arg[2], false, lmp);
    if (midpoint_iter < 2) error->all(FLERR, "Illegal fix_modify llggeom midpoint_iter command (must be >= 2)");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_tol_r") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom midpoint_tol_r command");
    midpoint_tol_r = utils::numeric(FLERR, arg[2], false, lmp);
    if (midpoint_tol_r < 0.0) error->all(FLERR, "Illegal fix_modify llggeom midpoint_tol_r command (must be >= 0)");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_tol_e") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom midpoint_tol_e command");
    midpoint_tol_e = utils::numeric(FLERR, arg[2], false, lmp);
    if (midpoint_tol_e < 0.0) error->all(FLERR, "Illegal fix_modify llggeom midpoint_tol_e command (must be >= 0)");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_relax") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom midpoint_relax command");
    midpoint_relax = utils::numeric(FLERR, arg[2], false, lmp);
    if (midpoint_relax <= 0.0 || midpoint_relax > 1.0)
      error->all(FLERR, "Illegal fix_modify llggeom midpoint_relax command (must satisfy 0 < value <= 1)");
    return 3;
  }
  if (strcmp(arg[1], "alpha") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom alpha command");
    alpha = utils::numeric(FLERR, arg[2], false, lmp);
    if (alpha < 0.0) error->all(FLERR, "Illegal fix_modify llggeom alpha command (must be >= 0)");
    return 3;
  }
  if (strcmp(arg[1], "gamma") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom gamma command");
    gamma = utils::numeric(FLERR, arg[2], false, lmp);
    if (gamma < 0.0) error->all(FLERR, "Illegal fix_modify llggeom gamma command (must be >= 0)");
    refresh_g_over_hbar();
    return 3;
  }
  if (strcmp(arg[1], "stemp") == 0 || strcmp(arg[1], "temp") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom stemp command");
    if (strcmp(arg[1], "temp") == 0) error->warning(FLERR, "Fix {}: fix_modify llggeom temp is deprecated; use stemp", style);
    spin_temperature = utils::numeric(FLERR, arg[2], false, lmp);
    if (spin_temperature < 0.0 && spin_temperature != -1.0)
      error->all(FLERR, "Illegal fix_modify llggeom stemp command (must be -1, 0, or > 0)");
    spin_temperature_cache_valid = 0;
    spin_temperature_cached_step = -1;
    return 3;
  }
  if (strcmp(arg[1], "seed") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom seed command");
    seed = utils::inumeric(FLERR, arg[2], false, lmp);
    if (seed <= 0) error->all(FLERR, "Illegal fix_modify llggeom seed command (seed must be > 0)");
    return 3;
  }
  if (strcmp(arg[1], "fm_units") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom fm_units command");
    if (strcmp(arg[2], "field") != 0)
      error->all(FLERR, "Illegal fix_modify llggeom fm_units command (must be 'field')");
    return 3;
  }
  if (strcmp(arg[1], "debug") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom debug command");
    debug_flag = utils::logical(FLERR, arg[2], false, lmp);
    if (!debug_flag)
      debug_close();
    else
      debug_open();
    return 3;
  }
  if (strcmp(arg[1], "debug_every") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom debug_every command");
    debug_every = utils::inumeric(FLERR, arg[2], false, lmp);
    if (debug_every < 1) error->all(FLERR, "Illegal fix_modify llggeom debug_every command (must be >= 1)");
    return 3;
  }
  if (strcmp(arg[1], "debug_rank") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom debug_rank command");
    debug_rank = utils::inumeric(FLERR, arg[2], false, lmp);
    if (debug_rank < 0 || debug_rank >= comm->nprocs)
      error->all(FLERR, "Illegal fix_modify llggeom debug_rank command (must be between 0 and nprocs-1)");
    debug_close();
    debug_open();
    return 3;
  }
  if (strcmp(arg[1], "debug_flush") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom debug_flush command");
    debug_flush = utils::logical(FLERR, arg[2], false, lmp);
    if (debug_fp) setvbuf(debug_fp, nullptr, debug_flush ? _IOLBF : _IOFBF, 0);
    return 3;
  }
  if (strcmp(arg[1], "debug_start") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom debug_start command");
    debug_start = utils::bnumeric(FLERR, arg[2], false, lmp);
    return 3;
  }
  if (strcmp(arg[1], "debug_file") == 0) {
    if (narg < 3) error->all(FLERR, "Illegal fix_modify llggeom debug_file command");
    debug_file = arg[2];
    debug_close();
    debug_open();
    return 3;
  }

  error->all(FLERR, "Illegal fix_modify llggeom command");
  return 0;
}

int FixLLGGeomNH::nh_payload_size_from_list(const double *list, int max_n)
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

int FixLLGGeomNH::pack_restart_payload_v1(double *list) const
{
  int n = 0;
  if (list) list[n] = static_cast<double>(lattice_flag);
  n++;
  if (list) list[n] = static_cast<double>(midpoint_iter);
  n++;
  if (list) list[n] = midpoint_tol_r;
  n++;
  if (list) list[n] = midpoint_tol_e;
  n++;
  if (list) list[n] = midpoint_relax;
  n++;
  if (list) list[n] = alpha;
  n++;
  if (list) list[n] = spin_temperature;
  n++;
  if (list) list[n] = static_cast<double>(seed);
  n++;
  if (list) list[n] = spin_temperature_cached;
  n++;
  if (list) list[n] = static_cast<double>(spin_temperature_cached_step);
  n++;
  if (list) list[n] = static_cast<double>(spin_temperature_cache_valid);
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

void FixLLGGeomNH::unpack_restart_payload_v1(const double *list)
{
  int n = 0;
  lattice_flag = static_cast<int>(list[n++]);
  midpoint_iter = static_cast<int>(list[n++]);
  midpoint_tol_r = list[n++];
  midpoint_tol_e = list[n++];
  midpoint_relax = list[n++];
  alpha = list[n++];
  spin_temperature = list[n++];
  seed = static_cast<int>(list[n++]);
  spin_temperature_cached = list[n++];
  spin_temperature_cached_step = static_cast<bigint>(list[n++]);
  spin_temperature_cache_valid = static_cast<int>(list[n++]);
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

int FixLLGGeomNH::pack_restart_payload_v2(double *list) const
{
  const int n = pack_restart_payload_v1(list);
  if (list) list[n] = gamma;
  return n + 1;
}

void FixLLGGeomNH::unpack_restart_payload_v2(const double *list)
{
  unpack_restart_payload_v1(list);
  gamma = list[pack_restart_payload_v1(nullptr)];
  refresh_g_over_hbar();
}

int FixLLGGeomNH::size_restart_global()
{
  return 2 + 1 + FixNH::size_restart_global() + 1 + pack_restart_payload_v2(nullptr);
}

int FixLLGGeomNH::pack_restart_data(double *list)
{
  int n = 0;
  list[n++] = RESTART_MAGIC;
  list[n++] = static_cast<double>(RESTART_VERSION);

  const int nh_n = FixNH::pack_restart_data(list + n + 1);
  list[n] = static_cast<double>(nh_n);
  n += nh_n + 1;

  const int llggeom_n = pack_restart_payload_v2(list + n + 1);
  list[n] = static_cast<double>(llggeom_n);
  n += llggeom_n + 1;
  return n;
}

void FixLLGGeomNH::restart(char *buf)
{
  auto *list = reinterpret_cast<double *>(buf);
  restart_from_legacy = 0;

  if (list[0] != RESTART_MAGIC) {
    FixNH::restart(buf);
    restart_from_legacy = 1;
    error->warning(
        FLERR,
        "Fix {} style {} read legacy restart payload without USER-TSPIN llggeom state; compatibility fallback reconstruction will be used",
        id, style);
    return;
  }

  const int version = static_cast<int>(list[1]);
  if (version != 2 && version != RESTART_VERSION)
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

  const int llggeom_n = static_cast<int>(list[n++]);
  const int expected = (version == RESTART_VERSION) ? pack_restart_payload_v2(nullptr) : pack_restart_payload_v1(nullptr);
  if (llggeom_n != expected)
    error->all(FLERR, "Fix {} style {} restart payload llggeom size mismatch (stored {}, expected {})", id, style,
               llggeom_n, expected);

  if (version == RESTART_VERSION)
    unpack_restart_payload_v2(list + n);
  else
    unpack_restart_payload_v1(list + n);
  restart_from_legacy = 0;
}
