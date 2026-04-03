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

#include "fix_glsd_nh_kokkos.h"

#ifdef LMP_KOKKOS

#include "angle.h"
#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
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
#include "memory_kokkos.h"
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

// Keep integer values consistent with FixNH's internal enums in src/fix_nh.cpp and src/KOKKOS/fix_nh_kokkos.cpp.
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

KOKKOS_INLINE_FUNCTION std::uint64_t splitmix64(std::uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

KOKKOS_INLINE_FUNCTION double gaussian_u64(std::uint64_t seed64, tagint tag, std::uint64_t step, int phase,
                                          int component)
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
  return sqrt(-2.0 * log(u1)) * cos(MathConst::MY_2PI * u2);
}
}    // namespace

template <class DeviceType>
typename FixGLSDNHKokkos<DeviceType>::ParsedArgs FixGLSDNHKokkos<DeviceType>::parse_glsd_trailing_block(
    LAMMPS *lmp, int narg, char **arg)
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

template <class DeviceType>
FixGLSDNHKokkos<DeviceType>::FixGLSDNHKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixGLSDNHKokkos(lmp, parse_glsd_trailing_block(lmp, narg, arg))
{
}

template <class DeviceType>
FixGLSDNHKokkos<DeviceType>::FixGLSDNHKokkos(LAMMPS *lmp, ParsedArgs parsed) :
    FixNHKokkos<DeviceType>(lmp, parsed.fixnh_narg, parsed.fixnh_argv.data()),
    size_vector_nh(this->size_vector),
    lattice_flag(parsed.lattice_flag),
    midpoint_iter(parsed.midpoint_iter),
    midpoint_tol(parsed.midpoint_tol),
    lambda(parsed.gammas),
    alpha(parsed.alpha),
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
    k_fm_cache(),
    fm_cache(nullptr),
    k_s0_cache(),
    s0_cache(nullptr),
    k_s_guess_device(),
    s_guess_device(nullptr),
    k_s_map_device(),
    s_map_device(nullptr),
    nmax_old(0),
    grow_callback_added(0),
    restart_callback_added(0),
    restart_from_legacy(0),
    d_fm_cache(),
    d_s0_cache(),
    d_s_guess(),
    d_s_map(),
    midpoint_backend_last(MidpointBackend::HOST_FALLBACK),
    midpoint_fallback_last(MidpointFallbackReason::NON_DEVICE_EXECUTION),
    nmax_s0(0),
    s0(nullptr),
    nmax_s_guess(0),
    s_guess(nullptr),
    nmax_s_map(0),
    s_map(nullptr),
    replay_fixes(),
    replay_fix_indices(),
    mask(),
    tag(),
    sp(),
    fm(),
    nsend_tmp(0),
    nrecv1_tmp(0),
    nextrarecv1_tmp(0),
    d_buf(),
    d_exchange_sendlist(),
    d_copylist(),
    d_indices()
{
  if (this->lmp->kokkos == nullptr || this->lmp->atomKK == nullptr)
    this->error->all(
        FLERR,
        "Fix {} (Kokkos) requires Kokkos to be enabled at runtime (use '-k on ...' or 'package kokkos', and do not use '-sf kk' by itself)",
        this->style);

  if (!this->atom->sp_flag)
    this->error->all(FLERR, "Fix {} requires atom_style spin/kk (or spin with Kokkos enabled)", this->style);

  this->maxexchange = 6;
  this->restart_peratom = 1;
  this->atom->add_callback(Atom::GROW);
  this->atom->add_callback(Atom::RESTART);
  grow_callback_added = 1;
  restart_callback_added = 1;
  this->grow_arrays(this->atom->nmax);

  this->sort_device = 1;
  this->exchange_comm_device = 1;
}

template <class DeviceType>
FixGLSDNHKokkos<DeviceType>::~FixGLSDNHKokkos()
{
  if (this->copymode) return;

  debug_close();
  this->memory->destroy(s0);
  this->memory->destroy(s_guess);
  this->memory->destroy(s_map);

  if (grow_callback_added && this->modify && (this->modify->find_fix(this->id) >= 0))
    this->atom->delete_callback(this->id, Atom::GROW);
  if (restart_callback_added && this->modify && (this->modify->find_fix(this->id) >= 0))
    this->atom->delete_callback(this->id, Atom::RESTART);

  this->memoryKK->destroy_kokkos(k_fm_cache, fm_cache);
  this->memoryKK->destroy_kokkos(k_s0_cache, s0_cache);
  this->memoryKK->destroy_kokkos(k_s_guess_device, s_guess_device);
  this->memoryKK->destroy_kokkos(k_s_map_device, s_map_device);
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::setmask()
{
  int mask = FixNHKokkos<DeviceType>::setmask();
  mask |= POST_FORCE;
  if (!lattice_flag) mask &= ~PRE_EXCHANGE;
  return mask;
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::grow_arrays(int nmax)
{
  this->memoryKK->grow_kokkos(k_fm_cache, fm_cache, nmax, 3, "glsd/nh/kk:fm_cache");
  this->memoryKK->grow_kokkos(k_s0_cache, s0_cache, nmax, 3, "glsd/nh/kk:s0_cache");
  this->memoryKK->grow_kokkos(k_s_guess_device, s_guess_device, nmax, 3, "glsd/nh/kk:s_guess_device");
  this->memoryKK->grow_kokkos(k_s_map_device, s_map_device, nmax, 3, "glsd/nh/kk:s_map_device");

  k_fm_cache.sync_host();
  k_s0_cache.sync_host();

  for (int i = nmax_old; i < nmax; i++) {
    if (fm_cache) fm_cache[i][0] = fm_cache[i][1] = fm_cache[i][2] = 0.0;
    if (s0_cache) s0_cache[i][0] = s0_cache[i][1] = s0_cache[i][2] = 0.0;
  }
  nmax_old = nmax;

  k_fm_cache.modify_host();
  k_s0_cache.modify_host();

  k_fm_cache.sync<DeviceType>();
  k_s0_cache.sync<DeviceType>();
  k_s_guess_device.sync<DeviceType>();
  k_s_map_device.sync<DeviceType>();

  d_fm_cache = k_fm_cache.view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();
  d_s_guess = k_s_guess_device.view<DeviceType>();
  d_s_map = k_s_map_device.view<DeviceType>();
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::copy_arrays(int i, int j, int /*delflag*/)
{
  k_fm_cache.sync_host();
  k_s0_cache.sync_host();

  fm_cache[j][0] = fm_cache[i][0];
  fm_cache[j][1] = fm_cache[i][1];
  fm_cache[j][2] = fm_cache[i][2];

  s0_cache[j][0] = s0_cache[i][0];
  s0_cache[j][1] = s0_cache[i][1];
  s0_cache[j][2] = s0_cache[i][2];

  k_fm_cache.modify_host();
  k_s0_cache.modify_host();

  k_fm_cache.sync<DeviceType>();
  k_s0_cache.sync<DeviceType>();

  d_fm_cache = k_fm_cache.view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::pack_exchange(int i, double *buf)
{
  k_fm_cache.sync_host();
  k_s0_cache.sync_host();

  buf[0] = fm_cache[i][0];
  buf[1] = fm_cache[i][1];
  buf[2] = fm_cache[i][2];
  buf[3] = s0_cache[i][0];
  buf[4] = s0_cache[i][1];
  buf[5] = s0_cache[i][2];
  return 6;
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::unpack_exchange(int nlocal, double *buf)
{
  k_fm_cache.sync_host();
  k_s0_cache.sync_host();

  fm_cache[nlocal][0] = buf[0];
  fm_cache[nlocal][1] = buf[1];
  fm_cache[nlocal][2] = buf[2];
  s0_cache[nlocal][0] = buf[3];
  s0_cache[nlocal][1] = buf[4];
  s0_cache[nlocal][2] = buf[5];

  k_fm_cache.modify_host();
  k_s0_cache.modify_host();

  k_fm_cache.sync<DeviceType>();
  k_s0_cache.sync<DeviceType>();

  d_fm_cache = k_fm_cache.view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();

  return 6;
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::pack_restart(int i, double *buf)
{
  k_fm_cache.sync_host();
  k_s0_cache.sync_host();

  buf[0] = 7.0;
  buf[1] = fm_cache[i][0];
  buf[2] = fm_cache[i][1];
  buf[3] = fm_cache[i][2];
  buf[4] = s0_cache[i][0];
  buf[5] = s0_cache[i][1];
  buf[6] = s0_cache[i][2];
  return 7;
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::unpack_restart(int nlocal, int nth)
{
  double **extra = this->atom->extra;
  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int>(extra[nlocal][m]);

  k_fm_cache.sync_host();
  k_s0_cache.sync_host();

  const int nvals = static_cast<int>(extra[nlocal][m++]);
  if (nvals <= 1) {
    fm_cache[nlocal][0] = fm_cache[nlocal][1] = fm_cache[nlocal][2] = 0.0;
    s0_cache[nlocal][0] = s0_cache[nlocal][1] = s0_cache[nlocal][2] = 0.0;
  } else if (nvals < 4) {
    this->error->warning(
        FLERR,
        "Fix {} style {} encountered truncated per-atom restart payload; using safe fallback for atom {}",
        this->id, this->style, nlocal);
    fm_cache[nlocal][0] = fm_cache[nlocal][1] = fm_cache[nlocal][2] = 0.0;
    s0_cache[nlocal][0] = s0_cache[nlocal][1] = s0_cache[nlocal][2] = 0.0;
  } else {
    fm_cache[nlocal][0] = extra[nlocal][m++];
    fm_cache[nlocal][1] = extra[nlocal][m++];
    fm_cache[nlocal][2] = extra[nlocal][m++];
    if (nvals >= 7) {
      s0_cache[nlocal][0] = extra[nlocal][m++];
      s0_cache[nlocal][1] = extra[nlocal][m++];
      s0_cache[nlocal][2] = extra[nlocal][m++];
    } else {
      s0_cache[nlocal][0] = s0_cache[nlocal][1] = s0_cache[nlocal][2] = 0.0;
    }
  }

  k_fm_cache.modify_host();
  k_s0_cache.modify_host();
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::maxsize_restart()
{
  return 7;
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::size_restart(int /*nlocal*/)
{
  return 7;
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &k_buf,
                                                     DAT::tdual_int_1d k_sendlist, DAT::tdual_int_1d k_copylist,
                                                     ExecutionSpace /*space*/)
{
  if (nsend == 0) return 0;

  k_buf.sync<DeviceType>();
  k_sendlist.sync<DeviceType>();
  k_copylist.sync<DeviceType>();

  k_fm_cache.template sync<DeviceType>();
  k_s0_cache.template sync<DeviceType>();

  d_exchange_sendlist = k_sendlist.view<DeviceType>();
  d_copylist = k_copylist.view<DeviceType>();

  d_buf = typename AT::t_xfloat_1d_um(k_buf.view<DeviceType>().data(), k_buf.extent(0) * k_buf.extent(1));
  d_fm_cache = k_fm_cache.view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();

  nsend_tmp = nsend;
  const int stride = this->maxexchange;

  auto exchange_sendlist = d_exchange_sendlist;
  auto copylist = d_copylist;
  auto buf = d_buf;
  auto fm_cache_ = d_fm_cache;
  auto s0_cache_ = d_s0_cache;

  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nsend), LAMMPS_LAMBDA(const int mysend) {
    const int i = exchange_sendlist(mysend);
    const int j = copylist(mysend);

    int m = mysend * stride;
    buf(m++) = static_cast<X_FLOAT>(fm_cache_(i, 0));
    buf(m++) = static_cast<X_FLOAT>(fm_cache_(i, 1));
    buf(m++) = static_cast<X_FLOAT>(fm_cache_(i, 2));
    buf(m++) = static_cast<X_FLOAT>(s0_cache_(i, 0));
    buf(m++) = static_cast<X_FLOAT>(s0_cache_(i, 1));
    buf(m++) = static_cast<X_FLOAT>(s0_cache_(i, 2));

    if (j > -1) {
      fm_cache_(i, 0) = fm_cache_(j, 0);
      fm_cache_(i, 1) = fm_cache_(j, 1);
      fm_cache_(i, 2) = fm_cache_(j, 2);
      s0_cache_(i, 0) = s0_cache_(j, 0);
      s0_cache_(i, 1) = s0_cache_(j, 1);
      s0_cache_(i, 2) = s0_cache_(j, 2);
    }
  });
  this->copymode = 0;

  k_buf.modify<DeviceType>();
  k_fm_cache.template modify<DeviceType>();
  k_s0_cache.template modify<DeviceType>();

  return nsend * stride;
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d &indices,
                                                        int nrecv, int nrecv1, int nextrarecv1,
                                                        ExecutionSpace /*space*/)
{
  if (nrecv == 0) return;

  k_buf.sync<DeviceType>();
  indices.sync<DeviceType>();

  k_fm_cache.template sync<DeviceType>();
  k_s0_cache.template sync<DeviceType>();

  d_indices = indices.view<DeviceType>();
  d_buf = typename AT::t_xfloat_1d_um(k_buf.view<DeviceType>().data(), k_buf.extent(0) * k_buf.extent(1));
  d_fm_cache = k_fm_cache.view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();

  nrecv1_tmp = nrecv1;
  nextrarecv1_tmp = nextrarecv1;
  const int stride = this->maxexchange;

  const int nrecv1_tmp_ = nrecv1_tmp;
  const int nextrarecv1_tmp_ = nextrarecv1_tmp;
  auto indices_ = d_indices;
  auto buf = d_buf;
  auto fm_cache_ = d_fm_cache;
  auto s0_cache_ = d_s0_cache;

  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nrecv), LAMMPS_LAMBDA(const int ii) {
    const int index = indices_(ii);
    if (index < 0) return;

    int m;
    if (ii < nrecv1_tmp_)
      m = ii * stride;
    else
      m = nextrarecv1_tmp_ + (ii - nrecv1_tmp_) * stride;

    fm_cache_(index, 0) = static_cast<double>(buf(m++));
    fm_cache_(index, 1) = static_cast<double>(buf(m++));
    fm_cache_(index, 2) = static_cast<double>(buf(m++));
    s0_cache_(index, 0) = static_cast<double>(buf(m++));
    s0_cache_(index, 1) = static_cast<double>(buf(m++));
    s0_cache_(index, 2) = static_cast<double>(buf(m++));
  });
  this->copymode = 0;

  k_fm_cache.template modify<DeviceType>();
  k_s0_cache.template modify<DeviceType>();
}

/* ----------------------------------------------------------------------
   sort local atom-based arrays
------------------------------------------------------------------------- */

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  k_fm_cache.sync_device();
  k_s0_cache.sync_device();
  k_s_guess_device.sync_device();
  k_s_map_device.sync_device();

  Sorter.sort(LMPDeviceType(), k_fm_cache.d_view);
  Sorter.sort(LMPDeviceType(), k_s0_cache.d_view);
  Sorter.sort(LMPDeviceType(), k_s_guess_device.d_view);
  Sorter.sort(LMPDeviceType(), k_s_map_device.d_view);

  k_fm_cache.modify_device();
  k_s0_cache.modify_device();
  k_s_guess_device.modify_device();
  k_s_map_device.modify_device();

  d_fm_cache = k_fm_cache.view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();
  d_s_guess = k_s_guess_device.view<DeviceType>();
  d_s_map = k_s_map_device.view<DeviceType>();
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::mc_sync_from_atom(int local_index)
{
  if (local_index < 0) return;
  if (!this->atom->sp_flag) return;

  const int nlocal = this->atom->nlocal;
  if (local_index >= nlocal) return;

  // Ensure required device views are current.
  mask = this->atomKK->k_mask.template view<DeviceType>();
  sp = this->atomKK->k_sp.template view<DeviceType>();
  this->atomKK->sync(this->execution_space, MASK_MASK | SP_MASK);

  k_fm_cache.template sync<DeviceType>();
  k_s0_cache.template sync<DeviceType>();
  d_fm_cache = k_fm_cache.view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();

  const int groupbit = this->groupbit;
  const int idx = local_index;
  auto mask_ = mask;
  auto sp_ = sp;
  auto fm_cache_ = d_fm_cache;
  auto s0_cache_ = d_s0_cache;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, 1), LAMMPS_LAMBDA(const int) {
    const int i = idx;
    if (!(mask_(i) & groupbit)) return;

    const double Smag = static_cast<double>(sp_(i, 3));
    const double sx = static_cast<double>(sp_(i, 0));
    const double sy = static_cast<double>(sp_(i, 1));
    const double sz = static_cast<double>(sp_(i, 2));
    const double snorm = sqrt(sx * sx + sy * sy + sz * sz);

    if (Smag <= 0.0 || snorm <= SPIN_EPS) {
      s0_cache_(i, 0) = 0.0;
      s0_cache_(i, 1) = 0.0;
      s0_cache_(i, 2) = 0.0;
    } else {
      const double inv = 1.0 / snorm;
      s0_cache_(i, 0) = Smag * sx * inv;
      s0_cache_(i, 1) = Smag * sy * inv;
      s0_cache_(i, 2) = Smag * sz * inv;
    }

    // Conservative: invalidate cached field after a type/spin change.
    fm_cache_(i, 0) = 0.0;
    fm_cache_(i, 1) = 0.0;
    fm_cache_(i, 2) = 0.0;
  });

  k_s0_cache.template modify<DeviceType>();
  k_fm_cache.template modify<DeviceType>();
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::init()
{
  FixNHKokkos<DeviceType>::init();

  if (!this->atom->sp_flag) this->error->all(FLERR, "Fix {} requires atom_style spin", this->style);
  if (!this->atom->fm) this->error->all(FLERR, "Fix {} requires atom_style spin with fm allocated", this->style);

  const int my_index = this->modify->find_fix(this->id);

  for (int ifix = 0; ifix < this->modify->nfix; ifix++) {
    Fix *f = this->modify->fix[ifix];
    if (f == this) continue;
    if (!f->time_integrate) continue;
    if (f->igroup == this->igroup) {
      this->error->all(FLERR, "Fix {} cannot be used together with time integration fix {} on the same group",
                       this->style, f->id);
    }
  }

  if (utils::strmatch(this->update->integrate_style, "^respa")) {
    this->error->all(FLERR, "Fix {} is not supported with rRESPA", this->style);
  }

  if (lambda < 0.0) this->error->all(FLERR, "Fix {} glsd lambda must be >= 0", this->style);
  if (spin_temperature < 0.0 && spin_temperature != -1.0)
    this->error->all(FLERR, "Fix {} glsd stemp must be -1 (no noise), 0 (follow lattice temperature), or > 0",
                     this->style);
  if (seed <= 0) this->error->all(FLERR, "Fix {} glsd seed must be > 0", this->style);

  hbar = this->force->hplanck / MathConst::MY_2PI;
  g_over_hbar = 0.0;
  if (hbar != 0.0) {
    constexpr double g = 2.0;
    g_over_hbar = g / hbar;
  }

  if (debug_every < 1) this->error->all(FLERR, "Fix {} glsd debug_every must be >= 1", this->style);
  if (debug_rank < 0 || debug_rank >= this->comm->nprocs)
    this->error->all(FLERR, "Fix {} glsd debug_rank must be between 0 and nprocs-1", this->style);

  if (hbar == 0.0) this->error->all(FLERR, "Fix {} requires nonzero hbar (use physical units)", this->style);

  if (alpha >= 0.0) {
    const double lambda_from_alpha = alpha * g_over_hbar;
    if (lambda != 0.0) {
      const double scale = std::max(1.0, std::max(std::fabs(lambda), std::fabs(lambda_from_alpha)));
      const double tol = 1.0e-12 * scale;
      if (std::fabs(lambda - lambda_from_alpha) > tol)
        this->error->all(
            FLERR,
            "Fix {} has inconsistent glsd lambda ({}) and alpha ({}) after unit conversion "
            "(expected lambda={} from alpha*g/hbar)",
            this->style, lambda, alpha, lambda_from_alpha);
    }
    lambda = lambda_from_alpha;
  }

  for (int ifix = 0; ifix < this->modify->nfix; ifix++) {
    if (utils::strmatch(this->modify->fix[ifix]->style, "^langevin/spin"))
      this->error->all(FLERR, "Fix {} cannot be combined with fix langevin/spin", this->style);
  }

  if (!lattice_flag && this->pstat_flag) {
    this->error->warning(FLERR, "Fix {} with glsd lattice off disables pressure control (barostat is inactive)",
                         this->style);
  }

  if (midpoint_iter < 2) this->error->all(FLERR, "Fix {} glsd midpoint_iter must be >= 2", this->style);
  if (midpoint_tol < 0.0) this->error->all(FLERR, "Fix {} glsd midpoint_tol must be >= 0", this->style);

  if (my_index >= 0) {
    for (int ifix = my_index + 1; ifix < this->modify->nfix; ifix++) {
      Fix *f = this->modify->fix[ifix];
      if (f && f->time_integrate)
        this->error->all(FLERR, "Fix {} with glsd midpoint_iter must be the last time integration fix (found {} after it)",
                         this->style, f->id);
    }
  }

  debug_open();

  replay_fixes.clear();
  replay_fix_indices.clear();

  for (int ifix = 0; ifix < this->modify->nfix; ifix++) {
    const char *s = this->modify->fix[ifix]->style;
    if (utils::strmatch(s, "^precession/spin")) {
      this->error->all(
          FLERR,
          "Fix {} (USER-TSPIN) does not support fix precession/spin; use fix tspin/precession/spin or setforce/spin instead",
          this->style);
    }
    if (utils::strmatch(s, "^setforce/spin") || utils::strmatch(s, "^tspin/precession/spin")) {
      replay_fixes.push_back(this->modify->fix[ifix]);
      replay_fix_indices.push_back(ifix);
      if (my_index >= 0 && ifix > my_index) {
        this->error->all(FLERR, "Fix {} must be defined after {} to preserve external fields in glsd recompute",
                         this->style, this->modify->fix[ifix]->id);
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
      Fix *f = this->modify->fix[ifix];
      if (!f || f == this) continue;
      if (!(this->modify->fmask[ifix] & POST_FORCE)) continue;
      if (is_replayed(f)) continue;
      this->error->all(
          FLERR,
          "Fix {} performs internal force/field recomputes for glsd midpoint iterations and will overwrite "
          "contributions from earlier post_force fix {} (style {}). Define that fix after {}.",
          this->style, f->id, f->style, this->id);
    }
  }
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::setup(int vflag)
{
  FixNHKokkos<DeviceType>::setup(vflag);
  cache_current_fm_device();
  pe_prev_end = current_pe_total();
}

template <class DeviceType>
double FixGLSDNHKokkos<DeviceType>::compute_spin_temperature()
{
  if (spin_temperature > 0.0) return spin_temperature;
  if (spin_temperature < 0.0) return 0.0;

  if (!this->temperature)
    this->error->all(FLERR, "Fix {} glsd stemp=0 requires a valid lattice temperature compute", this->style);

  const bigint step = this->update->ntimestep;
  if (spin_temperature_cache_valid && spin_temperature_cached_step == step) return spin_temperature_cached;

  this->atomKK->sync(this->temperature->execution_space, this->temperature->datamask_read);
  spin_temperature_cached = this->temperature->compute_scalar();
  this->atomKK->modified(this->temperature->execution_space, this->temperature->datamask_modify);
  this->atomKK->sync(this->execution_space, this->temperature->datamask_modify);

  spin_temperature_cached_step = step;
  spin_temperature_cache_valid = 1;
  return spin_temperature_cached;
}

template <class DeviceType>
double FixGLSDNHKokkos<DeviceType>::fm_to_frequency(double fm_component) const
{
  return g_over_hbar * fm_component;
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::cache_current_fm_device()
{
  this->atomKK->sync(this->execution_space, FM_MASK);
  k_fm_cache.template sync<DeviceType>();

  fm = this->atomKK->k_fm.template view<DeviceType>();
  d_fm_cache = k_fm_cache.view<DeviceType>();

  const int nlocal = this->atom->nlocal;
  auto fm_ = fm;
  auto fm_cache_ = d_fm_cache;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    fm_cache_(i, 0) = static_cast<double>(fm_(i, 0));
    fm_cache_(i, 1) = static_cast<double>(fm_(i, 1));
    fm_cache_(i, 2) = static_cast<double>(fm_(i, 2));
  });
  k_fm_cache.template modify<DeviceType>();
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::save_s0_cache_device()
{
  this->atomKK->sync(this->execution_space, MASK_MASK | SP_MASK);
  k_s0_cache.template sync<DeviceType>();

  const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;

  mask = this->atomKK->k_mask.template view<DeviceType>();
  sp = this->atomKK->k_sp.template view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();

  const int groupbit = this->groupbit;
  auto mask_ = mask;
  auto sp_ = sp;
  auto s0_cache_ = d_s0_cache;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!(mask_(i) & groupbit)) return;

    const double Smag = static_cast<double>(sp_(i, 3));
    if (Smag == 0.0) {
      s0_cache_(i, 0) = 0.0;
      s0_cache_(i, 1) = 0.0;
      s0_cache_(i, 2) = 0.0;
      return;
    }

    const double sx_dir = static_cast<double>(sp_(i, 0));
    const double sy_dir = static_cast<double>(sp_(i, 1));
    const double sz_dir = static_cast<double>(sp_(i, 2));
    const double snorm = sqrt(sx_dir * sx_dir + sy_dir * sy_dir + sz_dir * sz_dir);
    if (snorm < SPIN_EPS) {
      s0_cache_(i, 0) = 0.0;
      s0_cache_(i, 1) = 0.0;
      s0_cache_(i, 2) = 0.0;
      return;
    }
    const double inv = 1.0 / snorm;
    s0_cache_(i, 0) = Smag * sx_dir * inv;
    s0_cache_(i, 1) = Smag * sy_dir * inv;
    s0_cache_(i, 2) = Smag * sz_dir * inv;
  });

  k_s0_cache.template modify<DeviceType>();
}

template <class DeviceType>
bool FixGLSDNHKokkos<DeviceType>::ensure_midpoint_device_scratch()
{
  const int nmax = this->atom->nmax;
  if (nmax <= 0) return false;

  if (static_cast<int>(k_s_guess_device.extent(0)) < nmax || static_cast<int>(k_s_map_device.extent(0)) < nmax) {
    this->memoryKK->grow_kokkos(k_s_guess_device, s_guess_device, nmax, 3, "glsd/nh/kk:s_guess_device");
    this->memoryKK->grow_kokkos(k_s_map_device, s_map_device, nmax, 3, "glsd/nh/kk:s_map_device");
  }

  k_s_guess_device.template sync<DeviceType>();
  k_s_map_device.template sync<DeviceType>();
  d_s_guess = k_s_guess_device.view<DeviceType>();
  d_s_map = k_s_map_device.view<DeviceType>();

  if (!d_s_guess.data()) return false;
  if (!d_s_map.data()) return false;
  return true;
}

template <class DeviceType>
const char *FixGLSDNHKokkos<DeviceType>::midpoint_backend_string(MidpointBackend backend) const
{
  return (backend == MidpointBackend::DEVICE) ? "DEVICE" : "HOST";
}

template <class DeviceType>
const char *FixGLSDNHKokkos<DeviceType>::fallback_reason_string(MidpointFallbackReason reason) const
{
  switch (reason) {
    case MidpointFallbackReason::NONE: return "none";
    case MidpointFallbackReason::HOST_REPLAY_FIX: return "host_replay_fix";
    case MidpointFallbackReason::NON_DEVICE_EXECUTION: return "non_device_execution";
    case MidpointFallbackReason::SCRATCH_UNAVAILABLE: return "scratch_unavailable";
  }
  return "unknown";
}

template <class DeviceType>
typename FixGLSDNHKokkos<DeviceType>::MidpointBackend FixGLSDNHKokkos<DeviceType>::select_midpoint_backend(
    bool /*lattice_mode*/)
{
  midpoint_fallback_last = MidpointFallbackReason::NONE;

  bool have_host_replay_fix = false;
  for (auto *f : replay_fixes) {
    const char *s = f->style;
    if (utils::strmatch(s, "^.*/kk") && !utils::strmatch(s, "^.*/kk/host")) continue;
    have_host_replay_fix = true;
    break;
  }
  if (have_host_replay_fix) {
    midpoint_fallback_last = MidpointFallbackReason::HOST_REPLAY_FIX;
    return MidpointBackend::HOST_FALLBACK;
  }

#if !defined(LMP_KOKKOS_GPU)
  midpoint_fallback_last = MidpointFallbackReason::NON_DEVICE_EXECUTION;
  return MidpointBackend::HOST_FALLBACK;
#else
  if (this->execution_space != Device) {
    midpoint_fallback_last = MidpointFallbackReason::NON_DEVICE_EXECUTION;
    return MidpointBackend::HOST_FALLBACK;
  }
#endif

  if (!ensure_midpoint_device_scratch()) {
    midpoint_fallback_last = MidpointFallbackReason::SCRATCH_UNAVAILABLE;
    return MidpointBackend::HOST_FALLBACK;
  }

  return MidpointBackend::DEVICE;
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::glsd_map_device_kernel(double dt, typename AT::t_double_2d s_in,
                                                         typename AT::t_double_2d s_out, bool use_fm_cache,
                                                         int noise_phase)
{
  const double temp_use = compute_spin_temperature();
  const double kbt = this->force->boltz * temp_use;

  const double gamma_eff = lambda;
  const double mu_s = (gamma_eff > 0.0 && temp_use > 0.0) ? (2.0 * gamma_eff * kbt) : 0.0;
  const double sigma_half = (mu_s > 0.0) ? sqrt(0.5 * mu_s * dt) : 0.0;

  const std::uint64_t seed64 = static_cast<std::uint64_t>(seed);
  const std::uint64_t step64 = static_cast<std::uint64_t>(this->update->ntimestep);
  const int tag_enable = this->atom->tag_enable;

  int need = MASK_MASK;
  if (tag_enable) need |= TAG_MASK;
  if (use_fm_cache)
    k_fm_cache.template sync<DeviceType>();
  else
    need |= FM_MASK;

  this->atomKK->sync(this->execution_space, need);

  mask = this->atomKK->k_mask.template view<DeviceType>();
  if (tag_enable) tag = this->atomKK->k_tag.template view<DeviceType>();
  if (!use_fm_cache) fm = this->atomKK->k_fm.template view<DeviceType>();
  if (use_fm_cache) d_fm_cache = k_fm_cache.view<DeviceType>();

  const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;
  const int groupbit = this->groupbit;
  const double g_over_hbar_ = g_over_hbar;

  auto mask_ = mask;
  auto tag_ = tag;
  auto fm_ = fm;
  auto fm_cache_ = d_fm_cache;
  auto s_in_ = s_in;
  auto s_out_ = s_out;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!(mask_(i) & groupbit)) return;

    double Sx = s_in_(i, 0);
    double Sy = s_in_(i, 1);
    double Sz = s_in_(i, 2);

    double fm_x, fm_y, fm_z;
    if (use_fm_cache) {
      fm_x = fm_cache_(i, 0);
      fm_y = fm_cache_(i, 1);
      fm_z = fm_cache_(i, 2);
    } else {
      fm_x = static_cast<double>(fm_(i, 0));
      fm_y = static_cast<double>(fm_(i, 1));
      fm_z = static_cast<double>(fm_(i, 2));
    }

    const double Hx_w = g_over_hbar_ * fm_x;
    const double Hy_w = g_over_hbar_ * fm_y;
    const double Hz_w = g_over_hbar_ * fm_z;

    const double Hx = fm_x;
    const double Hy = fm_y;
    const double Hz = fm_z;

    if (gamma_eff != 0.0) {
      Sx += 0.5 * dt * gamma_eff * Hx;
      Sy += 0.5 * dt * gamma_eff * Hy;
      Sz += 0.5 * dt * gamma_eff * Hz;
    }

    if (sigma_half > 0.0) {
      const tagint ti = tag_enable ? tag_(i) : static_cast<tagint>(i + 1);
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
      const tagint ti = tag_enable ? tag_(i) : static_cast<tagint>(i + 1);
      const int phase1 = noise_phase * 2 + 1;
      Sx += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 0);
      Sy += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 1);
      Sz += sigma_half * gaussian_u64(seed64, ti, step64, phase1, 2);
    }

    s_out_(i, 0) = Sx;
    s_out_(i, 1) = Sy;
    s_out_(i, 2) = Sz;
  });
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::initial_integrate(int vflag)
{
  const int do_pstat = this->pstat_flag && lattice_flag;

  if (do_pstat && this->mpchain) this->nhc_press_integrate();

  if (this->tstat_flag) this->compute_temp_target();
  if (this->tstat_flag && lattice_flag) this->nhc_temp_integrate();

  if (do_pstat) {
    this->atomKK->sync(this->temperature->execution_space, this->temperature->datamask_read);
    this->atomKK->sync(this->pressure->execution_space, this->pressure->datamask_read);
    if (this->pstyle == ISO) {
      this->temperature->compute_scalar();
      this->pressure->compute_scalar();
    } else {
      this->temperature->compute_vector();
      this->pressure->compute_vector();
    }
    this->atomKK->modified(this->temperature->execution_space, this->temperature->datamask_modify);
    this->atomKK->modified(this->pressure->execution_space, this->pressure->datamask_modify);
    this->atomKK->sync(this->execution_space, this->temperature->datamask_modify);
    this->atomKK->sync(this->execution_space, this->pressure->datamask_modify);
    this->couple();
    this->pressure->addstep(this->update->ntimestep + 1);
  }

  if (do_pstat) {
    this->compute_press_target();
    this->nh_omega_dot();
    this->nh_v_press();
  }

  if (midpoint_iter > 1 && lattice_flag) save_s0_cache_device();

  if (!lattice_flag && midpoint_iter > 1) {
    midpoint_backend_last = select_midpoint_backend(false);
    if (midpoint_backend_last == MidpointBackend::DEVICE) {
      if (!solve_spin_midpoint_device(false, vflag, 0.0)) {
        midpoint_backend_last = MidpointBackend::HOST_FALLBACK;
        if (midpoint_fallback_last == MidpointFallbackReason::NONE)
          midpoint_fallback_last = MidpointFallbackReason::SCRATCH_UNAVAILABLE;
        solve_spin_midpoint_host(false, vflag, 0.0);
      }
    } else {
      solve_spin_midpoint_host(false, vflag, 0.0);
    }
    return;
  }

  if (lattice_flag) this->nve_v();
  if (do_pstat) this->remap();
  if (lattice_flag) this->nve_x();

  if (do_pstat) {
    this->remap();
    if (this->kspace_flag) this->force->kspace->setup();
  }
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::post_force(int vflag)
{
  if (lattice_flag && midpoint_iter > 1) {
    const double pe_mid = (this->update->eflag_global == this->update->ntimestep) ? current_pe_total() : 0.0;
    midpoint_backend_last = select_midpoint_backend(true);
    if (midpoint_backend_last == MidpointBackend::DEVICE) {
      if (!solve_spin_midpoint_device(true, vflag, pe_mid)) {
        midpoint_backend_last = MidpointBackend::HOST_FALLBACK;
        if (midpoint_fallback_last == MidpointFallbackReason::NONE)
          midpoint_fallback_last = MidpointFallbackReason::SCRATCH_UNAVAILABLE;
        solve_spin_midpoint_host(true, vflag, pe_mid);
      }
    } else {
      solve_spin_midpoint_host(true, vflag, pe_mid);
    }
    return;
  }

  if (!lattice_flag && midpoint_iter > 1) {
    cache_current_fm_device();
    const double pe_end = (this->update->eflag_global == this->update->ntimestep) ? current_pe_total() : 0.0;
    debug_log_energy(pe_end, pe_end, midpoint_backend_last, midpoint_fallback_last);
    pe_prev_end = pe_end;
    return;
  }

  // midpoint_iter>=2 => handled by the two early-return branches above
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::final_integrate()
{
  const int do_pstat = this->pstat_flag && lattice_flag;

  if (lattice_flag) this->nve_v();

  if (this->which == BIAS && this->neighbor->ago == 0) {
    this->atomKK->sync(this->temperature->execution_space, this->temperature->datamask_read);
    this->t_current = this->temperature->compute_scalar();
    this->atomKK->modified(this->temperature->execution_space, this->temperature->datamask_modify);
    this->atomKK->sync(this->execution_space, this->temperature->datamask_modify);
  }

  if (do_pstat) this->nh_v_press();

  this->atomKK->sync(this->temperature->execution_space, this->temperature->datamask_read);
  this->t_current = this->temperature->compute_scalar();
  this->atomKK->modified(this->temperature->execution_space, this->temperature->datamask_modify);
  this->atomKK->sync(this->execution_space, this->temperature->datamask_modify);
  this->tdof = this->temperature->dof;

  if (do_pstat) {
    if (this->pstyle == ISO) {
      this->atomKK->sync(this->pressure->execution_space, this->pressure->datamask_read);
      this->pressure->compute_scalar();
      this->atomKK->modified(this->pressure->execution_space, this->pressure->datamask_modify);
      this->atomKK->sync(this->execution_space, this->pressure->datamask_modify);
    } else {
      this->atomKK->sync(this->temperature->execution_space, this->temperature->datamask_read);
      this->atomKK->sync(this->pressure->execution_space, this->pressure->datamask_read);
      this->temperature->compute_vector();
      this->pressure->compute_vector();
      this->atomKK->modified(this->temperature->execution_space, this->temperature->datamask_modify);
      this->atomKK->modified(this->pressure->execution_space, this->pressure->datamask_modify);
      this->atomKK->sync(this->execution_space, this->temperature->datamask_modify);
      this->atomKK->sync(this->execution_space, this->pressure->datamask_modify);
    }
    this->couple();
    this->pressure->addstep(this->update->ntimestep + 1);
  }

  if (do_pstat) this->nh_omega_dot();

  if (this->tstat_flag && lattice_flag) this->nhc_temp_integrate();
  if (do_pstat && this->mpchain) this->nhc_press_integrate();
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::modify_param(int narg, char **arg)
{
  if (narg < 1) return FixNH::modify_param(narg, arg);
  if (strcmp(arg[0], "glsd") != 0) return FixNH::modify_param(narg, arg);
  if (narg < 2) this->error->all(FLERR, "Illegal fix_modify glsd command");

  if (strcmp(arg[1], "lattice") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd lattice command");
    lattice_flag = parse_on_off(arg[2], this->lmp, "fix_modify glsd lattice");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_iter") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd midpoint_iter command");
    midpoint_iter = utils::inumeric(FLERR, arg[2], false, this->lmp);
    if (midpoint_iter < 2) this->error->all(FLERR, "Illegal fix_modify glsd midpoint_iter command (must be >= 2)");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_tol") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd midpoint_tol command");
    midpoint_tol = utils::numeric(FLERR, arg[2], false, this->lmp);
    if (midpoint_tol < 0.0) this->error->all(FLERR, "Illegal fix_modify glsd midpoint_tol command (must be >= 0)");
    return 3;
  }
  if (strcmp(arg[1], "midpoint_anderson") == 0 || strcmp(arg[1], "midpoint_anderson_reg") == 0) {
    this->error->all(FLERR, "Illegal fix_modify glsd {} command (option removed; use midpoint_iter and midpoint_tol)",
                     arg[1]);
  }
  if (strcmp(arg[1], "gammas") == 0 || strcmp(arg[1], "lambda") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd lambda command");
    lambda = utils::numeric(FLERR, arg[2], false, this->lmp);
    if (lambda < 0.0) this->error->all(FLERR, "Illegal fix_modify glsd lambda command (must be >= 0)");
    if (strcmp(arg[1], "gammas") == 0)
      this->error->warning(FLERR, "Fix {}: fix_modify glsd gammas is deprecated; use fix_modify glsd lambda",
                           this->style);
    alpha = -1.0;
    return 3;
  }
  if (strcmp(arg[1], "alpha") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd alpha command");
    alpha = utils::numeric(FLERR, arg[2], false, this->lmp);
    if (alpha < 0.0) this->error->all(FLERR, "Illegal fix_modify glsd alpha command (must be >= 0)");
    if (g_over_hbar == 0.0) this->error->all(FLERR, "Illegal fix_modify glsd alpha command (requires nonzero hbar)");
    lambda = alpha * g_over_hbar;
    return 3;
  }
  if (strcmp(arg[1], "stemp") == 0 || strcmp(arg[1], "temp") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd stemp command");
    if (strcmp(arg[1], "temp") == 0)
      this->error->warning(FLERR, "Fix {}: fix_modify glsd temp is deprecated; use stemp", this->style);
    spin_temperature = utils::numeric(FLERR, arg[2], false, this->lmp);
    if (spin_temperature < 0.0 && spin_temperature != -1.0)
      this->error->all(FLERR, "Illegal fix_modify glsd stemp command (must be -1, 0, or > 0)");
    spin_temperature_cache_valid = 0;
    spin_temperature_cached_step = -1;
    return 3;
  }
  if (strcmp(arg[1], "seed") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd seed command");
    seed = utils::inumeric(FLERR, arg[2], false, this->lmp);
    if (seed <= 0) this->error->all(FLERR, "Illegal fix_modify glsd seed command (seed must be > 0)");
    return 3;
  }
  if (strcmp(arg[1], "fm_units") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd fm_units command");
    if (strcmp(arg[2], "field") != 0)
      this->error->all(FLERR, "Illegal fix_modify glsd fm_units command (must be 'field': H = -dE/dM in eV/μB)");
    return 3;
  }
  if (strcmp(arg[1], "debug") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd debug command");
    debug_flag = utils::logical(FLERR, arg[2], false, this->lmp);
    if (!debug_flag) debug_close();
    else debug_open();
    return 3;
  }
  if (strcmp(arg[1], "debug_every") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd debug_every command");
    debug_every = utils::inumeric(FLERR, arg[2], false, this->lmp);
    if (debug_every < 1) this->error->all(FLERR, "Illegal fix_modify glsd debug_every command (must be >= 1)");
    return 3;
  }
  if (strcmp(arg[1], "debug_rank") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd debug_rank command");
    debug_rank = utils::inumeric(FLERR, arg[2], false, this->lmp);
    if (debug_rank < 0 || debug_rank >= this->comm->nprocs)
      this->error->all(FLERR, "Illegal fix_modify glsd debug_rank command (must be between 0 and nprocs-1)");
    debug_close();
    debug_open();
    return 3;
  }
  if (strcmp(arg[1], "debug_flush") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd debug_flush command");
    debug_flush = utils::logical(FLERR, arg[2], false, this->lmp);
    if (debug_fp) setvbuf(debug_fp, nullptr, debug_flush ? _IOLBF : _IOFBF, 0);
    return 3;
  }
  if (strcmp(arg[1], "debug_start") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd debug_start command");
    debug_start = utils::bnumeric(FLERR, arg[2], false, this->lmp);
    return 3;
  }
  if (strcmp(arg[1], "debug_file") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify glsd debug_file command");
    debug_file = arg[2];
    debug_close();
    debug_open();
    return 3;
  }

  return FixNH::modify_param(narg, arg);
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::nh_payload_size_from_list(const double *list, int max_n)
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

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::pack_restart_payload_v1(double *list) const
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
  if (list) list[n] = spin_temperature_cached;
  n++;
  if (list) list[n] = static_cast<double>(spin_temperature_cached_step);
  n++;
  if (list) list[n] = static_cast<double>(spin_temperature_cache_valid);
  n++;
  if (list) list[n] = static_cast<double>(static_cast<int>(midpoint_backend_last));
  n++;
  if (list) list[n] = static_cast<double>(static_cast<int>(midpoint_fallback_last));
  n++;
  return n;
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::unpack_restart_payload_v1(const double *list)
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
  spin_temperature_cached = list[n++];
  spin_temperature_cached_step = static_cast<bigint>(list[n++]);
  spin_temperature_cache_valid = static_cast<int>(list[n++]);
  midpoint_backend_last = static_cast<MidpointBackend>(static_cast<int>(list[n++]));
  midpoint_fallback_last = static_cast<MidpointFallbackReason>(static_cast<int>(list[n++]));
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::size_restart_global()
{
  return 2 + 1 + FixNH::size_restart_global() + 1 + pack_restart_payload_v1(nullptr);
}

template <class DeviceType>
int FixGLSDNHKokkos<DeviceType>::pack_restart_data(double *list)
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

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::restart(char *buf)
{
  auto *list = reinterpret_cast<double *>(buf);
  restart_from_legacy = 0;

  if (list[0] != RESTART_MAGIC) {
    FixNH::restart(buf);
    restart_from_legacy = 1;
    this->error->warning(
        FLERR,
        "Fix {} style {} read legacy restart payload without USER-TSPIN glsd state; compatibility fallback reconstruction will be used",
        this->id, this->style);
    return;
  }

  const int version = static_cast<int>(list[1]);
  if (version != RESTART_VERSION)
    this->error->all(FLERR, "Fix {} style {} restart payload version {} is not supported", this->id, this->style,
                     version);

  int n = 2;
  const int nh_n = static_cast<int>(list[n++]);
  if (nh_n <= 0)
    this->error->all(FLERR, "Fix {} style {} restart payload has invalid NH size {}", this->id, this->style, nh_n);

  const int nh_parsed = nh_payload_size_from_list(list + n, nh_n);
  if (nh_parsed != nh_n)
    this->error->all(FLERR, "Fix {} style {} restart payload NH size mismatch (stored {}, parsed {})", this->id,
                     this->style, nh_n, nh_parsed);

  FixNH::restart(reinterpret_cast<char *>(list + n));
  n += nh_n;

  const int glsd_n = static_cast<int>(list[n++]);
  const int expected = pack_restart_payload_v1(nullptr);
  if (glsd_n != expected)
    this->error->all(FLERR, "Fix {} style {} restart payload glsd size mismatch (stored {}, expected {})", this->id,
                     this->style, glsd_n, expected);

  unpack_restart_payload_v1(list + n);
  restart_from_legacy = 0;
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::clear_force_arrays_device()
{
  const int nclear = this->force->newton ? (this->atom->nlocal + this->atom->nghost) : this->atom->nlocal;
  const int has_fml = (this->atom->fm_long != nullptr);

  int need = F_MASK | FM_MASK;
  if (has_fml) need |= FML_MASK;
  this->atomKK->sync(this->execution_space, need);

  auto f_ = this->atomKK->k_f.template view<DeviceType>();
  auto fm_ = this->atomKK->k_fm.template view<DeviceType>();
  auto fml_ = has_fml ? this->atomKK->k_fm_long.template view<DeviceType>() : decltype(f_)();

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nclear), LAMMPS_LAMBDA(const int i) {
    f_(i, 0) = 0.0;
    f_(i, 1) = 0.0;
    f_(i, 2) = 0.0;
    fm_(i, 0) = 0.0;
    fm_(i, 1) = 0.0;
    fm_(i, 2) = 0.0;
    if (has_fml) {
      fml_(i, 0) = 0.0;
      fml_(i, 1) = 0.0;
      fml_(i, 2) = 0.0;
    }
  });

  this->atomKK->modified(this->execution_space, need);
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::replay_external_spin_fields(int vflag)
{
  if (replay_fixes.empty()) return;

  bool have_host_fixes = false;
  bool have_device_fixes = false;
  for (auto *f : replay_fixes) {
    const char *s = f->style;
    if (utils::strmatch(s, "^.*/kk/host")) {
      have_host_fixes = true;
    } else if (utils::strmatch(s, "^.*/kk")) {
      have_device_fixes = true;
    } else {
      have_host_fixes = true;
    }
  }

  // If we have any host-side replay fixes (e.g. setforce/spin), run them on Host and then
  // sync FM back to the active execution space. Device-side replay fixes are applied afterwards
  // to avoid being overwritten by the Host->Device sync.
  if (have_host_fixes) {
    this->atomKK->sync(Host, FM_MASK);
    for (auto *f : replay_fixes) {
      const char *s = f->style;
      if (utils::strmatch(s, "^.*/kk") && !utils::strmatch(s, "^.*/kk/host")) continue;
      f->post_force(vflag);
    }
    this->atomKK->modified(Host, FM_MASK);
    this->atomKK->sync(this->execution_space, FM_MASK);
  }

  if (have_device_fixes) {
    for (auto *f : replay_fixes) {
      const char *s = f->style;
      if (!utils::strmatch(s, "^.*/kk") || utils::strmatch(s, "^.*/kk/host")) continue;
      f->post_force(vflag);
    }
    this->atomKK->modified(this->execution_space, FM_MASK);
  }
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::recompute_force_and_field(int eflag, int vflag)
{
  this->comm->forward_comm();
  clear_force_arrays_device();

  if (this->force->pair) this->force->pair->compute(eflag, vflag);

  if (this->atom->molecular != Atom::ATOMIC) {
    if (this->force->bond) this->force->bond->compute(eflag, vflag);
    if (this->force->angle) this->force->angle->compute(eflag, vflag);
    if (this->force->dihedral) this->force->dihedral->compute(eflag, vflag);
    if (this->force->improper) this->force->improper->compute(eflag, vflag);
  }

  if (this->force->kspace) this->force->kspace->compute(eflag, vflag);

  if (this->force->newton) this->comm->reverse_comm();

  replay_external_spin_fields(vflag);
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::glsd_map_host(double dt, double **s_in, double **fm_use, int noise_phase,
                                               double **s_out)
{
  auto *mask_h = this->atom->mask;
  auto *tag_h = this->atom->tag;
  const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;

  const double temp_use = compute_spin_temperature();

  const double kbt = this->force->boltz * temp_use;
  const std::uint64_t seed64 = static_cast<std::uint64_t>(seed);
  const std::uint64_t step64 = static_cast<std::uint64_t>(this->update->ntimestep);

  for (int i = 0; i < nlocal; i++) {
    if (!(mask_h[i] & this->groupbit)) continue;

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
      const tagint ti = tag_h ? tag_h[i] : static_cast<tagint>(i + 1);
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
      const tagint ti = tag_h ? tag_h[i] : static_cast<tagint>(i + 1);
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

template <class DeviceType>
bool FixGLSDNHKokkos<DeviceType>::solve_spin_midpoint_device(bool lattice_mode, int vflag, double pe_mid)
{
  if (midpoint_iter <= 1) return false;
  if (lattice_mode != (lattice_flag != 0)) return false;
  if (!ensure_midpoint_device_scratch()) return false;

  this->atomKK->sync(this->execution_space, MASK_MASK | SP_MASK);
  k_s0_cache.template sync<DeviceType>();
  k_s_guess_device.template sync<DeviceType>();
  k_s_map_device.template sync<DeviceType>();

  const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;
  const int groupbit = this->groupbit;
  const double dt = this->update->dt;

  mask = this->atomKK->k_mask.template view<DeviceType>();
  sp = this->atomKK->k_sp.template view<DeviceType>();
  d_s0_cache = k_s0_cache.view<DeviceType>();
  d_s_guess = k_s_guess_device.view<DeviceType>();
  d_s_map = k_s_map_device.view<DeviceType>();

  auto mask_ = mask;
  auto sp_ = sp;
  auto s0_ = d_s0_cache;

  if (!lattice_mode) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
      if (!(mask_(i) & groupbit)) return;

      const double Smag = static_cast<double>(sp_(i, 3));
      if (Smag == 0.0) {
        s0_(i, 0) = 0.0;
        s0_(i, 1) = 0.0;
        s0_(i, 2) = 0.0;
        return;
      }

      const double sx_dir = static_cast<double>(sp_(i, 0));
      const double sy_dir = static_cast<double>(sp_(i, 1));
      const double sz_dir = static_cast<double>(sp_(i, 2));
      const double snorm = sqrt(sx_dir * sx_dir + sy_dir * sy_dir + sz_dir * sz_dir);
      if (snorm < SPIN_EPS) {
        s0_(i, 0) = 0.0;
        s0_(i, 1) = 0.0;
        s0_(i, 2) = 0.0;
        return;
      }
      const double inv = 1.0 / snorm;
      s0_(i, 0) = Smag * sx_dir * inv;
      s0_(i, 1) = Smag * sy_dir * inv;
      s0_(i, 2) = Smag * sz_dir * inv;
    });
    k_s0_cache.template modify<DeviceType>();
  }

  // Initial guess: explicit full-step mapping using best available field estimate.
  glsd_map_device_kernel(dt, d_s0_cache, d_s_guess, !lattice_mode, 0);
  k_s_guess_device.template modify<DeviceType>();

  for (int iter = 0; iter < midpoint_iter; iter++) {
    // Midpoint spins: normalize 0.5*(S0 + S_guess)
    auto s_guess_ = d_s_guess;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
      if (!(mask_(i) & groupbit)) return;

      const double midx = 0.5 * (s0_(i, 0) + s_guess_(i, 0));
      const double midy = 0.5 * (s0_(i, 1) + s_guess_(i, 1));
      const double midz = 0.5 * (s0_(i, 2) + s_guess_(i, 2));
      const double midn = sqrt(midx * midx + midy * midy + midz * midz);
      if (midn > SPIN_EPS) {
        const double inv = 1.0 / midn;
        sp_(i, 0) = static_cast<X_FLOAT>(midx * inv);
        sp_(i, 1) = static_cast<X_FLOAT>(midy * inv);
        sp_(i, 2) = static_cast<X_FLOAT>(midz * inv);
        sp_(i, 3) = static_cast<X_FLOAT>(midn);
      } else {
        sp_(i, 0) = 0.0;
        sp_(i, 1) = 0.0;
        sp_(i, 2) = 0.0;
        sp_(i, 3) = 0.0;
      }
    });

    this->atomKK->modified(this->execution_space, SP_MASK);
    recompute_force_and_field(1, 0);

    glsd_map_device_kernel(dt, d_s0_cache, d_s_map, false, 0);
    k_s_map_device.template modify<DeviceType>();

    auto s_map_ = d_s_map;
    double max_dev = 0.0;
    constexpr double alpha_mix = 1.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType>(0, nlocal),
        LAMMPS_LAMBDA(const int i, double &max_local) {
          if (!(mask_(i) & groupbit)) return;

          const double dx = s_map_(i, 0) - s_guess_(i, 0);
          const double dy = s_map_(i, 1) - s_guess_(i, 1);
          const double dz = s_map_(i, 2) - s_guess_(i, 2);

          const double adx = fabs(dx);
          const double ady = fabs(dy);
          const double adz = fabs(dz);
          if (adx > max_local) max_local = adx;
          if (ady > max_local) max_local = ady;
          if (adz > max_local) max_local = adz;

          s_guess_(i, 0) += alpha_mix * dx;
          s_guess_(i, 1) += alpha_mix * dy;
          s_guess_(i, 2) += alpha_mix * dz;
        },
        Kokkos::Max<double>(max_dev));
    k_s_guess_device.template modify<DeviceType>();

    double max_dev_all = 0.0;
    MPI_Allreduce(&max_dev, &max_dev_all, 1, MPI_DOUBLE, MPI_MAX, this->world);
    if (midpoint_tol > 0.0 && max_dev_all < midpoint_tol) break;
  }

  // Final spins: normalize S_guess.
  auto s_guess_ = d_s_guess;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!(mask_(i) & groupbit)) return;

    const double Gx = s_guess_(i, 0);
    const double Gy = s_guess_(i, 1);
    const double Gz = s_guess_(i, 2);
    const double Gmag = sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
    if (Gmag > SPIN_EPS) {
      const double inv = 1.0 / Gmag;
      sp_(i, 3) = static_cast<X_FLOAT>(Gmag);
      sp_(i, 0) = static_cast<X_FLOAT>(Gx * inv);
      sp_(i, 1) = static_cast<X_FLOAT>(Gy * inv);
      sp_(i, 2) = static_cast<X_FLOAT>(Gz * inv);
    } else {
      sp_(i, 3) = 0.0;
    }
  });

  this->atomKK->modified(this->execution_space, SP_MASK);

  if (lattice_mode) {
    recompute_force_and_field(1, vflag);
    cache_current_fm_device();

    const double pe_end = (this->update->eflag_global == this->update->ntimestep) ? current_pe_total() : 0.0;
    debug_log_energy(pe_mid, pe_end, midpoint_backend_last, midpoint_fallback_last);
    pe_prev_end = pe_end;
  } else {
    this->comm->forward_comm();
    clear_force_arrays_device();
  }

  return true;
}

template <class DeviceType>
bool FixGLSDNHKokkos<DeviceType>::solve_spin_midpoint_host(bool lattice_mode, int vflag, double pe_mid)
{
  if (midpoint_iter <= 1) return false;
  if (lattice_mode != (lattice_flag != 0)) return false;

  // Ensure host pointers are current for the solver.
  this->atomKK->sync(Host, MASK_MASK | SP_MASK | FM_MASK);
  k_fm_cache.sync_host();
  k_s0_cache.sync_host();

  int nlocal = this->atom->nlocal;
  if (this->igroup == this->atom->firstgroup) nlocal = this->atom->nfirst;

  double **sp_h = this->atom->sp;
  int *mask_h = this->atom->mask;
  const double dt = this->update->dt;
  const double eps = 1.0e-12;

  double **s0_src = nullptr;
  if (lattice_mode) {
    s0_src = s0_cache;
  } else {
    if (this->atom->nmax > nmax_s0) {
      nmax_s0 = this->atom->nmax;
      this->memory->grow(s0, nmax_s0, 3, "glsd/nh/kk:s0");
    }
    for (int i = 0; i < nlocal; i++) {
      if (!(mask_h[i] & this->groupbit)) continue;
      const double Smag = sp_h[i][3];
      if (Smag == 0.0) {
        s0[i][0] = s0[i][1] = s0[i][2] = 0.0;
        continue;
      }
      const double sx_dir = sp_h[i][0];
      const double sy_dir = sp_h[i][1];
      const double sz_dir = sp_h[i][2];
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

  if (this->atom->nmax > nmax_s_guess) {
    nmax_s_guess = this->atom->nmax;
    this->memory->grow(s_guess, nmax_s_guess, 3, "glsd/nh/kk:s_guess");
  }
  if (this->atom->nmax > nmax_s_map) {
    nmax_s_map = this->atom->nmax;
    this->memory->grow(s_map, nmax_s_map, 3, "glsd/nh/kk:s_map");
  }

  // Initial guess: explicit full-step mapping using best available field estimate.
  double **fm_pred = lattice_mode ? this->atom->fm : fm_cache;
  glsd_map_host(dt, s0_src, fm_pred, 0, s_guess);

  for (int iter = 0; iter < midpoint_iter; iter++) {
    // Midpoint spins: normalize 0.5*(S0 + S_guess)
    for (int i = 0; i < nlocal; i++) {
      if (!(mask_h[i] & this->groupbit)) continue;

      const double S0x = s0_src[i][0];
      const double S0y = s0_src[i][1];
      const double S0z = s0_src[i][2];

      const double midx = 0.5 * (S0x + s_guess[i][0]);
      const double midy = 0.5 * (S0y + s_guess[i][1]);
      const double midz = 0.5 * (S0z + s_guess[i][2]);
      const double midn = std::sqrt(midx * midx + midy * midy + midz * midz);
      if (midn > eps) {
        const double inv = 1.0 / midn;
        sp_h[i][0] = midx * inv;
        sp_h[i][1] = midy * inv;
        sp_h[i][2] = midz * inv;
        sp_h[i][3] = midn;
      } else {
        sp_h[i][0] = sp_h[i][1] = sp_h[i][2] = 0.0;
        sp_h[i][3] = 0.0;
      }
    }

    // Field evaluation at midpoint spins.
    this->atomKK->modified(Host, SP_MASK);
    this->atomKK->sync(this->execution_space, SP_MASK);
    recompute_force_and_field(1, 0);
    this->atomKK->sync(Host, FM_MASK);
    double **fm_mid = this->atom->fm;

    double max_dev = 0.0;
    constexpr double alpha_mix = 1.0;

    glsd_map_host(dt, s0_src, fm_mid, 0, s_map);
    for (int i = 0; i < nlocal; i++) {
      if (!(mask_h[i] & this->groupbit)) continue;

      const double Gx = s_guess[i][0];
      const double Gy = s_guess[i][1];
      const double Gz = s_guess[i][2];

      const double Sx = s_map[i][0];
      const double Sy = s_map[i][1];
      const double Sz = s_map[i][2];

      const double dx = Sx - Gx;
      const double dy = Sy - Gy;
      const double dz = Sz - Gz;

      max_dev = std::max(max_dev, std::abs(dx));
      max_dev = std::max(max_dev, std::abs(dy));
      max_dev = std::max(max_dev, std::abs(dz));

      s_guess[i][0] = Gx + alpha_mix * dx;
      s_guess[i][1] = Gy + alpha_mix * dy;
      s_guess[i][2] = Gz + alpha_mix * dz;
    }

    double max_dev_all = 0.0;
    MPI_Allreduce(&max_dev, &max_dev_all, 1, MPI_DOUBLE, MPI_MAX, this->world);
    if (midpoint_tol > 0.0 && max_dev_all < midpoint_tol) break;
  }

  // Final spins: normalize S_guess.
  for (int i = 0; i < nlocal; i++) {
    if (!(mask_h[i] & this->groupbit)) continue;

    const double Gx = s_guess[i][0];
    const double Gy = s_guess[i][1];
    const double Gz = s_guess[i][2];
    const double Gmag = std::sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
    if (Gmag > eps) {
      const double inv = 1.0 / Gmag;
      sp_h[i][3] = Gmag;
      sp_h[i][0] = Gx * inv;
      sp_h[i][1] = Gy * inv;
      sp_h[i][2] = Gz * inv;
    } else {
      sp_h[i][3] = 0.0;
    }
  }

  this->atomKK->modified(Host, SP_MASK);
  this->atomKK->sync(this->execution_space, SP_MASK);

  if (lattice_mode) {
    recompute_force_and_field(1, vflag);
    cache_current_fm_device();

    const double pe_end = (this->update->eflag_global == this->update->ntimestep) ? current_pe_total() : 0.0;
    debug_log_energy(pe_mid, pe_end, midpoint_backend_last, midpoint_fallback_last);
    pe_prev_end = pe_end;
  } else {
    this->comm->forward_comm();
    clear_force_arrays_device();
  }

  return true;
}
template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::debug_open()
{
  if (!debug_flag) return;
  if (debug_fp) return;
  if (this->comm->me != debug_rank) return;

  std::string fname = debug_file;
  if (fname.empty()) {
    fname = "glsd_nh_debug.";
    fname += this->id;
    fname += ".log";
  }

  debug_fp = fopen(fname.c_str(), "w");
  if (!debug_fp)
    this->error->one(FLERR, "Fix {} could not open debug_file {}: {}", this->style, fname, utils::getsyserror());

  debug_header_printed = 0;
  if (debug_flush) setvbuf(debug_fp, nullptr, _IOLBF, 0);
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::debug_close()
{
  if (!debug_fp) return;
  fclose(debug_fp);
  debug_fp = nullptr;
}

template <class DeviceType>
double FixGLSDNHKokkos<DeviceType>::current_pe_total() const
{
  double one = 0.0;
  if (this->force->pair) one += this->force->pair->eng_vdwl + this->force->pair->eng_coul;

  if (this->atom->molecular != Atom::ATOMIC) {
    if (this->force->bond) one += this->force->bond->energy;
    if (this->force->angle) one += this->force->angle->energy;
    if (this->force->dihedral) one += this->force->dihedral->energy;
    if (this->force->improper) one += this->force->improper->energy;
  }

  double scalar = 0.0;
  MPI_Allreduce(&one, &scalar, 1, MPI_DOUBLE, MPI_SUM, this->world);

  if (this->force->kspace) scalar += this->force->kspace->energy;

  if (this->force->pair && this->force->pair->tail_flag) {
    const double volume = this->domain->xprd * this->domain->yprd * this->domain->zprd;
    scalar += this->force->pair->etail / volume;
  }

  if (this->modify->n_energy_global) scalar += this->modify->energy_global();
  return scalar;
}

template <class DeviceType>
void FixGLSDNHKokkos<DeviceType>::debug_log_energy(double pe_mid, double pe_end, MidpointBackend backend,
                                                   MidpointFallbackReason reason)
{
  if (!debug_flag) return;
  if (this->update->ntimestep < debug_start) return;
  if ((debug_every > 1) && ((this->update->ntimestep % debug_every) != 0)) return;
  if (this->comm->me != debug_rank) return;
  debug_open();
  if (!debug_fp) return;

  if (!debug_header_printed) {
    fprintf(debug_fp,
            "# fix %s energy diagnostics\n"
            "# columns:\n"
            "# step time dt pe_prev_end pe_mid pe_end dE_step (pe_end-pe_prev_end) dE_mid_end (pe_end-pe_mid)"
            " midpoint_backend fallback_reason\n",
            this->style);
    debug_header_printed = 1;
  }

  const double dE_step = pe_end - pe_prev_end;
  const double dE_mid_end = pe_end - pe_mid;

  fprintf(debug_fp, "%lld %.16g %.16g %.16g %.16g %.16g %.16g %.16g %s %s\n",
          static_cast<long long>(this->update->ntimestep), this->update->atime, this->update->dt, pe_prev_end, pe_mid,
          pe_end, dE_step, dE_mid_end, midpoint_backend_string(backend), fallback_reason_string(reason));
  if (debug_flush) fflush(debug_fp);
}

namespace LAMMPS_NS {
template class FixGLSDNHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixGLSDNHKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

#endif
