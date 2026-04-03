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

#include "fix_tspin_nh_kokkos.h"

#ifdef LMP_KOKKOS

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "math_const.h"
#include "modify.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "random_park.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

// Keep integer values consistent with FixNH's internal enums in src/fix_nh.cpp and src/KOKKOS/fix_nh_kokkos.cpp.
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
  return (state >> 11) * (1.0 / 9007199254740992.0);
}

static inline double gaussian_tag(std::uint64_t seed, tagint tag, int component)
{
  std::uint64_t state = seed;
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

template <class DeviceType>
typename FixTSPINNHKokkos<DeviceType>::ParsedArgs FixTSPINNHKokkos<DeviceType>::parse_tspin_trailing_block(
    LAMMPS *lmp, int narg, char **arg)
{
  ParsedArgs out;
  out.fixnh_strings.reserve(narg);

  int tspin_pos = -1;
  for (int i = 3; i < narg; i++) {
    if (strcmp(arg[i], "tspin") == 0) {
      tspin_pos = i;
      break;
    }
  }

  const int pass_end = (tspin_pos >= 0) ? tspin_pos : narg;
  for (int i = 0; i < pass_end; i++) out.fixnh_strings.emplace_back(arg[i]);

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
      } else if (strcmp(arg[iarg], "fm_units") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin fm_units", lmp->error);
        if (strcmp(arg[iarg + 1], "field") == 0) out.fm_is_frequency = 0;
        else if (strcmp(arg[iarg + 1], "energy") == 0) {
          lmp->error->warning(FLERR, "fix tspin ... tspin fm_units 'energy' is deprecated; use 'field' (eV/μB)");
          out.fm_is_frequency = 0;
        } else if (strcmp(arg[iarg + 1], "frequency") == 0) out.fm_is_frequency = 1;
        else lmp->error->all(FLERR, "fix tspin ... tspin fm_units must be 'field' or 'frequency'");
        iarg += 2;
      } else if (strcmp(arg[iarg], "ghost") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin ghost", lmp->error);
        const int ghost = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        if (ghost != 0 && ghost != 1) lmp->error->all(FLERR, "fix tspin ... tspin ghost must be 0 or 1");
        if (ghost != 0) lmp->error->all(FLERR, "fix tspin ... tspin ghost 1 is not supported by tspin/nh/kk");
        iarg += 2;
      } else if (strcmp(arg[iarg], "dtf") == 0) {
        lmp->error->all(FLERR, "fix tspin ... tspin dtf is no longer supported (spin updates always use dtf)");
      } else if (strcmp(arg[iarg], "mu") == 0) {
        lmp->error->all(FLERR, "fix tspin ... tspin mu is no longer supported (use tspin mass)");
      } else if (strcmp(arg[iarg], "mass") == 0) {
        if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix tspin ... tspin mass", lmp->error);
        const int ntypes = lmp->atom->ntypes;
        int count = 0;
        while ((iarg + 1 + count) < narg && utils::is_double(arg[iarg + 1 + count])) count++;

        if (count == 1) {
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
      } else {
        lmp->error->all(FLERR, "Illegal fix tspin ... tspin option: {}", arg[iarg]);
      }
    }
  }

  out.fixnh_argv.reserve(out.fixnh_strings.size());
  for (auto &s : out.fixnh_strings) out.fixnh_argv.push_back(s.data());
  out.fixnh_narg = static_cast<int>(out.fixnh_argv.size());

  return out;
}

template <class DeviceType>
FixTSPINNHKokkos<DeviceType>::FixTSPINNHKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixTSPINNHKokkos(lmp, parse_tspin_trailing_block(lmp, narg, arg))
{
}

template <class DeviceType>
FixTSPINNHKokkos<DeviceType>::FixTSPINNHKokkos(LAMMPS *lmp, ParsedArgs parsed) :
    FixNHKokkos<DeviceType>(lmp, parsed.fixnh_narg, parsed.fixnh_argv.data()),
    size_vector_nh(this->size_vector),
    lattice_flag(parsed.lattice_flag),
    spin_flag(parsed.spin_flag),
    spin_dof(0),
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
    etas_mass(),
    k_vs(),
    k_sreal(),
    k_smass(),
    k_isspin(),
    vs(nullptr),
    sreal(nullptr),
    smass(nullptr),
    isspin(nullptr),
    nmax_old(0),
    grow_callback_added(0),
    restart_callback_added(0),
    restart_from_legacy(0),
    k_mass_factor(),
    mass_factor_buf(nullptr),
    d_vs(),
    d_sreal(),
    d_smass(),
    d_mass_factor(),
    d_isspin(),
    mask(),
    type(),
    sp(),
    fm(),
    mass_type(),
    nsend_tmp(0),
    nrecv1_tmp(0),
    nextrarecv1_tmp(0),
    d_buf(),
    d_exchange_sendlist(),
    d_copylist(),
    d_indices()
{
  // If a user accidentally activates /kk styles (e.g. via -sf kk) without enabling
  // Kokkos at runtime (-k on or package kokkos), AtomKokkos is not in use and
  // this fix would otherwise crash when accessing atomKK views.
  if (this->lmp->kokkos == nullptr || this->lmp->atomKK == nullptr)
    this->error->all(
        FLERR,
        "Fix {} (Kokkos) requires Kokkos to be enabled at runtime (use '-k on ...' or 'package kokkos', and do not use '-sf kk' by itself)",
        this->style);

  if (!this->atom->sp_flag) spin_flag = 0;

  // Append two vector components for spin outputs without changing FixNH indexing.
  this->size_vector += 2;

  // per-atom state (vs, sreal, smass) is stored in this fix so Kokkos exchange can migrate it.
  this->maxexchange = 8;
  this->restart_peratom = 1;
  this->atom->add_callback(Atom::GROW);
  this->atom->add_callback(Atom::RESTART);
  grow_callback_added = 1;
  restart_callback_added = 1;
  this->grow_arrays(this->atom->nmax);

  // Support Kokkos device-side sorting for this fix's per-atom arrays.
  this->sort_device = 1;

  // enable device-side exchange of this fix's per-atom Kokkos arrays
  this->exchange_comm_device = 1;
}

template <class DeviceType>
FixTSPINNHKokkos<DeviceType>::~FixTSPINNHKokkos()
{
  if (this->copymode) return;

  // When a fix constructor throws (e.g., illegal command), the fix may not be
  // registered in Modify yet, so Atom::delete_callback() would not find it and
  // would hard-error. Guard against that failure mode.
  if (grow_callback_added && this->modify && (this->modify->find_fix(this->id) >= 0))
    this->atom->delete_callback(this->id, Atom::GROW);
  if (restart_callback_added && this->modify && (this->modify->find_fix(this->id) >= 0))
    this->atom->delete_callback(this->id, Atom::RESTART);

  this->memoryKK->destroy_kokkos(k_vs, vs);
  this->memoryKK->destroy_kokkos(k_sreal, sreal);
  this->memoryKK->destroy_kokkos(k_smass, smass);
  this->memoryKK->destroy_kokkos(k_isspin, isspin);
  this->memoryKK->destroy_kokkos(k_mass_factor, mass_factor_buf);
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::grow_arrays(int nmax)
{
  // preserve existing data and update host pointers
  this->memoryKK->grow_kokkos(k_vs, vs, nmax, 3, "tspin/nh/kk:vs");
  this->memoryKK->grow_kokkos(k_sreal, sreal, nmax, 3, "tspin/nh/kk:sreal");
  this->memoryKK->grow_kokkos(k_smass, smass, nmax, "tspin/nh/kk:smass");
  this->memoryKK->grow_kokkos(k_isspin, isspin, nmax, "tspin/nh/kk:isspin");

  // ensure host views are current before changing new entries
  k_vs.sync_host();
  k_sreal.sync_host();
  k_smass.sync_host();
  k_isspin.sync_host();

  // initialize new entries
  for (int i = nmax_old; i < nmax; i++) {
    if (vs) vs[i][0] = vs[i][1] = vs[i][2] = 0.0;
    if (sreal) sreal[i][0] = sreal[i][1] = sreal[i][2] = 0.0;
    if (smass) smass[i] = 0.0;
    if (isspin) isspin[i] = 0;
  }
  nmax_old = nmax;

  k_vs.modify_host();
  k_sreal.modify_host();
  k_smass.modify_host();
  k_isspin.modify_host();

  // propagate host initialization to the active execution space
  k_vs.sync<DeviceType>();
  k_sreal.sync<DeviceType>();
  k_smass.sync<DeviceType>();
  k_isspin.sync<DeviceType>();

  // refresh device views
  d_vs = k_vs.view<DeviceType>();
  d_sreal = k_sreal.view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::copy_arrays(int i, int j, int /*delflag*/)
{
  k_vs.sync_host();
  k_sreal.sync_host();
  k_smass.sync_host();
  k_isspin.sync_host();

  for (int k = 0; k < 3; k++) {
    vs[j][k] = vs[i][k];
    sreal[j][k] = sreal[i][k];
  }
  smass[j] = smass[i];
  isspin[j] = isspin[i];

  k_vs.modify_host();
  k_sreal.modify_host();
  k_smass.modify_host();
  k_isspin.modify_host();

  k_vs.sync<DeviceType>();
  k_sreal.sync<DeviceType>();
  k_smass.sync<DeviceType>();
  k_isspin.sync<DeviceType>();

  d_vs = k_vs.view<DeviceType>();
  d_sreal = k_sreal.view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::pack_exchange(int i, double *buf)
{
  k_vs.sync_host();
  k_sreal.sync_host();
  k_smass.sync_host();
  k_isspin.sync_host();

  buf[0] = vs[i][0];
  buf[1] = vs[i][1];
  buf[2] = vs[i][2];
  buf[3] = sreal[i][0];
  buf[4] = sreal[i][1];
  buf[5] = sreal[i][2];
  buf[6] = smass[i];
  buf[7] = isspin[i];

  return 8;
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::unpack_exchange(int nlocal, double *buf)
{
  k_vs.sync_host();
  k_sreal.sync_host();
  k_smass.sync_host();
  k_isspin.sync_host();

  vs[nlocal][0] = buf[0];
  vs[nlocal][1] = buf[1];
  vs[nlocal][2] = buf[2];
  sreal[nlocal][0] = buf[3];
  sreal[nlocal][1] = buf[4];
  sreal[nlocal][2] = buf[5];
  smass[nlocal] = buf[6];
  isspin[nlocal] = static_cast<int>(buf[7]);

  k_vs.modify_host();
  k_sreal.modify_host();
  k_smass.modify_host();
  k_isspin.modify_host();

  k_vs.sync<DeviceType>();
  k_sreal.sync<DeviceType>();
  k_smass.sync<DeviceType>();
  k_isspin.sync<DeviceType>();

  d_vs = k_vs.view<DeviceType>();
  d_sreal = k_sreal.view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();

  return 8;
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::pack_restart(int i, double *buf)
{
  k_vs.sync_host();
  k_sreal.sync_host();
  k_smass.sync_host();
  k_isspin.sync_host();

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

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::unpack_restart(int nlocal, int nth)
{
  double **extra = this->atom->extra;
  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int>(extra[nlocal][m]);

  k_vs.sync_host();
  k_sreal.sync_host();
  k_smass.sync_host();
  k_isspin.sync_host();

  const int nvals = static_cast<int>(extra[nlocal][m++]);
  if (nvals <= 1) {
    vs[nlocal][0] = vs[nlocal][1] = vs[nlocal][2] = 0.0;
    sreal[nlocal][0] = sreal[nlocal][1] = sreal[nlocal][2] = 0.0;
    smass[nlocal] = 0.0;
    isspin[nlocal] = 0;
  } else if (nvals < 9) {
    this->error->warning(
        FLERR,
        "Fix {} style {} encountered truncated per-atom restart payload; using safe fallback for atom {}",
        this->id, this->style, nlocal);
    vs[nlocal][0] = vs[nlocal][1] = vs[nlocal][2] = 0.0;
    sreal[nlocal][0] = sreal[nlocal][1] = sreal[nlocal][2] = 0.0;
    smass[nlocal] = 0.0;
    isspin[nlocal] = 0;
  } else {
    vs[nlocal][0] = extra[nlocal][m++];
    vs[nlocal][1] = extra[nlocal][m++];
    vs[nlocal][2] = extra[nlocal][m++];
    sreal[nlocal][0] = extra[nlocal][m++];
    sreal[nlocal][1] = extra[nlocal][m++];
    sreal[nlocal][2] = extra[nlocal][m++];
    smass[nlocal] = extra[nlocal][m++];
    isspin[nlocal] = static_cast<int>(extra[nlocal][m++]);
  }

  k_vs.modify_host();
  k_sreal.modify_host();
  k_smass.modify_host();
  k_isspin.modify_host();
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::maxsize_restart()
{
  return 9;
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::size_restart(int /*nlocal*/)
{
  return 9;
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &k_buf,
                                                      DAT::tdual_int_1d k_sendlist, DAT::tdual_int_1d k_copylist,
                                                      ExecutionSpace /*space*/)
{
  if (nsend == 0) return 0;

  k_buf.sync<DeviceType>();
  k_sendlist.sync<DeviceType>();
  k_copylist.sync<DeviceType>();

  k_vs.template sync<DeviceType>();
  k_sreal.template sync<DeviceType>();
  k_smass.template sync<DeviceType>();
  k_isspin.template sync<DeviceType>();

  d_exchange_sendlist = k_sendlist.view<DeviceType>();
  d_copylist = k_copylist.view<DeviceType>();

  d_buf = typename AT::t_xfloat_1d_um(k_buf.view<DeviceType>().data(), k_buf.extent(0) * k_buf.extent(1));
  d_vs = k_vs.view<DeviceType>();
  d_sreal = k_sreal.view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();

  nsend_tmp = nsend;
  const int stride = this->maxexchange;

  auto exchange_sendlist = d_exchange_sendlist;
  auto copylist = d_copylist;
  auto buf = d_buf;
  auto vs_ = d_vs;
  auto sreal_ = d_sreal;
  auto smass_ = d_smass;
  auto isspin_ = d_isspin;

  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nsend), LAMMPS_LAMBDA(const int mysend) {
    const int i = exchange_sendlist(mysend);
    const int j = copylist(mysend);

    int m = mysend * stride;
    buf(m++) = static_cast<X_FLOAT>(vs_(i, 0));
    buf(m++) = static_cast<X_FLOAT>(vs_(i, 1));
    buf(m++) = static_cast<X_FLOAT>(vs_(i, 2));
    buf(m++) = static_cast<X_FLOAT>(sreal_(i, 0));
    buf(m++) = static_cast<X_FLOAT>(sreal_(i, 1));
    buf(m++) = static_cast<X_FLOAT>(sreal_(i, 2));
    buf(m++) = static_cast<X_FLOAT>(smass_(i));
    buf(m++) = static_cast<X_FLOAT>(isspin_(i));

    if (j > -1) {
      vs_(i, 0) = vs_(j, 0);
      vs_(i, 1) = vs_(j, 1);
      vs_(i, 2) = vs_(j, 2);
      sreal_(i, 0) = sreal_(j, 0);
      sreal_(i, 1) = sreal_(j, 1);
      sreal_(i, 2) = sreal_(j, 2);
      smass_(i) = smass_(j);
      isspin_(i) = isspin_(j);
    }
  });
  this->copymode = 0;

  k_buf.modify<DeviceType>();
  k_vs.template modify<DeviceType>();
  k_sreal.template modify<DeviceType>();
  k_smass.template modify<DeviceType>();
  k_isspin.template modify<DeviceType>();

  return nsend * stride;
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d &indices,
                                                         int nrecv, int nrecv1, int nextrarecv1,
                                                         ExecutionSpace /*space*/)
{
  if (nrecv == 0) return;

  k_buf.sync<DeviceType>();
  indices.sync<DeviceType>();

  k_vs.template sync<DeviceType>();
  k_sreal.template sync<DeviceType>();
  k_smass.template sync<DeviceType>();
  k_isspin.template sync<DeviceType>();

  d_indices = indices.view<DeviceType>();
  d_buf = typename AT::t_xfloat_1d_um(k_buf.view<DeviceType>().data(), k_buf.extent(0) * k_buf.extent(1));
  d_vs = k_vs.view<DeviceType>();
  d_sreal = k_sreal.view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();

  nrecv1_tmp = nrecv1;
  nextrarecv1_tmp = nextrarecv1;
  const int stride = this->maxexchange;

  const int nrecv1_tmp_ = nrecv1_tmp;
  const int nextrarecv1_tmp_ = nextrarecv1_tmp;
  auto indices_ = d_indices;
  auto buf = d_buf;
  auto vs_ = d_vs;
  auto sreal_ = d_sreal;
  auto smass_ = d_smass;
  auto isspin_ = d_isspin;

  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nrecv), LAMMPS_LAMBDA(const int ii) {
    const int index = indices_(ii);
    if (index < 0) return;

    int m;
    if (ii < nrecv1_tmp_)
      m = ii * stride;
    else
      m = nextrarecv1_tmp_ + (ii - nrecv1_tmp_) * stride;

    vs_(index, 0) = static_cast<double>(buf(m++));
    vs_(index, 1) = static_cast<double>(buf(m++));
    vs_(index, 2) = static_cast<double>(buf(m++));
    sreal_(index, 0) = static_cast<double>(buf(m++));
    sreal_(index, 1) = static_cast<double>(buf(m++));
    sreal_(index, 2) = static_cast<double>(buf(m++));
    smass_(index) = static_cast<double>(buf(m++));
    isspin_(index) = static_cast<int>(buf(m++));
  });
  this->copymode = 0;

  k_vs.template modify<DeviceType>();
  k_sreal.template modify<DeviceType>();
  k_smass.template modify<DeviceType>();
  k_isspin.template modify<DeviceType>();
}

/* ----------------------------------------------------------------------
   sort local atom-based arrays
------------------------------------------------------------------------- */

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  k_vs.sync_device();
  k_sreal.sync_device();
  k_smass.sync_device();
  k_isspin.sync_device();

  Sorter.sort(LMPDeviceType(), k_vs.d_view);
  Sorter.sort(LMPDeviceType(), k_sreal.d_view);
  Sorter.sort(LMPDeviceType(), k_smass.d_view);
  Sorter.sort(LMPDeviceType(), k_isspin.d_view);

  k_vs.modify_device();
  k_sreal.modify_device();
  k_smass.modify_device();
  k_isspin.modify_device();

  d_vs = k_vs.view<DeviceType>();
  d_sreal = k_sreal.view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::init()
{
  FixNHKokkos<DeviceType>::init();

  for (int ifix = 0; ifix < this->modify->nfix; ifix++) {
    if (utils::strmatch(this->modify->fix[ifix]->style, "^precession/spin")) {
      this->error->all(FLERR,
                       "Fix {} (USER-TSPIN) does not support fix precession/spin; use fix tspin/precession/spin or setforce/spin instead",
                       this->style);
    }
  }

  if (!lattice_flag && this->pstat_flag)
    this->error->all(FLERR, "Fix {} cannot disable lattice integration with pressure control enabled", this->style);

  if (!spin_flag) return;

  ensure_mass_factor_kokkos();

  const int mchain = this->mtchain;
  const bool chain_from_restart = (!restart_from_legacy) && (static_cast<int>(etas.size()) == mchain) &&
      (static_cast<int>(etas_dot.size()) == (mchain + 1)) && (static_cast<int>(etas_dotdot.size()) == mchain) &&
      (static_cast<int>(etas_mass.size()) == mchain);
  if (!chain_from_restart) {
    etas.assign(mchain, 0.0);
    etas_dot.assign(mchain + 1, 0.0);
    etas_dotdot.assign(mchain, 0.0);
    etas_mass.assign(mchain, 0.0);
  }
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::ensure_mass_factor_kokkos()
{
  const int ntypes = this->atom->ntypes;
  if (ntypes <= 0) return;

  if (mass_factor.empty()) mass_factor.assign(ntypes + 1, 1.0);
  else if (static_cast<int>(mass_factor.size()) == 2) {
    const double v = mass_factor[1];
    mass_factor.assign(ntypes + 1, v);
  } else if (static_cast<int>(mass_factor.size()) != ntypes + 1) {
    mass_factor.assign(ntypes + 1, 1.0);
  }

  this->memoryKK->grow_kokkos(k_mass_factor, mass_factor_buf, ntypes + 1, "tspin/nh/kk:mass_factor");
  for (int t = 0; t <= ntypes; t++) mass_factor_buf[t] = mass_factor[t];
  k_mass_factor.modify_host();
  k_mass_factor.sync<DeviceType>();
  d_mass_factor = k_mass_factor.view<DeviceType>();
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::setup(int vflag)
{
  FixNHKokkos<DeviceType>::setup(vflag);

  if (!spin_flag) return;
  if (!this->atom->sp_flag) return;

  // Defensive: some workflows (e.g., `run ... pre no`) may bypass init().
  ensure_mass_factor_kokkos();

  if (this->atom->ntypes <= 0) this->error->all(FLERR, "Fix {} requires atom types to be defined", this->style);
  if (this->atom->mass == nullptr)
    this->error->all(FLERR, "Fix {} requires per-type masses (use 'mass' command or Masses section in data file)",
                     this->style);

  // Ensure we have device views and required atom fields.
  mask = this->atomKK->k_mask.template view<DeviceType>();
  type = this->atomKK->k_type.template view<DeviceType>();
  sp = this->atomKK->k_sp.template view<DeviceType>();
  mass_type = this->atomKK->k_mass.template view<DeviceType>();

  this->atomKK->sync(this->execution_space, MASK_MASK | TYPE_MASK | SP_MASK);

  // Validate masses for atom types that appear in the fix group.
  // This avoids hard-to-debug crashes if masses were never set.
  if (this->atom->mass_setflag) {
    this->atomKK->sync(Host, MASK_MASK | TYPE_MASK);
    auto *h_mask = this->atom->mask;
    auto *h_type = this->atom->type;
    for (int i = 0; i < this->atom->nlocal; i++) {
      if (!(h_mask[i] & this->groupbit)) continue;
      const int itype = h_type[i];
      if (itype < 1 || itype > this->atom->ntypes)
        this->error->all(FLERR, "Fix {} encountered invalid atom type {}", this->style, itype);
      if (!this->atom->mass_setflag[itype])
        this->error->all(FLERR, "Fix {} requires mass for atom type {} (use 'mass' command or Masses section)",
                         this->style, itype);
    }
  }

  // Ensure per-type mass is available on the execution space.
  // (FixNHKokkos::init() does this, but `run ... pre no` bypasses init.)
  this->atomKK->k_mass.template modify<LMPHostType>();
  this->atomKK->k_mass.template sync<DeviceType>();
  mass_type = this->atomKK->k_mass.template view<DeviceType>();

  // (re)initialize smass on device: mu_i = atom_mass[type] * mass_factor[type]
  d_smass = k_smass.view<DeviceType>();
  d_mass_factor = k_mass_factor.view<DeviceType>();
  const int groupbit = this->groupbit;
  auto mask_ = mask;
  auto type_ = type;
  auto mass_type_ = mass_type;
  auto smass_ = d_smass;
  auto mass_factor_ = d_mass_factor;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, this->atom->nlocal), LAMMPS_LAMBDA(const int i) {
    if (!(mask_(i) & groupbit)) {
      smass_(i) = 0.0;
      return;
    }
    const int itype = type_(i);
    smass_(i) = static_cast<double>(mass_type_(itype)) * mass_factor_(itype);
  });
  k_smass.template modify<DeviceType>();

  // initialize sreal on device from sp
  d_sreal = k_sreal.view<DeviceType>();
  auto sp_ = sp;
  auto sreal_ = d_sreal;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, this->atom->nlocal), LAMMPS_LAMBDA(const int i) {
    const double smag = static_cast<double>(sp_(i, 3));
    sreal_(i, 0) = static_cast<double>(sp_(i, 0)) * smag;
    sreal_(i, 1) = static_cast<double>(sp_(i, 1)) * smag;
    sreal_(i, 2) = static_cast<double>(sp_(i, 2)) * smag;
  });
  k_sreal.template modify<DeviceType>();

  update_spin_dof_device();
  if (!spin_state_initialized || reinit_spin_vel) init_spin_state_and_velocities_host();
  spin_state_initialized = 1;
  compute_twoKs_global_device();
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::update_spin_dof_device()
{
  if (!spin_flag) {
    spin_dof = 0;
    return;
  }

  this->atomKK->sync(this->execution_space, MASK_MASK | SP_MASK);
  k_isspin.template sync<DeviceType>();

  const int nlocal = this->atom->nlocal;
  mask = this->atomKK->k_mask.template view<DeviceType>();
  sp = this->atomKK->k_sp.template view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();

  const int groupbit = this->groupbit;
  auto mask_ = mask;
  auto sp_ = sp;
  auto isspin_ = d_isspin;

  int local_dof = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<DeviceType>(0, nlocal),
      LAMMPS_LAMBDA(const int i, int &sum) {
        int flag = 0;
        if (mask_(i) & groupbit) {
          const double smag = static_cast<double>(sp_(i, 3));
          if (smag > SPIN_MAG_EPS) {
            flag = 1;
            sum += 3;
          }
        }
        isspin_(i) = flag;
      },
      local_dof);
  k_isspin.template modify<DeviceType>();

  MPI_Allreduce(&local_dof, &spin_dof, 1, MPI_INT, MPI_SUM, this->world);
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::mc_sync_from_atom(int local_index)
{
  if (!spin_flag) return;
  if (local_index < 0) return;
  if (!this->atom->sp_flag) return;

  ensure_mass_factor_kokkos();

  const int nlocal = this->atom->nlocal;
  if (local_index >= nlocal) return;

  // Ensure device views are up to date.
  mask = this->atomKK->k_mask.template view<DeviceType>();
  type = this->atomKK->k_type.template view<DeviceType>();
  sp = this->atomKK->k_sp.template view<DeviceType>();
  mass_type = this->atomKK->k_mass.template view<DeviceType>();

  this->atomKK->sync(this->execution_space, MASK_MASK | TYPE_MASK | SP_MASK);
  this->atomKK->k_mass.template modify<LMPHostType>();
  this->atomKK->k_mass.template sync<DeviceType>();
  mass_type = this->atomKK->k_mass.template view<DeviceType>();

  k_vs.template sync<DeviceType>();
  k_sreal.template sync<DeviceType>();
  k_smass.template sync<DeviceType>();
  k_isspin.template sync<DeviceType>();
  k_mass_factor.template sync<DeviceType>();

  d_vs = k_vs.view<DeviceType>();
  d_sreal = k_sreal.view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();
  d_mass_factor = k_mass_factor.view<DeviceType>();

  const int groupbit = this->groupbit;
  const int idx = local_index;
  auto mask_ = mask;
  auto type_ = type;
  auto sp_ = sp;
  auto mass_type_ = mass_type;
  auto vs_ = d_vs;
  auto sreal_ = d_sreal;
  auto smass_ = d_smass;
  auto isspin_ = d_isspin;
  auto mass_factor_ = d_mass_factor;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, 1), LAMMPS_LAMBDA(const int) {
    const int i = idx;
    if (!(mask_(i) & groupbit)) return;

    const double smag = static_cast<double>(sp_(i, 3));
    sreal_(i, 0) = static_cast<double>(sp_(i, 0)) * smag;
    sreal_(i, 1) = static_cast<double>(sp_(i, 1)) * smag;
    sreal_(i, 2) = static_cast<double>(sp_(i, 2)) * smag;

    // Conservative: zero out auxiliary spin velocities after MC acceptance to avoid stale state.
    vs_(i, 0) = 0.0;
    vs_(i, 1) = 0.0;
    vs_(i, 2) = 0.0;

    const int itype = type_(i);
    smass_(i) = static_cast<double>(mass_type_(itype)) * mass_factor_(itype);
    isspin_(i) = (smag > SPIN_MAG_EPS) ? 1 : 0;
  });

  k_sreal.template modify<DeviceType>();
  k_vs.template modify<DeviceType>();
  k_smass.template modify<DeviceType>();
  k_isspin.template modify<DeviceType>();
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::init_spin_state_and_velocities_host()
{
  if (seed <= 0) this->error->all(FLERR, "Fix {} requires tspin seed > 0", this->style);

  // Fixed policy: seed stream is always keyed by atom ID (tag), so it is invariant
  // under MPI decomposition changes.
  if (this->atom->tag_enable == 0)
    this->error->all(FLERR, "Fix {} requires atoms to have IDs (tag_enable) for tspin initialization", this->style);

  // Initialize vs on host using legacy RanPark and rescale per-type to match t_start.
  // This only affects initial conditions; subsequent dynamics/thermostatting runs on the device.

  const int nlocal = this->atom->nlocal;
  this->atomKK->sync(Host, MASK_MASK | TYPE_MASK | SP_MASK | TAG_MASK);

  k_smass.sync_host();
  k_sreal.sync_host();
  k_vs.sync_host();
  k_isspin.sync_host();

  auto *h_mask = this->atom->mask;
  auto *h_type = this->atom->type;
  auto *h_tag = this->atom->tag;
  auto *h_isspin = isspin;

  constexpr double vscale0 = 0.5;

  const std::uint64_t base_seed = static_cast<std::uint64_t>(seed);
  for (int i = 0; i < nlocal; i++) {
    if (!(h_mask[i] & this->groupbit)) continue;
    if (!h_isspin[i]) {
      vs[i][0] = vs[i][1] = vs[i][2] = 0.0;
      continue;
    }
    const tagint tid = h_tag[i];
    vs[i][0] = vscale0 * gaussian_tag(base_seed, tid, 0);
    vs[i][1] = vscale0 * gaussian_tag(base_seed, tid, 1);
    vs[i][2] = vscale0 * gaussian_tag(base_seed, tid, 2);
  }

  const int ntypes = this->atom->ntypes;
  for (int itype = 1; itype <= ntypes; itype++) {
    double twoK_local_units = 0.0;
    int dof_local = 0;

    for (int i = 0; i < nlocal; i++) {
      if (!(h_mask[i] & this->groupbit)) continue;
      if (h_type[i] != itype) continue;
      if (!h_isspin[i]) continue;

      const double mu_i = smass[i];
      if (mu_i <= 0.0) this->error->all(FLERR, "Fix {} requires positive mu_i for spin atoms", this->style);

      const double v2 = vs[i][0] * vs[i][0] + vs[i][1] * vs[i][1] + vs[i][2] * vs[i][2];
      twoK_local_units += mu_i * v2;
      dof_local += 3;
    }

    double twoK_global_units = 0.0;
    int dof_global = 0;
    MPI_Allreduce(&twoK_local_units, &twoK_global_units, 1, MPI_DOUBLE, MPI_SUM, this->world);
    MPI_Allreduce(&dof_local, &dof_global, 1, MPI_INT, MPI_SUM, this->world);

    if (dof_global == 0) continue;
    const double twoK_energy = twoK_global_units * this->force->mvv2e;
    const double t_type = twoK_energy / (static_cast<double>(dof_global) * this->force->boltz);
    if (t_type <= 0.0) continue;

    const double rescale = std::sqrt(this->t_start / t_type);
    for (int i = 0; i < nlocal; i++) {
      if (!(h_mask[i] & this->groupbit)) continue;
      if (h_type[i] != itype) continue;
      if (!h_isspin[i]) continue;
      vs[i][0] *= rescale;
      vs[i][1] *= rescale;
      vs[i][2] *= rescale;
    }
  }

  k_vs.modify_host();
  k_vs.sync<DeviceType>();
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::compute_twoKs_global_device()
{
  if (!spin_flag) {
    twoKs_global = Ks_global = 0.0;
    return;
  }

  k_vs.template sync<DeviceType>();
  k_smass.template sync<DeviceType>();
  k_isspin.template sync<DeviceType>();

  const int nlocal = this->atom->nlocal;
  d_vs = k_vs.view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();

  auto vs_ = d_vs;
  auto smass_ = d_smass;
  auto isspin_ = d_isspin;

  double twoKs_local_energy = 0.0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<DeviceType>(0, nlocal),
      LAMMPS_LAMBDA(const int i, double &sum) {
        if (!isspin_(i)) return;
        const double v2 = vs_(i, 0) * vs_(i, 0) + vs_(i, 1) * vs_(i, 1) + vs_(i, 2) * vs_(i, 2);
        sum += smass_(i) * v2;
      },
      twoKs_local_energy);
  twoKs_local_energy *= this->force->mvv2e;

  MPI_Allreduce(&twoKs_local_energy, &twoKs_global, 1, MPI_DOUBLE, MPI_SUM, this->world);
  Ks_global = 0.5 * twoKs_global;
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::nhc_spin_integrate()
{
  if (!spin_flag) return;
  if (spin_dof <= 0) return;

  compute_twoKs_global_device();
  const double twoK_target_spin = static_cast<double>(spin_dof) * this->force->boltz * this->t_target;

  // mimic FixNH eta_mass update pattern
  etas_mass[0] = twoK_target_spin / (this->t_freq * this->t_freq);
  for (int ich = 1; ich < this->mtchain; ich++)
    etas_mass[ich] = this->force->boltz * this->t_target / (this->t_freq * this->t_freq);

  etas_dotdot[0] = (etas_mass[0] > 0.0) ? ((twoKs_global - twoK_target_spin) / etas_mass[0]) : 0.0;

  const double ncfac = 1.0 / this->nc_tchain;
  for (int iloop = 0; iloop < this->nc_tchain; iloop++) {
    for (int ich = this->mtchain - 1; ich > 0; ich--) {
      const double expfac = std::exp(-ncfac * this->dt8 * etas_dot[ich + 1]);
      etas_dot[ich] *= expfac;
      etas_dot[ich] += etas_dotdot[ich] * ncfac * this->dt4;
      etas_dot[ich] *= this->tdrag_factor;
      etas_dot[ich] *= expfac;
    }

    double expfac = std::exp(-ncfac * this->dt8 * etas_dot[1]);
    etas_dot[0] *= expfac;
    etas_dot[0] += etas_dotdot[0] * ncfac * this->dt4;
    etas_dot[0] *= this->tdrag_factor;
    etas_dot[0] *= expfac;

    const double factor = std::exp(-ncfac * this->dthalf * etas_dot[0]);

    k_vs.template sync<DeviceType>();
    k_isspin.template sync<DeviceType>();

    d_vs = k_vs.view<DeviceType>();
    d_isspin = k_isspin.view<DeviceType>();

    const double factor_ = factor;
    auto vs_ = d_vs;
    auto isspin_ = d_isspin;
    const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
      if (!isspin_(i)) return;
      vs_(i, 0) *= factor_;
      vs_(i, 1) *= factor_;
      vs_(i, 2) *= factor_;
    });

    k_vs.template modify<DeviceType>();

    twoKs_global *= factor * factor;
    Ks_global = 0.5 * twoKs_global;

    etas_dotdot[0] = (etas_mass[0] > 0.0) ? ((twoKs_global - twoK_target_spin) / etas_mass[0]) : 0.0;
    for (int ich = 0; ich < this->mtchain; ich++) etas[ich] += ncfac * this->dthalf * etas_dot[ich];

    etas_dot[0] *= expfac;
    etas_dot[0] += etas_dotdot[0] * ncfac * this->dt4;
    etas_dot[0] *= expfac;

    for (int ich = 1; ich < this->mtchain; ich++) {
      expfac = std::exp(-ncfac * this->dt8 * etas_dot[ich + 1]);
      etas_dot[ich] *= expfac;
      etas_dotdot[ich] = (etas_mass[ich] > 0.0) ?
          ((etas_mass[ich - 1] * etas_dot[ich - 1] * etas_dot[ich - 1] - this->force->boltz * this->t_target) / etas_mass[ich]) :
          0.0;
      etas_dot[ich] += etas_dotdot[ich] * ncfac * this->dt4;
      etas_dot[ich] *= expfac;
    }
  }
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::nve_v_spin_device()
{
  this->atomKK->sync(this->execution_space, FM_MASK);
  k_smass.template sync<DeviceType>();
  k_vs.template sync<DeviceType>();
  k_isspin.template sync<DeviceType>();

  const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;

  fm = this->atomKK->k_fm.template view<DeviceType>();
  d_smass = k_smass.view<DeviceType>();
  d_vs = k_vs.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();

  auto fm_ = fm;
  auto smass_ = d_smass;
  auto vs_ = d_vs;
  auto isspin_ = d_isspin;

  constexpr double g = 2.0;
  double dt_spin = this->dtf;
  double fm_scale = 1.0;
  if (fm_is_frequency) {
    const double hbar = this->force->hplanck / MathConst::MY_2PI;
    if (hbar == 0.0)
      this->error->all(FLERR, "Fix {} tspin fm_units frequency requires nonzero hbar (use physical units)", this->style);
    dt_spin = this->dthalf;
    fm_scale = hbar / g;
  }
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!isspin_(i)) return;
    const double mu_i = smass_(i);
    if (mu_i <= 0.0) return;
    const double dtmu = dt_spin / mu_i;
    vs_(i, 0) += dtmu * fm_scale * static_cast<double>(fm_(i, 0));
    vs_(i, 1) += dtmu * fm_scale * static_cast<double>(fm_(i, 1));
    vs_(i, 2) += dtmu * fm_scale * static_cast<double>(fm_(i, 2));
  });

  k_vs.template modify<DeviceType>();
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::nve_s_spin_device()
{
  this->atomKK->sync(this->execution_space, SP_MASK);
  k_sreal.template sync<DeviceType>();
  k_vs.template sync<DeviceType>();
  k_isspin.template sync<DeviceType>();

  const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;

  sp = this->atomKK->k_sp.template view<DeviceType>();
  d_sreal = k_sreal.view<DeviceType>();
  d_vs = k_vs.view<DeviceType>();
  d_isspin = k_isspin.view<DeviceType>();

  auto sp_ = sp;
  auto sreal_ = d_sreal;
  auto vs_ = d_vs;
  auto isspin_ = d_isspin;

  const double dtv = this->dtv;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!isspin_(i)) return;

    sreal_(i, 0) += dtv * vs_(i, 0);
    sreal_(i, 1) += dtv * vs_(i, 1);
    sreal_(i, 2) += dtv * vs_(i, 2);

    const double sx = sreal_(i, 0);
    const double sy = sreal_(i, 1);
    const double sz = sreal_(i, 2);
    const double smag = sqrt(sx * sx + sy * sy + sz * sz);

    sp_(i, 3) = static_cast<X_FLOAT>(smag);
    if (smag > SPIN_MAG_EPS) {
      sp_(i, 0) = static_cast<X_FLOAT>(sx / smag);
      sp_(i, 1) = static_cast<X_FLOAT>(sy / smag);
      sp_(i, 2) = static_cast<X_FLOAT>(sz / smag);
    }
  });

  k_sreal.template modify<DeviceType>();
  this->atomKK->modified(this->execution_space, SP_MASK);
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::initial_integrate(int /*vflag*/)
{
  // Mirror FixNHKokkos<DeviceType>::initial_integrate() and insert tspin updates.

  if (this->pstat_flag && this->mpchain) this->nhc_press_integrate();

  if (this->tstat_flag) this->compute_temp_target();

  if (spin_flag) nhc_spin_integrate();
  if (this->tstat_flag && lattice_flag) this->nhc_temp_integrate();

  if (this->pstat_flag) {
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

  if (this->pstat_flag) {
    this->compute_press_target();
    this->nh_omega_dot();
    this->nh_v_press();
  }

  if (spin_flag) nve_v_spin_device();
  if (lattice_flag) this->nve_v();

  if (this->pstat_flag) this->remap();

  if (spin_flag) nve_s_spin_device();
  if (lattice_flag) this->nve_x();

  if (this->pstat_flag) {
    this->remap();
    if (this->kspace_flag) this->force->kspace->setup();
  }
}

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::final_integrate()
{
  // Mirror FixNHKokkos<DeviceType>::final_integrate() and insert tspin updates.

  if (lattice_flag) this->nve_v();
  if (spin_flag) nve_v_spin_device();

  if (this->which == BIAS && this->neighbor->ago == 0) {
    this->atomKK->sync(this->temperature->execution_space, this->temperature->datamask_read);
    this->t_current = this->temperature->compute_scalar();
    this->atomKK->modified(this->temperature->execution_space, this->temperature->datamask_modify);
    this->atomKK->sync(this->execution_space, this->temperature->datamask_modify);
  }

  if (this->pstat_flag) this->nh_v_press();

  this->atomKK->sync(this->temperature->execution_space, this->temperature->datamask_read);
  this->t_current = this->temperature->compute_scalar();
  this->atomKK->modified(this->temperature->execution_space, this->temperature->datamask_modify);
  this->atomKK->sync(this->execution_space, this->temperature->datamask_modify);
  this->tdof = this->temperature->dof;

  if (this->pstat_flag) {
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

  if (this->pstat_flag) this->nh_omega_dot();

  if (this->tstat_flag && lattice_flag) this->nhc_temp_integrate();
  if (spin_flag) nhc_spin_integrate();
  if (this->pstat_flag && this->mpchain) this->nhc_press_integrate();
}

template <class DeviceType>
double FixTSPINNHKokkos<DeviceType>::compute_scalar()
{
  double energy = FixNH::compute_scalar();
  if (!spin_flag) return energy;

  compute_twoKs_global_device();

  energy += Ks_global;

  const double kt = this->boltz * this->t_target;
  const double twoK_target_spin = static_cast<double>(spin_dof) * this->boltz * this->t_target;

  if (this->mtchain > 0 && !etas.empty()) {
    energy += twoK_target_spin * etas[0] + 0.5 * etas_mass[0] * etas_dot[0] * etas_dot[0];
    for (int ich = 1; ich < this->mtchain; ich++)
      energy += kt * etas[ich] + 0.5 * etas_mass[ich] * etas_dot[ich] * etas_dot[ich];
  }
  return energy;
}

template <class DeviceType>
double FixTSPINNHKokkos<DeviceType>::compute_vector(int n)
{
  if (n < size_vector_nh) return FixNH::compute_vector(n);

  if (!spin_flag) return 0.0;
  compute_twoKs_global_device();

  if (n == size_vector_nh) return Ks_global;
  if (n == size_vector_nh + 1) return twoKs_global;
  return 0.0;
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::modify_param(int narg, char **arg)
{
  if (narg < 1) return FixNH::modify_param(narg, arg);
  if (strcmp(arg[0], "tspin") != 0) return FixNH::modify_param(narg, arg);
  if (narg < 2) this->error->all(FLERR, "Illegal fix_modify tspin command");

  if ((strcmp(arg[1], "on") == 0) || (strcmp(arg[1], "yes") == 0)) {
    spin_flag = 1;
    return 2;
  }
  if ((strcmp(arg[1], "off") == 0) || (strcmp(arg[1], "no") == 0)) {
    spin_flag = 0;
    return 2;
  }

  if (strcmp(arg[1], "lattice") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify tspin lattice command");
    if ((strcmp(arg[2], "yes") == 0) || (strcmp(arg[2], "on") == 0) || (strcmp(arg[2], "1") == 0)) {
      lattice_flag = 1;
    } else if ((strcmp(arg[2], "no") == 0) || (strcmp(arg[2], "off") == 0) || (strcmp(arg[2], "0") == 0)) {
      lattice_flag = 0;
    } else {
      this->error->all(FLERR, "Illegal fix_modify tspin lattice command (must be 'on' or 'off')");
    }
    if (!lattice_flag && this->pstat_flag)
      this->error->all(FLERR, "Fix {} cannot disable lattice integration with pressure control enabled", this->style);
    return 3;
  }

  if (strcmp(arg[1], "reinit") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify tspin reinit command");
    if ((strcmp(arg[2], "yes") == 0) || (strcmp(arg[2], "on") == 0) || (strcmp(arg[2], "1") == 0)) {
      reinit_spin_vel = 1;
    } else if ((strcmp(arg[2], "no") == 0) || (strcmp(arg[2], "off") == 0) || (strcmp(arg[2], "0") == 0)) {
      reinit_spin_vel = 0;
    } else {
      this->error->all(FLERR, "Illegal fix_modify tspin reinit command (must be 'on' or 'off')");
    }
    return 3;
  }

  if (strcmp(arg[1], "seed") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify tspin seed command");
    seed = utils::inumeric(FLERR, arg[2], false, this->lmp);
    if (seed <= 0) this->error->all(FLERR, "Illegal fix_modify tspin seed command (seed must be > 0)");
    return 3;
  }

  if (strcmp(arg[1], "mu") == 0) {
    this->error->all(FLERR, "Illegal fix_modify tspin mu command (use 'mass')");
  }

  if (strcmp(arg[1], "mass") == 0) {
    const int ntypes = this->atom->ntypes;
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify tspin mass command");

    int count = 0;
    while (count < ntypes && (2 + count) < narg && utils::is_double(arg[2 + count])) count++;
    if (count != 1 && count != ntypes) this->error->all(FLERR, "Illegal fix_modify tspin mass values");

    mass_factor.assign(ntypes + 1, 1.0);
    if (count == 1) {
      const double v = utils::numeric(FLERR, arg[2], false, this->lmp);
      for (int t = 1; t <= ntypes; t++) mass_factor[t] = v;
    } else {
      for (int t = 1; t <= ntypes; t++) mass_factor[t] = utils::numeric(FLERR, arg[1 + t], false, this->lmp);
    }

    this->memoryKK->grow_kokkos(k_mass_factor, mass_factor_buf, ntypes + 1, "tspin/nh/kk:mass_factor");
    for (int t = 0; t <= ntypes; t++) mass_factor_buf[t] = mass_factor[t];
    k_mass_factor.modify_host();
    k_mass_factor.sync<DeviceType>();
    d_mass_factor = k_mass_factor.view<DeviceType>();
    return 2 + count;
  }

  if (strcmp(arg[1], "fm_units") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify tspin fm_units command");
    if (strcmp(arg[2], "frequency") == 0) {
      fm_is_frequency = 1;
    } else if (strcmp(arg[2], "field") == 0) {
      fm_is_frequency = 0;
    } else if (strcmp(arg[2], "energy") == 0) {
      this->error->warning(FLERR, "Fix {} tspin fm_units 'energy' is deprecated; use 'field' (eV/μB)", this->style);
      fm_is_frequency = 0;
    } else {
      this->error->all(FLERR, "Illegal fix_modify tspin fm_units command (must be 'field' or 'frequency')");
    }
    return 3;
  }

  if (strcmp(arg[1], "dtf") == 0) {
    this->error->all(FLERR, "Illegal fix_modify tspin dtf command (spin updates always use dtf)");
  }

  if (strcmp(arg[1], "ghost") == 0) {
    if (narg < 3) this->error->all(FLERR, "Illegal fix_modify tspin ghost command");
    const int ghost = utils::inumeric(FLERR, arg[2], false, this->lmp);
    if (ghost != 0 && ghost != 1) this->error->all(FLERR, "fix_modify tspin ghost must be 0 or 1");
    if (ghost != 0) this->error->all(FLERR, "fix_modify tspin ghost 1 is not supported by tspin/nh/kk");
    return 3;
  }

  this->error->all(FLERR, "Illegal fix_modify tspin command");
  return 0;
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::nh_payload_size_from_list(const double *list, int max_n)
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
int FixTSPINNHKokkos<DeviceType>::pack_restart_payload_v1(double *list) const
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

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::unpack_restart_payload_v1(const double *list)
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
  if (mchain < 0)
    this->error->all(FLERR, "Fix {} style {} restart payload has invalid tspin chain length {}", this->id, this->style,
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

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::size_restart_global()
{
  return 2 + 1 + FixNH::size_restart_global() + 1 + pack_restart_payload_v1(nullptr);
}

template <class DeviceType>
int FixTSPINNHKokkos<DeviceType>::pack_restart_data(double *list)
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

template <class DeviceType>
void FixTSPINNHKokkos<DeviceType>::restart(char *buf)
{
  auto *list = reinterpret_cast<double *>(buf);
  restart_from_legacy = 0;

  if (list[0] != RESTART_MAGIC) {
    FixNH::restart(buf);
    restart_from_legacy = 1;
    spin_state_initialized = 0;
    this->error->warning(
        FLERR,
        "Fix {} style {} read legacy restart payload without USER-TSPIN tspin state; compatibility fallback initialization will be used",
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

  const int tspin_n = static_cast<int>(list[n++]);
  const double *payload = list + n;
  if (tspin_n < 10)
    this->error->all(FLERR, "Fix {} style {} restart payload has invalid tspin size {}", this->id, this->style,
                     tspin_n);
  const int mchain = static_cast<int>(payload[9]);
  if (mchain < 0)
    this->error->all(FLERR, "Fix {} style {} restart payload has invalid tspin chain length {}", this->id, this->style,
                     mchain);
  const int parsed_payload = 10 + mchain + (mchain + 1) + mchain + mchain;
  if (tspin_n != parsed_payload)
    this->error->all(FLERR, "Fix {} style {} restart payload tspin size mismatch (stored {}, parsed {})", this->id,
                     this->style, tspin_n, parsed_payload);

  unpack_restart_payload_v1(payload);
  restart_from_legacy = 0;
}

namespace LAMMPS_NS {
template class FixTSPINNHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixTSPINNHKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

#endif    // LMP_KOKKOS
