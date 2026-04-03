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

#include "fix_llgmid_nh_kokkos.h"

#ifdef LMP_KOKKOS

#include "angle.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "dihedral.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "memory_kokkos.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;

enum { NOBIAS, BIAS };
enum { ISO, ANISO, TRICLINIC };

namespace {
constexpr double KK_LLGMID_SPIN_EPS = 1.0e-12;

KOKKOS_INLINE_FUNCTION double kk_llgmid_norm3(double x, double y, double z)
{
  return sqrt(x * x + y * y + z * z);
}

KOKKOS_INLINE_FUNCTION void kk_llgmid_normalize3(double &x, double &y, double &z)
{
  const double n = kk_llgmid_norm3(x, y, z);
  if (n <= KK_LLGMID_SPIN_EPS) {
    x = y = z = 0.0;
    return;
  }
  const double inv = 1.0 / n;
  x *= inv;
  y *= inv;
  z *= inv;
}

KOKKOS_INLINE_FUNCTION double kk_llgmid_atanh(double x)
{
  return 0.5 * log((1.0 + x) / (1.0 - x));
}

KOKKOS_INLINE_FUNCTION void kk_llgmid_exact_flow_direction(double &ex, double &ey, double &ez, double hx, double hy,
                                                           double hz, double dt, double alpha, double g_over_hbar)
{
  const double e0n = kk_llgmid_norm3(ex, ey, ez);
  if (e0n <= KK_LLGMID_SPIN_EPS) {
    ex = ey = ez = 0.0;
    return;
  }

  const double hmag = kk_llgmid_norm3(hx, hy, hz);
  if (hmag <= KK_LLGMID_SPIN_EPS) {
    const double inv = 1.0 / e0n;
    ex *= inv;
    ey *= inv;
    ez *= inv;
    return;
  }

  const double inv_e = 1.0 / e0n;
  ex *= inv_e;
  ey *= inv_e;
  ez *= inv_e;
  hx /= hmag;
  hy /= hmag;
  hz /= hmag;

  const double u0 = fmax(-1.0, fmin(1.0, ex * hx + ey * hy + ez * hz));
  double px = ex - u0 * hx;
  double py = ey - u0 * hy;
  double pz = ez - u0 * hz;
  const double pnorm = kk_llgmid_norm3(px, py, pz);
  if (pnorm <= KK_LLGMID_SPIN_EPS) return;

  const double inv_p = 1.0 / pnorm;
  px *= inv_p;
  py *= inv_p;
  pz *= inv_p;

  const double cx = hy * pz - hz * py;
  const double cy = hz * px - hx * pz;
  const double cz = hx * py - hy * px;

  const double denom = 1.0 + alpha * alpha;
  const double a = g_over_hbar / denom;
  const double b = alpha * g_over_hbar / denom;
  const double uclip = fmax(-1.0 + 1.0e-15, fmin(1.0 - 1.0e-15, u0));
  const double arg0 = kk_llgmid_atanh(uclip);
  const double u1 = tanh(arg0 + b * hmag * dt);
  const double dphi = -a * hmag * dt;
  const double sint = sqrt(fmax(0.0, 1.0 - u1 * u1));
  const double cphi = cos(dphi);
  const double sphi = sin(dphi);

  ex = u1 * hx + sint * (cphi * px + sphi * cx);
  ey = u1 * hy + sint * (cphi * py + sphi * cy);
  ez = u1 * hz + sint * (cphi * pz + sphi * cz);
  kk_llgmid_normalize3(ex, ey, ez);
}
}    // namespace

template <class DeviceType>
FixLLGMidNHKokkos<DeviceType>::FixLLGMidNHKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixLLGMidNH(lmp, narg, arg),
    atomKK(nullptr),
    midpoint_backend_last(MidpointBackend::HOST_FALLBACK),
    midpoint_fallback_last(MidpointFallbackReason::NON_DEVICE_EXECUTION),
    x0_device_host(nullptr),
    v0_device_host(nullptr),
    f0_device_host(nullptr),
    s0_device_host(nullptr),
    x_mid_device_host(nullptr),
    e_mid_device_host(nullptr),
    s_guess_device_host(nullptr),
    x_end_device_host(nullptr),
    fm_cache_device_host(nullptr),
    d_buf(),
    d_exchange_sendlist(),
    d_copylist(),
    d_indices(),
    nsend_tmp(0),
    nrecv1_tmp(0),
    nextrarecv1_tmp(0)
{
  if (lmp->kokkos == nullptr || lmp->atomKK == nullptr)
    error->all(
        FLERR,
        "Fix {} (Kokkos) requires Kokkos to be enabled at runtime (use '-k on ...' or 'package kokkos', and do not use '-sf kk' by itself)",
        style);

  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK = dynamic_cast<AtomKokkos *>(atom);
  if (!atomKK) error->all(FLERR, "Fix {} requires atom_style spin/kk (or spin with Kokkos enabled)", style);
  sort_device = 1;
  exchange_comm_device = 1;
  grow_arrays(atom->nmax);
}

template <class DeviceType>
FixLLGMidNHKokkos<DeviceType>::~FixLLGMidNHKokkos()
{
  if (copymode) return;
  memoryKK->destroy_kokkos(k_x0_device, x0_device_host);
  memoryKK->destroy_kokkos(k_v0_device, v0_device_host);
  memoryKK->destroy_kokkos(k_f0_device, f0_device_host);
  memoryKK->destroy_kokkos(k_s0_device, s0_device_host);
  memoryKK->destroy_kokkos(k_x_mid_device, x_mid_device_host);
  memoryKK->destroy_kokkos(k_e_mid_device, e_mid_device_host);
  memoryKK->destroy_kokkos(k_s_guess_device, s_guess_device_host);
  memoryKK->destroy_kokkos(k_x_end_device, x_end_device_host);
  memoryKK->destroy_kokkos(k_fm_cache_device, fm_cache_device_host);
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::sync_host_all()
{
  atomKK->sync(Host, X_MASK | V_MASK | F_MASK | MASK_MASK | TYPE_MASK | TAG_MASK | SP_MASK | FM_MASK | FML_MASK | RMASS_MASK);
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::mark_host_all_modified()
{
  atomKK->modified(Host, X_MASK | V_MASK | F_MASK | SP_MASK | FM_MASK | FML_MASK);
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::grow_arrays(int nmax)
{
  const int old_nmax = nmax_old;
  FixLLGMidNH::grow_arrays(nmax);

  memoryKK->grow_kokkos(k_fm_cache_device, fm_cache_device_host, nmax, 3, "llgmid/nh/kk:fm_cache");
  memoryKK->grow_kokkos(k_s0_device, s0_device_host, nmax, 3, "llgmid/nh/kk:s0_device");
  memoryKK->grow_kokkos(k_x0_device, x0_device_host, nmax, 3, "llgmid/nh/kk:x0_device");
  memoryKK->grow_kokkos(k_v0_device, v0_device_host, nmax, 3, "llgmid/nh/kk:v0_device");
  memoryKK->grow_kokkos(k_f0_device, f0_device_host, nmax, 3, "llgmid/nh/kk:f0_device");

  k_fm_cache_device.sync_host();
  k_s0_device.sync_host();
  k_x0_device.sync_host();
  k_v0_device.sync_host();
  k_f0_device.sync_host();

  for (int i = old_nmax; i < nmax; i++) {
    if (fm_cache_device_host) fm_cache_device_host[i][0] = fm_cache_device_host[i][1] = fm_cache_device_host[i][2] = 0.0;
    if (s0_device_host) s0_device_host[i][0] = s0_device_host[i][1] = s0_device_host[i][2] = 0.0;
    if (x0_device_host) x0_device_host[i][0] = x0_device_host[i][1] = x0_device_host[i][2] = 0.0;
    if (v0_device_host) v0_device_host[i][0] = v0_device_host[i][1] = v0_device_host[i][2] = 0.0;
    if (f0_device_host) f0_device_host[i][0] = f0_device_host[i][1] = f0_device_host[i][2] = 0.0;
  }

  k_fm_cache_device.modify_host();
  k_s0_device.modify_host();
  k_x0_device.modify_host();
  k_v0_device.modify_host();
  k_f0_device.modify_host();

  k_fm_cache_device.template sync<DeviceType>();
  k_s0_device.template sync<DeviceType>();
  k_x0_device.template sync<DeviceType>();
  k_v0_device.template sync<DeviceType>();
  k_f0_device.template sync<DeviceType>();

  d_fm_cache_device = k_fm_cache_device.view<DeviceType>();
  d_s0_device = k_s0_device.view<DeviceType>();
  d_x0_device = k_x0_device.view<DeviceType>();
  d_v0_device = k_v0_device.view<DeviceType>();
  d_f0_device = k_f0_device.view<DeviceType>();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::copy_arrays(int i, int j, int /*delflag*/)
{
  FixLLGMidNH::copy_arrays(i, j, 0);

  k_fm_cache_device.sync_host();
  k_s0_device.sync_host();
  k_x0_device.sync_host();
  k_v0_device.sync_host();
  k_f0_device.sync_host();

  for (int k = 0; k < 3; k++) {
    fm_cache_device_host[j][k] = fm_cache_device_host[i][k];
    s0_device_host[j][k] = s0_device_host[i][k];
    x0_device_host[j][k] = x0_device_host[i][k];
    v0_device_host[j][k] = v0_device_host[i][k];
    f0_device_host[j][k] = f0_device_host[i][k];
  }

  k_fm_cache_device.modify_host();
  k_s0_device.modify_host();
  k_x0_device.modify_host();
  k_v0_device.modify_host();
  k_f0_device.modify_host();

  k_fm_cache_device.template sync<DeviceType>();
  k_s0_device.template sync<DeviceType>();
  k_x0_device.template sync<DeviceType>();
  k_v0_device.template sync<DeviceType>();
  k_f0_device.template sync<DeviceType>();
}

template <class DeviceType>
int FixLLGMidNHKokkos<DeviceType>::pack_exchange(int i, double *buf)
{
  return FixLLGMidNH::pack_exchange(i, buf);
}

template <class DeviceType>
int FixLLGMidNHKokkos<DeviceType>::unpack_exchange(int nlocal, double *buf)
{
  FixLLGMidNH::unpack_exchange(nlocal, buf);

  k_fm_cache_device.sync_host();
  k_s0_device.sync_host();
  k_x0_device.sync_host();
  k_v0_device.sync_host();
  k_f0_device.sync_host();

  for (int k = 0; k < 3; k++) {
    fm_cache_device_host[nlocal][k] = fm_cache[nlocal][k];
    s0_device_host[nlocal][k] = s0_cache[nlocal][k];
    x0_device_host[nlocal][k] = x0_cache[nlocal][k];
    v0_device_host[nlocal][k] = v0_cache[nlocal][k];
    f0_device_host[nlocal][k] = f0_cache[nlocal][k];
  }

  k_fm_cache_device.modify_host();
  k_s0_device.modify_host();
  k_x0_device.modify_host();
  k_v0_device.modify_host();
  k_f0_device.modify_host();

  k_fm_cache_device.template sync<DeviceType>();
  k_s0_device.template sync<DeviceType>();
  k_x0_device.template sync<DeviceType>();
  k_v0_device.template sync<DeviceType>();
  k_f0_device.template sync<DeviceType>();
  return maxexchange;
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::unpack_restart(int nlocal, int nth)
{
  FixLLGMidNH::unpack_restart(nlocal, nth);

  k_fm_cache_device.sync_host();
  k_s0_device.sync_host();
  k_x0_device.sync_host();
  k_v0_device.sync_host();
  k_f0_device.sync_host();

  for (int k = 0; k < 3; k++) {
    fm_cache_device_host[nlocal][k] = fm_cache[nlocal][k];
    s0_device_host[nlocal][k] = s0_cache[nlocal][k];
    x0_device_host[nlocal][k] = x0_cache[nlocal][k];
    v0_device_host[nlocal][k] = v0_cache[nlocal][k];
    f0_device_host[nlocal][k] = f0_cache[nlocal][k];
  }

  k_fm_cache_device.modify_host();
  k_s0_device.modify_host();
  k_x0_device.modify_host();
  k_v0_device.modify_host();
  k_f0_device.modify_host();
}

template <class DeviceType>
int FixLLGMidNHKokkos<DeviceType>::pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &k_buf,
                                                        DAT::tdual_int_1d k_sendlist, DAT::tdual_int_1d k_copylist,
                                                        ExecutionSpace /*space*/)
{
  if (nsend == 0) return 0;

  k_buf.sync<DeviceType>();
  k_sendlist.sync<DeviceType>();
  k_copylist.sync<DeviceType>();

  k_fm_cache_device.template sync<DeviceType>();
  k_s0_device.template sync<DeviceType>();
  k_x0_device.template sync<DeviceType>();
  k_v0_device.template sync<DeviceType>();
  k_f0_device.template sync<DeviceType>();

  d_exchange_sendlist = k_sendlist.view<DeviceType>();
  d_copylist = k_copylist.view<DeviceType>();
  d_buf = typename AT::t_xfloat_1d_um(k_buf.view<DeviceType>().data(), k_buf.extent(0) * k_buf.extent(1));

  auto exchange_sendlist = d_exchange_sendlist;
  auto copylist = d_copylist;
  auto buf = d_buf;
  auto fm_cache_ = d_fm_cache_device;
  auto s0_ = d_s0_device;
  auto x0_ = d_x0_device;
  auto v0_ = d_v0_device;
  auto f0_ = d_f0_device;
  const int stride = maxexchange;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nsend), LAMMPS_LAMBDA(const int mysend) {
    const int i = exchange_sendlist(mysend);
    const int j = copylist(mysend);
    int m = mysend * stride;
    for (int k = 0; k < 3; k++) buf(m++) = static_cast<X_FLOAT>(fm_cache_(i, k));
    for (int k = 0; k < 3; k++) buf(m++) = static_cast<X_FLOAT>(s0_(i, k));
    for (int k = 0; k < 3; k++) buf(m++) = static_cast<X_FLOAT>(x0_(i, k));
    for (int k = 0; k < 3; k++) buf(m++) = static_cast<X_FLOAT>(v0_(i, k));
    for (int k = 0; k < 3; k++) buf(m++) = static_cast<X_FLOAT>(f0_(i, k));

    if (j > -1) {
      for (int k = 0; k < 3; k++) {
        fm_cache_(i, k) = fm_cache_(j, k);
        s0_(i, k) = s0_(j, k);
        x0_(i, k) = x0_(j, k);
        v0_(i, k) = v0_(j, k);
        f0_(i, k) = f0_(j, k);
      }
    }
  });
  copymode = 0;

  k_buf.modify<DeviceType>();
  k_fm_cache_device.template modify<DeviceType>();
  k_s0_device.template modify<DeviceType>();
  k_x0_device.template modify<DeviceType>();
  k_v0_device.template modify<DeviceType>();
  k_f0_device.template modify<DeviceType>();
  return nsend * stride;
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d &indices,
                                                           int nrecv, int nrecv1, int nextrarecv1,
                                                           ExecutionSpace /*space*/)
{
  if (nrecv == 0) return;

  k_buf.sync<DeviceType>();
  indices.sync<DeviceType>();

  k_fm_cache_device.template sync<DeviceType>();
  k_s0_device.template sync<DeviceType>();
  k_x0_device.template sync<DeviceType>();
  k_v0_device.template sync<DeviceType>();
  k_f0_device.template sync<DeviceType>();

  d_indices = indices.view<DeviceType>();
  d_buf = typename AT::t_xfloat_1d_um(k_buf.view<DeviceType>().data(), k_buf.extent(0) * k_buf.extent(1));

  auto indices_ = d_indices;
  auto buf = d_buf;
  auto fm_cache_ = d_fm_cache_device;
  auto s0_ = d_s0_device;
  auto x0_ = d_x0_device;
  auto v0_ = d_v0_device;
  auto f0_ = d_f0_device;
  const int stride = maxexchange;

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nrecv), LAMMPS_LAMBDA(const int ii) {
    const int index = indices_(ii);
    if (index < 0) return;
    int m = (ii < nrecv1) ? (ii * stride) : (nextrarecv1 + (ii - nrecv1) * stride);
    for (int k = 0; k < 3; k++) fm_cache_(index, k) = static_cast<double>(buf(m++));
    for (int k = 0; k < 3; k++) s0_(index, k) = static_cast<double>(buf(m++));
    for (int k = 0; k < 3; k++) x0_(index, k) = static_cast<double>(buf(m++));
    for (int k = 0; k < 3; k++) v0_(index, k) = static_cast<double>(buf(m++));
    for (int k = 0; k < 3; k++) f0_(index, k) = static_cast<double>(buf(m++));
  });
  copymode = 0;

  k_fm_cache_device.template modify<DeviceType>();
  k_s0_device.template modify<DeviceType>();
  k_x0_device.template modify<DeviceType>();
  k_v0_device.template modify<DeviceType>();
  k_f0_device.template modify<DeviceType>();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  k_fm_cache_device.sync_device();
  k_s0_device.sync_device();
  k_x0_device.sync_device();
  k_v0_device.sync_device();
  k_f0_device.sync_device();

  Sorter.sort(LMPDeviceType(), k_fm_cache_device.d_view);
  Sorter.sort(LMPDeviceType(), k_s0_device.d_view);
  Sorter.sort(LMPDeviceType(), k_x0_device.d_view);
  Sorter.sort(LMPDeviceType(), k_v0_device.d_view);
  Sorter.sort(LMPDeviceType(), k_f0_device.d_view);

  k_fm_cache_device.modify_device();
  k_s0_device.modify_device();
  k_x0_device.modify_device();
  k_v0_device.modify_device();
  k_f0_device.modify_device();

  d_fm_cache_device = k_fm_cache_device.view<DeviceType>();
  d_s0_device = k_s0_device.view<DeviceType>();
  d_x0_device = k_x0_device.view<DeviceType>();
  d_v0_device = k_v0_device.view<DeviceType>();
  d_f0_device = k_f0_device.view<DeviceType>();
}

template <class DeviceType>
typename FixLLGMidNHKokkos<DeviceType>::MidpointBackend FixLLGMidNHKokkos<DeviceType>::select_midpoint_backend()
{
  midpoint_fallback_last = MidpointFallbackReason::NONE;
  if (!replay_fixes.empty()) {
    midpoint_fallback_last = MidpointFallbackReason::HOST_REPLAY_FIX;
    return MidpointBackend::HOST_FALLBACK;
  }
#if !defined(LMP_KOKKOS_GPU)
  midpoint_fallback_last = MidpointFallbackReason::NON_DEVICE_EXECUTION;
  return MidpointBackend::HOST_FALLBACK;
#else
  if (execution_space != Device) {
    midpoint_fallback_last = MidpointFallbackReason::NON_DEVICE_EXECUTION;
    return MidpointBackend::HOST_FALLBACK;
  }
#endif
  if (!ensure_device_scratch()) {
    midpoint_fallback_last = MidpointFallbackReason::SCRATCH_UNAVAILABLE;
    return MidpointBackend::HOST_FALLBACK;
  }
  return MidpointBackend::DEVICE;
}

template <class DeviceType>
bool FixLLGMidNHKokkos<DeviceType>::ensure_device_scratch()
{
  const int nmax = atom->nmax;
  if (nmax <= 0) return false;

  memoryKK->grow_kokkos(k_x0_device, x0_device_host, nmax, 3, "llgmid/nh/kk:x0_device");
  memoryKK->grow_kokkos(k_v0_device, v0_device_host, nmax, 3, "llgmid/nh/kk:v0_device");
  memoryKK->grow_kokkos(k_f0_device, f0_device_host, nmax, 3, "llgmid/nh/kk:f0_device");
  memoryKK->grow_kokkos(k_s0_device, s0_device_host, nmax, 3, "llgmid/nh/kk:s0_device");
  memoryKK->grow_kokkos(k_x_mid_device, x_mid_device_host, nmax, 3, "llgmid/nh/kk:x_mid_device");
  memoryKK->grow_kokkos(k_e_mid_device, e_mid_device_host, nmax, 3, "llgmid/nh/kk:e_mid_device");
  memoryKK->grow_kokkos(k_s_guess_device, s_guess_device_host, nmax, 3, "llgmid/nh/kk:s_guess_device");
  memoryKK->grow_kokkos(k_x_end_device, x_end_device_host, nmax, 3, "llgmid/nh/kk:x_end_device");
  memoryKK->grow_kokkos(k_fm_cache_device, fm_cache_device_host, nmax, 3, "llgmid/nh/kk:fm_cache");

  d_x0_device = k_x0_device.view<DeviceType>();
  d_v0_device = k_v0_device.view<DeviceType>();
  d_f0_device = k_f0_device.view<DeviceType>();
  d_s0_device = k_s0_device.view<DeviceType>();
  d_x_mid_device = k_x_mid_device.view<DeviceType>();
  d_e_mid_device = k_e_mid_device.view<DeviceType>();
  d_s_guess_device = k_s_guess_device.view<DeviceType>();
  d_x_end_device = k_x_end_device.view<DeviceType>();
  d_fm_cache_device = k_fm_cache_device.view<DeviceType>();

  return d_x0_device.data() && d_v0_device.data() && d_f0_device.data() && d_s0_device.data() &&
      d_x_mid_device.data() && d_e_mid_device.data() && d_s_guess_device.data() && d_x_end_device.data() &&
      d_fm_cache_device.data();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::init()
{
  sync_host_all();
  FixLLGMidNH::init();
  mark_host_all_modified();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::setup(int vflag)
{
  sync_host_all();
  FixLLGMidNH::setup(vflag);
  k_fm_cache_device.sync_host();
  for (int i = 0; i < atom->nlocal; i++) {
    fm_cache_device_host[i][0] = fm_cache[i][0];
    fm_cache_device_host[i][1] = fm_cache[i][1];
    fm_cache_device_host[i][2] = fm_cache[i][2];
  }
  k_fm_cache_device.modify_host();
  k_fm_cache_device.template sync<DeviceType>();
  d_fm_cache_device = k_fm_cache_device.view<DeviceType>();
  mark_host_all_modified();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::initial_integrate(int vflag)
{
  const int do_pstat = pstat_flag && lattice_flag;

  if (do_pstat && mpchain) nhc_press_integrate();
  if (tstat_flag) compute_temp_target();
  if (tstat_flag && lattice_flag) nhc_temp_integrate();

  if (do_pstat) {
    atomKK->sync(temperature->execution_space, temperature->datamask_read);
    atomKK->sync(pressure->execution_space, pressure->datamask_read);
    if (pstyle == ISO) {
      temperature->compute_scalar();
      pressure->compute_scalar();
    } else {
      temperature->compute_vector();
      pressure->compute_vector();
    }
    atomKK->modified(temperature->execution_space, temperature->datamask_modify);
    atomKK->modified(pressure->execution_space, pressure->datamask_modify);
    atomKK->sync(execution_space, temperature->datamask_modify);
    atomKK->sync(execution_space, pressure->datamask_modify);
    couple();
    pressure->addstep(update->ntimestep + 1);
    compute_press_target();
    nh_omega_dot();
    nh_v_press();
  }

  if (lattice_flag) {
    if (midpoint_iter > 1) cache_lattice_moving_step_start_state_device();
    this->nve_v();
    if (do_pstat) this->remap();
    this->nve_x();
  } else {
    build_predictor_midpoint_state_device();
  }

  if (do_pstat) {
    this->remap();
    if (this->kspace_flag) this->force->kspace->setup();
  }
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::cache_lattice_moving_step_start_state_device()
{
  if (!fm_cache) ensure_custom_peratom();
  if (!ensure_device_scratch()) error->all(FLERR, "Fix {} could not allocate llgmid Kokkos scratch arrays", style);

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  int need = X_MASK | V_MASK | F_MASK | MASK_MASK | TYPE_MASK | SP_MASK;
  if (atom->rmass) need |= RMASS_MASK;
  atomKK->sync(execution_space, need);
  atomKK->k_mass.template modify<LMPHostType>();
  atomKK->k_mass.template sync<DeviceType>();

  mask_view = atomKK->k_mask.template view<DeviceType>();
  x_view = atomKK->k_x.template view<DeviceType>();
  v_view = atomKK->k_v.template view<DeviceType>();
  f_view = atomKK->k_f.template view<DeviceType>();
  sp_view = atomKK->k_sp.template view<DeviceType>();

  auto mask_ = mask_view;
  auto x_ = x_view;
  auto v_ = v_view;
  auto f_ = f_view;
  auto sp_ = sp_view;
  auto x0_ = d_x0_device;
  auto v0_ = d_v0_device;
  auto f0_ = d_f0_device;
  auto s0_ = d_s0_device;
  auto xmid_ = d_x_mid_device;
  auto emid_ = d_e_mid_device;
  auto sguess_ = d_s_guess_device;
  auto xend_ = d_x_end_device;
  const int groupbit_local = groupbit;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    for (int k = 0; k < 3; k++) {
      x0_(i, k) = static_cast<double>(x_(i, k));
      v0_(i, k) = static_cast<double>(v_(i, k));
      f0_(i, k) = static_cast<double>(f_(i, k));
      xmid_(i, k) = x0_(i, k);
      xend_(i, k) = x0_(i, k);
      emid_(i, k) = 0.0;
      sguess_(i, k) = 0.0;
    }
    s0_(i, 0) = s0_(i, 1) = s0_(i, 2) = 0.0;
    if (!(mask_(i) & groupbit_local)) return;

    const double mag = static_cast<double>(sp_(i, 3));
    if (mag > KK_LLGMID_SPIN_EPS) {
      const double dirn =
          kk_llgmid_norm3(static_cast<double>(sp_(i, 0)), static_cast<double>(sp_(i, 1)), static_cast<double>(sp_(i, 2)));
      if (dirn > KK_LLGMID_SPIN_EPS) {
        const double scale = mag / dirn;
        s0_(i, 0) = static_cast<double>(sp_(i, 0)) * scale;
        s0_(i, 1) = static_cast<double>(sp_(i, 1)) * scale;
        s0_(i, 2) = static_cast<double>(sp_(i, 2)) * scale;
      }
    }
  });

  atomKK->modified(execution_space, X_MASK | V_MASK | F_MASK | SP_MASK);
  k_x0_device.template modify<DeviceType>();
  k_v0_device.template modify<DeviceType>();
  k_f0_device.template modify<DeviceType>();
  k_s0_device.template modify<DeviceType>();
  k_x_mid_device.template modify<DeviceType>();
  k_e_mid_device.template modify<DeviceType>();
  k_s_guess_device.template modify<DeviceType>();
  k_x_end_device.template modify<DeviceType>();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::build_predictor_midpoint_state_device()
{
  if (!fm_cache) ensure_custom_peratom();
  if (!ensure_device_scratch()) error->all(FLERR, "Fix {} could not allocate llgmid Kokkos scratch arrays", style);

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  int need = X_MASK | V_MASK | F_MASK | MASK_MASK | TYPE_MASK | SP_MASK;
  if (atom->rmass) need |= RMASS_MASK;
  atomKK->sync(execution_space, need);
  atomKK->k_mass.template modify<LMPHostType>();
  atomKK->k_mass.template sync<DeviceType>();
  k_fm_cache_device.template sync<DeviceType>();

  mask_view = atomKK->k_mask.template view<DeviceType>();
  type_view = atomKK->k_type.template view<DeviceType>();
  mass_type_view = atomKK->k_mass.template view<DeviceType>();
  x_view = atomKK->k_x.template view<DeviceType>();
  v_view = atomKK->k_v.template view<DeviceType>();
  f_view = atomKK->k_f.template view<DeviceType>();
  sp_view = atomKK->k_sp.template view<DeviceType>();

  auto mask_ = mask_view;
  auto type_ = type_view;
  auto mass_type_ = mass_type_view;
  auto x_ = x_view;
  auto v_ = v_view;
  auto f_ = f_view;
  auto sp_ = sp_view;
  auto x0_ = d_x0_device;
  auto v0_ = d_v0_device;
  auto f0_ = d_f0_device;
  auto s0_ = d_s0_device;
  auto xmid_ = d_x_mid_device;
  auto emid_ = d_e_mid_device;
  auto fm_cache_ = d_fm_cache_device;
  auto rmass_view = atomKK->k_rmass.template view<DeviceType>();
  const int groupbit_local = groupbit;
  const int lattice_mode = lattice_flag;
  const int has_rmass = (atom->rmass != nullptr);
  const double dt = update->dt;
  const double alpha_ = alpha;
  const double g_over_hbar_ = g_over_hbar;
  const double ftm2v = force->ftm2v;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    for (int k = 0; k < 3; k++) {
      x0_(i, k) = static_cast<double>(x_(i, k));
      v0_(i, k) = static_cast<double>(v_(i, k));
      f0_(i, k) = static_cast<double>(f_(i, k));
      xmid_(i, k) = x0_(i, k);
    }
    s0_(i, 0) = s0_(i, 1) = s0_(i, 2) = 0.0;
    emid_(i, 0) = emid_(i, 1) = emid_(i, 2) = 0.0;
    if (!(mask_(i) & groupbit_local)) return;

    const double mag = static_cast<double>(sp_(i, 3));
    if (mag > KK_LLGMID_SPIN_EPS) {
      const double dirn =
          kk_llgmid_norm3(static_cast<double>(sp_(i, 0)), static_cast<double>(sp_(i, 1)), static_cast<double>(sp_(i, 2)));
      if (dirn > KK_LLGMID_SPIN_EPS) {
        const double scale = mag / dirn;
        s0_(i, 0) = static_cast<double>(sp_(i, 0)) * scale;
        s0_(i, 1) = static_cast<double>(sp_(i, 1)) * scale;
        s0_(i, 2) = static_cast<double>(sp_(i, 2)) * scale;
      }
    }

    if (lattice_mode) {
      const double mass = has_rmass ? static_cast<double>(rmass_view(i)) : static_cast<double>(mass_type_(type_(i)));
      const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
      for (int k = 0; k < 3; k++)
        xmid_(i, k) = x0_(i, k) + 0.5 * dt * v0_(i, k) + 0.125 * dt * dt * ftm2v * f0_(i, k) * inv_mass;
    }

    const double smag = kk_llgmid_norm3(s0_(i, 0), s0_(i, 1), s0_(i, 2));
    if (smag > KK_LLGMID_SPIN_EPS) {
      double ex = s0_(i, 0) / smag;
      double ey = s0_(i, 1) / smag;
      double ez = s0_(i, 2) / smag;
      kk_llgmid_exact_flow_direction(ex, ey, ez, fm_cache_(i, 0), fm_cache_(i, 1), fm_cache_(i, 2), 0.5 * dt, alpha_,
                                     g_over_hbar_);
      double mx = s0_(i, 0) / smag + ex;
      double my = s0_(i, 1) / smag + ey;
      double mz = s0_(i, 2) / smag + ez;
      kk_llgmid_normalize3(mx, my, mz);
      emid_(i, 0) = smag * mx;
      emid_(i, 1) = smag * my;
      emid_(i, 2) = smag * mz;
      sp_(i, 0) = static_cast<X_FLOAT>(mx);
      sp_(i, 1) = static_cast<X_FLOAT>(my);
      sp_(i, 2) = static_cast<X_FLOAT>(mz);
      sp_(i, 3) = static_cast<X_FLOAT>(smag);
    } else {
      sp_(i, 0) = sp_(i, 1) = sp_(i, 2) = sp_(i, 3) = 0.0;
    }

    if (lattice_mode) {
      x_(i, 0) = static_cast<X_FLOAT>(xmid_(i, 0));
      x_(i, 1) = static_cast<X_FLOAT>(xmid_(i, 1));
      x_(i, 2) = static_cast<X_FLOAT>(xmid_(i, 2));
    }
  });

  atomKK->modified(execution_space, X_MASK | SP_MASK);
  k_x0_device.template modify<DeviceType>();
  k_v0_device.template modify<DeviceType>();
  k_f0_device.template modify<DeviceType>();
  k_s0_device.template modify<DeviceType>();
  k_x_mid_device.template modify<DeviceType>();
  k_e_mid_device.template modify<DeviceType>();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::clear_force_arrays_device()
{
  const int nclear = force->newton ? (atom->nlocal + atom->nghost) : atom->nlocal;
  const int has_fml = (atom->fm_long != nullptr);
  int need = F_MASK | FM_MASK;
  if (has_fml) need |= FML_MASK;
  atomKK->sync(execution_space, need);

  auto f_ = atomKK->k_f.template view<DeviceType>();
  auto fm_ = atomKK->k_fm.template view<DeviceType>();
  auto fml_ = has_fml ? atomKK->k_fm_long.template view<DeviceType>() : decltype(f_)();

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
  atomKK->modified(execution_space, need);
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::rebuild_neighbors_for_current_positions_device()
{
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  if (atom->sortfreq > 0 && update->ntimestep >= atomKK->nextsort) atomKK->sort();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::recompute_force_and_field_device(int eflag, int vflag)
{
  comm->forward_comm();
  clear_force_arrays_device();
  if (force->pair) force->pair->compute(eflag, vflag);
  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }
  if (force->kspace) force->kspace->compute(eflag, vflag);
  if (force->newton) comm->reverse_comm();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::cache_current_fm_device()
{
  atomKK->sync(execution_space, FM_MASK);
  k_fm_cache_device.template sync<DeviceType>();

  auto fm_ = atomKK->k_fm.template view<DeviceType>();
  auto fm_cache_ = d_fm_cache_device;
  const int nlocal = atom->nlocal;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    fm_cache_(i, 0) = static_cast<double>(fm_(i, 0));
    fm_cache_(i, 1) = static_cast<double>(fm_(i, 1));
    fm_cache_(i, 2) = static_cast<double>(fm_(i, 2));
  });
  k_fm_cache_device.template modify<DeviceType>();
}

template <class DeviceType>
bool FixLLGMidNHKokkos<DeviceType>::solve_spin_midpoint_device(bool lattice_mode, int vflag, double pe_mid)
{
  if (midpoint_iter <= 1) return false;
  if (lattice_mode != (lattice_flag != 0)) return false;
  if (!ensure_device_scratch()) return false;

  if (!fm_cache) ensure_custom_peratom();

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const int groupbit_local = groupbit;
  const double dt = update->dt;
  const double midpoint_tol_local = midpoint_tol;

  int need = X_MASK | V_MASK | F_MASK | FM_MASK | MASK_MASK | TYPE_MASK | SP_MASK;
  if (atom->rmass) need |= RMASS_MASK;
  atomKK->sync(execution_space, need);
  atomKK->k_mass.template modify<LMPHostType>();
  atomKK->k_mass.template sync<DeviceType>();

  mask_view = atomKK->k_mask.template view<DeviceType>();
  type_view = atomKK->k_type.template view<DeviceType>();
  mass_type_view = atomKK->k_mass.template view<DeviceType>();
  x_view = atomKK->k_x.template view<DeviceType>();
  v_view = atomKK->k_v.template view<DeviceType>();
  f_view = atomKK->k_f.template view<DeviceType>();
  sp_view = atomKK->k_sp.template view<DeviceType>();
  fm_view = atomKK->k_fm.template view<DeviceType>();
  auto rmass_view = atomKK->k_rmass.template view<DeviceType>();

  auto mask_ = mask_view;
  auto type_ = type_view;
  auto mass_type_ = mass_type_view;
  auto x_ = x_view;
  auto v_ = v_view;
  auto f_ = f_view;
  auto sp_ = sp_view;
  auto fm_ = fm_view;
  auto x0_ = d_x0_device;
  auto v0_ = d_v0_device;
  auto s0_ = d_s0_device;
  auto xmid_ = d_x_mid_device;
  auto emid_ = d_e_mid_device;
  auto sguess_ = d_s_guess_device;
  auto xend_ = d_x_end_device;
  const int has_rmass = (atom->rmass != nullptr);
  const double alpha_ = alpha;
  const double g_over_hbar_ = g_over_hbar;
  const double ftm2v = force->ftm2v;
  const int update_lattice = 0;

  for (int iter = 0; iter < midpoint_iter; iter++) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
      if (!(mask_(i) & groupbit_local)) return;

      if (update_lattice) {
        const double mass = has_rmass ? static_cast<double>(rmass_view(i)) : static_cast<double>(mass_type_(type_(i)));
        const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
        for (int k = 0; k < 3; k++) {
          xend_(i, k) = x0_(i, k) + dt * v0_(i, k) + 0.5 * dt * dt * ftm2v * static_cast<double>(f_(i, k)) * inv_mass;
          xmid_(i, k) = 0.5 * (x0_(i, k) + xend_(i, k));
          v_(i, k) = static_cast<X_FLOAT>(v0_(i, k) + dt * ftm2v * static_cast<double>(f_(i, k)) * inv_mass);
        }
      } else {
        for (int k = 0; k < 3; k++) {
          xend_(i, k) = x0_(i, k);
          xmid_(i, k) = x0_(i, k);
          v_(i, k) = static_cast<X_FLOAT>(v0_(i, k));
        }
      }

      const double smag = kk_llgmid_norm3(s0_(i, 0), s0_(i, 1), s0_(i, 2));
      if (smag > KK_LLGMID_SPIN_EPS) {
        double ex = s0_(i, 0) / smag;
        double ey = s0_(i, 1) / smag;
        double ez = s0_(i, 2) / smag;
        kk_llgmid_exact_flow_direction(ex, ey, ez, static_cast<double>(fm_(i, 0)), static_cast<double>(fm_(i, 1)),
                                       static_cast<double>(fm_(i, 2)), dt, alpha_, g_over_hbar_);
        sguess_(i, 0) = smag * ex;
        sguess_(i, 1) = smag * ey;
        sguess_(i, 2) = smag * ez;

        double mx = s0_(i, 0) / smag + ex;
        double my = s0_(i, 1) / smag + ey;
        double mz = s0_(i, 2) / smag + ez;
        kk_llgmid_normalize3(mx, my, mz);
        emid_(i, 0) = smag * mx;
        emid_(i, 1) = smag * my;
        emid_(i, 2) = smag * mz;
      } else {
        sguess_(i, 0) = sguess_(i, 1) = sguess_(i, 2) = 0.0;
        emid_(i, 0) = emid_(i, 1) = emid_(i, 2) = 0.0;
      }
    });

    double max_dev_local = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType>(0, nlocal),
        LAMMPS_LAMBDA(const int i, double &maxv) {
          if (!(mask_(i) & groupbit_local)) return;
          if (update_lattice) {
            const double dx = xmid_(i, 0) - static_cast<double>(x_(i, 0));
            const double dy = xmid_(i, 1) - static_cast<double>(x_(i, 1));
            const double dz = xmid_(i, 2) - static_cast<double>(x_(i, 2));
            const double dr = kk_llgmid_norm3(dx, dy, dz);
            if (dr > maxv) maxv = dr;
          }

          const double smag = static_cast<double>(sp_(i, 3));
          const double dsx = emid_(i, 0) - smag * static_cast<double>(sp_(i, 0));
          const double dsy = emid_(i, 1) - smag * static_cast<double>(sp_(i, 1));
          const double dsz = emid_(i, 2) - smag * static_cast<double>(sp_(i, 2));
          const double ds = kk_llgmid_norm3(dsx, dsy, dsz);
          if (ds > maxv) maxv = ds;
        },
        Kokkos::Max<double>(max_dev_local));

    double max_dev_all = max_dev_local;
    if (midpoint_tol_local > 0.0) MPI_Allreduce(&max_dev_local, &max_dev_all, 1, MPI_DOUBLE, MPI_MAX, world);
    if ((iter == midpoint_iter - 1) || (midpoint_tol_local > 0.0 && max_dev_all <= midpoint_tol_local)) break;

    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
      if (!(mask_(i) & groupbit_local)) return;
      if (update_lattice) {
        x_(i, 0) = static_cast<X_FLOAT>(xmid_(i, 0));
        x_(i, 1) = static_cast<X_FLOAT>(xmid_(i, 1));
        x_(i, 2) = static_cast<X_FLOAT>(xmid_(i, 2));
      }

      const double smag = kk_llgmid_norm3(s0_(i, 0), s0_(i, 1), s0_(i, 2));
      if (smag <= KK_LLGMID_SPIN_EPS) {
        sp_(i, 0) = sp_(i, 1) = sp_(i, 2) = sp_(i, 3) = 0.0;
        return;
      }

      double sx = emid_(i, 0);
      double sy = emid_(i, 1);
      double sz = emid_(i, 2);
      kk_llgmid_normalize3(sx, sy, sz);
      sp_(i, 0) = static_cast<X_FLOAT>(sx);
      sp_(i, 1) = static_cast<X_FLOAT>(sy);
      sp_(i, 2) = static_cast<X_FLOAT>(sz);
      sp_(i, 3) = static_cast<X_FLOAT>(smag);
    });
    atomKK->modified(execution_space, X_MASK | V_MASK | SP_MASK);
    recompute_force_and_field_device(1, 0);
  }

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!(mask_(i) & groupbit_local)) return;
    if (update_lattice) {
      x_(i, 0) = static_cast<X_FLOAT>(xend_(i, 0));
      x_(i, 1) = static_cast<X_FLOAT>(xend_(i, 1));
      x_(i, 2) = static_cast<X_FLOAT>(xend_(i, 2));
    }

    const double smag = kk_llgmid_norm3(s0_(i, 0), s0_(i, 1), s0_(i, 2));
    if (smag <= KK_LLGMID_SPIN_EPS) {
      sp_(i, 0) = sp_(i, 1) = sp_(i, 2) = sp_(i, 3) = 0.0;
      return;
    }

    double sx = sguess_(i, 0);
    double sy = sguess_(i, 1);
    double sz = sguess_(i, 2);
    kk_llgmid_normalize3(sx, sy, sz);
    sp_(i, 0) = static_cast<X_FLOAT>(sx);
    sp_(i, 1) = static_cast<X_FLOAT>(sy);
    sp_(i, 2) = static_cast<X_FLOAT>(sz);
    sp_(i, 3) = static_cast<X_FLOAT>(smag);
  });
  atomKK->modified(execution_space, X_MASK | V_MASK | SP_MASK);

  recompute_force_and_field_device(1, vflag);
  cache_current_fm_device();
  const double pe_end = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
  debug_log_energy(pe_mid, pe_end);
  pe_prev_end = pe_end;
  return true;
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::post_force(int vflag)
{
  midpoint_backend_last = select_midpoint_backend();
  if (midpoint_backend_last == MidpointBackend::DEVICE) {
    const double pe_mid = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
    if (solve_spin_midpoint_device(lattice_flag != 0, vflag, pe_mid)) return;
    midpoint_backend_last = MidpointBackend::HOST_FALLBACK;
    if (midpoint_fallback_last == MidpointFallbackReason::NONE)
      midpoint_fallback_last = MidpointFallbackReason::SCRATCH_UNAVAILABLE;
  }

  sync_host_all();
  FixLLGMidNH::post_force(vflag);
  k_fm_cache_device.sync_host();
  for (int i = 0; i < atom->nlocal; i++) {
    fm_cache_device_host[i][0] = fm_cache[i][0];
    fm_cache_device_host[i][1] = fm_cache[i][1];
    fm_cache_device_host[i][2] = fm_cache[i][2];
  }
  k_fm_cache_device.modify_host();
  k_fm_cache_device.template sync<DeviceType>();
  d_fm_cache_device = k_fm_cache_device.view<DeviceType>();
  mark_host_all_modified();
}

template <class DeviceType>
void FixLLGMidNHKokkos<DeviceType>::final_integrate()
{
  const int do_pstat = pstat_flag && lattice_flag;

  if (lattice_flag) this->nve_v();
  if (which == BIAS && neighbor->ago == 0) {
    atomKK->sync(temperature->execution_space, temperature->datamask_read);
    t_current = temperature->compute_scalar();
    atomKK->modified(temperature->execution_space, temperature->datamask_modify);
    atomKK->sync(execution_space, temperature->datamask_modify);
  }

  if (do_pstat) nh_v_press();

  atomKK->sync(temperature->execution_space, temperature->datamask_read);
  t_current = temperature->compute_scalar();
  atomKK->modified(temperature->execution_space, temperature->datamask_modify);
  atomKK->sync(execution_space, temperature->datamask_modify);
  tdof = temperature->dof;

  if (do_pstat) {
    if (pstyle == ISO) {
      atomKK->sync(pressure->execution_space, pressure->datamask_read);
      pressure->compute_scalar();
      atomKK->modified(pressure->execution_space, pressure->datamask_modify);
      atomKK->sync(execution_space, pressure->datamask_modify);
    } else {
      atomKK->sync(temperature->execution_space, temperature->datamask_read);
      atomKK->sync(pressure->execution_space, pressure->datamask_read);
      temperature->compute_vector();
      pressure->compute_vector();
      atomKK->modified(temperature->execution_space, temperature->datamask_modify);
      atomKK->modified(pressure->execution_space, pressure->datamask_modify);
      atomKK->sync(execution_space, temperature->datamask_modify);
      atomKK->sync(execution_space, pressure->datamask_modify);
    }
    couple();
    pressure->addstep(update->ntimestep + 1);
    nh_omega_dot();
  }

  if (tstat_flag && lattice_flag) nhc_temp_integrate();
  if (do_pstat && mpchain) nhc_press_integrate();
}

namespace LAMMPS_NS {
template class FixLLGMidNHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixLLGMidNHKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

#endif
