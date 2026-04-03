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

#include "fix_llggeom_nh_kokkos.h"

#ifdef LMP_KOKKOS

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "force.h"
#include "improper.h"
#include "angle.h"
#include "dihedral.h"
#include "kspace.h"
#include "math_const.h"
#include "modify.h"
#include "pair.h"
#include "error.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;

enum { NOBIAS, BIAS };
enum { ISO, ANISO, TRICLINIC };

namespace {
constexpr double KK_SPIN_EPS = 1.0e-12;
constexpr double KK_FIELD_EPS = 1.0e-12;
constexpr double KK_PERP_EPS = 1.0e-12;
constexpr double KK_THERMAL_EPS = 1.0e-30;

KOKKOS_INLINE_FUNCTION double kk_norm3(double x, double y, double z)
{
  return sqrt(x * x + y * y + z * z);
}

KOKKOS_INLINE_FUNCTION void kk_normalize3(double &x, double &y, double &z)
{
  const double n = kk_norm3(x, y, z);
  if (n <= KK_SPIN_EPS) {
    x = y = z = 0.0;
    return;
  }
  const double inv = 1.0 / n;
  x *= inv;
  y *= inv;
  z *= inv;
}

KOKKOS_INLINE_FUNCTION void kk_damping_half_step(double &ex, double &ey, double &ez, double hx, double hy, double hz,
                                                 double tau, double alpha, double g_over_hbar)
{
  kk_normalize3(ex, ey, ez);
  const double hmag = kk_norm3(hx, hy, hz);
  if (hmag < KK_FIELD_EPS || tau == 0.0 || alpha == 0.0) return;

  const double hux = hx / hmag;
  const double huy = hy / hmag;
  const double huz = hz / hmag;
  const double u = fmax(-1.0, fmin(1.0, ex * hux + ey * huy + ez * huz));
  const double wx = ex - u * hux;
  const double wy = ey - u * huy;
  const double wz = ez - u * huz;
  const double rho = kk_norm3(wx, wy, wz);
  if (rho < KK_PERP_EPS) {
    const double sign = (u >= 0.0) ? 1.0 : -1.0;
    ex = sign * hux;
    ey = sign * huy;
    ez = sign * huz;
    return;
  }

  const double b = alpha * g_over_hbar / (1.0 + alpha * alpha);
  const double q = ((1.0 + u) > KK_PERP_EPS) ? (rho / (1.0 + u)) : ((1.0 - u) / rho);
  const double qp = q * exp(-b * hmag * tau);
  const double qp2 = qp * qp;
  const double up = (1.0 - qp2) / (1.0 + qp2);
  const double rhop = 2.0 * qp / (1.0 + qp2);
  const double scale = rhop / rho;

  ex = up * hux + scale * wx;
  ey = up * huy + scale * wy;
  ez = up * huz + scale * wz;
  kk_normalize3(ex, ey, ez);
}

KOKKOS_INLINE_FUNCTION void kk_boris_step(double &ex, double &ey, double &ez, double hx, double hy, double hz,
                                          double dt, double alpha, double g_over_hbar)
{
  kk_normalize3(ex, ey, ez);
  const double a = g_over_hbar / (1.0 + alpha * alpha);
  const double tx = -0.5 * dt * a * hx;
  const double ty = -0.5 * dt * a * hy;
  const double tz = -0.5 * dt * a * hz;
  const double t2 = tx * tx + ty * ty + tz * tz;

  const double epx = ex + (ey * tz - ez * ty);
  const double epy = ey + (ez * tx - ex * tz);
  const double epz = ez + (ex * ty - ey * tx);

  const double sx = 2.0 * tx / (1.0 + t2);
  const double sy = 2.0 * ty / (1.0 + t2);
  const double sz = 2.0 * tz / (1.0 + t2);

  ex = ex + (epy * sz - epz * sy);
  ey = ey + (epz * sx - epx * sz);
  ez = ez + (epx * sy - epy * sx);
  kk_normalize3(ex, ey, ez);
}

KOKKOS_INLINE_FUNCTION void kk_spin_map(double &ex, double &ey, double &ez, double hx, double hy, double hz, double dt,
                                        double alpha, double g_over_hbar)
{
  kk_damping_half_step(ex, ey, ez, hx, hy, hz, 0.5 * dt, alpha, g_over_hbar);
  kk_boris_step(ex, ey, ez, hx, hy, hz, dt, alpha, g_over_hbar);
  kk_damping_half_step(ex, ey, ez, hx, hy, hz, 0.5 * dt, alpha, g_over_hbar);
  kk_normalize3(ex, ey, ez);
}

KOKKOS_INLINE_FUNCTION std::uint64_t kk_splitmix64(std::uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

KOKKOS_INLINE_FUNCTION double kk_gaussian_u64(std::uint64_t seed64, tagint tag, std::uint64_t step, int component)
{
  std::uint64_t state = seed64;
  state ^= static_cast<std::uint64_t>(tag) * 0xd1b54a32d192ed03ULL;
  state ^= step * 0x9e3779b97f4a7c15ULL;
  state ^= static_cast<std::uint64_t>(component) * 0x94d049bb133111ebULL;

  double u1 = 0.0;
  do {
    state = kk_splitmix64(state);
    u1 = (state >> 11) * (1.0 / 9007199254740992.0);
  } while (u1 <= 0.0);
  state = kk_splitmix64(state);
  const double u2 = (state >> 11) * (1.0 / 9007199254740992.0);
  return sqrt(-2.0 * log(u1)) * cos(MathConst::MY_2PI * u2);
}
}    // namespace

template <class DeviceType>
FixLLGGeomNHKokkos<DeviceType>::FixLLGGeomNHKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixLLGGeomNH(lmp, narg, arg),
    atomKK((AtomKokkos *) atom),
    x0_device_host(nullptr),
    v0_device_host(nullptr),
    f0_device_host(nullptr),
    s0_device_host(nullptr),
    r_mid_guess_device_host(nullptr),
    e_mid_guess_device_host(nullptr),
    e_pred_device_host(nullptr),
    f_mid_device_host(nullptr),
    h_mid_device_host(nullptr),
    r_new_device_host(nullptr),
    v_new_device_host(nullptr),
    e_new_device_host(nullptr),
    h_th_device_host(nullptr),
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
  if (!atomKK) error->all(FLERR, "Fix {} requires atom_style spin/kk (or spin with Kokkos enabled)", style);
  sort_device = 1;
  exchange_comm_device = 1;
  grow_arrays(atom->nmax);
}

template <class DeviceType>
FixLLGGeomNHKokkos<DeviceType>::~FixLLGGeomNHKokkos()
{
  if (copymode) return;
  this->memoryKK->destroy_kokkos(k_x0_device, x0_device_host);
  this->memoryKK->destroy_kokkos(k_v0_device, v0_device_host);
  this->memoryKK->destroy_kokkos(k_f0_device, f0_device_host);
  this->memoryKK->destroy_kokkos(k_s0_device, s0_device_host);
  this->memoryKK->destroy_kokkos(k_r_mid_guess_device, r_mid_guess_device_host);
  this->memoryKK->destroy_kokkos(k_e_mid_guess_device, e_mid_guess_device_host);
  this->memoryKK->destroy_kokkos(k_e_pred_device, e_pred_device_host);
  this->memoryKK->destroy_kokkos(k_f_mid_device, f_mid_device_host);
  this->memoryKK->destroy_kokkos(k_h_mid_device, h_mid_device_host);
  this->memoryKK->destroy_kokkos(k_r_new_device, r_new_device_host);
  this->memoryKK->destroy_kokkos(k_v_new_device, v_new_device_host);
  this->memoryKK->destroy_kokkos(k_e_new_device, e_new_device_host);
  this->memoryKK->destroy_kokkos(k_h_th_device, h_th_device_host);
  this->memoryKK->destroy_kokkos(k_fm_cache_device, fm_cache_device_host);
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::grow_arrays(int nmax)
{
  const int old_nmax = nmax_old;
  FixLLGGeomNH::grow_arrays(nmax);

  this->memoryKK->grow_kokkos(k_fm_cache_device, fm_cache_device_host, nmax, 3, "llggeom/nh/kk:fm_cache");
  this->memoryKK->grow_kokkos(k_s0_device, s0_device_host, nmax, 3, "llggeom/nh/kk:s0_device");
  this->memoryKK->grow_kokkos(k_x0_device, x0_device_host, nmax, 3, "llggeom/nh/kk:x0_device");
  this->memoryKK->grow_kokkos(k_v0_device, v0_device_host, nmax, 3, "llggeom/nh/kk:v0_device");
  this->memoryKK->grow_kokkos(k_f0_device, f0_device_host, nmax, 3, "llggeom/nh/kk:f0_device");

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
void FixLLGGeomNHKokkos<DeviceType>::copy_arrays(int i, int j, int /*delflag*/)
{
  FixLLGGeomNH::copy_arrays(i, j, 0);

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
int FixLLGGeomNHKokkos<DeviceType>::pack_exchange(int i, double *buf)
{
  return FixLLGGeomNH::pack_exchange(i, buf);
}

template <class DeviceType>
int FixLLGGeomNHKokkos<DeviceType>::unpack_exchange(int nlocal, double *buf)
{
  FixLLGGeomNH::unpack_exchange(nlocal, buf);

  k_fm_cache_device.sync_host();
  k_s0_device.sync_host();
  k_x0_device.sync_host();
  k_v0_device.sync_host();
  k_f0_device.sync_host();

  for (int k = 0; k < 3; k++) {
    fm_cache_device_host[nlocal][k] = this->fm_cache[nlocal][k];
    s0_device_host[nlocal][k] = this->s0_cache[nlocal][k];
    x0_device_host[nlocal][k] = this->x0_cache[nlocal][k];
    v0_device_host[nlocal][k] = this->v0_cache[nlocal][k];
    f0_device_host[nlocal][k] = this->f0_cache[nlocal][k];
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
void FixLLGGeomNHKokkos<DeviceType>::unpack_restart(int nlocal, int nth)
{
  FixLLGGeomNH::unpack_restart(nlocal, nth);

  k_fm_cache_device.sync_host();
  k_s0_device.sync_host();
  k_x0_device.sync_host();
  k_v0_device.sync_host();
  k_f0_device.sync_host();

  for (int k = 0; k < 3; k++) {
    fm_cache_device_host[nlocal][k] = this->fm_cache[nlocal][k];
    s0_device_host[nlocal][k] = this->s0_cache[nlocal][k];
    x0_device_host[nlocal][k] = this->x0_cache[nlocal][k];
    v0_device_host[nlocal][k] = this->v0_cache[nlocal][k];
    f0_device_host[nlocal][k] = this->f0_cache[nlocal][k];
  }

  k_fm_cache_device.modify_host();
  k_s0_device.modify_host();
  k_x0_device.modify_host();
  k_v0_device.modify_host();
  k_f0_device.modify_host();
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::sync_host_all()
{
  atomKK->sync(Host, X_MASK | V_MASK | F_MASK | MASK_MASK | TYPE_MASK | TAG_MASK | SP_MASK | FM_MASK | FML_MASK | RMASS_MASK);
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::mark_host_all_modified()
{
  atomKK->modified(Host, X_MASK | V_MASK | F_MASK | SP_MASK | FM_MASK | FML_MASK);
}

template <class DeviceType>
int FixLLGGeomNHKokkos<DeviceType>::pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &k_buf,
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
void FixLLGGeomNHKokkos<DeviceType>::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d &indices,
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
void FixLLGGeomNHKokkos<DeviceType>::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
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
void FixLLGGeomNHKokkos<DeviceType>::validate_replay_fixes()
{
  for (auto *fix : replay_fixes) {
    if (utils::strmatch(fix->style, "^.*/kk")) continue;
    error->all(FLERR,
               "Fix {} requires replayed spin-field fixes to be Kokkos styles when using {} (found {} style {})",
               style, style, fix->id, fix->style);
  }
}

template <class DeviceType>
bool FixLLGGeomNHKokkos<DeviceType>::ensure_device_scratch()
{
  const int nmax = atom->nmax;
  if (nmax <= 0) return false;

  this->memoryKK->grow_kokkos(k_x0_device, x0_device_host, nmax, 3, "llggeom/nh/kk:x0_device");
  this->memoryKK->grow_kokkos(k_v0_device, v0_device_host, nmax, 3, "llggeom/nh/kk:v0_device");
  this->memoryKK->grow_kokkos(k_f0_device, f0_device_host, nmax, 3, "llggeom/nh/kk:f0_device");
  this->memoryKK->grow_kokkos(k_s0_device, s0_device_host, nmax, 3, "llggeom/nh/kk:s0_device");
  this->memoryKK->grow_kokkos(k_r_mid_guess_device, r_mid_guess_device_host, nmax, 3, "llggeom/nh/kk:r_mid_guess");
  this->memoryKK->grow_kokkos(k_e_mid_guess_device, e_mid_guess_device_host, nmax, 3, "llggeom/nh/kk:e_mid_guess");
  this->memoryKK->grow_kokkos(k_e_pred_device, e_pred_device_host, nmax, 3, "llggeom/nh/kk:e_pred");
  this->memoryKK->grow_kokkos(k_f_mid_device, f_mid_device_host, nmax, 3, "llggeom/nh/kk:f_mid");
  this->memoryKK->grow_kokkos(k_h_mid_device, h_mid_device_host, nmax, 3, "llggeom/nh/kk:h_mid");
  this->memoryKK->grow_kokkos(k_r_new_device, r_new_device_host, nmax, 3, "llggeom/nh/kk:r_new");
  this->memoryKK->grow_kokkos(k_v_new_device, v_new_device_host, nmax, 3, "llggeom/nh/kk:v_new");
  this->memoryKK->grow_kokkos(k_e_new_device, e_new_device_host, nmax, 3, "llggeom/nh/kk:e_new");
  this->memoryKK->grow_kokkos(k_h_th_device, h_th_device_host, nmax, 3, "llggeom/nh/kk:h_th");
  this->memoryKK->grow_kokkos(k_fm_cache_device, fm_cache_device_host, nmax, 3, "llggeom/nh/kk:fm_cache");

  d_x0_device = k_x0_device.view<DeviceType>();
  d_v0_device = k_v0_device.view<DeviceType>();
  d_f0_device = k_f0_device.view<DeviceType>();
  d_s0_device = k_s0_device.view<DeviceType>();
  d_r_mid_guess_device = k_r_mid_guess_device.view<DeviceType>();
  d_e_mid_guess_device = k_e_mid_guess_device.view<DeviceType>();
  d_e_pred_device = k_e_pred_device.view<DeviceType>();
  d_f_mid_device = k_f_mid_device.view<DeviceType>();
  d_h_mid_device = k_h_mid_device.view<DeviceType>();
  d_r_new_device = k_r_new_device.view<DeviceType>();
  d_v_new_device = k_v_new_device.view<DeviceType>();
  d_e_new_device = k_e_new_device.view<DeviceType>();
  d_h_th_device = k_h_th_device.view<DeviceType>();
  d_fm_cache_device = k_fm_cache_device.view<DeviceType>();

  return d_x0_device.data() && d_v0_device.data() && d_f0_device.data() && d_s0_device.data() &&
      d_r_mid_guess_device.data() && d_e_mid_guess_device.data() && d_f_mid_device.data() && d_h_mid_device.data() &&
      d_r_new_device.data() && d_v_new_device.data() && d_e_new_device.data() && d_h_th_device.data() &&
      d_fm_cache_device.data();
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::init()
{
  sync_host_all();
  FixLLGGeomNH::init();
  validate_replay_fixes();
  mark_host_all_modified();
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::setup(int vflag)
{
  sync_host_all();
  FixLLGGeomNH::setup(vflag);
  k_fm_cache_device.sync_host();
  for (int i = 0; i < atom->nlocal; i++) {
    fm_cache_device_host[i][0] = this->fm_cache[i][0];
    fm_cache_device_host[i][1] = this->fm_cache[i][1];
    fm_cache_device_host[i][2] = this->fm_cache[i][2];
  }
  k_fm_cache_device.modify_host();
  k_fm_cache_device.template sync<DeviceType>();
  d_fm_cache_device = k_fm_cache_device.view<DeviceType>();
  mark_host_all_modified();
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::initial_integrate(int vflag)
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
    build_predictor_midpoint_state();
  }

  if (do_pstat) {
    this->remap();
    if (this->kspace_flag) this->force->kspace->setup();
  }
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::cache_lattice_moving_step_start_state_device()
{
  if (!this->fm_cache) this->ensure_custom_peratom();
  this->ensure_solver_arrays(this->atom->nmax);
  if (!ensure_device_scratch()) error->all(FLERR, "Fix {} could not allocate llggeom Kokkos scratch arrays", this->style);

  this->spin_temperature_cache_valid = 0;
  const double temp_use = this->compute_spin_temperature();
  const double dt = this->update->dt;
  const double prefactor_base =
      (this->alpha > 0.0 && temp_use > 0.0 && this->g_over_hbar > 0.0 && dt > 0.0) ?
      (2.0 * this->alpha * this->force->boltz * temp_use) :
      0.0;

  const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;
  int need = X_MASK | V_MASK | F_MASK | MASK_MASK | TYPE_MASK | TAG_MASK | SP_MASK;
  if (this->atom->rmass) need |= RMASS_MASK;
  atomKK->sync(execution_space, need);
  atomKK->k_mass.template modify<LMPHostType>();
  atomKK->k_mass.template sync<DeviceType>();

  mask_view = atomKK->k_mask.template view<DeviceType>();
  tag_view = atomKK->k_tag.template view<DeviceType>();
  type_view = atomKK->k_type.template view<DeviceType>();
  mass_type_view = atomKK->k_mass.template view<DeviceType>();
  x_view = atomKK->k_x.template view<DeviceType>();
  v_view = atomKK->k_v.template view<DeviceType>();
  f_view = atomKK->k_f.template view<DeviceType>();
  sp_view = atomKK->k_sp.template view<DeviceType>();

  auto rmass_view = atomKK->k_rmass.template view<DeviceType>();
  auto mask_ = mask_view;
  auto tag_ = tag_view;
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
  auto rmid_ = d_r_mid_guess_device;
  auto emid_ = d_e_mid_guess_device;
  auto epred_ = d_e_pred_device;
  auto hth_ = d_h_th_device;
  auto fmid_ = d_f_mid_device;
  auto hmid_ = d_h_mid_device;
  auto rnew_ = d_r_new_device;
  auto vnew_ = d_v_new_device;
  auto enew_ = d_e_new_device;
  const int groupbit_local = this->groupbit;
  const int has_rmass = (this->atom->rmass != nullptr);
  const std::uint64_t seed64 = static_cast<std::uint64_t>(this->seed);
  const std::uint64_t step64 = static_cast<std::uint64_t>(this->update->ntimestep);
  const double g_over_hbar_ = this->g_over_hbar;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    for (int k = 0; k < 3; k++) {
      x0_(i, k) = static_cast<double>(x_(i, k));
      v0_(i, k) = static_cast<double>(v_(i, k));
      f0_(i, k) = static_cast<double>(f_(i, k));
      rmid_(i, k) = x0_(i, k);
      rnew_(i, k) = x0_(i, k);
      vnew_(i, k) = v0_(i, k);
      emid_(i, k) = 0.0;
      epred_(i, k) = 0.0;
      fmid_(i, k) = 0.0;
      hmid_(i, k) = 0.0;
      enew_(i, k) = 0.0;
    }
    hth_(i, 0) = hth_(i, 1) = hth_(i, 2) = 0.0;
    s0_(i, 0) = s0_(i, 1) = s0_(i, 2) = 0.0;
    if (!(mask_(i) & groupbit_local)) return;

    const double mag = static_cast<double>(sp_(i, 3));
    if (mag > FixLLGGeomNH::SPIN_EPS) {
      const double dirn =
          kk_norm3(static_cast<double>(sp_(i, 0)), static_cast<double>(sp_(i, 1)), static_cast<double>(sp_(i, 2)));
      if (dirn > FixLLGGeomNH::SPIN_EPS) {
        const double scale = mag / dirn;
        s0_(i, 0) = static_cast<double>(sp_(i, 0)) * scale;
        s0_(i, 1) = static_cast<double>(sp_(i, 1)) * scale;
        s0_(i, 2) = static_cast<double>(sp_(i, 2)) * scale;
      }
    }

    const double mu_s = kk_norm3(s0_(i, 0), s0_(i, 1), s0_(i, 2));
    if (prefactor_base > 0.0 && mu_s > KK_THERMAL_EPS) {
      const double sigma = sqrt(prefactor_base / (g_over_hbar_ * mu_s * dt));
      const tagint ti = tag_(i) ? tag_(i) : static_cast<tagint>(i + 1);
      hth_(i, 0) = sigma * kk_gaussian_u64(seed64, ti, step64, 0);
      hth_(i, 1) = sigma * kk_gaussian_u64(seed64, ti, step64, 1);
      hth_(i, 2) = sigma * kk_gaussian_u64(seed64, ti, step64, 2);
    }
  });

  atomKK->modified(execution_space, X_MASK | V_MASK | F_MASK | SP_MASK);
  k_x0_device.template modify<DeviceType>();
  k_v0_device.template modify<DeviceType>();
  k_f0_device.template modify<DeviceType>();
  k_s0_device.template modify<DeviceType>();
  k_h_th_device.template modify<DeviceType>();
  k_r_mid_guess_device.template modify<DeviceType>();
  k_r_new_device.template modify<DeviceType>();
  k_v_new_device.template modify<DeviceType>();
  k_e_mid_guess_device.template modify<DeviceType>();
  k_e_pred_device.template modify<DeviceType>();
  k_f_mid_device.template modify<DeviceType>();
  k_h_mid_device.template modify<DeviceType>();
  k_e_new_device.template modify<DeviceType>();
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::build_predictor_midpoint_state()
{
  if (!this->fm_cache) this->ensure_custom_peratom();
  this->ensure_solver_arrays(this->atom->nmax);
  if (!ensure_device_scratch()) error->all(FLERR, "Fix {} could not allocate llggeom Kokkos scratch arrays", this->style);

  this->spin_temperature_cache_valid = 0;
  const double temp_use = this->compute_spin_temperature();
  const double dt = this->update->dt;
  const double prefactor_base =
      (this->alpha > 0.0 && temp_use > 0.0 && this->g_over_hbar > 0.0 && dt > 0.0) ?
      (2.0 * this->alpha * this->force->boltz * temp_use) :
      0.0;

  const int nlocal = (this->igroup == this->atom->firstgroup) ? this->atom->nfirst : this->atom->nlocal;
  int need = X_MASK | V_MASK | F_MASK | MASK_MASK | TYPE_MASK | TAG_MASK | SP_MASK;
  if (this->atom->rmass) need |= RMASS_MASK;
  atomKK->sync(execution_space, need);
  atomKK->k_mass.template modify<LMPHostType>();
  atomKK->k_mass.template sync<DeviceType>();

  mask_view = atomKK->k_mask.template view<DeviceType>();
  tag_view = atomKK->k_tag.template view<DeviceType>();
  type_view = atomKK->k_type.template view<DeviceType>();
  mass_type_view = atomKK->k_mass.template view<DeviceType>();
  x_view = atomKK->k_x.template view<DeviceType>();
  v_view = atomKK->k_v.template view<DeviceType>();
  f_view = atomKK->k_f.template view<DeviceType>();
  sp_view = atomKK->k_sp.template view<DeviceType>();
  auto rmass_view = atomKK->k_rmass.template view<DeviceType>();

  auto mask_ = mask_view;
  auto tag_ = tag_view;
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
  auto rmid_ = d_r_mid_guess_device;
  auto emid_ = d_e_mid_guess_device;
  auto epred_ = d_e_pred_device;
  auto hth_ = d_h_th_device;
  auto fm_cache_ = d_fm_cache_device;
  const int groupbit_local = this->groupbit;
  const int lattice_mode = this->lattice_flag;
  const int has_rmass = (this->atom->rmass != nullptr);
  const double alpha_ = this->alpha;
  const double g_over_hbar_ = this->g_over_hbar;
  const double ftm2v = this->force->ftm2v;
  const std::uint64_t seed64 = static_cast<std::uint64_t>(this->seed);
  const std::uint64_t step64 = static_cast<std::uint64_t>(this->update->ntimestep);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    for (int k = 0; k < 3; k++) {
      x0_(i, k) = static_cast<double>(x_(i, k));
      v0_(i, k) = static_cast<double>(v_(i, k));
      f0_(i, k) = static_cast<double>(f_(i, k));
    }
    hth_(i, 0) = hth_(i, 1) = hth_(i, 2) = 0.0;
    epred_(i, 0) = epred_(i, 1) = epred_(i, 2) = 0.0;
    emid_(i, 0) = emid_(i, 1) = emid_(i, 2) = 0.0;
    s0_(i, 0) = s0_(i, 1) = s0_(i, 2) = 0.0;
    if (!(mask_(i) & groupbit_local)) return;

    const double mag = static_cast<double>(sp_(i, 3));
    if (mag > FixLLGGeomNH::SPIN_EPS) {
      const double dirn =
          kk_norm3(static_cast<double>(sp_(i, 0)), static_cast<double>(sp_(i, 1)), static_cast<double>(sp_(i, 2)));
      if (dirn > FixLLGGeomNH::SPIN_EPS) {
        const double scale = mag / dirn;
        s0_(i, 0) = static_cast<double>(sp_(i, 0)) * scale;
        s0_(i, 1) = static_cast<double>(sp_(i, 1)) * scale;
        s0_(i, 2) = static_cast<double>(sp_(i, 2)) * scale;
      }
    }

    const double mu_s = kk_norm3(s0_(i, 0), s0_(i, 1), s0_(i, 2));
    if (prefactor_base > 0.0 && mu_s > KK_THERMAL_EPS) {
      const double sigma = sqrt(prefactor_base / (g_over_hbar_ * mu_s * dt));
      const tagint ti = tag_(i) ? tag_(i) : static_cast<tagint>(i + 1);
      hth_(i, 0) = sigma * kk_gaussian_u64(seed64, ti, step64, 0);
      hth_(i, 1) = sigma * kk_gaussian_u64(seed64, ti, step64, 1);
      hth_(i, 2) = sigma * kk_gaussian_u64(seed64, ti, step64, 2);
    }

    if (lattice_mode) {
      const double mass = has_rmass ? static_cast<double>(rmass_view(i)) : static_cast<double>(mass_type_(type_(i)));
      const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
      for (int k = 0; k < 3; k++) {
        rmid_(i, k) = x0_(i, k) + 0.5 * dt * v0_(i, k) + 0.125 * dt * dt * ftm2v * f0_(i, k) * inv_mass;
        x_(i, k) = rmid_(i, k);
      }
    }

    if (mu_s > FixLLGGeomNH::SPIN_EPS) {
      double ex = s0_(i, 0) / mu_s;
      double ey = s0_(i, 1) / mu_s;
      double ez = s0_(i, 2) / mu_s;
      kk_spin_map(ex, ey, ez, fm_cache_(i, 0) + hth_(i, 0), fm_cache_(i, 1) + hth_(i, 1), fm_cache_(i, 2) + hth_(i, 2),
                  0.5 * dt, alpha_, g_over_hbar_);
      epred_(i, 0) = ex;
      epred_(i, 1) = ey;
      epred_(i, 2) = ez;
      double mx = s0_(i, 0) / mu_s + ex;
      double my = s0_(i, 1) / mu_s + ey;
      double mz = s0_(i, 2) / mu_s + ez;
      kk_normalize3(mx, my, mz);
      emid_(i, 0) = mx;
      emid_(i, 1) = my;
      emid_(i, 2) = mz;
      sp_(i, 0) = static_cast<X_FLOAT>(mx);
      sp_(i, 1) = static_cast<X_FLOAT>(my);
      sp_(i, 2) = static_cast<X_FLOAT>(mz);
      sp_(i, 3) = static_cast<X_FLOAT>(mu_s);
    } else {
      sp_(i, 0) = sp_(i, 1) = sp_(i, 2) = sp_(i, 3) = 0.0;
    }
  });

  atomKK->modified(execution_space, X_MASK | SP_MASK);
  k_x0_device.template modify<DeviceType>();
  k_v0_device.template modify<DeviceType>();
  k_f0_device.template modify<DeviceType>();
  k_s0_device.template modify<DeviceType>();
  k_h_th_device.template modify<DeviceType>();
  k_r_mid_guess_device.template modify<DeviceType>();
  k_e_mid_guess_device.template modify<DeviceType>();
  k_e_pred_device.template modify<DeviceType>();
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::clear_force_arrays_device()
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
void FixLLGGeomNHKokkos<DeviceType>::rebuild_neighbors_for_current_positions_device()
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
void FixLLGGeomNHKokkos<DeviceType>::recompute_force_and_field_device(int eflag, int vflag)
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
  replay_external_spin_fields_device(vflag);
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::cache_current_fm_device()
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
void FixLLGGeomNHKokkos<DeviceType>::replay_external_spin_fields_device(int vflag)
{
  if (replay_fixes.empty()) return;

  bool have_host_kk_fixes = false;
  bool have_device_kk_fixes = false;
  for (auto *fix : replay_fixes) {
    if (utils::strmatch(fix->style, "^.*/kk/host"))
      have_host_kk_fixes = true;
    else
      have_device_kk_fixes = true;
  }

  if (have_host_kk_fixes) {
    atomKK->sync(Host, FM_MASK);
    for (auto *fix : replay_fixes) {
      if (!utils::strmatch(fix->style, "^.*/kk/host")) continue;
      fix->post_force(vflag);
    }
    atomKK->modified(Host, FM_MASK);
    atomKK->sync(execution_space, FM_MASK);
  }

  if (have_device_kk_fixes) {
    for (auto *fix : replay_fixes) {
      if (utils::strmatch(fix->style, "^.*/kk/host")) continue;
      fix->post_force(vflag);
    }
    atomKK->modified(execution_space, FM_MASK);
  }
}

template <class DeviceType>
bool FixLLGGeomNHKokkos<DeviceType>::solve_spin_midpoint_device(bool lattice_mode, int vflag, double pe_mid)
{
  if (midpoint_iter <= 1) return false;
  if (lattice_mode != (lattice_flag != 0)) return false;
  if (!ensure_device_scratch()) return false;

  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const int groupbit_local = this->groupbit;
  const double dt = update->dt;
  const double tol_r = midpoint_tol_r * dt * dt;
  const double tol_e = midpoint_tol_e;
  const double omega = midpoint_relax;
  const int update_lattice = 0;

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
  auto rmid_ = d_r_mid_guess_device;
  auto emid_ = d_e_mid_guess_device;
  auto fmid_ = d_f_mid_device;
  auto hmid_ = d_h_mid_device;
  auto hth_ = d_h_th_device;
  auto rnew_ = d_r_new_device;
  auto vnew_ = d_v_new_device;
  auto enew_ = d_e_new_device;
  const int has_rmass = (atom->rmass != nullptr);
  const double alpha_ = alpha;
  const double g_over_hbar_ = g_over_hbar;
  const double ftm2v = force->ftm2v;

  for (int iter = 0; iter < midpoint_iter; iter++) {
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
      if (!(mask_(i) & groupbit_local)) return;

      fmid_(i, 0) = static_cast<double>(f_(i, 0));
      fmid_(i, 1) = static_cast<double>(f_(i, 1));
      fmid_(i, 2) = static_cast<double>(f_(i, 2));
      hmid_(i, 0) = static_cast<double>(fm_(i, 0)) + hth_(i, 0);
      hmid_(i, 1) = static_cast<double>(fm_(i, 1)) + hth_(i, 1);
      hmid_(i, 2) = static_cast<double>(fm_(i, 2)) + hth_(i, 2);

      if (update_lattice) {
        const double mass = has_rmass ? static_cast<double>(rmass_view(i)) : static_cast<double>(mass_type_(type_(i)));
        const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
        for (int k = 0; k < 3; k++) {
          vnew_(i, k) = v0_(i, k) + dt * ftm2v * fmid_(i, k) * inv_mass;
          rnew_(i, k) = x0_(i, k) + 0.5 * dt * (v0_(i, k) + vnew_(i, k));
          rmid_(i, k) = 0.5 * (x0_(i, k) + rnew_(i, k));
        }
      } else {
        for (int k = 0; k < 3; k++) {
          vnew_(i, k) = v0_(i, k);
          rnew_(i, k) = x0_(i, k);
          rmid_(i, k) = x0_(i, k);
        }
      }

      double ex = s0_(i, 0);
      double ey = s0_(i, 1);
      double ez = s0_(i, 2);
      const double smag = kk_norm3(ex, ey, ez);
      if (smag > FixLLGGeomNH::SPIN_EPS) {
        ex /= smag;
        ey /= smag;
        ez /= smag;
        kk_spin_map(ex, ey, ez, hmid_(i, 0), hmid_(i, 1), hmid_(i, 2), dt, alpha_, g_over_hbar_);
        enew_(i, 0) = ex;
        enew_(i, 1) = ey;
        enew_(i, 2) = ez;
        double mx = s0_(i, 0) / smag + ex;
        double my = s0_(i, 1) / smag + ey;
        double mz = s0_(i, 2) / smag + ez;
        kk_normalize3(mx, my, mz);
        emid_(i, 0) = mx;
        emid_(i, 1) = my;
        emid_(i, 2) = mz;
      } else {
        enew_(i, 0) = enew_(i, 1) = enew_(i, 2) = 0.0;
        emid_(i, 0) = emid_(i, 1) = emid_(i, 2) = 0.0;
      }
    });

    double rr_local = 0.0;
    double re_local = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType>(0, nlocal),
        LAMMPS_LAMBDA(const int i, double &maxv) {
          if (!(mask_(i) & groupbit_local) || !update_lattice) return;
          const double dx = rmid_(i, 0) - static_cast<double>(x_(i, 0));
          const double dy = rmid_(i, 1) - static_cast<double>(x_(i, 1));
          const double dz = rmid_(i, 2) - static_cast<double>(x_(i, 2));
          const double dr = kk_norm3(dx, dy, dz);
          if (dr > maxv) maxv = dr;
        },
        Kokkos::Max<double>(rr_local));
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<DeviceType>(0, nlocal),
        LAMMPS_LAMBDA(const int i, double &maxv) {
          if (!(mask_(i) & groupbit_local)) return;
          const double en = kk_norm3(emid_(i, 0), emid_(i, 1), emid_(i, 2));
          const double sn = kk_norm3(static_cast<double>(sp_(i, 0)), static_cast<double>(sp_(i, 1)),
                                     static_cast<double>(sp_(i, 2)));
          double de = 0.0;
          if (en > KK_SPIN_EPS && sn > KK_SPIN_EPS) {
            double dot = (emid_(i, 0) * static_cast<double>(sp_(i, 0)) + emid_(i, 1) * static_cast<double>(sp_(i, 1)) +
                          emid_(i, 2) * static_cast<double>(sp_(i, 2))) /
                (en * sn);
            dot = fmax(-1.0, fmin(1.0, dot));
            de = sqrt(fmax(0.0, 0.5 * (1.0 - dot)));
          } else {
            const double dsx = emid_(i, 0) - static_cast<double>(sp_(i, 0));
            const double dsy = emid_(i, 1) - static_cast<double>(sp_(i, 1));
            const double dsz = emid_(i, 2) - static_cast<double>(sp_(i, 2));
            de = kk_norm3(dsx, dsy, dsz);
          }
          if (de > maxv) maxv = de;
        },
        Kokkos::Max<double>(re_local));

    double rr_all = rr_local;
    double re_all = re_local;
    MPI_Allreduce(&rr_local, &rr_all, 1, MPI_DOUBLE, MPI_MAX, world);
    MPI_Allreduce(&re_local, &re_all, 1, MPI_DOUBLE, MPI_MAX, world);
    if ((iter == midpoint_iter - 1) || ((rr_all <= tol_r) && (re_all <= tol_e))) break;

    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
      if (!(mask_(i) & groupbit_local)) return;
      if (update_lattice) {
        for (int k = 0; k < 3; k++) x_(i, k) = omega * rmid_(i, k) + (1.0 - omega) * static_cast<double>(x_(i, k));
      }

      double mx = omega * emid_(i, 0) + (1.0 - omega) * static_cast<double>(sp_(i, 0));
      double my = omega * emid_(i, 1) + (1.0 - omega) * static_cast<double>(sp_(i, 1));
      double mz = omega * emid_(i, 2) + (1.0 - omega) * static_cast<double>(sp_(i, 2));
      kk_normalize3(mx, my, mz);
      const double smag = static_cast<double>(sp_(i, 3));
      sp_(i, 0) = static_cast<X_FLOAT>(mx);
      sp_(i, 1) = static_cast<X_FLOAT>(my);
      sp_(i, 2) = static_cast<X_FLOAT>(mz);
      sp_(i, 3) = static_cast<X_FLOAT>(smag);
    });
    atomKK->modified(execution_space, X_MASK | SP_MASK);
    recompute_force_and_field_device(1, 0);
  }

  // Refresh the field on the final corrected midpoint before committing the endpoint.
  // Without this extra recompute, the endpoint uses atom->fm from the previous midpoint
  // iterate, which introduces a systematic one-iteration lag.
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!(mask_(i) & groupbit_local)) return;
    if (update_lattice) {
      x_(i, 0) = rmid_(i, 0);
      x_(i, 1) = rmid_(i, 1);
      x_(i, 2) = rmid_(i, 2);
    }

    const double smag = kk_norm3(s0_(i, 0), s0_(i, 1), s0_(i, 2));
    if (smag > FixLLGGeomNH::SPIN_EPS) {
      sp_(i, 0) = static_cast<X_FLOAT>(emid_(i, 0));
      sp_(i, 1) = static_cast<X_FLOAT>(emid_(i, 1));
      sp_(i, 2) = static_cast<X_FLOAT>(emid_(i, 2));
      sp_(i, 3) = static_cast<X_FLOAT>(smag);
    } else {
      sp_(i, 0) = sp_(i, 1) = sp_(i, 2) = sp_(i, 3) = 0.0;
    }
  });
  atomKK->modified(execution_space, X_MASK | SP_MASK);
  recompute_force_and_field_device(1, 0);

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!(mask_(i) & groupbit_local)) return;

    fmid_(i, 0) = static_cast<double>(f_(i, 0));
    fmid_(i, 1) = static_cast<double>(f_(i, 1));
    fmid_(i, 2) = static_cast<double>(f_(i, 2));
    hmid_(i, 0) = static_cast<double>(fm_(i, 0)) + hth_(i, 0);
    hmid_(i, 1) = static_cast<double>(fm_(i, 1)) + hth_(i, 1);
    hmid_(i, 2) = static_cast<double>(fm_(i, 2)) + hth_(i, 2);

    if (update_lattice) {
      const double mass = has_rmass ? static_cast<double>(rmass_view(i)) : static_cast<double>(mass_type_(type_(i)));
      const double inv_mass = (mass > 0.0) ? (1.0 / mass) : 0.0;
      for (int k = 0; k < 3; k++) {
        vnew_(i, k) = v0_(i, k) + dt * ftm2v * fmid_(i, k) * inv_mass;
        rnew_(i, k) = x0_(i, k) + 0.5 * dt * (v0_(i, k) + vnew_(i, k));
        rmid_(i, k) = 0.5 * (x0_(i, k) + rnew_(i, k));
      }
    } else {
      for (int k = 0; k < 3; k++) {
        vnew_(i, k) = v0_(i, k);
        rnew_(i, k) = x0_(i, k);
        rmid_(i, k) = x0_(i, k);
      }
    }

    double ex = s0_(i, 0);
    double ey = s0_(i, 1);
    double ez = s0_(i, 2);
    const double smag = kk_norm3(ex, ey, ez);
    if (smag > FixLLGGeomNH::SPIN_EPS) {
      ex /= smag;
      ey /= smag;
      ez /= smag;
      kk_spin_map(ex, ey, ez, hmid_(i, 0), hmid_(i, 1), hmid_(i, 2), dt, alpha_, g_over_hbar_);
      enew_(i, 0) = ex;
      enew_(i, 1) = ey;
      enew_(i, 2) = ez;
      double mx = s0_(i, 0) / smag + ex;
      double my = s0_(i, 1) / smag + ey;
      double mz = s0_(i, 2) / smag + ez;
      kk_normalize3(mx, my, mz);
      emid_(i, 0) = mx;
      emid_(i, 1) = my;
      emid_(i, 2) = mz;
    } else {
      enew_(i, 0) = enew_(i, 1) = enew_(i, 2) = 0.0;
      emid_(i, 0) = emid_(i, 1) = emid_(i, 2) = 0.0;
    }
  });

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, nlocal), LAMMPS_LAMBDA(const int i) {
    if (!(mask_(i) & groupbit_local)) return;
    if (update_lattice) {
      for (int k = 0; k < 3; k++) {
        x_(i, k) = rnew_(i, k);
        v_(i, k) = vnew_(i, k);
      }
    }
    const double smag = kk_norm3(s0_(i, 0), s0_(i, 1), s0_(i, 2));
    if (smag > FixLLGGeomNH::SPIN_EPS) {
      sp_(i, 0) = static_cast<X_FLOAT>(enew_(i, 0));
      sp_(i, 1) = static_cast<X_FLOAT>(enew_(i, 1));
      sp_(i, 2) = static_cast<X_FLOAT>(enew_(i, 2));
      sp_(i, 3) = static_cast<X_FLOAT>(smag);
    } else {
      sp_(i, 0) = sp_(i, 1) = sp_(i, 2) = sp_(i, 3) = 0.0;
    }
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
void FixLLGGeomNHKokkos<DeviceType>::post_force(int vflag)
{
  const double pe_mid = (update->eflag_global == update->ntimestep) ? current_pe_total() : 0.0;
  if (!solve_spin_midpoint_device(lattice_flag != 0, vflag, pe_mid))
    error->all(FLERR, "Fix {} failed to execute llggeom midpoint solve on Kokkos", style);
}

template <class DeviceType>
void FixLLGGeomNHKokkos<DeviceType>::final_integrate()
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
template class FixLLGGeomNHKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixLLGGeomNHKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

#endif
