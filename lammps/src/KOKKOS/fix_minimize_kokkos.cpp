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

#include "fix_minimize_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "memory_kokkos.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMinimizeKokkos::FixMinimizeKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixMinimize(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;

  // This fix owns atom-based arrays (minimizer vectors) which must migrate with
  // atoms during reneighboring. Enable Kokkos device exchange/border comm by
  // providing pack/unpack kernels for these arrays.
  exchange_comm_device = 1;
  maxexchange_dynamic = 1;
}

/* ---------------------------------------------------------------------- */

FixMinimizeKokkos::~FixMinimizeKokkos()
{
  memoryKK->destroy_kokkos(k_vectors,vectors);
  vectors = nullptr;
}

/* ----------------------------------------------------------------------
   allocate/initialize memory for a new vector with 3 elements per atom
------------------------------------------------------------------------- */

void FixMinimizeKokkos::add_vector_kokkos()
{
  int n = 3;

  memory->grow(peratom,nvector+1,"minimize:peratom");
  peratom[nvector] = n;

  // d_vectors needs to be LayoutRight for subviews

  k_vectors.sync<LMPDeviceType>();

  memoryKK->grow_kokkos(k_vectors,vectors,nvector+1,atom->nmax*n,
                      "minimize:vectors");
  d_vectors = k_vectors.d_view;
  h_vectors = k_vectors.h_view;

  k_vectors.modify<LMPDeviceType>();

  nvector++;
  maxexchange = 3 * nvector;
}

/* ----------------------------------------------------------------------
   return a pointer to the Mth vector
------------------------------------------------------------------------- */

DAT::t_ffloat_1d FixMinimizeKokkos::request_vector_kokkos(int m)
{
  k_vectors.sync<LMPDeviceType>();

  return Kokkos::subview(d_vectors,m,Kokkos::ALL);
}

/* ----------------------------------------------------------------------
   Kokkos exchange pack/unpack for atom-based minimizer vectors
   called by CommKokkos::exchange_device() for fixes in atom->extra_grow
------------------------------------------------------------------------- */

template<class DeviceType>
struct FixMinimizeKokkos_PackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_float_2d _vectors;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d _sendlist;
  typename AT::t_int_1d _copylist;
  int _nvector;

  FixMinimizeKokkos_PackExchangeFunctor(const DAT::tdual_float_2d &vectors,
                                       const DAT::tdual_xfloat_2d &buf,
                                       const DAT::tdual_int_1d &sendlist,
                                       const DAT::tdual_int_1d &copylist,
                                       int nvector) :
      _vectors(vectors.view<DeviceType>()),
      _sendlist(sendlist.view<DeviceType>()),
      _copylist(copylist.view<DeviceType>()),
      _nvector(nvector)
  {
    const int peratom = 3 * _nvector;
    const int maxsendlist = (buf.view<DeviceType>().extent(0) * buf.view<DeviceType>().extent(1)) / peratom;
    buffer_view<DeviceType>(_buf, buf, maxsendlist, peratom);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &mysend) const
  {
    const int i = _sendlist(mysend);
    int col = 0;
    for (int m = 0; m < _nvector; m++) {
      const int base = 3 * i;
      _buf(mysend, col++) = _vectors(m, base + 0);
      _buf(mysend, col++) = _vectors(m, base + 1);
      _buf(mysend, col++) = _vectors(m, base + 2);
    }

    const int j = _copylist(mysend);
    if (j > -1) {
      for (int m = 0; m < _nvector; m++) {
        const int ib = 3 * i;
        const int jb = 3 * j;
        _vectors(m, ib + 0) = _vectors(m, jb + 0);
        _vectors(m, ib + 1) = _vectors(m, jb + 1);
        _vectors(m, ib + 2) = _vectors(m, jb + 2);
      }
    }
  }
};

int FixMinimizeKokkos::pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &k_buf,
                                           DAT::tdual_int_1d k_sendlist,
                                           DAT::tdual_int_1d k_copylist,
                                           ExecutionSpace space)
{
  const int peratom = 3 * nvector;
  if (nsend == 0 || peratom == 0) return 0;

  // ensure buffer is large enough for nsend atoms
  if (nsend > (int) (k_buf.view<LMPHostType>().extent(0) * k_buf.view<LMPHostType>().extent(1)) / peratom) {
    const int newsize = nsend * peratom / (int) k_buf.view<LMPHostType>().extent(1) + 1;
    k_buf.resize(newsize, k_buf.view<LMPHostType>().extent(1));
  }

  if (space == Host) {
    k_vectors.sync<LMPHostType>();
    FixMinimizeKokkos_PackExchangeFunctor<LMPHostType>
      f(k_vectors, k_buf, k_sendlist, k_copylist, nvector);
    Kokkos::parallel_for(nsend, f);
    k_vectors.modify<LMPHostType>();
  } else {
    k_vectors.sync<LMPDeviceType>();
    FixMinimizeKokkos_PackExchangeFunctor<LMPDeviceType>
      f(k_vectors, k_buf, k_sendlist, k_copylist, nvector);
    Kokkos::parallel_for(nsend, f);
    k_vectors.modify<LMPDeviceType>();
  }

  return nsend * peratom;
}

template<class DeviceType>
struct FixMinimizeKokkos_UnpackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  typename AT::t_float_2d _vectors;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d _indices;
  int _nvector;

  FixMinimizeKokkos_UnpackExchangeFunctor(const DAT::tdual_float_2d &vectors,
                                         const DAT::tdual_xfloat_2d &buf,
                                         const DAT::tdual_int_1d &indices,
                                         int nvector) :
      _vectors(vectors.view<DeviceType>()),
      _indices(indices.view<DeviceType>()),
      _nvector(nvector)
  {
    const int peratom = 3 * _nvector;
    const int maxsendlist = (buf.view<DeviceType>().extent(0) * buf.view<DeviceType>().extent(1)) / peratom;
    buffer_view<DeviceType>(_buf, buf, maxsendlist, peratom);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &myrecv) const
  {
    const int i = _indices(myrecv);
    if (i < 0) return;

    int col = 0;
    for (int m = 0; m < _nvector; m++) {
      const int base = 3 * i;
      _vectors(m, base + 0) = _buf(myrecv, col++);
      _vectors(m, base + 1) = _buf(myrecv, col++);
      _vectors(m, base + 2) = _buf(myrecv, col++);
    }
  }
};

void FixMinimizeKokkos::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d &indices,
                                              int nrecv, int /*nrecv1*/, int /*nextrarecv1*/,
                                              ExecutionSpace space)
{
  const int peratom = 3 * nvector;
  if (nrecv == 0 || peratom == 0) return;

  if (space == Host) {
    k_vectors.sync<LMPHostType>();
    indices.sync<LMPHostType>();
    FixMinimizeKokkos_UnpackExchangeFunctor<LMPHostType> f(k_vectors, k_buf, indices, nvector);
    Kokkos::parallel_for(nrecv, f);
    k_vectors.modify<LMPHostType>();
  } else {
    k_vectors.sync<LMPDeviceType>();
    indices.sync<LMPDeviceType>();
    FixMinimizeKokkos_UnpackExchangeFunctor<LMPDeviceType> f(k_vectors, k_buf, indices, nvector);
    Kokkos::parallel_for(nrecv, f);
    k_vectors.modify<LMPDeviceType>();
  }
}

/* ----------------------------------------------------------------------
   reset x0 for atoms that moved across PBC via reneighboring in line search
   x0 = 1st vector
   must do minimum_image using original box stored at beginning of line search
   swap & set_global_box() change to original box, then restore current box
------------------------------------------------------------------------- */

void FixMinimizeKokkos::reset_coords()
{
  box_swap();
  domain->set_global_box();

  int nlocal = atom->nlocal;

  atomKK->sync(Device,X_MASK);
  k_vectors.sync<LMPDeviceType>();

  {
    // local variables for lambda capture

    auto triclinic = domain->triclinic;
    auto xperiodic = domain->xperiodic;
    auto xprd_half = domain->xprd_half;
    auto xprd = domain->xprd;
    auto yperiodic = domain->yperiodic;
    auto yprd_half = domain->yprd_half;
    auto yprd = domain->yprd;
    auto zperiodic = domain->zperiodic;
    auto zprd_half = domain->zprd_half;
    auto zprd = domain->zprd;
    auto xy = domain->xy;
    auto xz = domain->xz;
    auto yz = domain->yz;
    auto l_x = atomKK->k_x.d_view;
    auto l_x0 = Kokkos::subview(d_vectors,0,Kokkos::ALL);

    Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int& i) {
      const int n = i*3;
      double dx0 = l_x(i,0) - l_x0[n];
      double dy0 = l_x(i,1) - l_x0[n+1];
      double dz0 = l_x(i,2) - l_x0[n+2];
      double dx = dx0;
      double dy = dy0;
      double dz = dz0;
      // domain->minimum_image(FLERR, dx,dy,dz);
      {
        if (triclinic == 0) {
          if (xperiodic) {
            if (fabs(dx) > xprd_half) {
              if (dx < 0.0) dx += xprd;
              else dx -= xprd;
            }
          }
          if (yperiodic) {
            if (fabs(dy) > yprd_half) {
              if (dy < 0.0) dy += yprd;
              else dy -= yprd;
            }
          }
          if (zperiodic) {
            if (fabs(dz) > zprd_half) {
              if (dz < 0.0) dz += zprd;
              else dz -= zprd;
            }
          }

        } else {
          if (zperiodic) {
            if (fabs(dz) > zprd_half) {
              if (dz < 0.0) {
                dz += zprd;
                dy += yz;
                dx += xz;
              } else {
                dz -= zprd;
                dy -= yz;
                dx -= xz;
              }
            }
          }
          if (yperiodic) {
            if (fabs(dy) > yprd_half) {
              if (dy < 0.0) {
                dy += yprd;
                dx += xy;
              } else {
                dy -= yprd;
                dx -= xy;
              }
            }
          }
          if (xperiodic) {
            if (fabs(dx) > xprd_half) {
              if (dx < 0.0) dx += xprd;
              else dx -= xprd;
            }
          }
        }
      } // end domain->minimum_image(FLERR, dx,dy,dz);
      if (dx != dx0) l_x0[n] = l_x(i,0) - dx;
      if (dy != dy0) l_x0[n+1] = l_x(i,1) - dy;
      if (dz != dz0) l_x0[n+2] = l_x(i,2) - dz;
    });
  }
  k_vectors.modify<LMPDeviceType>();

  box_swap();
  domain->set_global_box();
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixMinimizeKokkos::grow_arrays(int nmax)
{
  k_vectors.sync<LMPDeviceType>();
  memoryKK->grow_kokkos(k_vectors,vectors,nvector,3*nmax,"minimize:vector");
  d_vectors = k_vectors.d_view;
  h_vectors = k_vectors.h_view;
  k_vectors.modify<LMPDeviceType>();
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixMinimizeKokkos::copy_arrays(int i, int j, int /*delflag*/)
{
  int m,iper,nper,ni,nj;

  k_vectors.sync<LMPHostType>();

  for (m = 0; m < nvector; m++) {
    nper = 3;
    ni = nper*i;
    nj = nper*j;
    for (iper = 0; iper < nper; iper++) h_vectors(m,nj++) = h_vectors(m,ni++);
  }

  k_vectors.modify<LMPHostType>();
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixMinimizeKokkos::pack_exchange(int i, double *buf)
{
  int m,iper,nper,ni;

  k_vectors.sync<LMPHostType>();

  int n = 0;
  for (m = 0; m < nvector; m++) {
    nper = peratom[m];
    ni = nper*i;
    for (iper = 0; iper < nper; iper++) buf[n++] = h_vectors(m,ni++);
  }
  return n;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixMinimizeKokkos::unpack_exchange(int nlocal, double *buf)
{
  int m,iper,nper,ni;

  k_vectors.sync<LMPHostType>();

  int n = 0;
  for (m = 0; m < nvector; m++) {
    nper = peratom[m];
    ni = nper*nlocal;
    for (iper = 0; iper < nper; iper++) h_vectors(m,ni++) = buf[n++];
  }

  k_vectors.modify<LMPHostType>();

  return n;
}
