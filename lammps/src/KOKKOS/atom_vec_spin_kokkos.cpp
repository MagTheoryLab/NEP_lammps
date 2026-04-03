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

/* ------------------------------------------------------------------------
   Contributing authors: Julien Tranchida (SNL)
                         Aidan Thompson (SNL)

   Please cite the related publication:
   Tranchida, J., Plimpton, S. J., Thibaudeau, P., & Thompson, A. P. (2018).
   Massively parallel symplectic algorithm for coupled magnetic spin dynamics
   and molecular dynamics. Journal of Computational Physics.
------------------------------------------------------------------------- */

#include "atom_vec_spin_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "modify.h"

using namespace LAMMPS_NS;

static constexpr int DELTA = 10;

/* ---------------------------------------------------------------------- */

AtomVecSpinKokkos::AtomVecSpinKokkos(LAMMPS *lmp) : AtomVec(lmp),
AtomVecKokkos(lmp), AtomVecSpin(lmp)
{
  // Required so CommKokkos can keep device-side exchange/border communication
  // when fixes have atom-based arrays (atom->nextra_grow).
  unpack_exchange_indices_flag = 1;
  // Atom style spin has extra reverse-comm fields (fm/fm_long) and provides
  // Kokkos pack/unpack kernels for them, so CommKokkos can use device reverse comm.
  reverse_comm_kokkos_extra_flag = 1;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
------------------------------------------------------------------------- */

void AtomVecSpinKokkos::grow(int n)
{
  int step = MAX(DELTA,nmax*0.01);
  if (n == 0) nmax += step;
  else nmax = n;
  atomKK->nmax = nmax;
  if (nmax < 0 || nmax > MAXSMALLINT)
    error->one(FLERR,"Per-processor system is too big");

  atomKK->sync(Device,ALL_MASK);
  atomKK->modified(Device,ALL_MASK);

  memoryKK->grow_kokkos(atomKK->k_tag,atomKK->tag,nmax,"atom:tag");
  memoryKK->grow_kokkos(atomKK->k_type,atomKK->type,nmax,"atom:type");
  memoryKK->grow_kokkos(atomKK->k_mask,atomKK->mask,nmax,"atom:mask");
  memoryKK->grow_kokkos(atomKK->k_image,atomKK->image,nmax,"atom:image");

  // allocating mech. quantities

  memoryKK->grow_kokkos(atomKK->k_x,atomKK->x,nmax,"atom:x");
  memoryKK->grow_kokkos(atomKK->k_v,atomKK->v,nmax,"atom:v");
  memoryKK->grow_kokkos(atomKK->k_f,atomKK->f,nmax,"atom:f");

  // allocating mag. quantities

  memoryKK->grow_kokkos(atomKK->k_sp,atomKK->sp,nmax,"atom:sp");
  memoryKK->grow_kokkos(atomKK->k_fm,atomKK->fm,nmax,"atom:fm");
  memoryKK->grow_kokkos(atomKK->k_fm_long,atomKK->fm_long,nmax,"atom:fm_long");

  grow_pointers();
  atomKK->sync(Host,ALL_MASK);

  if (atom->nextra_grow)
    for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
      modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
------------------------------------------------------------------------- */

void AtomVecSpinKokkos::grow_pointers()
{
  tag = atomKK->tag;
  d_tag = atomKK->k_tag.d_view;
  h_tag = atomKK->k_tag.h_view;

  type = atomKK->type;
  d_type = atomKK->k_type.d_view;
  h_type = atomKK->k_type.h_view;
  mask = atomKK->mask;
  d_mask = atomKK->k_mask.d_view;
  h_mask = atomKK->k_mask.h_view;
  image = atomKK->image;
  d_image = atomKK->k_image.d_view;
  h_image = atomKK->k_image.h_view;

  x = atomKK->x;
  d_x = atomKK->k_x.d_view;
  h_x = atomKK->k_x.h_view;
  v = atomKK->v;
  d_v = atomKK->k_v.d_view;
  h_v = atomKK->k_v.h_view;
  f = atomKK->f;
  d_f = atomKK->k_f.d_view;
  h_f = atomKK->k_f.h_view;

  sp = atomKK->sp;
  d_sp = atomKK->k_sp.d_view;
  h_sp = atomKK->k_sp.h_view;
  fm = atomKK->fm;
  d_fm = atomKK->k_fm.d_view;
  h_fm = atomKK->k_fm.h_view;
  fm_long = atomKK->fm_long;
  d_fm_long = atomKK->k_fm_long.d_view;
  h_fm_long = atomKK->k_fm_long.h_view;
}

/* ----------------------------------------------------------------------
   sort atom arrays on device
------------------------------------------------------------------------- */

void AtomVecSpinKokkos::sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter)
{
  atomKK->sync(Device, TAG_MASK|TYPE_MASK|MASK_MASK|IMAGE_MASK|X_MASK|V_MASK|SP_MASK);

  Sorter.sort(LMPDeviceType(), d_tag);
  Sorter.sort(LMPDeviceType(), d_type);
  Sorter.sort(LMPDeviceType(), d_mask);
  Sorter.sort(LMPDeviceType(), d_image);
  Sorter.sort(LMPDeviceType(), d_x);
  Sorter.sort(LMPDeviceType(), d_v);
  Sorter.sort(LMPDeviceType(), d_sp);

  atomKK->modified(Device, TAG_MASK|TYPE_MASK|MASK_MASK|IMAGE_MASK|X_MASK|V_MASK|SP_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecSpinKokkos_PackComm {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_sp_array_randomread _sp;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_um _buf;
  typename ArrayTypes<DeviceType>::t_int_1d_const _list;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecSpinKokkos_PackComm(
      const typename DAT::tdual_x_array &x,
      const typename DAT::tdual_float_1d_4 &sp,
      const typename DAT::tdual_xfloat_2d &buf,
      const typename DAT::tdual_int_1d &list,
      const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
      const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz, const int* const pbc):
      _x(x.view<DeviceType>()),_sp(sp.view<DeviceType>()),
      _list(list.view<DeviceType>()),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz) {
        const size_t elements = 7;
        const size_t maxsend = (buf.view<DeviceType>().extent(0)*buf.view<DeviceType>().extent(1))/elements;
        buffer_view<DeviceType>(_buf,buf,maxsend,elements);
        _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
        _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
        const int j = _list(i);
      if (PBC_FLAG == 0) {
          _buf(i,0) = _x(j,0);
          _buf(i,1) = _x(j,1);
          _buf(i,2) = _x(j,2);
          _buf(i,3) = _sp(j,0);
          _buf(i,4) = _sp(j,1);
          _buf(i,5) = _sp(j,2);
          _buf(i,6) = _sp(j,3);
      } else {
        if (TRICLINIC == 0) {
          _buf(i,0) = _x(j,0) + _pbc[0]*_xprd;
          _buf(i,1) = _x(j,1) + _pbc[1]*_yprd;
          _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
          _buf(i,3) = _sp(j,0);
          _buf(i,4) = _sp(j,1);
          _buf(i,5) = _sp(j,2);
          _buf(i,6) = _sp(j,3);
        } else {
          _buf(i,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
          _buf(i,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
          _buf(i,2) = _x(j,2) + _pbc[2]*_zprd;
          _buf(i,3) = _sp(j,0);
          _buf(i,4) = _sp(j,1);
          _buf(i,5) = _sp(j,2);
          _buf(i,6) = _sp(j,3);
        }
      }
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG,int TRICLINIC>
struct AtomVecSpinKokkos_PackCommSelf {
  typedef DeviceType device_type;

  // KOKKOS self-communication note:
  // CommKokkos may use AtomVecKokkos::pack_comm_self() (and, for nprocs==1,
  // AtomVecKokkos::pack_comm_self_fused()) to build ghost atoms across periodic
  // boundaries without MPI.  The base AtomVecKokkos implementation only copies
  // coordinates.  For atom_style spin/kk, spin data (sp) must also be copied to
  // the ghost atoms; otherwise spin-aware pair styles can read stale sp values
  // on ghosts and produce nonphysical force/energy behavior (e.g. poor energy
  // conservation depending on processor grid).

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_x_array _xw;
  typename ArrayTypes<DeviceType>::t_sp_array_randomread _sp;
  typename ArrayTypes<DeviceType>::t_sp_array _spw;
  int _nfirst;
  typename ArrayTypes<DeviceType>::t_int_1d_const _list;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;
  X_FLOAT _pbc[6];

  AtomVecSpinKokkos_PackCommSelf(
      const typename DAT::tdual_x_array &x,
      const typename DAT::tdual_float_1d_4 &sp,
      const int &nfirst,
      const typename DAT::tdual_int_1d &list,
      const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
      const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz,
      const int* const pbc):
      _x(x.view<DeviceType>()),
      _xw(x.view<DeviceType>()),
      _sp(sp.view<DeviceType>()),
      _spw(sp.view<DeviceType>()),
      _nfirst(nfirst),
      _list(list.view<DeviceType>()),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz)
  {
    _pbc[0] = pbc[0]; _pbc[1] = pbc[1]; _pbc[2] = pbc[2];
    _pbc[3] = pbc[3]; _pbc[4] = pbc[4]; _pbc[5] = pbc[5];
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
    const int j = _list(i);
    if (PBC_FLAG == 0) {
      _xw(i+_nfirst,0) = _x(j,0);
      _xw(i+_nfirst,1) = _x(j,1);
      _xw(i+_nfirst,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
      } else {
        _xw(i+_nfirst,0) = _x(j,0) + _pbc[0]*_xprd + _pbc[5]*_xy + _pbc[4]*_xz;
        _xw(i+_nfirst,1) = _x(j,1) + _pbc[1]*_yprd + _pbc[3]*_yz;
        _xw(i+_nfirst,2) = _x(j,2) + _pbc[2]*_zprd;
      }
    }
    _spw(i+_nfirst,0) = _sp(j,0);
    _spw(i+_nfirst,1) = _sp(j,1);
    _spw(i+_nfirst,2) = _sp(j,2);
    _spw(i+_nfirst,3) = _sp(j,3);
  }
};

int AtomVecSpinKokkos::pack_comm_self(const int &n, const DAT::tdual_int_1d &list,
                                      const int nfirst, const int &pbc_flag, const int* const pbc)
{
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(Host, X_MASK | SP_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecSpinKokkos_PackCommSelf<LMPHostType, 1, 1> f(atomKK->k_x, atomKK->k_sp, nfirst, list,
                                                                  domain->xprd, domain->yprd, domain->zprd,
                                                                  domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      } else {
        struct AtomVecSpinKokkos_PackCommSelf<LMPHostType, 1, 0> f(atomKK->k_x, atomKK->k_sp, nfirst, list,
                                                                  domain->xprd, domain->yprd, domain->zprd,
                                                                  domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecSpinKokkos_PackCommSelf<LMPHostType, 0, 1> f(atomKK->k_x, atomKK->k_sp, nfirst, list,
                                                                  domain->xprd, domain->yprd, domain->zprd,
                                                                  domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      } else {
        struct AtomVecSpinKokkos_PackCommSelf<LMPHostType, 0, 0> f(atomKK->k_x, atomKK->k_sp, nfirst, list,
                                                                  domain->xprd, domain->yprd, domain->zprd,
                                                                  domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      }
    }
    atomKK->modified(Host, X_MASK | SP_MASK);
  } else {
    atomKK->sync(Device, X_MASK | SP_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecSpinKokkos_PackCommSelf<LMPDeviceType, 1, 1> f(atomKK->k_x, atomKK->k_sp, nfirst, list,
                                                                    domain->xprd, domain->yprd, domain->zprd,
                                                                    domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      } else {
        struct AtomVecSpinKokkos_PackCommSelf<LMPDeviceType, 1, 0> f(atomKK->k_x, atomKK->k_sp, nfirst, list,
                                                                    domain->xprd, domain->yprd, domain->zprd,
                                                                    domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecSpinKokkos_PackCommSelf<LMPDeviceType, 0, 1> f(atomKK->k_x, atomKK->k_sp, nfirst, list,
                                                                    domain->xprd, domain->yprd, domain->zprd,
                                                                    domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      } else {
        struct AtomVecSpinKokkos_PackCommSelf<LMPDeviceType, 0, 0> f(atomKK->k_x, atomKK->k_sp, nfirst, list,
                                                                    domain->xprd, domain->yprd, domain->zprd,
                                                                    domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      }
    }
    atomKK->modified(Device, X_MASK | SP_MASK);
  }

  return n * size_forward;
}

template<class DeviceType,int TRICLINIC>
struct AtomVecSpinKokkos_PackCommSelfFused {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  typename ArrayTypes<DeviceType>::t_x_array _xw;
  typename ArrayTypes<DeviceType>::t_sp_array_randomread _sp;
  typename ArrayTypes<DeviceType>::t_sp_array _spw;

  typename ArrayTypes<DeviceType>::t_int_2d_const _list;
  typename ArrayTypes<DeviceType>::t_int_2d_const _pbc;
  typename ArrayTypes<DeviceType>::t_int_1d_const _pbc_flag;
  typename ArrayTypes<DeviceType>::t_int_1d_const _firstrecv;
  typename ArrayTypes<DeviceType>::t_int_1d_const _sendnum_scan;
  typename ArrayTypes<DeviceType>::t_int_1d_const _g2l;
  X_FLOAT _xprd,_yprd,_zprd,_xy,_xz,_yz;

  AtomVecSpinKokkos_PackCommSelfFused(
      const typename DAT::tdual_x_array &x,
      const typename DAT::tdual_float_1d_4 &sp,
      const typename DAT::tdual_int_2d &list,
      const typename DAT::tdual_int_2d &pbc,
      const typename DAT::tdual_int_1d &pbc_flag,
      const typename DAT::tdual_int_1d &firstrecv,
      const typename DAT::tdual_int_1d &sendnum_scan,
      const typename DAT::tdual_int_1d &g2l,
      const X_FLOAT &xprd, const X_FLOAT &yprd, const X_FLOAT &zprd,
      const X_FLOAT &xy, const X_FLOAT &xz, const X_FLOAT &yz):
      _x(x.view<DeviceType>()),
      _xw(x.view<DeviceType>()),
      _sp(sp.view<DeviceType>()),
      _spw(sp.view<DeviceType>()),
      _list(list.view<DeviceType>()),
      _pbc(pbc.view<DeviceType>()),
      _pbc_flag(pbc_flag.view<DeviceType>()),
      _firstrecv(firstrecv.view<DeviceType>()),
      _sendnum_scan(sendnum_scan.view<DeviceType>()),
      _g2l(g2l.view<DeviceType>()),
      _xprd(xprd),_yprd(yprd),_zprd(zprd),
      _xy(xy),_xz(xz),_yz(yz) {};

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii) const {
    int iswap = 0;
    while (ii >= _sendnum_scan[iswap]) iswap++;
    int i = ii;
    if (iswap > 0) i = ii - _sendnum_scan[iswap-1];

    const int nfirst = _firstrecv[iswap];
    const int nlocal = _firstrecv[0];

    int j = _list(iswap,i);
    if (j >= nlocal) j = _g2l(j-nlocal);

    if (_pbc_flag(ii) == 0) {
      _xw(i+nfirst,0) = _x(j,0);
      _xw(i+nfirst,1) = _x(j,1);
      _xw(i+nfirst,2) = _x(j,2);
    } else {
      if (TRICLINIC == 0) {
        _xw(i+nfirst,0) = _x(j,0) + _pbc(ii,0)*_xprd;
        _xw(i+nfirst,1) = _x(j,1) + _pbc(ii,1)*_yprd;
        _xw(i+nfirst,2) = _x(j,2) + _pbc(ii,2)*_zprd;
      } else {
        _xw(i+nfirst,0) = _x(j,0) + _pbc(ii,0)*_xprd + _pbc(ii,5)*_xy + _pbc(ii,4)*_xz;
        _xw(i+nfirst,1) = _x(j,1) + _pbc(ii,1)*_yprd + _pbc(ii,3)*_yz;
        _xw(i+nfirst,2) = _x(j,2) + _pbc(ii,2)*_zprd;
      }
    }

    _spw(i+nfirst,0) = _sp(j,0);
    _spw(i+nfirst,1) = _sp(j,1);
    _spw(i+nfirst,2) = _sp(j,2);
    _spw(i+nfirst,3) = _sp(j,3);
  }
};

int AtomVecSpinKokkos::pack_comm_self_fused(const int &n, const DAT::tdual_int_2d &list,
                                           const DAT::tdual_int_1d &sendnum_scan,
                                           const DAT::tdual_int_1d &firstrecv,
                                           const DAT::tdual_int_1d &pbc_flag,
                                           const DAT::tdual_int_2d &pbc,
                                           const DAT::tdual_int_1d &g2l)
{
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(Host, X_MASK | SP_MASK);
    if (domain->triclinic) {
      struct AtomVecSpinKokkos_PackCommSelfFused<LMPHostType,1> f(atomKK->k_x, atomKK->k_sp, list, pbc, pbc_flag,
                                                                 firstrecv, sendnum_scan, g2l,
                                                                 domain->xprd, domain->yprd, domain->zprd,
                                                                 domain->xy, domain->xz, domain->yz);
      Kokkos::parallel_for(n, f);
    } else {
      struct AtomVecSpinKokkos_PackCommSelfFused<LMPHostType,0> f(atomKK->k_x, atomKK->k_sp, list, pbc, pbc_flag,
                                                                 firstrecv, sendnum_scan, g2l,
                                                                 domain->xprd, domain->yprd, domain->zprd,
                                                                 domain->xy, domain->xz, domain->yz);
      Kokkos::parallel_for(n, f);
    }
    atomKK->modified(Host, X_MASK | SP_MASK);
  } else {
    atomKK->sync(Device, X_MASK | SP_MASK);
    if (domain->triclinic) {
      struct AtomVecSpinKokkos_PackCommSelfFused<LMPDeviceType,1> f(atomKK->k_x, atomKK->k_sp, list, pbc, pbc_flag,
                                                                   firstrecv, sendnum_scan, g2l,
                                                                   domain->xprd, domain->yprd, domain->zprd,
                                                                   domain->xy, domain->xz, domain->yz);
      Kokkos::parallel_for(n, f);
    } else {
      struct AtomVecSpinKokkos_PackCommSelfFused<LMPDeviceType,0> f(atomKK->k_x, atomKK->k_sp, list, pbc, pbc_flag,
                                                                   firstrecv, sendnum_scan, g2l,
                                                                   domain->xprd, domain->yprd, domain->zprd,
                                                                   domain->xy, domain->xz, domain->yz);
      Kokkos::parallel_for(n, f);
    }
    atomKK->modified(Device, X_MASK | SP_MASK);
  }

  return n * size_forward;
}

int AtomVecSpinKokkos::pack_comm_kokkos(const int &n, const DAT::tdual_int_1d &list,
                                       const DAT::tdual_xfloat_2d &buf, const int &pbc_flag,
                                       const int *const pbc)
{
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(Host, X_MASK | SP_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecSpinKokkos_PackComm<LMPHostType, 1, 1> f(atomKK->k_x, atomKK->k_sp, buf, list,
                                                              domain->xprd, domain->yprd, domain->zprd,
                                                              domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      } else {
        struct AtomVecSpinKokkos_PackComm<LMPHostType, 1, 0> f(atomKK->k_x, atomKK->k_sp, buf, list,
                                                              domain->xprd, domain->yprd, domain->zprd,
                                                              domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecSpinKokkos_PackComm<LMPHostType, 0, 1> f(atomKK->k_x, atomKK->k_sp, buf, list,
                                                              domain->xprd, domain->yprd, domain->zprd,
                                                              domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      } else {
        struct AtomVecSpinKokkos_PackComm<LMPHostType, 0, 0> f(atomKK->k_x, atomKK->k_sp, buf, list,
                                                              domain->xprd, domain->yprd, domain->zprd,
                                                              domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      }
    }
  } else {
    atomKK->sync(Device, X_MASK | SP_MASK);
    if (pbc_flag) {
      if (domain->triclinic) {
        struct AtomVecSpinKokkos_PackComm<LMPDeviceType, 1, 1> f(atomKK->k_x, atomKK->k_sp, buf, list,
                                                                domain->xprd, domain->yprd, domain->zprd,
                                                                domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      } else {
        struct AtomVecSpinKokkos_PackComm<LMPDeviceType, 1, 0> f(atomKK->k_x, atomKK->k_sp, buf, list,
                                                                domain->xprd, domain->yprd, domain->zprd,
                                                                domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      }
    } else {
      if (domain->triclinic) {
        struct AtomVecSpinKokkos_PackComm<LMPDeviceType, 0, 1> f(atomKK->k_x, atomKK->k_sp, buf, list,
                                                                domain->xprd, domain->yprd, domain->zprd,
                                                                domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      } else {
        struct AtomVecSpinKokkos_PackComm<LMPDeviceType, 0, 0> f(atomKK->k_x, atomKK->k_sp, buf, list,
                                                                domain->xprd, domain->yprd, domain->zprd,
                                                                domain->xy, domain->xz, domain->yz, pbc);
        Kokkos::parallel_for(n, f);
      }
    }
  }

  return n * size_forward;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
struct AtomVecSpinKokkos_UnpackComm {
  using device_type = DeviceType;

  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_sp_array _sp;
  typename ArrayTypes<DeviceType>::t_xfloat_2d_const _buf;
  int _first;

  AtomVecSpinKokkos_UnpackComm(const typename DAT::tdual_x_array &x, const typename DAT::tdual_float_1d_4 &sp,
                               const typename DAT::tdual_xfloat_2d &buf, const int &first) :
      _x(x.view<DeviceType>()), _sp(sp.view<DeviceType>()), _first(first)
  {
    const size_t elements = 7;
    const int maxsend =
        (buf.template view<DeviceType>().extent(0) * buf.template view<DeviceType>().extent(1)) / elements;
    buffer_view<DeviceType>(_buf, buf, maxsend, elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const
  {
    _x(i + _first, 0) = _buf(i, 0);
    _x(i + _first, 1) = _buf(i, 1);
    _x(i + _first, 2) = _buf(i, 2);
    _sp(i + _first, 0) = _buf(i, 3);
    _sp(i + _first, 1) = _buf(i, 4);
    _sp(i + _first, 2) = _buf(i, 5);
    _sp(i + _first, 3) = _buf(i, 6);
  }
};

void AtomVecSpinKokkos::unpack_comm_kokkos(const int &n, const int &first, const DAT::tdual_xfloat_2d &buf)
{
  if (lmp->kokkos->forward_comm_on_host) {
    atomKK->sync(Host, X_MASK | SP_MASK);
    struct AtomVecSpinKokkos_UnpackComm<LMPHostType> f(atomKK->k_x, atomKK->k_sp, buf, first);
    Kokkos::parallel_for(n, f);
    atomKK->modified(Host, X_MASK | SP_MASK);
  } else {
    atomKK->sync(Device, X_MASK | SP_MASK);
    struct AtomVecSpinKokkos_UnpackComm<LMPDeviceType> f(atomKK->k_x, atomKK->k_sp, buf, first);
    Kokkos::parallel_for(n, f);
    atomKK->modified(Device, X_MASK | SP_MASK);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType,int PBC_FLAG>
struct AtomVecSpinKokkos_PackBorder {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_xfloat_2d _buf;
  const typename ArrayTypes<DeviceType>::t_int_1d_const _list;
  const typename ArrayTypes<DeviceType>::t_x_array_randomread _x;
  const typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  const typename ArrayTypes<DeviceType>::t_int_1d _type;
  const typename ArrayTypes<DeviceType>::t_int_1d _mask;
  const typename ArrayTypes<DeviceType>::t_sp_array_randomread _sp;
  X_FLOAT _dx,_dy,_dz;

  AtomVecSpinKokkos_PackBorder(
      const typename ArrayTypes<DeviceType>::t_xfloat_2d &buf,
      const typename ArrayTypes<DeviceType>::t_int_1d_const &list,
      const typename ArrayTypes<DeviceType>::t_x_array &x,
      const typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
      const typename ArrayTypes<DeviceType>::t_int_1d &type,
      const typename ArrayTypes<DeviceType>::t_int_1d &mask,
      const typename ArrayTypes<DeviceType>::t_sp_array &sp,
      const X_FLOAT &dx, const X_FLOAT &dy, const X_FLOAT &dz):
  _buf(buf),_list(list),
    _x(x),_tag(tag),_type(type),_mask(mask),_sp(sp),
    _dx(dx),_dy(dy),_dz(dz) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
      const int j = _list(i);
      if (PBC_FLAG == 0) {
          _buf(i,0) = _x(j,0);
          _buf(i,1) = _x(j,1);
          _buf(i,2) = _x(j,2);
          _buf(i,3) = d_ubuf(_tag(j)).d;
          _buf(i,4) = d_ubuf(_type(j)).d;
          _buf(i,5) = d_ubuf(_mask(j)).d;
          _buf(i,6) = _sp(j,0);
          _buf(i,7) = _sp(j,1);
          _buf(i,8) = _sp(j,2);
          _buf(i,9) = _sp(j,3);
      } else {
          _buf(i,0) = _x(j,0) + _dx;
          _buf(i,1) = _x(j,1) + _dy;
          _buf(i,2) = _x(j,2) + _dz;
          _buf(i,3) = d_ubuf(_tag(j)).d;
          _buf(i,4) = d_ubuf(_type(j)).d;
          _buf(i,5) = d_ubuf(_mask(j)).d;
          _buf(i,6) = _sp(j,0);
          _buf(i,7) = _sp(j,1);
          _buf(i,8) = _sp(j,2);
          _buf(i,9) = _sp(j,3);
      }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSpinKokkos::pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_xfloat_2d buf,
                               int pbc_flag, int *pbc, ExecutionSpace space)
{
  X_FLOAT dx,dy,dz;

  if (pbc_flag != 0) {
    if (domain->triclinic == 0) {
      dx = pbc[0]*domain->xprd;
      dy = pbc[1]*domain->yprd;
      dz = pbc[2]*domain->zprd;
    } else {
      dx = pbc[0];
      dy = pbc[1];
      dz = pbc[2];
    }
    if(space==Host) {
      AtomVecSpinKokkos_PackBorder<LMPHostType,1> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        h_x,h_tag,h_type,h_mask,h_sp,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecSpinKokkos_PackBorder<LMPDeviceType,1> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        d_x,d_tag,d_type,d_mask,d_sp,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }

  } else {
    dx = dy = dz = 0;
    if(space==Host) {
      AtomVecSpinKokkos_PackBorder<LMPHostType,0> f(
        buf.view<LMPHostType>(), k_sendlist.view<LMPHostType>(),
        h_x,h_tag,h_type,h_mask,h_sp,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    } else {
      AtomVecSpinKokkos_PackBorder<LMPDeviceType,0> f(
        buf.view<LMPDeviceType>(), k_sendlist.view<LMPDeviceType>(),
        d_x,d_tag,d_type,d_mask,d_sp,dx,dy,dz);
      Kokkos::parallel_for(n,f);
    }
  }
  return n*size_border;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSpinKokkos_UnpackBorder {
  typedef DeviceType device_type;

  const typename ArrayTypes<DeviceType>::t_xfloat_2d_const _buf;
  typename ArrayTypes<DeviceType>::t_x_array _x;
  typename ArrayTypes<DeviceType>::t_tagint_1d _tag;
  typename ArrayTypes<DeviceType>::t_int_1d _type;
  typename ArrayTypes<DeviceType>::t_int_1d _mask;
  typename ArrayTypes<DeviceType>::t_sp_array _sp;
  int _first;


  AtomVecSpinKokkos_UnpackBorder(
      const typename ArrayTypes<DeviceType>::t_xfloat_2d_const &buf,
      typename ArrayTypes<DeviceType>::t_x_array &x,
      typename ArrayTypes<DeviceType>::t_tagint_1d &tag,
      typename ArrayTypes<DeviceType>::t_int_1d &type,
      typename ArrayTypes<DeviceType>::t_int_1d &mask,
      typename ArrayTypes<DeviceType>::t_sp_array &sp,
      const int& first):
    _buf(buf),_x(x),_tag(tag),_type(type),_mask(mask),_sp(sp),_first(first){
  };

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const {
      _x(i+_first,0) = _buf(i,0);
      _x(i+_first,1) = _buf(i,1);
      _x(i+_first,2) = _buf(i,2);
      _tag(i+_first) = (tagint) d_ubuf(_buf(i,3)).i;
      _type(i+_first) = (int) d_ubuf(_buf(i,4)).i;
      _mask(i+_first) = (int) d_ubuf(_buf(i,5)).i;
      _sp(i+_first,0) = _buf(i,6);
      _sp(i+_first,1) = _buf(i,7);
      _sp(i+_first,2) = _buf(i,8);
      _sp(i+_first,3) = _buf(i,9);
  }
};

/* ---------------------------------------------------------------------- */

void AtomVecSpinKokkos::unpack_border_kokkos(const int &n, const int &first,
                     const DAT::tdual_xfloat_2d &buf,ExecutionSpace space) {
  if (first+n >= nmax) {
    grow(first+n+100);
  }
  if(space==Host) {
    struct AtomVecSpinKokkos_UnpackBorder<LMPHostType>
      f(buf.view<LMPHostType>(),h_x,h_tag,h_type,h_mask,h_sp,first);
    Kokkos::parallel_for(n,f);
  } else {
    struct AtomVecSpinKokkos_UnpackBorder<LMPDeviceType>
      f(buf.view<LMPDeviceType>(),d_x,d_tag,d_type,d_mask,d_sp,first);
    Kokkos::parallel_for(n,f);
  }
  atomKK->modified(space,X_MASK|TAG_MASK|TYPE_MASK|MASK_MASK|SP_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSpinKokkos_PackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array_randomread _x;
  typename AT::t_v_array_randomread _v;
  typename AT::t_tagint_1d_randomread _tag;
  typename AT::t_int_1d_randomread _type;
  typename AT::t_int_1d_randomread _mask;
  typename AT::t_imageint_1d_randomread _image;
  typename AT::t_sp_array_randomread _sp;
  typename AT::t_x_array _xw;
  typename AT::t_v_array _vw;
  typename AT::t_tagint_1d _tagw;
  typename AT::t_int_1d _typew;
  typename AT::t_int_1d _maskw;
  typename AT::t_imageint_1d _imagew;
  typename AT::t_sp_array _spw;

  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d_const _sendlist;
  typename AT::t_int_1d_const _copylist;
  int _size_exchange;

  AtomVecSpinKokkos_PackExchangeFunctor(
      const AtomKokkos* atom,
      const typename AT::tdual_xfloat_2d buf,
      typename AT::tdual_int_1d sendlist,
      typename AT::tdual_int_1d copylist):
    _x(atom->k_x.view<DeviceType>()),
    _v(atom->k_v.view<DeviceType>()),
    _tag(atom->k_tag.view<DeviceType>()),
    _type(atom->k_type.view<DeviceType>()),
    _mask(atom->k_mask.view<DeviceType>()),
    _image(atom->k_image.view<DeviceType>()),
    _sp(atom->k_sp.view<DeviceType>()),
    _xw(atom->k_x.view<DeviceType>()),
    _vw(atom->k_v.view<DeviceType>()),
    _tagw(atom->k_tag.view<DeviceType>()),
    _typew(atom->k_type.view<DeviceType>()),
    _maskw(atom->k_mask.view<DeviceType>()),
    _imagew(atom->k_image.view<DeviceType>()),
    _spw(atom->k_sp.view<DeviceType>()),
    _sendlist(sendlist.template view<DeviceType>()),
    _copylist(copylist.template view<DeviceType>()),
    _size_exchange(atom->avecKK->size_exchange) {
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*
                             buf.template view<DeviceType>().extent(1))/_size_exchange;
    buffer_view<DeviceType>(_buf,buf,maxsendlist,_size_exchange);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &mysend) const {
    const int i = _sendlist(mysend);
    _buf(mysend,0) = _size_exchange;
    _buf(mysend,1) = _x(i,0);
    _buf(mysend,2) = _x(i,1);
    _buf(mysend,3) = _x(i,2);
    _buf(mysend,4) = _v(i,0);
    _buf(mysend,5) = _v(i,1);
    _buf(mysend,6) = _v(i,2);
    _buf(mysend,7) = d_ubuf(_tag[i]).d;
    _buf(mysend,8) = d_ubuf(_type[i]).d;
    _buf(mysend,9) = d_ubuf(_mask[i]).d;
    _buf(mysend,10) = d_ubuf(_image[i]).d;
    _buf(mysend,11) = _sp(i,0);
    _buf(mysend,12) = _sp(i,1);
    _buf(mysend,13) = _sp(i,2);
    _buf(mysend,14) = _sp(i,3);
    const int j = _copylist(mysend);

    if (j>-1) {
    _xw(i,0) = _x(j,0);
    _xw(i,1) = _x(j,1);
    _xw(i,2) = _x(j,2);
    _vw(i,0) = _v(j,0);
    _vw(i,1) = _v(j,1);
    _vw(i,2) = _v(j,2);
    _tagw(i) = _tag(j);
    _typew(i) = _type(j);
    _maskw(i) = _mask(j);
    _imagew(i) = _image(j);
    _spw(i,0) = _sp(j,0);
    _spw(i,1) = _sp(j,1);
    _spw(i,2) = _sp(j,2);
    _spw(i,3) = _sp(j,3);
    }
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSpinKokkos::pack_exchange_kokkos(const int &nsend,DAT::tdual_xfloat_2d &k_buf,
                                              DAT::tdual_int_1d k_sendlist,
                                              DAT::tdual_int_1d k_copylist,
                                              ExecutionSpace space)
{
  size_exchange = 15;

  if (nsend > (int) (k_buf.view<LMPHostType>().extent(0)*k_buf.view<LMPHostType>().extent(1))/size_exchange) {
    int newsize = nsend*size_exchange/k_buf.view<LMPHostType>().extent(1)+1;
    k_buf.resize(newsize,k_buf.view<LMPHostType>().extent(1));
  }
  if (space == Host) {
    AtomVecSpinKokkos_PackExchangeFunctor<LMPHostType>
      f(atomKK,k_buf,k_sendlist,k_copylist);
    Kokkos::parallel_for(nsend,f);
    return nsend*size_exchange;
  } else {
    AtomVecSpinKokkos_PackExchangeFunctor<LMPDeviceType>
      f(atomKK,k_buf,k_sendlist,k_copylist);
    Kokkos::parallel_for(nsend,f);
    return nsend*size_exchange;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSpinKokkos_UnpackExchangeFunctor {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typename AT::t_x_array _x;
  typename AT::t_v_array _v;
  typename AT::t_tagint_1d _tag;
  typename AT::t_int_1d _type;
  typename AT::t_int_1d _mask;
  typename AT::t_imageint_1d _image;
  typename AT::t_sp_array _sp;
  typename AT::t_int_1d _indices;
  typename AT::t_xfloat_2d_um _buf;
  typename AT::t_int_1d _nlocal;
  int _dim;
  X_FLOAT _lo,_hi;
  int _size_exchange;

  AtomVecSpinKokkos_UnpackExchangeFunctor(
      const AtomKokkos* atom,
      const typename AT::tdual_xfloat_2d buf,
      typename AT::tdual_int_1d nlocal,
      const typename AT::tdual_int_1d indices,
      int dim, X_FLOAT lo, X_FLOAT hi):
      _x(atom->k_x.view<DeviceType>()),
      _v(atom->k_v.view<DeviceType>()),
      _tag(atom->k_tag.view<DeviceType>()),
      _type(atom->k_type.view<DeviceType>()),
      _mask(atom->k_mask.view<DeviceType>()),
      _image(atom->k_image.view<DeviceType>()),
      _sp(atom->k_sp.view<DeviceType>()),
      _indices(indices.template view<DeviceType>()),
      _nlocal(nlocal.template view<DeviceType>()),
      _dim(dim),_lo(lo),_hi(hi),_size_exchange(atom->avecKK->size_exchange) {
    const int maxsendlist = (buf.template view<DeviceType>().extent(0)*buf.template view<DeviceType>().extent(1))/_size_exchange;

    buffer_view<DeviceType>(_buf,buf,maxsendlist,_size_exchange);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &myrecv) const {
    X_FLOAT x = _buf(myrecv,_dim+1);
    int i = -1;
    if (x >= _lo && x < _hi) {
      i = Kokkos::atomic_fetch_add(&_nlocal(0),1);
      _x(i,0) = _buf(myrecv,1);
      _x(i,1) = _buf(myrecv,2);
      _x(i,2) = _buf(myrecv,3);
      _v(i,0) = _buf(myrecv,4);
      _v(i,1) = _buf(myrecv,5);
      _v(i,2) = _buf(myrecv,6);
      _tag[i] = (tagint) d_ubuf(_buf(myrecv,7)).i;
      _type[i] = (int) d_ubuf(_buf(myrecv,8)).i;
      _mask[i] = (int) d_ubuf(_buf(myrecv,9)).i;
      _image[i] = (imageint) d_ubuf(_buf(myrecv,10)).i;
      _sp(i,0) = _buf(myrecv,11);
      _sp(i,1) = _buf(myrecv,12);
      _sp(i,2) = _buf(myrecv,13);
      _sp(i,3) = _buf(myrecv,14);
    }
    if (_indices.data()) _indices(myrecv) = i;
  }
};

/* ---------------------------------------------------------------------- */

int AtomVecSpinKokkos::unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv, int nlocal,
                                              int dim, X_FLOAT lo, X_FLOAT hi, ExecutionSpace space,
                                              DAT::tdual_int_1d &k_indices)
{
  while (nlocal + nrecv/size_exchange >= nmax) grow(0);

  if(space == Host) {
    k_count.h_view(0) = nlocal;
    AtomVecSpinKokkos_UnpackExchangeFunctor<LMPHostType> f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
    Kokkos::parallel_for(nrecv/size_exchange,f);
    if (k_indices.h_view.data()) k_indices.modify<LMPHostType>();
    return k_count.h_view(0);
  } else {
    k_count.h_view(0) = nlocal;
    k_count.modify<LMPHostType>();
    k_count.sync<LMPDeviceType>();
    AtomVecSpinKokkos_UnpackExchangeFunctor<LMPDeviceType>
      f(atomKK,k_buf,k_count,k_indices,dim,lo,hi);
    Kokkos::parallel_for(nrecv/size_exchange,f);
    if (k_indices.h_view.data()) k_indices.modify<LMPDeviceType>();
    k_count.modify<LMPDeviceType>();
    k_count.sync<LMPHostType>();

    return k_count.h_view(0);
  }
}

/* ----------------------------------------------------------------------
   reverse communication pack/unpack on device/host (f + fm + fm_long)
------------------------------------------------------------------------- */

template<class DeviceType>
struct AtomVecSpinKokkos_PackReverse {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array_randomread _f;
  typename ArrayTypes<DeviceType>::t_fm_array_randomread _fm;
  typename ArrayTypes<DeviceType>::t_fm_long_array_randomread _fm_long;
  typename ArrayTypes<DeviceType>::t_ffloat_2d_um _buf;
  int _first;

  AtomVecSpinKokkos_PackReverse(
    const typename DAT::tdual_f_array &f,
    const typename DAT::tdual_f_array &fm,
    const typename DAT::tdual_f_array &fm_long,
    const typename DAT::tdual_ffloat_2d &buf,
    const int &first) :
      _f(f.view<DeviceType>()),
      _fm(fm.view<DeviceType>()),
      _fm_long(fm_long.view<DeviceType>()),
      _first(first)
  {
    const size_t elements = 9;
    const size_t maxsend = (buf.view<DeviceType>().extent(0) * buf.view<DeviceType>().extent(1)) / elements;
    buffer_view<DeviceType>(_buf, buf, maxsend, elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const
  {
    const int j = i + _first;
    _buf(i,0) = _f(j,0);
    _buf(i,1) = _f(j,1);
    _buf(i,2) = _f(j,2);
    _buf(i,3) = _fm(j,0);
    _buf(i,4) = _fm(j,1);
    _buf(i,5) = _fm(j,2);
    _buf(i,6) = _fm_long(j,0);
    _buf(i,7) = _fm_long(j,1);
    _buf(i,8) = _fm_long(j,2);
  }
};

int AtomVecSpinKokkos::pack_reverse_kokkos(const int &n, const int &first,
                                          const DAT::tdual_ffloat_2d &buf)
{
  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(Host, F_MASK | FM_MASK | FML_MASK);
    AtomVecSpinKokkos_PackReverse<LMPHostType> f(atomKK->k_f, atomKK->k_fm, atomKK->k_fm_long, buf, first);
    Kokkos::parallel_for(n, f);
  } else {
    atomKK->sync(Device, F_MASK | FM_MASK | FML_MASK);
    AtomVecSpinKokkos_PackReverse<LMPDeviceType> f(atomKK->k_f, atomKK->k_fm, atomKK->k_fm_long, buf, first);
    Kokkos::parallel_for(n, f);
  }

  return n * size_reverse;
}

template<class DeviceType>
struct AtomVecSpinKokkos_UnpackReverse {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array _f;
  typename ArrayTypes<DeviceType>::t_fm_array _fm;
  typename ArrayTypes<DeviceType>::t_fm_long_array _fm_long;
  typename ArrayTypes<DeviceType>::t_ffloat_2d_const _buf;
  typename ArrayTypes<DeviceType>::t_int_1d_const _list;

  AtomVecSpinKokkos_UnpackReverse(
    const typename DAT::tdual_f_array &f,
    const typename DAT::tdual_f_array &fm,
    const typename DAT::tdual_f_array &fm_long,
    const typename DAT::tdual_ffloat_2d &buf,
    const typename DAT::tdual_int_1d &list) :
      _f(f.view<DeviceType>()),
      _fm(fm.view<DeviceType>()),
      _fm_long(fm_long.view<DeviceType>()),
      _list(list.view<DeviceType>())
  {
    const size_t elements = 9;
    const size_t maxsend = (buf.view<DeviceType>().extent(0) * buf.view<DeviceType>().extent(1)) / elements;
    buffer_view<DeviceType>(_buf, buf, maxsend, elements);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const
  {
    const int j = _list(i);
    _f(j,0) += _buf(i,0);
    _f(j,1) += _buf(i,1);
    _f(j,2) += _buf(i,2);
    _fm(j,0) += _buf(i,3);
    _fm(j,1) += _buf(i,4);
    _fm(j,2) += _buf(i,5);
    _fm_long(j,0) += _buf(i,6);
    _fm_long(j,1) += _buf(i,7);
    _fm_long(j,2) += _buf(i,8);
  }
};

void AtomVecSpinKokkos::unpack_reverse_kokkos(const int &n, const DAT::tdual_int_1d &list,
                                              const DAT::tdual_ffloat_2d &buf)
{
  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(Host, F_MASK | FM_MASK | FML_MASK);
    AtomVecSpinKokkos_UnpackReverse<LMPHostType> f(atomKK->k_f, atomKK->k_fm, atomKK->k_fm_long, buf, list);
    Kokkos::parallel_for(n, f);
    atomKK->modified(Host, F_MASK | FM_MASK | FML_MASK);
  } else {
    atomKK->sync(Device, F_MASK | FM_MASK | FML_MASK);
    AtomVecSpinKokkos_UnpackReverse<LMPDeviceType> f(atomKK->k_f, atomKK->k_fm, atomKK->k_fm_long, buf, list);
    Kokkos::parallel_for(n, f);
    atomKK->modified(Device, F_MASK | FM_MASK | FML_MASK);
  }
}

template<class DeviceType>
struct AtomVecSpinKokkos_UnpackReverseSelf {
  typedef DeviceType device_type;

  typename ArrayTypes<DeviceType>::t_f_array_randomread _f;
  typename ArrayTypes<DeviceType>::t_f_array _fw;
  typename ArrayTypes<DeviceType>::t_fm_array_randomread _fm;
  typename ArrayTypes<DeviceType>::t_fm_array _fmw;
  typename ArrayTypes<DeviceType>::t_fm_long_array_randomread _fm_long;
  typename ArrayTypes<DeviceType>::t_fm_long_array _fm_long_w;

  int _nfirst;
  typename ArrayTypes<DeviceType>::t_int_1d_const _list;

  AtomVecSpinKokkos_UnpackReverseSelf(
    const typename DAT::tdual_f_array &f,
    const typename DAT::tdual_f_array &fm,
    const typename DAT::tdual_f_array &fm_long,
    const int &nfirst,
    const typename DAT::tdual_int_1d &list) :
      _f(f.view<DeviceType>()),
      _fw(f.view<DeviceType>()),
      _fm(fm.view<DeviceType>()),
      _fmw(fm.view<DeviceType>()),
      _fm_long(fm_long.view<DeviceType>()),
      _fm_long_w(fm_long.view<DeviceType>()),
      _nfirst(nfirst),
      _list(list.view<DeviceType>()) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const
  {
    const int j = _list(i);
    const int src = i + _nfirst;
    // Self-swap reverse comm can include multiple ghost images mapping to the
    // same owned atom (e.g. periodic boundaries when a proc grid dimension is 1).
    // Use atomics to avoid races on GPU/OpenMP backends.
    Kokkos::atomic_add(&_fw(j,0), _f(src,0));
    Kokkos::atomic_add(&_fw(j,1), _f(src,1));
    Kokkos::atomic_add(&_fw(j,2), _f(src,2));
    Kokkos::atomic_add(&_fmw(j,0), _fm(src,0));
    Kokkos::atomic_add(&_fmw(j,1), _fm(src,1));
    Kokkos::atomic_add(&_fmw(j,2), _fm(src,2));
    Kokkos::atomic_add(&_fm_long_w(j,0), _fm_long(src,0));
    Kokkos::atomic_add(&_fm_long_w(j,1), _fm_long(src,1));
    Kokkos::atomic_add(&_fm_long_w(j,2), _fm_long(src,2));
  }
};

int AtomVecSpinKokkos::pack_reverse_self(const int &n, const DAT::tdual_int_1d &list,
                                        const int nfirst)
{
  if (lmp->kokkos->reverse_comm_on_host) {
    atomKK->sync(Host, F_MASK | FM_MASK | FML_MASK);
    AtomVecSpinKokkos_UnpackReverseSelf<LMPHostType> f(atomKK->k_f, atomKK->k_fm, atomKK->k_fm_long, nfirst, list);
    Kokkos::parallel_for(n, f);
    atomKK->modified(Host, F_MASK | FM_MASK | FML_MASK);
  } else {
    atomKK->sync(Device, F_MASK | FM_MASK | FML_MASK);
    AtomVecSpinKokkos_UnpackReverseSelf<LMPDeviceType> f(atomKK->k_f, atomKK->k_fm, atomKK->k_fm_long, nfirst, list);
    Kokkos::parallel_for(n, f);
    atomKK->modified(Device, F_MASK | FM_MASK | FML_MASK);
  }

  return n * size_reverse;
}

/* ----------------------------------------------------------------------
   clear extra forces starting at atom N
   nbytes = # of bytes to clear for a per-atom vector
   include f b/c this is invoked from within SPIN pair styles
------------------------------------------------------------------------- */

void AtomVecSpinKokkos::force_clear(int /*n*/, size_t nbytes)
{
  int nzero = (double)nbytes/sizeof(double);

  if (nzero) {
    atomKK->k_fm.clear_sync_state(); // will be cleared below
    atomKK->k_fm_long.clear_sync_state(); // will be cleared below

    // local variables for lambda capture

    auto l_fm = atomKK->k_fm.d_view;
    auto l_fm_long = atomKK->k_fm_long.d_view;

    Kokkos::parallel_for(nzero, LAMMPS_LAMBDA(int i) {
      l_fm(i,0) = 0.0;
      l_fm(i,1) = 0.0;
      l_fm(i,2) = 0.0;
      l_fm_long(i,0) = 0.0;
      l_fm_long(i,1) = 0.0;
      l_fm_long(i,2) = 0.0;
    });

    atomKK->modified(Device,FM_MASK|FML_MASK);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecSpinKokkos::sync(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.sync<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPDeviceType>();
    if (mask & SP_MASK) atomKK->k_sp.sync<LMPDeviceType>();
    if (mask & FM_MASK) atomKK->k_fm.sync<LMPDeviceType>();
    if (mask & FML_MASK) atomKK->k_fm_long.sync<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK->k_x.sync<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.sync<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.sync<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.sync<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.sync<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.sync<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.sync<LMPHostType>();
    if (mask & SP_MASK) atomKK->k_sp.sync<LMPHostType>();
    if (mask & FM_MASK) atomKK->k_fm.sync<LMPHostType>();
    if (mask & FML_MASK) atomKK->k_fm_long.sync<LMPHostType>();
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecSpinKokkos::modified(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if (mask & X_MASK) atomKK->k_x.modify<LMPDeviceType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPDeviceType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPDeviceType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPDeviceType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPDeviceType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPDeviceType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPDeviceType>();
    if (mask & SP_MASK) atomKK->k_sp.modify<LMPDeviceType>();
    if (mask & FM_MASK) atomKK->k_fm.modify<LMPDeviceType>();
    if (mask & FML_MASK) atomKK->k_fm_long.modify<LMPDeviceType>();
  } else {
    if (mask & X_MASK) atomKK->k_x.modify<LMPHostType>();
    if (mask & V_MASK) atomKK->k_v.modify<LMPHostType>();
    if (mask & F_MASK) atomKK->k_f.modify<LMPHostType>();
    if (mask & TAG_MASK) atomKK->k_tag.modify<LMPHostType>();
    if (mask & TYPE_MASK) atomKK->k_type.modify<LMPHostType>();
    if (mask & MASK_MASK) atomKK->k_mask.modify<LMPHostType>();
    if (mask & IMAGE_MASK) atomKK->k_image.modify<LMPHostType>();
    if (mask & SP_MASK) atomKK->k_sp.modify<LMPHostType>();
    if (mask & FM_MASK) atomKK->k_fm.modify<LMPHostType>();
    if (mask & FML_MASK) atomKK->k_fm_long.modify<LMPHostType>();
  }
}

void AtomVecSpinKokkos::sync_overlapping_device(ExecutionSpace space, unsigned int mask)
{
  if (space == Device) {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x,space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v,space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f,space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type,space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask,space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space);
    if ((mask & SP_MASK) && atomKK->k_sp.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_float_1d_4>(atomKK->k_sp,space);
    if ((mask & FM_MASK) && atomKK->k_fm.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_fm,space);
    if ((mask & FML_MASK) && atomKK->k_fm_long.need_sync<LMPDeviceType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_fm_long,space);
  } else {
    if ((mask & X_MASK) && atomKK->k_x.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_x_array>(atomKK->k_x,space);
    if ((mask & V_MASK) && atomKK->k_v.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_v_array>(atomKK->k_v,space);
    if ((mask & F_MASK) && atomKK->k_f.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_f,space);
    if ((mask & TAG_MASK) && atomKK->k_tag.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_tagint_1d>(atomKK->k_tag,space);
    if ((mask & TYPE_MASK) && atomKK->k_type.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_type,space);
    if ((mask & MASK_MASK) && atomKK->k_mask.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_int_1d>(atomKK->k_mask,space);
    if ((mask & IMAGE_MASK) && atomKK->k_image.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_imageint_1d>(atomKK->k_image,space);
    if ((mask & SP_MASK) && atomKK->k_sp.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_float_1d_4>(atomKK->k_sp,space);
    if ((mask & FM_MASK) && atomKK->k_fm.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_fm,space);
    if ((mask & FML_MASK) && atomKK->k_fm_long.need_sync<LMPHostType>())
      perform_async_copy<DAT::tdual_f_array>(atomKK->k_fm_long,space);
  }
}
