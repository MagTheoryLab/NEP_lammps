/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(spin/kk,AtomVecSpinKokkos);
AtomStyle(spin/kk/device,AtomVecSpinKokkos);
AtomStyle(spin/kk/host,AtomVecSpinKokkos);
// clang-format on
#else

// clang-format off
#ifndef LMP_ATOM_VEC_SPIN_KOKKOS_H
#define LMP_ATOM_VEC_SPIN_KOKKOS_H

#include "atom_vec_kokkos.h"
#include "atom_vec_spin.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomVecSpinKokkos : public AtomVecKokkos, public AtomVecSpin {
 public:
  AtomVecSpinKokkos(class LAMMPS *);
  void grow(int) override;
  void grow_pointers() override;
  void force_clear(int, size_t) override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;
  int pack_comm_self(const int &n, const DAT::tdual_int_1d &list,
                     const int nfirst,
                     const int &pbc_flag, const int pbc[]) override;
  int pack_comm_self_fused(const int &n, const DAT::tdual_int_2d &list,
                           const DAT::tdual_int_1d &sendnum_scan,
                           const DAT::tdual_int_1d &firstrecv,
                           const DAT::tdual_int_1d &pbc_flag,
                           const DAT::tdual_int_2d &pbc,
                           const DAT::tdual_int_1d &g2l) override;
  int pack_comm_kokkos(const int &n, const DAT::tdual_int_1d &list,
                       const DAT::tdual_xfloat_2d &buf,
                       const int &pbc_flag, const int pbc[]) override;
  void unpack_comm_kokkos(const int &n, const int &nfirst,
                          const DAT::tdual_xfloat_2d &buf) override;
  int pack_border_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                         DAT::tdual_xfloat_2d buf,
                         int pbc_flag, int *pbc, ExecutionSpace space) override;
  void unpack_border_kokkos(const int &n, const int &nfirst,
                            const DAT::tdual_xfloat_2d &buf,
                            ExecutionSpace space) override;
  int pack_exchange_kokkos(const int &nsend,DAT::tdual_xfloat_2d &buf,
                           DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist,
                           ExecutionSpace space) override;
  int unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv,
                             int nlocal, int dim, X_FLOAT lo, X_FLOAT hi,
                             ExecutionSpace space,
                             DAT::tdual_int_1d &k_indices) override;

  int pack_reverse_self(const int &n, const DAT::tdual_int_1d &list,
                        const int nfirst) override;
  int pack_reverse_kokkos(const int &n, const int &nfirst,
                          const DAT::tdual_ffloat_2d &buf) override;
  void unpack_reverse_kokkos(const int &n, const DAT::tdual_int_1d &list,
                             const DAT::tdual_ffloat_2d &buf) override;

  void sync(ExecutionSpace space, unsigned int mask) override;
  void modified(ExecutionSpace space, unsigned int mask) override;
  void sync_overlapping_device(ExecutionSpace space, unsigned int mask) override;

 protected:
  DAT::t_tagint_1d d_tag;
  HAT::t_tagint_1d h_tag;

  DAT::t_int_1d d_type, d_mask;
  HAT::t_int_1d h_type, h_mask;

  DAT::t_imageint_1d d_image;
  HAT::t_imageint_1d h_image;

  DAT::t_x_array d_x;
  DAT::t_v_array d_v;
  DAT::t_f_array d_f;

  DAT::t_sp_array d_sp;
  DAT::t_fm_array d_fm;
  DAT::t_fm_long_array d_fm_long;

  HAT::t_sp_array h_sp;
  HAT::t_fm_array h_fm;
  HAT::t_fm_long_array h_fm_long;
};

}    // namespace LAMMPS_NS

#endif
#endif
