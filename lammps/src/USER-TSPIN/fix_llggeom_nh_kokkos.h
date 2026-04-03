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

#ifndef LMP_FIX_LLGGEOM_NH_KOKKOS_H
#define LMP_FIX_LLGGEOM_NH_KOKKOS_H

#ifdef LMP_KOKKOS

#include "fix_llggeom_nh.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomKokkos;

template <class DeviceType>
class FixLLGGeomNHKokkos : public FixLLGGeomNH, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixLLGGeomNHKokkos(class LAMMPS *, int, char **);
  ~FixLLGGeomNHKokkos() override;

  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void post_force(int) override;
  void final_integrate() override;
  void cache_lattice_moving_step_start_state_device();
  void build_predictor_midpoint_state() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  void unpack_restart(int, int) override;
  int pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist, ExecutionSpace space) override;
  void unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d &indices, int nrecv, int nrecv1,
                              int nextrarecv1, ExecutionSpace space) override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;

  bool ensure_device_scratch();
  bool solve_spin_midpoint_device(bool, int, double);
  void clear_force_arrays_device();
  void rebuild_neighbors_for_current_positions_device();
  void recompute_force_and_field_device(int, int);
  void replay_external_spin_fields_device(int);
  void cache_current_fm_device();

 protected:
  AtomKokkos *atomKK;

  DAT::tdual_double_2d k_x0_device;
  double **x0_device_host;
  DAT::tdual_double_2d k_v0_device;
  double **v0_device_host;
  DAT::tdual_double_2d k_f0_device;
  double **f0_device_host;
  DAT::tdual_double_2d k_s0_device;
  double **s0_device_host;
  DAT::tdual_double_2d k_r_mid_guess_device;
  double **r_mid_guess_device_host;
  DAT::tdual_double_2d k_e_mid_guess_device;
  double **e_mid_guess_device_host;
  DAT::tdual_double_2d k_e_pred_device;
  double **e_pred_device_host;
  DAT::tdual_double_2d k_f_mid_device;
  double **f_mid_device_host;
  DAT::tdual_double_2d k_h_mid_device;
  double **h_mid_device_host;
  DAT::tdual_double_2d k_r_new_device;
  double **r_new_device_host;
  DAT::tdual_double_2d k_v_new_device;
  double **v_new_device_host;
  DAT::tdual_double_2d k_e_new_device;
  double **e_new_device_host;
  DAT::tdual_double_2d k_h_th_device;
  double **h_th_device_host;
  DAT::tdual_double_2d k_fm_cache_device;
  double **fm_cache_device_host;

  typename AT::t_double_2d d_x0_device;
  typename AT::t_double_2d d_v0_device;
  typename AT::t_double_2d d_f0_device;
  typename AT::t_double_2d d_s0_device;
  typename AT::t_double_2d d_r_mid_guess_device;
  typename AT::t_double_2d d_e_mid_guess_device;
  typename AT::t_double_2d d_e_pred_device;
  typename AT::t_double_2d d_f_mid_device;
  typename AT::t_double_2d d_h_mid_device;
  typename AT::t_double_2d d_r_new_device;
  typename AT::t_double_2d d_v_new_device;
  typename AT::t_double_2d d_e_new_device;
  typename AT::t_double_2d d_h_th_device;
  typename AT::t_double_2d d_fm_cache_device;

  typename AT::t_int_1d mask_view;
  typename AT::t_tagint_1d tag_view;
  typename AT::t_int_1d type_view;
  typename AT::t_float_1d mass_type_view;
  typename AT::t_x_array x_view;
  typename AT::t_v_array v_view;
  typename AT::t_f_array f_view;
  typename AT::t_sp_array sp_view;
  typename AT::t_f_array fm_view;
  typename AT::t_xfloat_1d_um d_buf;
  typename AT::t_int_1d d_exchange_sendlist;
  typename AT::t_int_1d d_copylist;
  typename AT::t_int_1d d_indices;
  int nsend_tmp;
  int nrecv1_tmp;
  int nextrarecv1_tmp;

  void sync_host_all();
  void mark_host_all_modified();
  void validate_replay_fixes();
};

}    // namespace LAMMPS_NS

#endif

#endif
