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

#ifndef LMP_FIX_TSPIN_NH_KOKKOS_H
#define LMP_FIX_TSPIN_NH_KOKKOS_H

#ifdef LMP_KOKKOS

#include "fix_nh_kokkos.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

template <class DeviceType>
class FixTSPINNHKokkos : public FixNHKokkos<DeviceType>, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixTSPINNHKokkos(class LAMMPS *, int, char **);
  ~FixTSPINNHKokkos() override;

  void init() override;
  void setup(int) override;

  void initial_integrate(int) override;
  void final_integrate() override;

  double compute_scalar() override;
  double compute_vector(int) override;

  int modify_param(int, char **) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int size_restart(int) override;
  int maxsize_restart() override;

  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;

  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  int pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d k_sendlist,
                           DAT::tdual_int_1d k_copylist, ExecutionSpace space) override;
  void unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, DAT::tdual_int_1d &indices, int nrecv, int nrecv1,
                              int nextrarecv1, ExecutionSpace space) override;
  void sort_kokkos(Kokkos::BinSort<KeyViewType, BinOp> &Sorter) override;

  // Called by MC swap fixes after an accepted type/spin update to keep this fix's
  // internal per-atom state consistent with atom->sp and atom->type.
  void mc_sync_from_atom(int local_index);

  // NOTE: These helpers run Kokkos extended lambdas; for CUDA builds they must
  // not be declared with private/protected access (NVCC limitation).
  void update_spin_dof_device();
  void init_spin_state_and_velocities_host();
  void compute_twoKs_global_device();
  void nhc_spin_integrate();
  void nve_v_spin_device();
  void nve_s_spin_device();

  void ensure_mass_factor_kokkos();

  protected:
    struct ParsedArgs {
      std::vector<std::string> fixnh_strings;
      std::vector<char *> fixnh_argv;
      int fixnh_narg = 0;

      int spin_flag = 1;
      int lattice_flag = 1;
      int seed = 12345;
      int reinit_spin_vel = 0;
      int fm_is_frequency = 0;
      std::vector<double> mass_factor;    // size 0 => default 1.0 for all types
    };

  static ParsedArgs parse_tspin_trailing_block(class LAMMPS *, int, char **);
  FixTSPINNHKokkos(class LAMMPS *, ParsedArgs);

  int size_vector_nh;
  int lattice_flag;

   int spin_flag;
   int spin_dof;
    std::vector<double> mass_factor;
    int seed;
    int reinit_spin_vel;
    int spin_state_initialized;
     int fm_is_frequency;

  double twoKs_global, Ks_global;
  std::vector<double> etas, etas_dot, etas_dotdot, etas_mass;

  DAT::tdual_double_2d k_vs;
  DAT::tdual_double_2d k_sreal;
  DAT::tdual_double_1d k_smass;
  DAT::tdual_int_1d k_isspin;
  double **vs;
  double **sreal;
  double *smass;
  int *isspin;
  int nmax_old;
  int grow_callback_added;
  int restart_callback_added;
  int restart_from_legacy;

  DAT::tdual_double_1d k_mass_factor;
  double *mass_factor_buf;

  typename AT::t_double_2d d_vs, d_sreal;
  typename AT::t_double_1d d_smass, d_mass_factor;
  typename AT::t_int_1d d_isspin;

  typename AT::t_int_1d mask;
  typename AT::t_int_1d type;
  typename AT::t_sp_array sp;
  typename AT::t_f_array fm;
  typename AT::t_float_1d mass_type;

  // exchange temporaries
  int nsend_tmp, nrecv1_tmp, nextrarecv1_tmp;
  typename AT::t_xfloat_1d_um d_buf;
  typename AT::t_int_1d d_exchange_sendlist, d_copylist, d_indices;

  static constexpr double RESTART_MAGIC = -6.019852413e8;
  static constexpr int RESTART_VERSION = 1;

  int size_restart_global() override;
  int pack_restart_data(double *) override;
  void restart(char *) override;

  static int nh_payload_size_from_list(const double *, int);
  int pack_restart_payload_v1(double *) const;
  void unpack_restart_payload_v1(const double *);
};

}    // namespace LAMMPS_NS

#endif    // LMP_KOKKOS

#endif    // LMP_FIX_TSPIN_NH_KOKKOS_H
