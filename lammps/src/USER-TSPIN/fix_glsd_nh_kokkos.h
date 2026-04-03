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

#ifndef LMP_FIX_GLSD_NH_KOKKOS_H
#define LMP_FIX_GLSD_NH_KOKKOS_H

#ifdef LMP_KOKKOS

#include "fix_nh_kokkos.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace LAMMPS_NS {

template <class DeviceType>
class FixGLSDNHKokkos : public FixNHKokkos<DeviceType>, public KokkosBase {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixGLSDNHKokkos(class LAMMPS *, int, char **);
  ~FixGLSDNHKokkos() override;

  int setmask() override;
  void init() override;
  void setup(int) override;

  void initial_integrate(int) override;
  void post_force(int) override;
  void final_integrate() override;

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

  // Called by MC swap fixes after an accepted type/spin update to keep GLSD caches
  // (S0/fm) consistent with the current atom->sp state for the affected local atom.
  void mc_sync_from_atom(int local_index);

  // NOTE: These helpers run Kokkos extended lambdas; for CUDA builds they must
  // not be declared with private/protected access (NVCC limitation).
  void cache_current_fm_device();
  void save_s0_cache_device();
  void glsd_map_device_kernel(double dt, typename AT::t_double_2d s_in, typename AT::t_double_2d s_out,
                              bool use_fm_cache, int noise_phase);
  bool solve_spin_midpoint_device(bool lattice_mode, int vflag, double pe_mid);
  void clear_force_arrays_device();

 protected:
  struct ParsedArgs {
    std::vector<std::string> fixnh_strings;
    std::vector<char *> fixnh_argv;
    int fixnh_narg = 0;

    int lattice_flag = 1;
    int midpoint_iter = 3;
    double midpoint_tol = 0.0;

    double gammas = 0.0;    // interpreted as lambda (direct); keyword aliases: gammas, lambda
    double alpha = -1.0;    // dimensionless: lambda = alpha*(g/ħ); alpha<0 means disabled
    double spin_temperature = 0.0;
    int seed = 12345;

    int debug_flag = 0;
    int debug_every = 1;
    int debug_rank = 0;
    int debug_flush = 0;
    bigint debug_start = 0;
    std::string debug_file;
  };

  static ParsedArgs parse_glsd_trailing_block(class LAMMPS *, int, char **);
  FixGLSDNHKokkos(class LAMMPS *, ParsedArgs);

  int size_vector_nh;
  int lattice_flag;
  int midpoint_iter;
  double midpoint_tol;

  enum class MidpointBackend : int { DEVICE = 0, HOST_FALLBACK = 1 };
  enum class MidpointFallbackReason : int {
    NONE = 0,
    HOST_REPLAY_FIX = 1,
    NON_DEVICE_EXECUTION = 2,
    SCRATCH_UNAVAILABLE = 3
  };

  double lambda;
  double alpha;
  double spin_temperature;
  double spin_temperature_cached;
  bigint spin_temperature_cached_step;
  int spin_temperature_cache_valid;
  int seed;
  double hbar;
  double g_over_hbar;

  int debug_flag;
  int debug_every;
  int debug_rank;
  int debug_flush;
  bigint debug_start;
  int debug_header_printed;
  std::string debug_file;
  FILE *debug_fp;
  double pe_prev_end;

  // per-atom cached total field/torque from the previous step (migrates with atoms)
  DAT::tdual_double_2d k_fm_cache;
  double **fm_cache;
  DAT::tdual_double_2d k_s0_cache;
  double **s0_cache;
  DAT::tdual_double_2d k_s_guess_device;
  double **s_guess_device;
  DAT::tdual_double_2d k_s_map_device;
  double **s_map_device;
  int nmax_old;
  int grow_callback_added;
  int restart_callback_added;
  int restart_from_legacy;

  typename AT::t_double_2d d_fm_cache;
  typename AT::t_double_2d d_s0_cache;
  typename AT::t_double_2d d_s_guess;
  typename AT::t_double_2d d_s_map;

  MidpointBackend midpoint_backend_last;
  MidpointFallbackReason midpoint_fallback_last;

  // Scratch storage for implicit-midpoint iterations (host only)
  int nmax_s0;
  double **s0;
  int nmax_s_guess;
  double **s_guess;
  int nmax_s_map;
  double **s_map;

  // Whitelist replay of external spin-field fixes during the recompute stage
  std::vector<class Fix *> replay_fixes;
  std::vector<int> replay_fix_indices;

  // AtomKokkos views used in device kernels
  typename AT::t_int_1d mask;
  typename AT::t_tagint_1d tag;
  typename AT::t_sp_array sp;
  typename AT::t_f_array fm;

  // exchange temporaries
  int nsend_tmp, nrecv1_tmp, nextrarecv1_tmp;
  typename AT::t_xfloat_1d_um d_buf;
  typename AT::t_int_1d d_exchange_sendlist, d_copylist, d_indices;

  static constexpr double RESTART_MAGIC = -6.019852414e8;
  static constexpr int RESTART_VERSION = 1;

  // helpers
  double compute_spin_temperature();
  void recompute_force_and_field(int eflag, int vflag);
  void replay_external_spin_fields(int vflag);
  MidpointBackend select_midpoint_backend(bool lattice_mode);
  const char *fallback_reason_string(MidpointFallbackReason reason) const;
  const char *midpoint_backend_string(MidpointBackend backend) const;
  bool ensure_midpoint_device_scratch();

  double fm_to_frequency(double fm_component) const;

  void glsd_map_host(double dt, double **s_in, double **fm_use, int noise_phase, double **s_out);
  bool solve_spin_midpoint_host(bool lattice_mode, int vflag, double pe_mid);

  void debug_open();
  void debug_close();
  void debug_log_energy(double pe_mid, double pe_end, MidpointBackend backend, MidpointFallbackReason reason);
  double current_pe_total() const;

  int size_restart_global() override;
  int pack_restart_data(double *) override;
  void restart(char *) override;

  static int nh_payload_size_from_list(const double *, int);
  int pack_restart_payload_v1(double *) const;
  void unpack_restart_payload_v1(const double *);
};

}    // namespace LAMMPS_NS

#endif    // LMP_KOKKOS

#endif    // LMP_FIX_GLSD_NH_KOKKOS_H
