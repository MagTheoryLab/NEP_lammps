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

#ifndef LMP_FIX_LLGMID_NH_H
#define LMP_FIX_LLGMID_NH_H

#include "fix_nh.h"

#include <cstdio>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixLLGMidNH : public FixNH {
 public:
  FixLLGMidNH(class LAMMPS *, int, char **);
  ~FixLLGMidNH() override;

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

 protected:
  struct ParsedArgs;
  static ParsedArgs parse_llgmid_trailing_block(class LAMMPS *, int, char **);
  FixLLGMidNH(class LAMMPS *, ParsedArgs);

  int size_vector_nh;
  int lattice_flag;
  int midpoint_iter;
  double midpoint_tol;
  double alpha;
  double gamma;
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

  int idx_fm_cache;
  double **fm_cache;
  int idx_s0_cache;
  double **s0_cache;
  int idx_x0_cache;
  double **x0_cache;
  int idx_v0_cache;
  double **v0_cache;
  int idx_f0_cache;
  double **f0_cache;
  int nmax_old;
  int grow_callback_added;
  int restart_callback_added;
  int restart_from_legacy;

  int nmax_s_guess;
  double **s_guess;
  int nmax_x_end;
  double **x_end;

  std::vector<class Fix *> replay_fixes;
  std::vector<int> replay_fix_indices;

  static constexpr double RESTART_MAGIC = -6.019852415e8;
  static constexpr int RESTART_VERSION = 2;

  void ensure_custom_peratom();
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  double get_mass(int) const;
  void refresh_g_over_hbar();
  void exact_flow_direction(const double *, const double *, double, double *) const;
  void write_spin_from_vector(int, const double *);
  void cache_lattice_moving_step_start_state();
  void build_predictor_midpoint_state();
  void apply_corrector_from_midpoint_field(double **x_mid, double **e_mid, int update_lattice);
  void cache_current_fm();
  bool solve_spin_midpoint(bool lattice_mode, int vflag, double pe_mid);

  void clear_force_arrays();
  void rebuild_neighbors_for_current_positions();
  void recompute_force_and_field(int eflag, int vflag);
  void replay_external_spin_fields(int vflag);

  void debug_open();
  void debug_close();
  void debug_log_energy(double pe_mid, double pe_end);
  double current_pe_total() const;

  int size_restart_global() override;
  int pack_restart_data(double *) override;
  void restart(char *) override;

  static int nh_payload_size_from_list(const double *, int);
  int pack_restart_payload_v1(double *) const;
  void unpack_restart_payload_v1(const double *);
  int pack_restart_payload_v2(double *) const;
  void unpack_restart_payload_v2(const double *);
};

}    // namespace LAMMPS_NS

#endif
