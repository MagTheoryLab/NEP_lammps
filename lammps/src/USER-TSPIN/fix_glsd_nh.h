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

#ifndef LMP_FIX_GLSD_NH_H
#define LMP_FIX_GLSD_NH_H

#include "fix_nh.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixGLSDNH : public FixNH {
 public:
  FixGLSDNH(class LAMMPS *, int, char **);
  ~FixGLSDNH() override;

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
  // Spin variable is treated as magnetic moment M in μB:
  // - atom->fm stores H = -dE/dM in eV/μB
  struct ParsedArgs;
  static ParsedArgs parse_glsd_trailing_block(class LAMMPS *, int, char **);
  FixGLSDNH(class LAMMPS *, ParsedArgs);

  int size_vector_nh;    // number of FixNH vector components
  int lattice_flag;      // 1 = integrate lattice (x/v) as in FixNH, 0 = freeze lattice and only evolve spins
  int midpoint_iter;     // spin-only implicit-midpoint fixed-point iterations (>=1)
  double midpoint_tol;      // >=0 convergence tolerance for midpoint iterations (0 = no early stop)

  // GLSD parameters (Ma & Dudarev PRB 86, 054416)
  // We directly integrate the magnetic moment M (μB) using:
  //   dM/dt = -(g/ħ) M×H + λ H + η
  // with <η(t)η(t')> = 2 λ kBT δ(t-t').
  // Here H = -dE/dM is in eV/μB (our convention), so the precession frequency is ω=(g/ħ)H.
  double lambda;              // λ (internal). Can be set via `glsd alpha`, or directly via `glsd lambda` (`glsd gammas` is a legacy alias).
  double alpha;               // dimensionless: lambda = alpha * (g/ħ) (in our μB-absorbed convention); alpha<0 disables
  double spin_temperature;    // spin thermostat temperature (K)
  int seed;               // RNG seed
  double hbar;            // Planck/(2pi) in current units
  double g_over_hbar;     // g/ħ

  // optional debug logging (energy diagnostics)
  int debug_flag;
  int debug_every;
  int debug_rank;
  int debug_flush;
  bigint debug_start;
  int debug_header_printed;
  std::string debug_file;
  FILE *debug_fp;
  double pe_prev_end;

  // Per-atom cached total field/torque from the previous step (migrates with atoms)
  int idx_fm_cache;
  double **fm_cache;
  int idx_s0_cache;      // per-atom starting spin S0 cache for lattice-moving midpoint iterations (migrates with atoms)
  double **s0_cache;
  int nmax_old;
  int grow_callback_added;
  int restart_callback_added;
  int restart_from_legacy;

  // Scratch storage for implicit-midpoint iterations (local only)
  int nmax_s0;
  double **s0;
  int nmax_s_guess;
  double **s_guess;
  int nmax_s_map;
  double **s_map;

  // Whitelist replay of external spin-field fixes during the recompute stage
  std::vector<class Fix *> replay_fixes;
  std::vector<int> replay_fix_indices;

  // deterministic gaussian noise helper
  static std::uint64_t splitmix64(std::uint64_t x);
  static double gaussian_u64(std::uint64_t seed, tagint tag, std::uint64_t step, int phase, int component);

  static constexpr double RESTART_MAGIC = -6.019852414e8;
  static constexpr int RESTART_VERSION = 1;

  // custom per-atom storage and migration hooks
  void ensure_custom_peratom();
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  // integration helpers
  double fm_to_frequency(double fm_component) const;
  void glsd_step(double dt, double **fm_use, int noise_phase);
  void glsd_map(double dt, double **s_in, double **fm_use, int noise_phase, double **s_out);
  void cache_current_fm();
  bool solve_spin_midpoint(bool lattice_mode, int vflag, double pe_mid);

  // recompute force/field at (x^{n+1}, S^{n+1}) without running modify->post_force() twice
  void clear_force_arrays();
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
};

}    // namespace LAMMPS_NS

#endif
