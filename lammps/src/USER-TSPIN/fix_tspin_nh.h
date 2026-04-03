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

#ifndef LMP_FIX_TSPIN_NH_H
#define LMP_FIX_TSPIN_NH_H

#include "fix_nh.h"

#include <vector>

namespace LAMMPS_NS {

class FixTSPINNH : public FixNH {
 public:
  FixTSPINNH(class LAMMPS *, int, char **);
  ~FixTSPINNH() override;

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

 protected:
  struct ParsedArgs;
  static ParsedArgs parse_tspin_trailing_block(class LAMMPS *, int, char **);
  FixTSPINNH(class LAMMPS *, ParsedArgs);

  int size_vector_nh;    // number of FixNH vector components (before appending spin outputs)
  int lattice_flag;      // 1 = integrate lattice (x/v) as in FixNH, 0 = freeze lattice and only evolve spins

  // --- Spin generalized-coordinate support (custom per-atom properties) ---
  // These exist only when spin is enabled and atom_style supports spins (atom->sp_flag).

  int spin_flag;                 // 1 if integrate+thermostat spins, 0 = ignore spins
  int spin_dof;                  // total spin DOF across MPI ranks

  // custom per-atom properties (via Atom::add_custom)
  int ghost_custom;              // 0 = no ghost comm (default)
  int idx_vs, idx_sreal, idx_smass, idx_isspin;
  // cached pointers to custom per-atom properties (updated on grow and on demand)
  double **vs, **sreal;
  double *smass;
  int *isspin;
  int nmax_old;
  int grow_callback_added;
  int restart_callback_added;
  int restart_from_legacy;

  // per-type factor for spin inertia mu_i ("mass" in legacy inputs):
  // mu_i = atom_mass[type] * mass_factor[type]
  std::vector<double> mass_factor;
  int seed;
  int reinit_spin_vel;
  int spin_state_initialized;

  // Interpretation of atom->fm for the spin integrator:
  // - 0: fm is a generalized "field/force" for the spin coordinate (default; NEP prints eV/μB)
  // - 1: fm is a precession frequency (1/time), converted to eV/μB via field = (ħ/g)*ω
  int fm_is_frequency;

  // global spin kinetic energies (energy units)
  // twoKs_global = 2*K_s; Ks_global = 0.5*twoKs_global
  double twoKs_global, Ks_global;

  // Nose-Hoover chain for spins (separate from FixNH particle chain)
  std::vector<double> etas, etas_dot, etas_dotdot, etas_mass;

  static constexpr double RESTART_MAGIC = -6.019852413e8;
  static constexpr int RESTART_VERSION = 1;

  // helpers
  void ensure_custom_peratom();
  void ensure_mass_factor();
  void update_spin_dof_and_flags();
  void refresh_spin_state_from_atom();
  void init_spin_velocities_legacy();
  void compute_twoKs_global();

  // atom migration support for tspin_* custom properties
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  // integration pieces
  void nhc_spin_integrate();
  void nve_v_spin();
  void nve_s_spin();

  int size_restart_global() override;
  int pack_restart_data(double *) override;
  void restart(char *) override;

  static int nh_payload_size_from_list(const double *, int);
  int pack_restart_payload_v1(double *) const;
  void unpack_restart_payload_v1(const double *);
};

}    // namespace LAMMPS_NS

#endif
