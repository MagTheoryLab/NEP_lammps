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

#ifdef MINIMIZE_CLASS
// clang-format off
MinimizeStyle(tspin/fire, MinTSPINFire);
// clang-format on
#else

#ifndef LMP_MIN_TSPIN_FIRE_H
#define LMP_MIN_TSPIN_FIRE_H

#include "min.h"

namespace LAMMPS_NS {

class MinTSPINFire : public Min {
 public:
  MinTSPINFire(class LAMMPS *);
  ~MinTSPINFire() override;

  void init() override;
  void setup_style() override;
  void reset_vectors() override;
  int modify_param(int, char **) override;
  int iterate(int) override;

  double fnorm_sqr() override;
  double fnorm_inf() override;
  double fnorm_max() override;

 private:
  double *v_fire;
  double *u_fire;
  double *g_spin;
  double *w_fire;
  double *g_rho;

  enum PresetStyle { PRESET_CONSERVATIVE = 0, PRESET_BALANCED = 1, PRESET_AGGRESSIVE = 2 };

  // channel step-scale weights
  double scale_lat;
  double scale_mag;
  double scale_spin;

  // timestep
  double dt_init;
  double dt_max;

  // per-step displacement/rotation/magnitude caps
  double disp_max;
  double angle_max;
  double mag_step;
  double mag_min;

  // power smoothing and channel power weights
  double p_smooth;
  double pw_lat;
  double pw_mag;
  double pw_spin;

  // FIRE internal parameters
  double mix_init;
  double dt_grow;
  double dt_shrink;
  double mix_decay;
  int n_accel;

  // numerical thresholds
  double eps_power;
  double mix_tol_lat;
  double mix_tol_mag;
  double mix_tol_spin;

  // convergence tolerances
  double ftol_mag;
  double ftol_spin;
  int ftol_mag_set;
  int ftol_spin_set;

  double eps_theta;
  int vary_mag;
  int expert;
  int preset;

  // explicit-set flags
  int scale_mag_set;
  int mag_step_set;
  int mag_min_set;
  int dt_init_set;
  int dt_max_set;
  int disp_max_set;
  int angle_max_set;
  int p_smooth_set;
  int dt_grow_set;
  int dt_shrink_set;
  int mix_decay_set;
  int n_accel_set;

  // FIRE runtime state
  double dt_fire;
  double mix_alpha;
  int n_pos;
  double p_smooth_bar;

  void apply_preset_defaults();
  void log_setup_summary() const;
  void compute_projected_spin_gradient();
  void compute_variable_spin_gradients();
  int iterate_fixed(int);
  int iterate_variable(int);
};

}    // namespace LAMMPS_NS

#endif
#endif
