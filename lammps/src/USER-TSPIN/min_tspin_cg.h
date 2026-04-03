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
MinimizeStyle(tspin/cg, MinTSPINCG);
// clang-format on
#else

#ifndef LMP_MIN_TSPIN_CG_H
#define LMP_MIN_TSPIN_CG_H

#include "min.h"

namespace LAMMPS_NS {

class MinTSPINCG : public Min {
 public:
  MinTSPINCG(class LAMMPS *);
  ~MinTSPINCG() override;

  void init() override;
  void setup_style() override;
  void reset_vectors() override;
  int modify_param(int, char **) override;
  int iterate(int) override;

  double fnorm_sqr() override;
  double fnorm_inf() override;
  double fnorm_max() override;

 private:
  // line search selector (subset of MinLineSearch)
  int (MinTSPINCG::*linemin)(double, double &);

  // per-atom vector storage (flattened as 1d vectors length 3*nlocal)
  double *x0;     // atom coords at start of line search
  double *g;      // atom gradient (negative) at previous step
  double *h;      // atom search direction
  double *d0;     // spin "displacement" coords at start of line search
  double *gd;     // spin gradient (negative)
  double *hd;     // spin search direction

  // extra global dof (from fixes), x0 is stored by fix/modify
  double *gextra;
  double *hextra;

  // scratch buffers (local, reallocated on nlocal growth)
  int nlocal_max;
  double *fspin;    // negative gradient for spin dof (length 3*nlocal_max)
  double *dtrial;   // scratch trial d (length 3*nlocal_max)

  // mapping/scaling between spin field and a length-like dof d
  // paper notation (PhysRevB 111, 134412): eta_zeta (length/muB)
  // d = eta_zeta * M, where M is magnetic moment vector in muB
  double eta_zeta;
  int eta_auto;            // if 1, adapt eta_zeta automatically (still allows manual initial value)
  double eta_auto_weight;  // smoothing factor in [0,1]
  double eta_auto_min;     // clamp lower bound (length/muB)
  double eta_auto_max;     // clamp upper bound (length/muB)
  double spin_dmax;
  int vary_mag;    // if 1, allow |M| to change (optimize full M vector)

  double estimate_eta_zeta() const;
  void update_eta_zeta();

  void compute_fspin();
  void compute_d_from_sp(double *dvec) const;
  void set_sp_from_d(const double *dvec);

  int linemin_backtrack(double, double &);
  int linemin_quadratic(double, double &);
  double alpha_step(double, int);
};

}    // namespace LAMMPS_NS

#endif
#endif
