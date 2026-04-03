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
#ifdef LMP_KOKKOS
MinimizeStyle(tspin/cg/kk,MinTSPINCGKokkos);
MinimizeStyle(tspin/cg/kk/device,MinTSPINCGKokkos);
MinimizeStyle(tspin/cg/kk/host,MinTSPINCGKokkos);
#endif
// clang-format on
#else

#ifndef LMP_MIN_TSPIN_CG_KOKKOS_H
#define LMP_MIN_TSPIN_CG_KOKKOS_H

#ifdef LMP_KOKKOS

#include "min_kokkos.h"

namespace LAMMPS_NS {

class MinTSPINCGKokkos : public MinKokkos {
 public:
  MinTSPINCGKokkos(class LAMMPS *);
  ~MinTSPINCGKokkos() override;

  void init() override;
  void setup_style() override;
  void reset_vectors() override;
  int modify_param(int, char **) override;
  int iterate(int) override;

 double fnorm_sqr() override;
 double fnorm_inf() override;
 double fnorm_max() override;

 private:
  // line search selector
  int (MinTSPINCGKokkos::*linemin)(double, double &);

  // per-atom vector storage (flattened 1d vectors length 3*nmax)
  DAT::t_ffloat_1d x0;    // atom coords at start of line search
  DAT::t_ffloat_1d g;     // atom gradient (negative) at previous step
  DAT::t_ffloat_1d h;     // atom search direction
  DAT::t_ffloat_1d d0;    // spin "displacement" coords at start of line search
  DAT::t_ffloat_1d gd;    // spin gradient (negative)
  DAT::t_ffloat_1d hd;    // spin search direction

  // extra global dof (from fixes), x0 is stored by fix/modify
  double *gextra;
  double *hextra;

  // scratch buffers
  DAT::t_ffloat_1d fspin;     // negative gradient for spin dof (length 3*nmax)
  DAT::t_ffloat_1d dtrial;    // scratch trial d (length 3*nmax)

  // mapping/scaling between spin field and a length-like dof d
  double eta_zeta;    // length/muB
  double spin_dmax;   // max step in d space (length)
  double spin_ftol;   // optional spin-only force tolerance (length-based, same units as fspin)
  int vary_mag;       // if 1, allow |M| to change (optimize full M vector)
  int eta_auto;            // if 1, adapt eta_zeta automatically (still allows manual initial value)
  double eta_auto_weight;  // smoothing factor in [0,1]
  double eta_auto_min;     // clamp lower bound (length/muB)
  double eta_auto_max;     // clamp upper bound (length/muB)

  // NOTE:
  // These member functions launch Kokkos kernels using CUDA extended lambdas.
  // NVCC requires the enclosing member function to have public access, so they
  // cannot be private/protected within this class (similar to KOKKOS package
  // classes that avoid protected/private with CUDA).
 public:
  void compute_fspin();
  double estimate_eta_zeta();
  void update_eta_zeta();
  void compute_d_from_sp(DAT::t_ffloat_1d dvec) const;
  void set_sp_from_d(const DAT::t_ffloat_1d &dvec);

  int linemin_backtrack(double, double &);
  int linemin_quadratic(double, double &);
  double alpha_step(double, int);
};

}    // namespace LAMMPS_NS

#endif

#endif
#endif
