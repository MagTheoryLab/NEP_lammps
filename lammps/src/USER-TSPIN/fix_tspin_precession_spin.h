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

#ifdef FIX_CLASS
// clang-format off
FixStyle(tspin/precession/spin,FixTSPINPrecessionSpin);
// clang-format on
#else

#ifndef LMP_FIX_TSPIN_PRECESSION_SPIN_H
#define LMP_FIX_TSPIN_PRECESSION_SPIN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixTSPINPrecessionSpin : public Fix {
 public:
  FixTSPINPrecessionSpin(class LAMMPS *, int, char **);
  ~FixTSPINPrecessionSpin() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double compute_scalar() override;

 protected:
  enum { CONSTANT, EQUAL };

  int zeeman_flag;

  int varflag;
  int magfieldstyle;
  int magvar;
  int tesla_value_logged;
  char *magstr;

  double H_input_raw;    // parsed/evaluated Zeeman input in Tesla
  double H_field;    // effective field H in eV/muB (added directly to atom->fm)
  double nhx, nhy, nhz;
  double hx, hy, hz;

  int ilevel_respa;
  int eflag;
  double eprec, eprec_all;

  void set_field_components();
  void update_variable_field_if_needed();
};

}    // namespace LAMMPS_NS

#endif
#endif
