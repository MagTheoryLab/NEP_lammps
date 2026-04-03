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

#ifdef LMP_KOKKOS

#include "min_tspin_fire_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "fix_minimize_kokkos.h"
#include "output.h"
#include "timer.h"
#include "update.h"
#include "utils.h"

#include <cmath>

using namespace LAMMPS_NS;

namespace {

struct FireStats {
  double f2, v2, g2, u2, fdotv, gdotu, fmax2, gmax2;

  KOKKOS_INLINE_FUNCTION
  FireStats() : f2(0.0), v2(0.0), g2(0.0), u2(0.0), fdotv(0.0), gdotu(0.0), fmax2(0.0), gmax2(0.0) {}

  KOKKOS_INLINE_FUNCTION
  FireStats &operator+=(const FireStats &rhs)
  {
    f2 += rhs.f2;
    v2 += rhs.v2;
    g2 += rhs.g2;
    u2 += rhs.u2;
    fdotv += rhs.fdotv;
    gdotu += rhs.gdotu;
    fmax2 = (fmax2 > rhs.fmax2) ? fmax2 : rhs.fmax2;
    gmax2 = (gmax2 > rhs.gmax2) ? gmax2 : rhs.gmax2;
    return *this;
  }
};

struct VariableFireStats {
  double f2, v2, rho2, w2, g2, u2, fdotv, rhodotw, gdotu, fmax2, rhomax2, gmax2;

  KOKKOS_INLINE_FUNCTION
  VariableFireStats() :
    f2(0.0), v2(0.0), rho2(0.0), w2(0.0), g2(0.0), u2(0.0), fdotv(0.0), rhodotw(0.0),
    gdotu(0.0), fmax2(0.0), rhomax2(0.0), gmax2(0.0)
  {}

  KOKKOS_INLINE_FUNCTION
  VariableFireStats &operator+=(const VariableFireStats &rhs)
  {
    f2 += rhs.f2;
    v2 += rhs.v2;
    rho2 += rhs.rho2;
    w2 += rhs.w2;
    g2 += rhs.g2;
    u2 += rhs.u2;
    fdotv += rhs.fdotv;
    rhodotw += rhs.rhodotw;
    gdotu += rhs.gdotu;
    fmax2 = (fmax2 > rhs.fmax2) ? fmax2 : rhs.fmax2;
    rhomax2 = (rhomax2 > rhs.rhomax2) ? rhomax2 : rhs.rhomax2;
    gmax2 = (gmax2 > rhs.gmax2) ? gmax2 : rhs.gmax2;
    return *this;
  }
};

struct MaxPair {
  double d0, d1;

  KOKKOS_INLINE_FUNCTION
  MaxPair() : d0(0.0), d1(0.0) {}

  KOKKOS_INLINE_FUNCTION
  MaxPair &operator+=(const MaxPair &rhs)
  {
    d0 = (d0 > rhs.d0) ? d0 : rhs.d0;
    d1 = (d1 > rhs.d1) ? d1 : rhs.d1;
    return *this;
  }
};

struct MaxTriple {
  double d0, d1, d2;

  KOKKOS_INLINE_FUNCTION
  MaxTriple() : d0(0.0), d1(0.0), d2(0.0) {}

  KOKKOS_INLINE_FUNCTION
  MaxTriple &operator+=(const MaxTriple &rhs)
  {
    d0 = (d0 > rhs.d0) ? d0 : rhs.d0;
    d1 = (d1 > rhs.d1) ? d1 : rhs.d1;
    d2 = (d2 > rhs.d2) ? d2 : rhs.d2;
    return *this;
  }
};

}    // namespace

MinTSPINFireKokkos::MinTSPINFireKokkos(LAMMPS *lmp) : MinKokkos(lmp)
{
  kokkosable = 1;
  searchflag = 0;

  scale_lat = 1.0;
  scale_mag = 1.0;
  scale_spin = 1.0;
  dt_init = 0.1;
  dt_max = 1.0;
  disp_max = 0.02;
  angle_max = 0.1;
  mag_step = 0.0;
  mag_min = 0.0;
  p_smooth = 0.0;
  pw_lat= 1.0;
  pw_mag = 1.0;
  pw_spin = 1.0;
  mix_init = 0.1;
  dt_grow = 1.1;
  dt_shrink = 0.5;
  mix_decay = 0.99;
  n_accel = 5;
  eps_power = 1.0e-30;
  mix_tol_lat = 1.0e-20;
  mix_tol_mag = 1.0e-20;
  mix_tol_spin = 1.0e-20;
  ftol_mag = 0.0;
  ftol_spin = 0.0;
  ftol_mag_set = 0;
  ftol_spin_set = 0;
  eps_theta = 1.0e-12;
  vary_mag = 0;
  expert = 0;
  preset = PRESET_BALANCED;
  scale_mag_set = 0;
  mag_step_set = 0;
  mag_min_set = 0;
  dt_init_set = 0;
  dt_max_set = 0;
  disp_max_set = 0;
  angle_max_set = 0;
  p_smooth_set = 0;
  dt_grow_set = 0;
  dt_shrink_set = 0;
  mix_decay_set = 0;
  n_accel_set = 0;

  dt_fire = dt_init;
  mix_alpha = mix_init;
  n_pos = 0;
  p_smooth_bar = 0.0;
}

void MinTSPINFireKokkos::init()
{
  MinKokkos::init();
  apply_preset_defaults();

  if (dt_init <= 0.0) error->all(FLERR, "min tspin/fire/kk requires dt_init > 0");
  if (dt_max <= 0.0) error->all(FLERR, "min tspin/fire/kk requires dt_max > 0");
  if (dt_max < dt_init) error->all(FLERR, "min tspin/fire/kk requires dt_max >= dt_init");
  if (disp_max <= 0.0) error->all(FLERR, "min tspin/fire/kk requires rmax > 0");
  if (p_smooth < 0.0 || p_smooth >= 1.0) error->all(FLERR, "min tspin/fire/kk requires p_smooth in [0,1)");
  if (mix_init < 0.0 || mix_init > 1.0)
    error->all(FLERR, "min tspin/fire/kk requires mix_init in [0,1]");
  if (dt_grow < 1.0) error->all(FLERR, "min tspin/fire/kk requires dt_grow >= 1");
  if (dt_shrink <= 0.0 || dt_shrink >= 1.0) error->all(FLERR, "min tspin/fire/kk requires dt_shrink in (0,1)");
  if (mix_decay <= 0.0 || mix_decay > 1.0) error->all(FLERR, "min tspin/fire/kk requires mix_decay in (0,1]");
  if (n_accel < 0) error->all(FLERR, "min tspin/fire/kk requires n_accel >= 0");
  if (eps_power <= 0.0) error->all(FLERR, "min tspin/fire/kk requires eps_power > 0");
  if (eps_theta <= 0.0) error->all(FLERR, "min tspin/fire/kk requires eps_theta > 0");

  if (vary_mag) {
    if (scale_lat < 0.0) error->all(FLERR, "min tspin/fire/kk requires scale_lat >= 0");
    if (scale_mag < 0.0) error->all(FLERR, "min tspin/fire/kk requires scale_mag >= 0");
    if (scale_spin < 0.0) error->all(FLERR, "min tspin/fire/kk requires scale_spin >= 0");
    if (scale_lat == 0.0 && scale_mag == 0.0 && scale_spin == 0.0)
      error->all(FLERR, "min tspin/fire/kk requires at least one of scale_lat/scale_mag/scale_spin to be > 0");
    if (angle_max <= 0.0) error->all(FLERR, "min tspin/fire/kk requires angle_max > 0");
    if (mag_step <= 0.0) error->all(FLERR, "min tspin/fire/kk requires mag_step > 0");
    if (mag_min < 0.0) error->all(FLERR, "min tspin/fire/kk requires mag_min >= 0");
    if (pw_lat < 0.0 || pw_mag < 0.0 || pw_spin < 0.0)
      error->all(FLERR, "min tspin/fire/kk requires pw_lat/pw_mag/pw_spin >= 0");
    if (mix_tol_lat < 0.0 || mix_tol_mag < 0.0 || mix_tol_spin < 0.0)
      error->all(FLERR, "min tspin/fire/kk requires mix_tol_lat/mix_tol_mag/mix_tol_spin >= 0");
    if (ftol_mag < 0.0) error->all(FLERR, "min tspin/fire/kk requires ftol_mag >= 0");
    if (ftol_spin < 0.0) error->all(FLERR, "min tspin/fire/kk requires ftol_spin >= 0");
    if (!ftol_mag_set) ftol_mag = update->ftol;
    if (!ftol_spin_set) ftol_spin = update->ftol;
  } else {
    if (scale_mag_set || mag_step_set || mag_min_set || ftol_mag_set)
      error->all(FLERR, "min tspin/fire/kk vary_mag no does not accept scale_mag/mag_step/mag_min/ftol_mag");
    if (scale_lat < 0.0) error->all(FLERR, "min tspin/fire/kk requires scale_lat >= 0");
    if (scale_spin < 0.0) error->all(FLERR, "min tspin/fire/kk requires scale_spin >= 0");
    if (scale_lat == 0.0 && scale_spin == 0.0)
      error->all(FLERR, "min tspin/fire/kk requires at least one of scale_lat or scale_spin to be > 0");
    if (angle_max <= 0.0) error->all(FLERR, "min tspin/fire/kk requires angle_max > 0");
    if (mix_tol_lat < 0.0 || mix_tol_spin < 0.0)
      error->all(FLERR, "min tspin/fire/kk requires mix_tol_lat/mix_tol_spin >= 0");
    if (!ftol_spin_set) ftol_spin = update->ftol;
  }
}

void MinTSPINFireKokkos::setup_style()
{
  if (nextra_atom)
    error->all(FLERR, "min tspin/fire/kk does not support additional extra per-atom DOFs");
  if (!atom->sp_flag)
    error->all(FLERR, "min tspin/fire/kk requires atom/spin style");
  if (!atom->fm)
    error->all(FLERR, "min tspin/fire/kk requires atom_style spin with fm allocated");

  fix_minimize_kk->add_vector_kokkos();
  fix_minimize_kk->add_vector_kokkos();
  fix_minimize_kk->add_vector_kokkos();
  fix_minimize_kk->add_vector_kokkos();
  fix_minimize_kk->add_vector_kokkos();

  log_setup_summary();
}

void MinTSPINFireKokkos::apply_preset_defaults()
{
  if (!dt_init_set) {
    if (preset == PRESET_CONSERVATIVE) dt_init = 0.01;
    else if (preset == PRESET_BALANCED) dt_init = 0.02;
    else dt_init = 0.05;
  }
  if (!dt_max_set) {
    if (preset == PRESET_CONSERVATIVE) dt_max = 0.1;
    else if (preset == PRESET_BALANCED) dt_max = 0.2;
    else dt_max = 0.5;
  }
  if (!disp_max_set) {
    if (preset == PRESET_CONSERVATIVE) disp_max = 0.01;
    else if (preset == PRESET_BALANCED) disp_max = 0.02;
    else disp_max = 0.05;
  }
  if (!angle_max_set) {
    if (preset == PRESET_CONSERVATIVE) angle_max = 0.05;
    else if (preset == PRESET_BALANCED) angle_max = 0.1;
    else angle_max = 0.2;
  }
  if (!p_smooth_set) {
    if (vary_mag) {
      if (preset == PRESET_CONSERVATIVE) p_smooth = 0.5;
      else if (preset == PRESET_BALANCED) p_smooth = 0.4;
      else p_smooth = 0.2;
    } else {
      if (preset == PRESET_CONSERVATIVE) p_smooth = 0.3;
      else if (preset == PRESET_BALANCED) p_smooth = 0.2;
      else p_smooth = 0.0;
    }
  }
  if (!dt_grow_set) {
    if (preset == PRESET_CONSERVATIVE) dt_grow = 1.05;
    else if (preset == PRESET_BALANCED) dt_grow = 1.1;
    else dt_grow = 1.2;
  }
  if (!dt_shrink_set) dt_shrink = 0.5;
  if (!mix_decay_set) {
    if (preset == PRESET_AGGRESSIVE) mix_decay = 0.98;
    else mix_decay = 0.99;
  }
  if (!n_accel_set) {
    if (preset == PRESET_CONSERVATIVE) n_accel = 8;
    else if (preset == PRESET_BALANCED) n_accel = 5;
    else n_accel = 3;
  }
  if (vary_mag && !mag_step_set) {
    if (preset == PRESET_CONSERVATIVE) mag_step = 0.01;
    else if (preset == PRESET_BALANCED) mag_step = 0.02;
    else mag_step = 0.05;
  }
}

void MinTSPINFireKokkos::log_setup_summary() const
{
  const char *preset_name = "balanced";
  if (preset == PRESET_CONSERVATIVE) preset_name = "conservative";
  else if (preset == PRESET_AGGRESSIVE) preset_name = "aggressive";

  utils::logmesg(lmp,
                 "  tspin/fire/kk summary: vary_mag={} preset={} expert={} scale_lat={} scale_spin={} "
                 "scale_mag={} dt_init={} dt_max={} disp_max={} angle_max={} mag_step={} "
                 "mag_min={} ftol_r={} ftol_mag={} ftol_spin={}\n",
                 vary_mag ? "yes" : "no", preset_name, expert ? "yes" : "no", scale_lat,
                 scale_spin, vary_mag ? scale_mag : 0.0, dt_init, dt_max, disp_max,
                 angle_max, vary_mag ? mag_step : 0.0, vary_mag ? mag_min : 0.0,
                 (scale_lat > 0.0) ? update->ftol : 0.0, vary_mag ? ftol_mag : 0.0, ftol_spin);
}

int MinTSPINFireKokkos::modify_param(int narg, char **arg)
{
  const auto require_expert = [this]() {
    if (!expert)
      error->all(FLERR, "min tspin/fire/kk expert parameter requires 'min_modify expert yes' first");
  };

  if (strcmp(arg[0], "preset") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    if (strcmp(arg[1], "conservative") == 0) preset = PRESET_CONSERVATIVE;
    else if (strcmp(arg[1], "balanced") == 0) preset = PRESET_BALANCED;
    else if (strcmp(arg[1], "aggressive") == 0) preset = PRESET_AGGRESSIVE;
    else error->all(FLERR, "min tspin/fire/kk requires preset = conservative, balanced, or aggressive");
    return 2;
  }
  if (strcmp(arg[0], "expert") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    expert = utils::logical(FLERR, arg[1], false, lmp);
    return 2;
  }
  if (strcmp(arg[0], "vary_mag") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    vary_mag = utils::logical(FLERR, arg[1], false, lmp);
    return 2;
  }
  if (strcmp(arg[0], "scale_lat") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    scale_lat = utils::numeric(FLERR, arg[1], false, lmp);
    if (scale_lat < 0.0) error->all(FLERR, "min tspin/fire/kk requires scale_lat >= 0");
    return 2;
  }
  if (strcmp(arg[0], "scale_spin") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    scale_spin = utils::numeric(FLERR, arg[1], false, lmp);
    if (scale_spin < 0.0) error->all(FLERR, "min tspin/fire/kk requires scale_spin >= 0");
    return 2;
  }
  if (strcmp(arg[0], "scale_mag") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    scale_mag = utils::numeric(FLERR, arg[1], false, lmp);
    if (scale_mag < 0.0) error->all(FLERR, "min tspin/fire/kk requires scale_mag >= 0");
    scale_mag_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "dt_init") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    dt_init = utils::numeric(FLERR, arg[1], false, lmp);
    if (dt_init <= 0.0) error->all(FLERR, "min tspin/fire/kk requires dt_init > 0");
    dt_init_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "dt_max") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    dt_max = utils::numeric(FLERR, arg[1], false, lmp);
    if (dt_max <= 0.0) error->all(FLERR, "min tspin/fire/kk requires dt_max > 0");
    dt_max_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "disp_max") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    disp_max = utils::numeric(FLERR, arg[1], false, lmp);
    if (disp_max <= 0.0) error->all(FLERR, "min tspin/fire/kk requires rmax > 0");
    disp_max_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "angle_max") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    angle_max = utils::numeric(FLERR, arg[1], false, lmp);
    if (angle_max <= 0.0) error->all(FLERR, "min tspin/fire/kk requires angle_max > 0");
    angle_max_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mag_step") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mag_step = utils::numeric(FLERR, arg[1], false, lmp);
    if (mag_step <= 0.0) error->all(FLERR, "min tspin/fire/kk requires mag_step > 0");
    mag_step_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mag_min") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mag_min = utils::numeric(FLERR, arg[1], false, lmp);
    if (mag_min < 0.0) error->all(FLERR, "min tspin/fire/kk requires mag_min >= 0");
    mag_min_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "pw_lat") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    pw_lat= utils::numeric(FLERR, arg[1], false, lmp);
    return 2;
  }
  if (strcmp(arg[0], "pw_mag") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    pw_mag = utils::numeric(FLERR, arg[1], false, lmp);
    return 2;
  }
  if (strcmp(arg[0], "pw_spin") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    pw_spin = utils::numeric(FLERR, arg[1], false, lmp);
    return 2;
  }
  if (strcmp(arg[0], "p_smooth") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    p_smooth = utils::numeric(FLERR, arg[1], false, lmp);
    if (p_smooth < 0.0 || p_smooth >= 1.0) error->all(FLERR, "min tspin/fire/kk requires p_smooth in [0,1)");
    p_smooth_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mix_init") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_init = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_init < 0.0 || mix_init > 1.0)
      error->all(FLERR, "min tspin/fire/kk requires mix_init in [0,1]");
    return 2;
  }
  if (strcmp(arg[0], "dt_grow") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    dt_grow = utils::numeric(FLERR, arg[1], false, lmp);
    if (dt_grow < 1.0) error->all(FLERR, "min tspin/fire/kk requires dt_grow >= 1");
    dt_grow_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "dt_shrink") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    dt_shrink = utils::numeric(FLERR, arg[1], false, lmp);
    if (dt_shrink <= 0.0 || dt_shrink >= 1.0) error->all(FLERR, "min tspin/fire/kk requires dt_shrink in (0,1)");
    dt_shrink_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mix_decay") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_decay = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_decay <= 0.0 || mix_decay > 1.0) error->all(FLERR, "min tspin/fire/kk requires mix_decay in (0,1]");
    mix_decay_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "n_accel") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    n_accel = utils::inumeric(FLERR, arg[1], false, lmp);
    if (n_accel < 0) error->all(FLERR, "min tspin/fire/kk requires n_accel >= 0");
    n_accel_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "eps_power") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eps_power = utils::numeric(FLERR, arg[1], false, lmp);
    if (eps_power <= 0.0) error->all(FLERR, "min tspin/fire/kk requires eps_power > 0");
    return 2;
  }
  if (strcmp(arg[0], "mix_tol_lat") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_tol_lat = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_tol_lat < 0.0) error->all(FLERR, "min tspin/fire/kk requires mix_tol_lat >= 0");
    return 2;
  }
  if (strcmp(arg[0], "mix_tol_mag") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_tol_mag = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_tol_mag < 0.0) error->all(FLERR, "min tspin/fire/kk requires mix_tol_mag >= 0");
    return 2;
  }
  if (strcmp(arg[0], "mix_tol_spin") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_tol_spin = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_tol_spin < 0.0) error->all(FLERR, "min tspin/fire/kk requires mix_tol_spin >= 0");
    return 2;
  }
  if (strcmp(arg[0], "ftol_mag") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    ftol_mag = utils::numeric(FLERR, arg[1], false, lmp);
    if (ftol_mag < 0.0) error->all(FLERR, "min tspin/fire/kk requires ftol_mag >= 0");
    ftol_mag_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "ftol_spin") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    ftol_spin = utils::numeric(FLERR, arg[1], false, lmp);
    if (ftol_spin < 0.0) error->all(FLERR, "min tspin/fire/kk requires ftol_spin >= 0");
    ftol_spin_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "eps_theta") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eps_theta = utils::numeric(FLERR, arg[1], false, lmp);
    if (eps_theta <= 0.0) error->all(FLERR, "min tspin/fire/kk requires eps_theta > 0");
    return 2;
  }

  if (strcmp(arg[0], "lambda_s") == 0)
    error->all(FLERR, "min tspin/fire/kk no longer accepts lambda_s; use scale_spin");
  if (strcmp(arg[0], "ws") == 0)
    error->all(FLERR, "min tspin/fire/kk no longer accepts ws; use pw_spin with 'min_modify expert yes'");
  if (strcmp(arg[0], "thetamax") == 0)
    error->all(FLERR, "min tspin/fire/kk no longer accepts thetamax; use angle_max");
  if (strcmp(arg[0], "spin_ftol") == 0)
    error->all(FLERR, "min tspin/fire/kk no longer accepts spin_ftol; use ftol_spin");
  if (strcmp(arg[0], "mix_eps") == 0)
    error->all(FLERR, "min tspin/fire/kk no longer accepts mix_eps; use mix_tol_lat and mix_tol_spin with 'min_modify expert yes'");
  return 0;
}

void MinTSPINFireKokkos::reset_vectors()
{
  nvec = 3 * atom->nlocal;

  atomKK->sync(Device, X_MASK | F_MASK);
  auto d_x = atomKK->k_x.d_view;
  auto d_f = atomKK->k_f.d_view;

  if (nvec) xvec = DAT::t_ffloat_1d(d_x.data(), d_x.size());
  if (nvec) fvec = DAT::t_ffloat_1d(d_f.data(), d_f.size());

  v_fire = fix_minimize_kk->request_vector_kokkos(0);
  u_fire = fix_minimize_kk->request_vector_kokkos(1);
  g_spin = fix_minimize_kk->request_vector_kokkos(2);
  w_fire = fix_minimize_kk->request_vector_kokkos(3);
  g_rho = fix_minimize_kk->request_vector_kokkos(4);

  fix_minimize_kk->k_vectors.modify<LMPDeviceType>();
}

void MinTSPINFireKokkos::compute_projected_spin_gradient()
{
  const int nlocal = atom->nlocal;
  atomKK->sync(Device, SP_MASK | FM_MASK);
  auto sp = atomKK->k_sp.d_view;
  auto fm = atomKK->k_fm.d_view;
  auto g = g_spin;

  Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
    const int j = 3 * i;
    if (sp(i, 3) <= 0.0) {
      g(j + 0) = 0.0;
      g(j + 1) = 0.0;
      g(j + 2) = 0.0;
      return;
    }

    const double sx = sp(i, 0);
    const double sy = sp(i, 1);
    const double sz = sp(i, 2);
    const double hdot = fm(i, 0) * sx + fm(i, 1) * sy + fm(i, 2) * sz;

    g(j + 0) = fm(i, 0) - hdot * sx;
    g(j + 1) = fm(i, 1) - hdot * sy;
    g(j + 2) = fm(i, 2) - hdot * sz;
  });
}

void MinTSPINFireKokkos::compute_variable_spin_gradients()
{
  const int nlocal = atom->nlocal;
  atomKK->sync(Device, SP_MASK | FM_MASK);
  auto sp = atomKK->k_sp.d_view;
  auto fm = atomKK->k_fm.d_view;
  auto g = g_spin;
  auto grho = g_rho;

  Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
    const int j = 3 * i;
    if (sp(i, 3) <= 0.0) {
      grho(j + 0) = 0.0;
      grho(j + 1) = 0.0;
      grho(j + 2) = 0.0;
      g(j + 0) = 0.0;
      g(j + 1) = 0.0;
      g(j + 2) = 0.0;
      return;
    }

    const double sx = sp(i, 0);
    const double sy = sp(i, 1);
    const double sz = sp(i, 2);
    const double rho = sp(i, 3);
    const double hpar = fm(i, 0) * sx + fm(i, 1) * sy + fm(i, 2) * sz;

    grho(j + 0) = hpar;
    grho(j + 1) = 0.0;
    grho(j + 2) = 0.0;

    g(j + 0) = rho * (fm(i, 0) - hpar * sx);
    g(j + 1) = rho * (fm(i, 1) - hpar * sy);
    g(j + 2) = rho * (fm(i, 2) - hpar * sz);
  });
}

double MinTSPINFireKokkos::fnorm_sqr()
{
  if (vary_mag) {
    atomKK->sync(Device, F_MASK | SP_MASK | FM_MASK);
    compute_variable_spin_gradients();

    auto f = atomKK->k_f.d_view;
    auto g = g_spin;
    auto grho = g_rho;
    const int nlocal = atom->nlocal;
    const int lattice_active = (scale_lat > 0.0) ? 1 : 0;
    const int rho_active = (scale_mag > 0.0) ? 1 : 0;
    const int dir_active = (scale_spin > 0.0) ? 1 : 0;
    double local_norm2 = 0.0;

    Kokkos::parallel_reduce(
      nlocal,
      LAMMPS_LAMBDA(const int &i, double &sum) {
        const int j = 3 * i;
        if (lattice_active) sum += f(i, 0) * f(i, 0) + f(i, 1) * f(i, 1) + f(i, 2) * f(i, 2);
        if (rho_active) sum += grho(j + 0) * grho(j + 0);
        if (dir_active) sum += g(j + 0) * g(j + 0) + g(j + 1) * g(j + 1) + g(j + 2) * g(j + 2);
      },
      local_norm2);

    double norm2 = local_norm2;
    MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, MPI_DOUBLE, MPI_SUM, world);
    if (nextra_global)
      for (int i = 0; i < nextra_global; i++) norm2 += fextra[i] * fextra[i];
    return norm2;
  }

  atomKK->sync(Device, F_MASK | SP_MASK | FM_MASK);
  compute_projected_spin_gradient();

  double local_norm2 = 0.0;
  auto g = g_spin;
  auto f = atomKK->k_f.d_view;
  const int nlocal = atom->nlocal;
  const int lattice_active = (scale_lat > 0.0) ? 1 : 0;
  const int spin_active = (scale_spin > 0.0) ? 1 : 0;

  Kokkos::parallel_reduce(
    nlocal,
    LAMMPS_LAMBDA(const int &i, double &sum) {
      const int j = 3 * i;
      if (lattice_active) sum += f(i, 0) * f(i, 0) + f(i, 1) * f(i, 1) + f(i, 2) * f(i, 2);
      if (spin_active) sum += g(j + 0) * g(j + 0) + g(j + 1) * g(j + 1) + g(j + 2) * g(j + 2);
    },
    local_norm2);

  double norm2 = local_norm2;
  MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm2 += fextra[i] * fextra[i];

  return norm2;
}

double MinTSPINFireKokkos::fnorm_inf()
{
  if (vary_mag) {
    atomKK->sync(Device, F_MASK | SP_MASK | FM_MASK);
    compute_variable_spin_gradients();

    auto f = atomKK->k_f.d_view;
    auto g = g_spin;
    auto grho = g_rho;
    const int nlocal = atom->nlocal;
    const int lattice_active = (scale_lat > 0.0) ? 1 : 0;
    const int rho_active = (scale_mag > 0.0) ? 1 : 0;
    const int dir_active = (scale_spin > 0.0) ? 1 : 0;
    MaxTriple local;

    Kokkos::parallel_reduce(
      nlocal,
      LAMMPS_LAMBDA(const int &i, MaxTriple &val) {
        const int j = 3 * i;
        if (lattice_active) {
          val.d0 = MAX(val.d0, f(i, 0) * f(i, 0));
          val.d0 = MAX(val.d0, f(i, 1) * f(i, 1));
          val.d0 = MAX(val.d0, f(i, 2) * f(i, 2));
        }
        if (rho_active) val.d1 = MAX(val.d1, grho(j + 0) * grho(j + 0));
        if (dir_active) {
          val.d2 = MAX(val.d2, g(j + 0) * g(j + 0));
          val.d2 = MAX(val.d2, g(j + 1) * g(j + 1));
          val.d2 = MAX(val.d2, g(j + 2) * g(j + 2));
        }
      },
      local);

    double local_vals[3] = {local.d0, local.d1, local.d2};
    double global_vals[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local_vals, global_vals, 3, MPI_DOUBLE, MPI_MAX, world);

    double norm_inf = MAX(global_vals[0], MAX(global_vals[1], global_vals[2]));
    if (nextra_global)
      for (int i = 0; i < nextra_global; i++) norm_inf = MAX(norm_inf, fextra[i] * fextra[i]);
    return norm_inf;
  }

  atomKK->sync(Device, F_MASK | SP_MASK | FM_MASK);
  compute_projected_spin_gradient();

  MaxPair local;
  auto g = g_spin;
  auto f = atomKK->k_f.d_view;
  const int nlocal = atom->nlocal;
  const int lattice_active = (scale_lat > 0.0) ? 1 : 0;
  const int spin_active = (scale_spin > 0.0) ? 1 : 0;

  Kokkos::parallel_reduce(
    nlocal,
    LAMMPS_LAMBDA(const int &i, MaxPair &val) {
      const int j = 3 * i;
      if (lattice_active) {
        val.d0 = MAX(val.d0, f(i, 0) * f(i, 0));
        val.d0 = MAX(val.d0, f(i, 1) * f(i, 1));
        val.d0 = MAX(val.d0, f(i, 2) * f(i, 2));
      }
      if (spin_active) {
        val.d1 = MAX(val.d1, g(j + 0) * g(j + 0));
        val.d1 = MAX(val.d1, g(j + 1) * g(j + 1));
        val.d1 = MAX(val.d1, g(j + 2) * g(j + 2));
      }
    },
    local);

  double local_vals[2] = {local.d0, local.d1};
  double global_vals[2] = {0.0, 0.0};
  MPI_Allreduce(local_vals, global_vals, 2, MPI_DOUBLE, MPI_MAX, world);

  double norm_inf = MAX(global_vals[0], global_vals[1]);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm_inf = MAX(norm_inf, fextra[i] * fextra[i]);

  return norm_inf;
}

double MinTSPINFireKokkos::fnorm_max()
{
  if (vary_mag) {
    atomKK->sync(Device, F_MASK | SP_MASK | FM_MASK);
    compute_variable_spin_gradients();

    auto f = atomKK->k_f.d_view;
    auto g = g_spin;
    auto grho = g_rho;
    const int nlocal = atom->nlocal;
    const int lattice_active = (scale_lat > 0.0) ? 1 : 0;
    const int rho_active = (scale_mag > 0.0) ? 1 : 0;
    const int dir_active = (scale_spin > 0.0) ? 1 : 0;
    MaxTriple local;

    Kokkos::parallel_reduce(
      nlocal,
      LAMMPS_LAMBDA(const int &i, MaxTriple &val) {
        const int j = 3 * i;
        const double f2 = lattice_active ? (f(i, 0) * f(i, 0) + f(i, 1) * f(i, 1) + f(i, 2) * f(i, 2)) : 0.0;
        const double rho2 = rho_active ? (grho(j + 0) * grho(j + 0)) : 0.0;
        const double g2 = dir_active ? (g(j + 0) * g(j + 0) + g(j + 1) * g(j + 1) + g(j + 2) * g(j + 2)) : 0.0;
        val.d0 = MAX(val.d0, f2);
        val.d1 = MAX(val.d1, rho2);
        val.d2 = MAX(val.d2, g2);
      },
      local);

    double local_vals[3] = {local.d0, local.d1, local.d2};
    double global_vals[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local_vals, global_vals, 3, MPI_DOUBLE, MPI_MAX, world);

    double norm_max = MAX(global_vals[0], MAX(global_vals[1], global_vals[2]));
    if (nextra_global)
      for (int i = 0; i < nextra_global; i++) norm_max = MAX(norm_max, fextra[i] * fextra[i]);
    return norm_max;
  }

  atomKK->sync(Device, F_MASK | SP_MASK | FM_MASK);
  compute_projected_spin_gradient();

  MaxPair local;
  auto g = g_spin;
  auto f = atomKK->k_f.d_view;
  const int nlocal = atom->nlocal;
  const int lattice_active = (scale_lat > 0.0) ? 1 : 0;
  const int spin_active = (scale_spin > 0.0) ? 1 : 0;

  Kokkos::parallel_reduce(
    nlocal,
    LAMMPS_LAMBDA(const int &i, MaxPair &val) {
      const int j = 3 * i;
      const double f2 = lattice_active ? (f(i, 0) * f(i, 0) + f(i, 1) * f(i, 1) + f(i, 2) * f(i, 2)) : 0.0;
      const double g2 = spin_active ? (g(j + 0) * g(j + 0) + g(j + 1) * g(j + 1) + g(j + 2) * g(j + 2)) : 0.0;
      val.d0 = MAX(val.d0, f2);
      val.d1 = MAX(val.d1, g2);
    },
    local);

  double local_vals[2] = {local.d0, local.d1};
  double global_vals[2] = {0.0, 0.0};
  MPI_Allreduce(local_vals, global_vals, 2, MPI_DOUBLE, MPI_MAX, world);

  double norm_max = MAX(global_vals[0], global_vals[1]);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm_max = MAX(norm_max, fextra[i] * fextra[i]);

  return norm_max;
}

int MinTSPINFireKokkos::iterate(int maxiter)
{
  return vary_mag ? iterate_variable(maxiter) : iterate_fixed(maxiter);
}

int MinTSPINFireKokkos::iterate_fixed(int maxiter)
{
  fix_minimize_kk->k_vectors.sync<LMPDeviceType>();
  fix_minimize_kk->k_vectors.modify<LMPDeviceType>();
  atomKK->sync(Device, X_MASK | F_MASK | SP_MASK | FM_MASK);

  Kokkos::deep_copy(v_fire, 0.0);
  Kokkos::deep_copy(u_fire, 0.0);
  Kokkos::deep_copy(g_spin, 0.0);
  Kokkos::deep_copy(w_fire, 0.0);
  Kokkos::deep_copy(g_rho, 0.0);

  dt_fire = dt_init;
  mix_alpha = mix_init;
  n_pos = 0;
  p_smooth_bar = 0.0;

  int nlocal = atom->nlocal;

  for (int iter = 0; iter < maxiter; iter++) {
    if (timer->check_timeout(niter)) return TIMEOUT;

    const bigint ntimestep = ++update->ntimestep;
    const double dtau_step = dt_fire;
    niter++;
    nlocal = atom->nlocal;

    {
      auto f = atomKK->k_f.d_view;
      auto sp = atomKK->k_sp.d_view;
      auto fm = atomKK->k_fm.d_view;
      auto v = v_fire;
      auto u = u_fire;
      auto g = g_spin;
      const double dtau = dtau_step;
      const double lr = scale_lat;
      const double ls = scale_spin;

      Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
        const int j = 3 * i;
        const double fx = f(i, 0);
        const double fy = f(i, 1);
        const double fz = f(i, 2);

        if (lr > 0.0) {
          v(j + 0) += dtau * lr * fx;
          v(j + 1) += dtau * lr * fy;
          v(j + 2) += dtau * lr * fz;
        } else {
          v(j + 0) = 0.0;
          v(j + 1) = 0.0;
          v(j + 2) = 0.0;
        }

        if (sp(i, 3) <= 0.0) {
          g(j + 0) = 0.0;
          g(j + 1) = 0.0;
          g(j + 2) = 0.0;
          u(j + 0) = 0.0;
          u(j + 1) = 0.0;
          u(j + 2) = 0.0;
          return;
        }

        const double sx = sp(i, 0);
        const double sy = sp(i, 1);
        const double sz = sp(i, 2);
        const double hdot = fm(i, 0) * sx + fm(i, 1) * sy + fm(i, 2) * sz;
        const double gx = fm(i, 0) - hdot * sx;
        const double gy = fm(i, 1) - hdot * sy;
        const double gz = fm(i, 2) - hdot * sz;

        g(j + 0) = gx;
        g(j + 1) = gy;
        g(j + 2) = gz;

        if (ls > 0.0) {
          double ux = u(j + 0) + dtau * ls * gx;
          double uy = u(j + 1) + dtau * ls * gy;
          double uz = u(j + 2) + dtau * ls * gz;
          const double udot = ux * sx + uy * sy + uz * sz;
          ux -= udot * sx;
          uy -= udot * sy;
          uz -= udot * sz;

          u(j + 0) = ux;
          u(j + 1) = uy;
          u(j + 2) = uz;
        } else {
          u(j + 0) = 0.0;
          u(j + 1) = 0.0;
          u(j + 2) = 0.0;
        }
      });
    }

    FireStats stats;
    {
      auto f = atomKK->k_f.d_view;
      auto sp = atomKK->k_sp.d_view;
      auto v = v_fire;
      auto u = u_fire;
      auto g = g_spin;
      const int lattice_active = (scale_lat > 0.0) ? 1 : 0;
      const int spin_active = (scale_spin > 0.0) ? 1 : 0;

      Kokkos::parallel_reduce(
        nlocal,
        LAMMPS_LAMBDA(const int &i, FireStats &sum) {
          const int j = 3 * i;
          const double fx = f(i, 0);
          const double fy = f(i, 1);
          const double fz = f(i, 2);
          const double vx = v(j + 0);
          const double vy = v(j + 1);
          const double vz = v(j + 2);

          const double fi2 = fx * fx + fy * fy + fz * fz;
          const double vi2 = vx * vx + vy * vy + vz * vz;
          if (lattice_active) {
            sum.f2 += fi2;
            sum.v2 += vi2;
            sum.fdotv += fx * vx + fy * vy + fz * vz;
            sum.fmax2 = MAX(sum.fmax2, fi2);
          }

          if (sp(i, 3) <= 0.0) return;

          const double gx = g(j + 0);
          const double gy = g(j + 1);
          const double gz = g(j + 2);
          const double ux = u(j + 0);
          const double uy = u(j + 1);
          const double uz = u(j + 2);
          const double gi2 = gx * gx + gy * gy + gz * gz;
          const double ui2 = ux * ux + uy * uy + uz * uz;
          if (spin_active) {
            sum.g2 += gi2;
            sum.u2 += ui2;
            sum.gdotu += gx * ux + gy * uy + gz * uz;
            sum.gmax2 = MAX(sum.gmax2, gi2);
          }
        },
        stats);
    }

    double local_stats[6] = {stats.f2, stats.v2, stats.g2, stats.u2, stats.fdotv, stats.gdotu};
    double global_stats[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(local_stats, global_stats, 6, MPI_DOUBLE, MPI_SUM, world);

    const double fnorm = sqrt(global_stats[0]);
    const double vnorm = sqrt(global_stats[1]);
    const double gnorm = sqrt(global_stats[2]);
    const double unorm = sqrt(global_stats[3]);
    const double pr = (scale_lat > 0.0) ? global_stats[4] / (fnorm * vnorm + eps_power) : 0.0;
    const double ps = (scale_spin > 0.0) ? global_stats[5] / (gnorm * unorm + eps_power) : 0.0;
    const double p = ((scale_lat > 0.0) ? pw_lat * pr : 0.0) + ((scale_spin > 0.0) ? pw_spin * ps : 0.0);

    if (p_smooth > 0.0) p_smooth_bar = p_smooth * p_smooth_bar + (1.0 - p_smooth) * p;
    else p_smooth_bar = p;
    const double p_use = (p_smooth > 0.0) ? p_smooth_bar : p;

    if (p_use > 0.0) {
      const int mix_r = (scale_lat > 0.0 && fnorm > mix_tol_lat && vnorm > mix_tol_lat) ? 1 : 0;
      const int mix_s = (scale_spin > 0.0 && gnorm > mix_tol_spin && unorm > mix_tol_spin) ? 1 : 0;
      const double keep = 1.0 - mix_alpha;
      const double mix_r_scale = mix_r ? mix_alpha * (vnorm / fnorm) : 0.0;
      const double mix_s_scale = mix_s ? mix_alpha * (unorm / gnorm) : 0.0;

      {
        auto f = atomKK->k_f.d_view;
        auto sp = atomKK->k_sp.d_view;
        auto v = v_fire;
        auto u = u_fire;
        auto g = g_spin;

        Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
          const int j = 3 * i;

          if (mix_r) {
            v(j + 0) = keep * v(j + 0) + mix_r_scale * f(i, 0);
            v(j + 1) = keep * v(j + 1) + mix_r_scale * f(i, 1);
            v(j + 2) = keep * v(j + 2) + mix_r_scale * f(i, 2);
          }

          if (sp(i, 3) <= 0.0) {
            u(j + 0) = 0.0;
            u(j + 1) = 0.0;
            u(j + 2) = 0.0;
            return;
          }

          if (mix_s) {
            double ux = keep * u(j + 0) + mix_s_scale * g(j + 0);
            double uy = keep * u(j + 1) + mix_s_scale * g(j + 1);
            double uz = keep * u(j + 2) + mix_s_scale * g(j + 2);
            const double sx = sp(i, 0);
            const double sy = sp(i, 1);
            const double sz = sp(i, 2);
            const double udot = ux * sx + uy * sy + uz * sz;
            u(j + 0) = ux - udot * sx;
            u(j + 1) = uy - udot * sy;
            u(j + 2) = uz - udot * sz;
          }
        });
      }

      if (n_pos >= n_accel) {
        dt_fire = MIN(dt_grow * dt_fire, dt_max);
        mix_alpha *= mix_decay;
      }
      n_pos++;
    } else {
      Kokkos::deep_copy(v_fire, 0.0);
      Kokkos::deep_copy(u_fire, 0.0);
      dt_fire *= dt_shrink;
      mix_alpha = mix_init;
      n_pos = 0;
    }

    {
      auto x = atomKK->k_x.d_view;
      auto sp = atomKK->k_sp.d_view;
      auto v = v_fire;
      auto u = u_fire;
      const double lr = scale_lat;
      const double ls = scale_spin;
      const double dtau = dtau_step;
      const double drmax = disp_max;
      const double dthetamax = angle_max;
      const double eps_small_theta = eps_theta;

      Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
        const int j = 3 * i;

        double drx = 0.0;
        double dry = 0.0;
        double drz = 0.0;
        if (lr > 0.0) {
          drx = dtau * v(j + 0);
          dry = dtau * v(j + 1);
          drz = dtau * v(j + 2);
          const double dr2 = drx * drx + dry * dry + drz * drz;
          if (dr2 > drmax * drmax) {
            const double scale = drmax / sqrt(dr2);
            drx *= scale;
            dry *= scale;
            drz *= scale;
          }
        }

        x(i, 0) += drx;
        x(i, 1) += dry;
        x(i, 2) += drz;

        if (sp(i, 3) <= 0.0) {
          u(j + 0) = 0.0;
          u(j + 1) = 0.0;
          u(j + 2) = 0.0;
          return;
        }

        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double delta2 = 0.0;
        if (ls > 0.0) {
          dx = dtau * u(j + 0);
          dy = dtau * u(j + 1);
          dz = dtau * u(j + 2);
          delta2 = dx * dx + dy * dy + dz * dz;
          if (delta2 > dthetamax * dthetamax) {
            const double scale = dthetamax / sqrt(delta2);
            dx *= scale;
            dy *= scale;
            dz *= scale;
            delta2 = dx * dx + dy * dy + dz * dz;
          }
        }

        const double sx = sp(i, 0);
        const double sy = sp(i, 1);
        const double sz = sp(i, 2);

        const double theta2 = dx * dx + dy * dy + dz * dz;
        double enx, eny, enz;

        if (theta2 > eps_small_theta * eps_small_theta) {
          const double theta = sqrt(theta2);
          const double c = cos(theta);
          const double s_over_theta = sin(theta) / theta;
          enx = c * sx + s_over_theta * dx;
          eny = c * sy + s_over_theta * dy;
          enz = c * sz + s_over_theta * dz;
        } else {
          enx = sx + dx;
          eny = sy + dy;
          enz = sz + dz;
        }

        const double enorm = sqrt(enx * enx + eny * eny + enz * enz);
        if (enorm > 0.0) {
          enx /= enorm;
          eny /= enorm;
          enz /= enorm;
        } else {
          enx = sx;
          eny = sy;
          enz = sz;
        }

        sp(i, 0) = enx;
        sp(i, 1) = eny;
        sp(i, 2) = enz;

        const double udot_new = u(j + 0) * enx + u(j + 1) * eny + u(j + 2) * enz;
        u(j + 0) -= udot_new * enx;
        u(j + 1) -= udot_new * eny;
        u(j + 2) -= udot_new * enz;
      });
    }

    atomKK->modified(Device, X_MASK | SP_MASK);
    fix_minimize_kk->k_vectors.modify<LMPDeviceType>();

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    if (neval >= update->max_eval) return MAXEVAL;

    nlocal = atom->nlocal;
    atomKK->sync(Device, F_MASK | SP_MASK | FM_MASK);
    compute_projected_spin_gradient();

    MaxPair current_max;
    {
      auto f = atomKK->k_f.d_view;
      auto g = g_spin;

      Kokkos::parallel_reduce(
        nlocal,
        LAMMPS_LAMBDA(const int &i, MaxPair &val) {
          const int j = 3 * i;
          const double f2 = f(i, 0) * f(i, 0) + f(i, 1) * f(i, 1) + f(i, 2) * f(i, 2);
          const double g2 = g(j + 0) * g(j + 0) + g(j + 1) * g(j + 1) + g(j + 2) * g(j + 2);
          val.d0 = MAX(val.d0, f2);
          val.d1 = MAX(val.d1, g2);
        },
        current_max);
    }

    double max_local[2] = {current_max.d0, current_max.d1};
    double max_global[2] = {0.0, 0.0};
    MPI_Allreduce(max_local, max_global, 2, MPI_DOUBLE, MPI_MAX, world);

    const double delta_e = fabs(ecurrent - eprevious);
    const double fmax = sqrt(max_global[0]);
    const double gmax = sqrt(max_global[1]);
    const double spin_ftol_spinff = (scale_spin > 0.0) ? ftol_spin : 0.0;
    const double ftol_spinff = (scale_lat > 0.0) ? update->ftol : 0.0;

    const int criteria_active = (update->etol > 0.0) || (ftol_spinff > 0.0) || (spin_ftol_spinff > 0.0);
    const int etol_ok = (update->etol <= 0.0) || (delta_e < update->etol);
    const int ftol_ok = (ftol_spinff <= 0.0) || (fmax < ftol_spinff);
    const int sftol_ok = (spin_ftol_spinff <= 0.0) || (gmax < spin_ftol_spinff);

    if (criteria_active && etol_ok && ftol_ok && sftol_ok) {
      if (ftol_spinff > 0.0 || spin_ftol_spinff > 0.0) return FTOL;
      return ETOL;
    }

    if (output->next == ntimestep) {
      atomKK->sync(Host, ALL_MASK);
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}

int MinTSPINFireKokkos::iterate_variable(int maxiter)
{
  fix_minimize_kk->k_vectors.sync<LMPDeviceType>();
  fix_minimize_kk->k_vectors.modify<LMPDeviceType>();
  atomKK->sync(Device, X_MASK | F_MASK | SP_MASK | FM_MASK);

  Kokkos::deep_copy(v_fire, 0.0);
  Kokkos::deep_copy(u_fire, 0.0);
  Kokkos::deep_copy(g_spin, 0.0);
  Kokkos::deep_copy(w_fire, 0.0);
  Kokkos::deep_copy(g_rho, 0.0);

  dt_fire = dt_init;
  mix_alpha = mix_init;
  n_pos = 0;
  p_smooth_bar = 0.0;

  for (int iter = 0; iter < maxiter; iter++) {
    if (timer->check_timeout(niter)) return TIMEOUT;

    const bigint ntimestep = ++update->ntimestep;
    const double dtau_step = dt_fire;
    niter++;
    const int nlocal = atom->nlocal;
    {
      auto f = atomKK->k_f.d_view;
      auto sp = atomKK->k_sp.d_view;
      auto fm = atomKK->k_fm.d_view;
      auto v = v_fire;
      auto u = u_fire;
      auto g = g_spin;
      auto w = w_fire;
      auto grho = g_rho;
      const double dtau = dtau_step;
      const double lr = scale_lat;
      const double lrho = scale_mag;
      const double le = scale_spin;

      Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
        const int j = 3 * i;
        const double sx = sp(i, 0);
        const double sy = sp(i, 1);
        const double sz = sp(i, 2);
        const double rho = sp(i, 3);
        const double hpar = fm(i, 0) * sx + fm(i, 1) * sy + fm(i, 2) * sz;
        const double gx = rho * (fm(i, 0) - hpar * sx);
        const double gy = rho * (fm(i, 1) - hpar * sy);
        const double gz = rho * (fm(i, 2) - hpar * sz);

        if (lr > 0.0) {
          v(j + 0) += dtau * lr * f(i, 0);
          v(j + 1) += dtau * lr * f(i, 1);
          v(j + 2) += dtau * lr * f(i, 2);
        } else {
          v(j + 0) = 0.0;
          v(j + 1) = 0.0;
          v(j + 2) = 0.0;
        }

        if (rho <= 0.0) {
          grho(j + 0) = 0.0;
          grho(j + 1) = 0.0;
          grho(j + 2) = 0.0;
          g(j + 0) = 0.0;
          g(j + 1) = 0.0;
          g(j + 2) = 0.0;
          w(j + 0) = 0.0;
          w(j + 1) = 0.0;
          w(j + 2) = 0.0;
          u(j + 0) = 0.0;
          u(j + 1) = 0.0;
          u(j + 2) = 0.0;
          return;
        }

        grho(j + 0) = hpar;
        grho(j + 1) = 0.0;
        grho(j + 2) = 0.0;
        g(j + 0) = gx;
        g(j + 1) = gy;
        g(j + 2) = gz;

        if (lrho > 0.0) w(j + 0) += dtau * lrho * hpar;
        else w(j + 0) = 0.0;
        w(j + 1) = 0.0;
        w(j + 2) = 0.0;

        if (le > 0.0) {
          double ux = u(j + 0) + dtau * le * gx;
          double uy = u(j + 1) + dtau * le * gy;
          double uz = u(j + 2) + dtau * le * gz;
          const double udot = ux * sx + uy * sy + uz * sz;
          u(j + 0) = ux - udot * sx;
          u(j + 1) = uy - udot * sy;
          u(j + 2) = uz - udot * sz;
        } else {
          u(j + 0) = 0.0;
          u(j + 1) = 0.0;
          u(j + 2) = 0.0;
        }
      });
    }

    VariableFireStats stats;
    {
      auto f = atomKK->k_f.d_view;
      auto sp = atomKK->k_sp.d_view;
      auto v = v_fire;
      auto u = u_fire;
      auto g = g_spin;
      auto w = w_fire;
      auto grho = g_rho;
      const int lattice_active = (scale_lat > 0.0) ? 1 : 0;
      const int rho_active = (scale_mag > 0.0) ? 1 : 0;
      const int dir_active = (scale_spin > 0.0) ? 1 : 0;

      Kokkos::parallel_reduce(
        nlocal,
        LAMMPS_LAMBDA(const int &i, VariableFireStats &sum) {
          const int j = 3 * i;
          if (lattice_active) {
            const double fx = f(i, 0);
            const double fy = f(i, 1);
            const double fz = f(i, 2);
            const double vx = v(j + 0);
            const double vy = v(j + 1);
            const double vz = v(j + 2);
            const double fi2 = fx * fx + fy * fy + fz * fz;
            const double vi2 = vx * vx + vy * vy + vz * vz;
            sum.f2 += fi2;
            sum.v2 += vi2;
            sum.fdotv += fx * vx + fy * vy + fz * vz;
            sum.fmax2 = MAX(sum.fmax2, fi2);
          }

          if (sp(i, 3) <= 0.0) return;

          if (rho_active) {
            const double gi = grho(j + 0);
            const double wi = w(j + 0);
            sum.rho2 += gi * gi;
            sum.w2 += wi * wi;
            sum.rhodotw += gi * wi;
            sum.rhomax2 = MAX(sum.rhomax2, gi * gi);
          }

          if (dir_active) {
            const double gx = g(j + 0);
            const double gy = g(j + 1);
            const double gz = g(j + 2);
            const double ux = u(j + 0);
            const double uy = u(j + 1);
            const double uz = u(j + 2);
            const double gi2 = gx * gx + gy * gy + gz * gz;
            const double ui2 = ux * ux + uy * uy + uz * uz;
            sum.g2 += gi2;
            sum.u2 += ui2;
            sum.gdotu += gx * ux + gy * uy + gz * uz;
            sum.gmax2 = MAX(sum.gmax2, gi2);
          }
        },
        stats);
    }

    double local_sum[9] = {stats.f2, stats.v2, stats.rho2, stats.w2, stats.g2,
                           stats.u2, stats.fdotv, stats.rhodotw, stats.gdotu};
    double global_sum[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(local_sum, global_sum, 9, MPI_DOUBLE, MPI_SUM, world);

    const double fnorm = sqrt(global_sum[0]);
    const double vnorm = sqrt(global_sum[1]);
    const double rhonorm = sqrt(global_sum[2]);
    const double wnorm = sqrt(global_sum[3]);
    const double gnorm = sqrt(global_sum[4]);
    const double unorm = sqrt(global_sum[5]);
    const double pr = (scale_lat > 0.0) ? global_sum[6] / (fnorm * vnorm + eps_power) : 0.0;
    const double prho = (scale_mag > 0.0) ? global_sum[7] / (rhonorm * wnorm + eps_power) : 0.0;
    const double pe = (scale_spin > 0.0) ? global_sum[8] / (gnorm * unorm + eps_power) : 0.0;
    const double p =
      ((scale_lat > 0.0) ? pw_lat * pr : 0.0) + ((scale_mag > 0.0) ? pw_mag * prho : 0.0) + ((scale_spin > 0.0) ? pw_spin * pe : 0.0);

    if (p_smooth > 0.0) p_smooth_bar = p_smooth * p_smooth_bar + (1.0 - p_smooth) * p;
    else p_smooth_bar = p;
    const double p_use = (p_smooth > 0.0) ? p_smooth_bar : p;

    if (p_use > 0.0) {
      const double keep = 1.0 - mix_alpha;
      const int mix_r = (scale_lat > 0.0 && fnorm > mix_tol_lat && vnorm > mix_tol_lat) ? 1 : 0;
      const int mix_rho = (scale_mag > 0.0 && rhonorm > mix_tol_mag && wnorm > mix_tol_mag) ? 1 : 0;
      const int mix_e = (scale_spin > 0.0 && gnorm > mix_tol_spin && unorm > mix_tol_spin) ? 1 : 0;
      const double mix_r_scale = mix_r ? mix_alpha * (vnorm / fnorm) : 0.0;
      const double mix_rho_scale = mix_rho ? mix_alpha * (wnorm / rhonorm) : 0.0;
      const double mix_e_scale = mix_e ? mix_alpha * (unorm / gnorm) : 0.0;

      auto f = atomKK->k_f.d_view;
      auto sp = atomKK->k_sp.d_view;
      auto v = v_fire;
      auto u = u_fire;
      auto g = g_spin;
      auto w = w_fire;
      auto grho = g_rho;

      Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
        const int j = 3 * i;

        if (mix_r) {
          v(j + 0) = keep * v(j + 0) + mix_r_scale * f(i, 0);
          v(j + 1) = keep * v(j + 1) + mix_r_scale * f(i, 1);
          v(j + 2) = keep * v(j + 2) + mix_r_scale * f(i, 2);
        }

        if (sp(i, 3) <= 0.0) {
          w(j + 0) = 0.0;
          w(j + 1) = 0.0;
          w(j + 2) = 0.0;
          u(j + 0) = 0.0;
          u(j + 1) = 0.0;
          u(j + 2) = 0.0;
          return;
        }

        if (mix_rho) w(j + 0) = keep * w(j + 0) + mix_rho_scale * grho(j + 0);

        if (mix_e) {
          double ux = keep * u(j + 0) + mix_e_scale * g(j + 0);
          double uy = keep * u(j + 1) + mix_e_scale * g(j + 1);
          double uz = keep * u(j + 2) + mix_e_scale * g(j + 2);
          const double udot = ux * sp(i, 0) + uy * sp(i, 1) + uz * sp(i, 2);
          u(j + 0) = ux - udot * sp(i, 0);
          u(j + 1) = uy - udot * sp(i, 1);
          u(j + 2) = uz - udot * sp(i, 2);
        }
      });

      if (n_pos >= n_accel) {
        dt_fire = MIN(dt_grow * dt_fire, dt_max);
        mix_alpha *= mix_decay;
      }
      n_pos++;
    } else {
      Kokkos::deep_copy(v_fire, 0.0);
      Kokkos::deep_copy(u_fire, 0.0);
      Kokkos::deep_copy(w_fire, 0.0);
      dt_fire *= dt_shrink;
      mix_alpha = mix_init;
      n_pos = 0;
    }

    {
      auto x = atomKK->k_x.d_view;
      auto sp = atomKK->k_sp.d_view;
      auto v = v_fire;
      auto u = u_fire;
      auto w = w_fire;
      const double lr = scale_lat;
      const double lrho = scale_mag;
      const double le = scale_spin;
      const double dtau = dtau_step;
      const double drmax = disp_max;
      const double drhomax = mag_step;
      const double dthetamax = angle_max;
      const double rho_floor = mag_min;
      const double eps_small_theta = eps_theta;

      Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
        const int j = 3 * i;

        double drx = 0.0;
        double dry = 0.0;
        double drz = 0.0;
        if (lr > 0.0) {
          drx = dtau * v(j + 0);
          dry = dtau * v(j + 1);
          drz = dtau * v(j + 2);
          const double dr2 = drx * drx + dry * dry + drz * drz;
          if (dr2 > drmax * drmax) {
            const double scale = drmax / sqrt(dr2);
            drx *= scale;
            dry *= scale;
            drz *= scale;
          }
        }

        x(i, 0) += drx;
        x(i, 1) += dry;
        x(i, 2) += drz;

        if (sp(i, 3) <= 0.0) {
          w(j + 0) = 0.0;
          w(j + 1) = 0.0;
          w(j + 2) = 0.0;
          u(j + 0) = 0.0;
          u(j + 1) = 0.0;
          u(j + 2) = 0.0;
          return;
        }

        double drho = 0.0;
        if (lrho > 0.0) {
          drho = dtau * w(j + 0);
          if (fabs(drho) > drhomax) drho = (drho > 0.0) ? drhomax : -drhomax;
        }

        double rho_new = sp(i, 3) + drho;
        if (rho_new < rho_floor) {
          rho_new = rho_floor;
          w(j + 0) = 0.0;
        }
        w(j + 1) = 0.0;
        w(j + 2) = 0.0;

        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double delta2 = 0.0;
        if (le > 0.0) {
          dx = dtau * u(j + 0);
          dy = dtau * u(j + 1);
          dz = dtau * u(j + 2);
          delta2 = dx * dx + dy * dy + dz * dz;
          if (delta2 > dthetamax * dthetamax) {
            const double scale = dthetamax / sqrt(delta2);
            dx *= scale;
            dy *= scale;
            dz *= scale;
            delta2 = dx * dx + dy * dy + dz * dz;
          }
        }

        const double sx = sp(i, 0);
        const double sy = sp(i, 1);
        const double sz = sp(i, 2);
        double enx, eny, enz;
        if (delta2 > eps_small_theta * eps_small_theta) {
          const double theta = sqrt(delta2);
          const double c = cos(theta);
          const double s_over_theta = sin(theta) / theta;
          enx = c * sx + s_over_theta * dx;
          eny = c * sy + s_over_theta * dy;
          enz = c * sz + s_over_theta * dz;
        } else {
          enx = sx + dx;
          eny = sy + dy;
          enz = sz + dz;
        }

        const double enorm = sqrt(enx * enx + eny * eny + enz * enz);
        if (enorm > 0.0) {
          enx /= enorm;
          eny /= enorm;
          enz /= enorm;
        } else {
          enx = sx;
          eny = sy;
          enz = sz;
        }

        sp(i, 0) = enx;
        sp(i, 1) = eny;
        sp(i, 2) = enz;
        sp(i, 3) = rho_new;

        const double udot_new = u(j + 0) * enx + u(j + 1) * eny + u(j + 2) * enz;
        u(j + 0) -= udot_new * enx;
        u(j + 1) -= udot_new * eny;
        u(j + 2) -= udot_new * enz;
      });
    }

    atomKK->modified(Device, X_MASK | SP_MASK);
    fix_minimize_kk->k_vectors.modify<LMPDeviceType>();

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    if (neval >= update->max_eval) return MAXEVAL;

    atomKK->sync(Device, F_MASK | SP_MASK | FM_MASK);
    compute_variable_spin_gradients();

    MaxTriple current_max;
    {
      auto f = atomKK->k_f.d_view;
      auto g = g_spin;
      auto grho = g_rho;

      Kokkos::parallel_reduce(
        nlocal,
        LAMMPS_LAMBDA(const int &i, MaxTriple &val) {
          const int j = 3 * i;
          const double f2 = f(i, 0) * f(i, 0) + f(i, 1) * f(i, 1) + f(i, 2) * f(i, 2);
          const double rho2 = grho(j + 0) * grho(j + 0);
          const double g2 = g(j + 0) * g(j + 0) + g(j + 1) * g(j + 1) + g(j + 2) * g(j + 2);
          val.d0 = MAX(val.d0, f2);
          val.d1 = MAX(val.d1, rho2);
          val.d2 = MAX(val.d2, g2);
        },
        current_max);
    }

    double local_conv[3] = {current_max.d0, current_max.d1, current_max.d2};
    double global_conv[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local_conv, global_conv, 3, MPI_DOUBLE, MPI_MAX, world);

    const double delta_e = fabs(ecurrent - eprevious);
    const double fmax = sqrt(global_conv[0]);
    const double rhomax = sqrt(global_conv[1]);
    const double gmax = sqrt(global_conv[2]);
    const double ftol_spinff = (scale_lat > 0.0) ? update->ftol : 0.0;
    const double ftol_mag_eff = (scale_mag > 0.0) ? ftol_mag : 0.0;
    const double ftol_spin_eff = (scale_spin > 0.0) ? ftol_spin : 0.0;
    const int criteria_active = (update->etol > 0.0) || (ftol_spinff > 0.0) || (ftol_mag_eff > 0.0) || (ftol_spin_eff > 0.0);
    const int etol_ok = (update->etol <= 0.0) || (delta_e < update->etol);
    const int ftol_ok = (ftol_spinff <= 0.0) || (fmax < ftol_spinff);
    const int rftol_ok = (ftol_mag_eff <= 0.0) || (rhomax < ftol_mag_eff);
    const int eftol_ok = (ftol_spin_eff <= 0.0) || (gmax < ftol_spin_eff);

    if (criteria_active && etol_ok && ftol_ok && rftol_ok && eftol_ok) {
      if (ftol_spinff > 0.0 || ftol_mag_eff > 0.0 || ftol_spin_eff > 0.0) return FTOL;
      return ETOL;
    }

    if (output->next == ntimestep) {
      atomKK->sync(Host, ALL_MASK);
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}

#endif
