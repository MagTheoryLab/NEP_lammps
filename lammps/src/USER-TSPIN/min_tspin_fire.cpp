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

#include "min_tspin_fire.h"

#include "atom.h"
#include "error.h"
#include "fix_minimize.h"
#include "output.h"
#include "timer.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

MinTSPINFire::MinTSPINFire(LAMMPS *lmp) :
  Min(lmp), v_fire(nullptr), u_fire(nullptr), g_spin(nullptr), w_fire(nullptr), g_rho(nullptr)
{
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

MinTSPINFire::~MinTSPINFire() = default;

void MinTSPINFire::init()
{
  Min::init();
  apply_preset_defaults();

  if (dt_init <= 0.0) error->all(FLERR, "min tspin/fire requires dt_init > 0");
  if (dt_max <= 0.0) error->all(FLERR, "min tspin/fire requires dt_max > 0");
  if (dt_max < dt_init) error->all(FLERR, "min tspin/fire requires dt_max >= dt_init");
  if (disp_max <= 0.0) error->all(FLERR, "min tspin/fire requires rmax > 0");
  if (p_smooth < 0.0 || p_smooth >= 1.0) error->all(FLERR, "min tspin/fire requires p_smooth in [0,1)");
  if (mix_init < 0.0 || mix_init > 1.0)
    error->all(FLERR, "min tspin/fire requires mix_init in [0,1]");
  if (dt_grow < 1.0) error->all(FLERR, "min tspin/fire requires dt_grow >= 1");
  if (dt_shrink <= 0.0 || dt_shrink >= 1.0) error->all(FLERR, "min tspin/fire requires dt_shrink in (0,1)");
  if (mix_decay <= 0.0 || mix_decay > 1.0) error->all(FLERR, "min tspin/fire requires mix_decay in (0,1]");
  if (n_accel < 0) error->all(FLERR, "min tspin/fire requires n_accel >= 0");
  if (eps_power <= 0.0) error->all(FLERR, "min tspin/fire requires eps_power > 0");
  if (eps_theta <= 0.0) error->all(FLERR, "min tspin/fire requires eps_theta > 0");

  if (vary_mag) {
    if (scale_lat < 0.0) error->all(FLERR, "min tspin/fire requires scale_lat >= 0");
    if (scale_mag < 0.0) error->all(FLERR, "min tspin/fire requires scale_mag >= 0");
    if (scale_spin < 0.0) error->all(FLERR, "min tspin/fire requires scale_spin >= 0");
    if (scale_lat == 0.0 && scale_mag == 0.0 && scale_spin == 0.0)
      error->all(FLERR, "min tspin/fire requires at least one of scale_lat/scale_mag/scale_spin to be > 0");
    if (angle_max <= 0.0) error->all(FLERR, "min tspin/fire requires angle_max > 0");
    if (mag_step <= 0.0) error->all(FLERR, "min tspin/fire requires mag_step > 0");
    if (mag_min < 0.0) error->all(FLERR, "min tspin/fire requires mag_min >= 0");
    if (pw_lat < 0.0 || pw_mag < 0.0 || pw_spin < 0.0)
      error->all(FLERR, "min tspin/fire requires pw_lat/pw_mag/pw_spin >= 0");
    if (mix_tol_lat < 0.0 || mix_tol_mag < 0.0 || mix_tol_spin < 0.0)
      error->all(FLERR, "min tspin/fire requires mix_tol_lat/mix_tol_mag/mix_tol_spin >= 0");
    if (ftol_mag < 0.0) error->all(FLERR, "min tspin/fire requires ftol_mag >= 0");
    if (ftol_spin < 0.0) error->all(FLERR, "min tspin/fire requires ftol_spin >= 0");
    if (!ftol_mag_set) ftol_mag = update->ftol;
    if (!ftol_spin_set) ftol_spin = update->ftol;
  } else {
    if (scale_mag_set || mag_step_set || mag_min_set || ftol_mag_set)
      error->all(FLERR, "min tspin/fire vary_mag no does not accept scale_mag/mag_step/mag_min/ftol_mag");
    if (scale_lat < 0.0) error->all(FLERR, "min tspin/fire requires scale_lat >= 0");
    if (scale_spin < 0.0) error->all(FLERR, "min tspin/fire requires scale_spin >= 0");
    if (scale_lat == 0.0 && scale_spin == 0.0)
      error->all(FLERR, "min tspin/fire requires at least one of scale_lat or scale_spin to be > 0");
    if (angle_max <= 0.0) error->all(FLERR, "min tspin/fire requires angle_max > 0");
    if (mix_tol_lat < 0.0 || mix_tol_spin < 0.0)
      error->all(FLERR, "min tspin/fire requires mix_tol_lat/mix_tol_spin >= 0");
    if (!ftol_spin_set) ftol_spin = update->ftol;
  }
}

void MinTSPINFire::setup_style()
{
  if (nextra_atom)
    error->all(FLERR, "min tspin/fire does not support additional extra per-atom DOFs");
  if (!atom->sp_flag)
    error->all(FLERR, "min tspin/fire requires atom/spin style");
  if (!atom->fm)
    error->all(FLERR, "min tspin/fire requires atom_style spin with fm allocated");

  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);

  log_setup_summary();
}

void MinTSPINFire::apply_preset_defaults()
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

void MinTSPINFire::log_setup_summary() const
{
  const char *preset_name = "balanced";
  if (preset == PRESET_CONSERVATIVE) preset_name = "conservative";
  else if (preset == PRESET_AGGRESSIVE) preset_name = "aggressive";

  utils::logmesg(lmp,
                 "  tspin/fire summary: vary_mag={} preset={} expert={} scale_lat={} scale_spin={} "
                 "scale_mag={} dt_init={} dt_max={} disp_max={} angle_max={} mag_step={} "
                 "mag_min={} ftol_r={} ftol_mag={} ftol_spin={}\n",
                 vary_mag ? "yes" : "no", preset_name, expert ? "yes" : "no", scale_lat,
                 scale_spin, vary_mag ? scale_mag : 0.0, dt_init, dt_max, disp_max,
                 angle_max, vary_mag ? mag_step : 0.0, vary_mag ? mag_min : 0.0,
                 (scale_lat > 0.0) ? update->ftol : 0.0, vary_mag ? ftol_mag : 0.0, ftol_spin);
}

int MinTSPINFire::modify_param(int narg, char **arg)
{
  const auto require_expert = [this]() {
    if (!expert)
      error->all(FLERR, "min tspin/fire expert parameter requires 'min_modify expert yes' first");
  };

  if (strcmp(arg[0], "preset") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    if (strcmp(arg[1], "conservative") == 0) preset = PRESET_CONSERVATIVE;
    else if (strcmp(arg[1], "balanced") == 0) preset = PRESET_BALANCED;
    else if (strcmp(arg[1], "aggressive") == 0) preset = PRESET_AGGRESSIVE;
    else error->all(FLERR, "min tspin/fire requires preset = conservative, balanced, or aggressive");
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
    if (scale_lat < 0.0) error->all(FLERR, "min tspin/fire requires scale_lat >= 0");
    return 2;
  }
  if (strcmp(arg[0], "scale_spin") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    scale_spin = utils::numeric(FLERR, arg[1], false, lmp);
    if (scale_spin < 0.0) error->all(FLERR, "min tspin/fire requires scale_spin >= 0");
    return 2;
  }
  if (strcmp(arg[0], "scale_mag") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    scale_mag = utils::numeric(FLERR, arg[1], false, lmp);
    if (scale_mag < 0.0) error->all(FLERR, "min tspin/fire requires scale_mag >= 0");
    scale_mag_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "dt_init") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    dt_init = utils::numeric(FLERR, arg[1], false, lmp);
    if (dt_init <= 0.0) error->all(FLERR, "min tspin/fire requires dt_init > 0");
    dt_init_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "dt_max") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    dt_max = utils::numeric(FLERR, arg[1], false, lmp);
    if (dt_max <= 0.0) error->all(FLERR, "min tspin/fire requires dt_max > 0");
    dt_max_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "disp_max") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    disp_max = utils::numeric(FLERR, arg[1], false, lmp);
    if (disp_max <= 0.0) error->all(FLERR, "min tspin/fire requires rmax > 0");
    disp_max_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "angle_max") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    angle_max = utils::numeric(FLERR, arg[1], false, lmp);
    if (angle_max <= 0.0) error->all(FLERR, "min tspin/fire requires angle_max > 0");
    angle_max_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mag_step") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mag_step = utils::numeric(FLERR, arg[1], false, lmp);
    if (mag_step <= 0.0) error->all(FLERR, "min tspin/fire requires mag_step > 0");
    mag_step_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mag_min") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mag_min = utils::numeric(FLERR, arg[1], false, lmp);
    if (mag_min < 0.0) error->all(FLERR, "min tspin/fire requires mag_min >= 0");
    mag_min_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "ftol_mag") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    ftol_mag = utils::numeric(FLERR, arg[1], false, lmp);
    if (ftol_mag < 0.0) error->all(FLERR, "min tspin/fire requires ftol_mag >= 0");
    ftol_mag_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "ftol_spin") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    ftol_spin = utils::numeric(FLERR, arg[1], false, lmp);
    if (ftol_spin < 0.0) error->all(FLERR, "min tspin/fire requires ftol_spin >= 0");
    ftol_spin_set = 1;
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
    if (p_smooth < 0.0 || p_smooth >= 1.0) error->all(FLERR, "min tspin/fire requires p_smooth in [0,1)");
    p_smooth_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mix_init") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_init = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_init < 0.0 || mix_init > 1.0)
      error->all(FLERR, "min tspin/fire requires mix_init in [0,1]");
    return 2;
  }
  if (strcmp(arg[0], "dt_grow") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    dt_grow = utils::numeric(FLERR, arg[1], false, lmp);
    if (dt_grow < 1.0) error->all(FLERR, "min tspin/fire requires dt_grow >= 1");
    dt_grow_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "dt_shrink") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    dt_shrink = utils::numeric(FLERR, arg[1], false, lmp);
    if (dt_shrink <= 0.0 || dt_shrink >= 1.0) error->all(FLERR, "min tspin/fire requires dt_shrink in (0,1)");
    dt_shrink_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mix_decay") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_decay = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_decay <= 0.0 || mix_decay > 1.0) error->all(FLERR, "min tspin/fire requires mix_decay in (0,1]");
    mix_decay_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "n_accel") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    n_accel = utils::inumeric(FLERR, arg[1], false, lmp);
    if (n_accel < 0) error->all(FLERR, "min tspin/fire requires n_accel >= 0");
    n_accel_set = 1;
    return 2;
  }
  if (strcmp(arg[0], "mix_tol_lat") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_tol_lat = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_tol_lat < 0.0) error->all(FLERR, "min tspin/fire requires mix_tol_lat >= 0");
    return 2;
  }
  if (strcmp(arg[0], "mix_tol_mag") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_tol_mag = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_tol_mag < 0.0) error->all(FLERR, "min tspin/fire requires mix_tol_mag >= 0");
    return 2;
  }
  if (strcmp(arg[0], "mix_tol_spin") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    mix_tol_spin = utils::numeric(FLERR, arg[1], false, lmp);
    if (mix_tol_spin < 0.0) error->all(FLERR, "min tspin/fire requires mix_tol_spin >= 0");
    return 2;
  }
  if (strcmp(arg[0], "eps_power") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eps_power = utils::numeric(FLERR, arg[1], false, lmp);
    if (eps_power <= 0.0) error->all(FLERR, "min tspin/fire requires eps_power > 0");
    return 2;
  }
  if (strcmp(arg[0], "eps_theta") == 0) {
    require_expert();
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eps_theta = utils::numeric(FLERR, arg[1], false, lmp);
    if (eps_theta <= 0.0) error->all(FLERR, "min tspin/fire requires eps_theta > 0");
    return 2;
  }

  if (strcmp(arg[0], "lambda_s") == 0)
    error->all(FLERR, "min tspin/fire no longer accepts lambda_s; use scale_spin");
  if (strcmp(arg[0], "ws") == 0)
    error->all(FLERR, "min tspin/fire no longer accepts ws; use pw_spin with 'min_modify expert yes'");
  if (strcmp(arg[0], "thetamax") == 0)
    error->all(FLERR, "min tspin/fire no longer accepts thetamax; use angle_max");
  if (strcmp(arg[0], "spin_ftol") == 0)
    error->all(FLERR, "min tspin/fire no longer accepts spin_ftol; use ftol_spin");
  if (strcmp(arg[0], "mix_eps") == 0)
    error->all(FLERR, "min tspin/fire no longer accepts mix_eps; use mix_tol_lat and mix_tol_spin with 'min_modify expert yes'");

  return 0;
}

void MinTSPINFire::reset_vectors()
{
  nvec = 3 * atom->nlocal;
  if (nvec) xvec = atom->x[0];
  if (nvec) fvec = atom->f[0];

  v_fire = fix_minimize->request_vector(0);
  u_fire = fix_minimize->request_vector(1);
  g_spin = fix_minimize->request_vector(2);
  w_fire = fix_minimize->request_vector(3);
  g_rho = fix_minimize->request_vector(4);
}

void MinTSPINFire::compute_projected_spin_gradient()
{
  const int nlocal = atom->nlocal;
  double **sp = atom->sp;
  double **fm = atom->fm;

  for (int i = 0; i < nlocal; i++) {
    const int j = 3 * i;
    if (sp[i][3] <= 0.0) {
      g_spin[j + 0] = 0.0;
      g_spin[j + 1] = 0.0;
      g_spin[j + 2] = 0.0;
      continue;
    }

    const double sx = sp[i][0];
    const double sy = sp[i][1];
    const double sz = sp[i][2];
    const double hdot = fm[i][0] * sx + fm[i][1] * sy + fm[i][2] * sz;
    g_spin[j + 0] = fm[i][0] - hdot * sx;
    g_spin[j + 1] = fm[i][1] - hdot * sy;
    g_spin[j + 2] = fm[i][2] - hdot * sz;
  }
}

void MinTSPINFire::compute_variable_spin_gradients()
{
  const int nlocal = atom->nlocal;
  double **sp = atom->sp;
  double **fm = atom->fm;

  for (int i = 0; i < nlocal; i++) {
    const int j = 3 * i;
    if (sp[i][3] <= 0.0) {
      g_rho[j + 0] = 0.0;
      g_rho[j + 1] = 0.0;
      g_rho[j + 2] = 0.0;
      g_spin[j + 0] = 0.0;
      g_spin[j + 1] = 0.0;
      g_spin[j + 2] = 0.0;
      continue;
    }

    const double sx = sp[i][0];
    const double sy = sp[i][1];
    const double sz = sp[i][2];
    const double rho = sp[i][3];
    const double hpar = fm[i][0] * sx + fm[i][1] * sy + fm[i][2] * sz;
    const double hx = fm[i][0] - hpar * sx;
    const double hy = fm[i][1] - hpar * sy;
    const double hz = fm[i][2] - hpar * sz;

    g_rho[j + 0] = hpar;
    g_rho[j + 1] = 0.0;
    g_rho[j + 2] = 0.0;

    g_spin[j + 0] = rho * hx;
    g_spin[j + 1] = rho * hy;
    g_spin[j + 2] = rho * hz;
  }
}

double MinTSPINFire::fnorm_sqr()
{
  double local_norm2 = 0.0;

  if (vary_mag) {
    compute_variable_spin_gradients();
    if (scale_lat > 0.0)
      for (int i = 0; i < nvec; i++) local_norm2 += fvec[i] * fvec[i];
    if (scale_mag > 0.0)
      for (int i = 0; i < nvec; i += 3) local_norm2 += g_rho[i] * g_rho[i];
    if (scale_spin > 0.0)
      for (int i = 0; i < nvec; i++) local_norm2 += g_spin[i] * g_spin[i];
  } else {
    compute_projected_spin_gradient();
    if (scale_lat > 0.0)
      for (int i = 0; i < nvec; i++) local_norm2 += fvec[i] * fvec[i];
    if (scale_spin > 0.0)
      for (int i = 0; i < nvec; i++) local_norm2 += g_spin[i] * g_spin[i];
  }

  double norm2 = local_norm2;
  MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm2 += fextra[i] * fextra[i];

  return norm2;
}

double MinTSPINFire::fnorm_inf()
{
  double local_norm_inf = 0.0;

  if (vary_mag) {
    compute_variable_spin_gradients();
    if (scale_lat > 0.0)
      for (int i = 0; i < nvec; i++) local_norm_inf = MAX(local_norm_inf, fvec[i] * fvec[i]);
    if (scale_mag > 0.0)
      for (int i = 0; i < nvec; i += 3) local_norm_inf = MAX(local_norm_inf, g_rho[i] * g_rho[i]);
    if (scale_spin > 0.0)
      for (int i = 0; i < nvec; i++) local_norm_inf = MAX(local_norm_inf, g_spin[i] * g_spin[i]);
  } else {
    compute_projected_spin_gradient();
    if (scale_lat > 0.0)
      for (int i = 0; i < nvec; i++) local_norm_inf = MAX(local_norm_inf, fvec[i] * fvec[i]);
    if (scale_spin > 0.0)
      for (int i = 0; i < nvec; i++) local_norm_inf = MAX(local_norm_inf, g_spin[i] * g_spin[i]);
  }

  double norm_inf = 0.0;
  MPI_Allreduce(&local_norm_inf, &norm_inf, 1, MPI_DOUBLE, MPI_MAX, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm_inf = MAX(norm_inf, fextra[i] * fextra[i]);

  return norm_inf;
}

double MinTSPINFire::fnorm_max()
{
  double local_norm_max = 0.0;

  if (vary_mag) {
    compute_variable_spin_gradients();
    for (int i = 0; i < nvec; i += 3) {
      const double f2 =
        (scale_lat > 0.0) ? (fvec[i] * fvec[i] + fvec[i + 1] * fvec[i + 1] + fvec[i + 2] * fvec[i + 2]) : 0.0;
      const double rho2 = (scale_mag > 0.0) ? (g_rho[i] * g_rho[i]) : 0.0;
      const double g2 = (scale_spin > 0.0)
        ? (g_spin[i] * g_spin[i] + g_spin[i + 1] * g_spin[i + 1] + g_spin[i + 2] * g_spin[i + 2])
        : 0.0;
      local_norm_max = MAX(local_norm_max, MAX(f2, MAX(rho2, g2)));
    }
  } else {
    compute_projected_spin_gradient();
    for (int i = 0; i < nvec; i += 3) {
      const double f2 =
        (scale_lat > 0.0) ? (fvec[i] * fvec[i] + fvec[i + 1] * fvec[i + 1] + fvec[i + 2] * fvec[i + 2]) : 0.0;
      const double g2 = (scale_spin > 0.0)
        ? (g_spin[i] * g_spin[i] + g_spin[i + 1] * g_spin[i + 1] + g_spin[i + 2] * g_spin[i + 2])
        : 0.0;
      local_norm_max = MAX(local_norm_max, MAX(f2, g2));
    }
  }

  double norm_max = 0.0;
  MPI_Allreduce(&local_norm_max, &norm_max, 1, MPI_DOUBLE, MPI_MAX, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm_max = MAX(norm_max, fextra[i] * fextra[i]);

  return norm_max;
}

int MinTSPINFire::iterate(int maxiter)
{
  return vary_mag ? iterate_variable(maxiter) : iterate_fixed(maxiter);
}

int MinTSPINFire::iterate_fixed(int maxiter)
{
  int nlocal = atom->nlocal;

  if (nvec) {
    std::memset(v_fire, 0, sizeof(double) * nvec);
    std::memset(u_fire, 0, sizeof(double) * nvec);
    std::memset(g_spin, 0, sizeof(double) * nvec);
    std::memset(w_fire, 0, sizeof(double) * nvec);
    std::memset(g_rho, 0, sizeof(double) * nvec);
  }

  dt_fire = dt_init;
  mix_alpha = mix_init;
  n_pos = 0;
  p_smooth_bar = 0.0;

  for (int iter = 0; iter < maxiter; iter++) {
    if (timer->check_timeout(niter)) return TIMEOUT;

    const bigint ntimestep = ++update->ntimestep;
    const double dtau_step = dt_fire;
    niter++;
    nlocal = atom->nlocal;

    double **x = atom->x;
    double **f = atom->f;
    double **sp = atom->sp;
    double **fm = atom->fm;

    for (int i = 0; i < nlocal; i++) {
      const int j = 3 * i;
      if (scale_lat > 0.0) {
        v_fire[j + 0] += dtau_step * scale_lat * f[i][0];
        v_fire[j + 1] += dtau_step * scale_lat * f[i][1];
        v_fire[j + 2] += dtau_step * scale_lat * f[i][2];
      } else {
        v_fire[j + 0] = 0.0;
        v_fire[j + 1] = 0.0;
        v_fire[j + 2] = 0.0;
      }

      if (sp[i][3] <= 0.0) {
        g_spin[j + 0] = 0.0;
        g_spin[j + 1] = 0.0;
        g_spin[j + 2] = 0.0;
        u_fire[j + 0] = 0.0;
        u_fire[j + 1] = 0.0;
        u_fire[j + 2] = 0.0;
        continue;
      }

      const double sx = sp[i][0];
      const double sy = sp[i][1];
      const double sz = sp[i][2];
      const double hdot = fm[i][0] * sx + fm[i][1] * sy + fm[i][2] * sz;
      const double gx = fm[i][0] - hdot * sx;
      const double gy = fm[i][1] - hdot * sy;
      const double gz = fm[i][2] - hdot * sz;
      g_spin[j + 0] = gx;
      g_spin[j + 1] = gy;
      g_spin[j + 2] = gz;

      if (scale_spin > 0.0) {
        double ux = u_fire[j + 0] + dtau_step * scale_spin * gx;
        double uy = u_fire[j + 1] + dtau_step * scale_spin * gy;
        double uz = u_fire[j + 2] + dtau_step * scale_spin * gz;
        const double udot = ux * sx + uy * sy + uz * sz;
        u_fire[j + 0] = ux - udot * sx;
        u_fire[j + 1] = uy - udot * sy;
        u_fire[j + 2] = uz - udot * sz;
      } else {
        u_fire[j + 0] = 0.0;
        u_fire[j + 1] = 0.0;
        u_fire[j + 2] = 0.0;
      }
    }

    double local_f2 = 0.0, local_v2 = 0.0, local_g2 = 0.0, local_u2 = 0.0;
    double local_fdotv = 0.0, local_gdotu = 0.0, local_fmax2 = 0.0, local_gmax2 = 0.0;

    for (int i = 0; i < nlocal; i++) {
      const int j = 3 * i;
      const double fi2 = f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2];
      const double vi2 = v_fire[j + 0] * v_fire[j + 0] + v_fire[j + 1] * v_fire[j + 1] +
        v_fire[j + 2] * v_fire[j + 2];
      if (scale_lat > 0.0) {
        local_f2 += fi2;
        local_v2 += vi2;
        local_fdotv += f[i][0] * v_fire[j + 0] + f[i][1] * v_fire[j + 1] + f[i][2] * v_fire[j + 2];
        local_fmax2 = MAX(local_fmax2, fi2);
      }

      if (sp[i][3] <= 0.0) continue;

      const double gi2 = g_spin[j + 0] * g_spin[j + 0] + g_spin[j + 1] * g_spin[j + 1] +
        g_spin[j + 2] * g_spin[j + 2];
      const double ui2 = u_fire[j + 0] * u_fire[j + 0] + u_fire[j + 1] * u_fire[j + 1] +
        u_fire[j + 2] * u_fire[j + 2];
      if (scale_spin > 0.0) {
        local_g2 += gi2;
        local_u2 += ui2;
        local_gdotu += g_spin[j + 0] * u_fire[j + 0] + g_spin[j + 1] * u_fire[j + 1] +
          g_spin[j + 2] * u_fire[j + 2];
        local_gmax2 = MAX(local_gmax2, gi2);
      }
    }

    double local_sum[6] = {local_f2, local_v2, local_g2, local_u2, local_fdotv, local_gdotu};
    double global_sum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(local_sum, global_sum, 6, MPI_DOUBLE, MPI_SUM, world);

    double local_max[2] = {local_fmax2, local_gmax2};
    double global_max[2] = {0.0, 0.0};
    MPI_Allreduce(local_max, global_max, 2, MPI_DOUBLE, MPI_MAX, world);

    const double fnorm = sqrt(global_sum[0]);
    const double vnorm = sqrt(global_sum[1]);
    const double gnorm = sqrt(global_sum[2]);
    const double unorm = sqrt(global_sum[3]);
    const double pr = (scale_lat > 0.0) ? global_sum[4] / (fnorm * vnorm + eps_power) : 0.0;
    const double ps = (scale_spin > 0.0) ? global_sum[5] / (gnorm * unorm + eps_power) : 0.0;
    const double p = ((scale_lat > 0.0) ? pw_lat * pr : 0.0) + ((scale_spin > 0.0) ? pw_spin * ps : 0.0);

    if (p_smooth > 0.0) p_smooth_bar = p_smooth * p_smooth_bar + (1.0 - p_smooth) * p;
    else p_smooth_bar = p;
    const double p_use = (p_smooth > 0.0) ? p_smooth_bar : p;

    if (p_use > 0.0) {
      if (scale_lat > 0.0 && fnorm > mix_tol_lat && vnorm > mix_tol_lat) {
        const double scale_keep = 1.0 - mix_alpha;
        const double scale_mix = mix_alpha * vnorm / fnorm;
        for (int i = 0; i < nlocal; i++) {
          const int j = 3 * i;
          v_fire[j + 0] = scale_keep * v_fire[j + 0] + scale_mix * f[i][0];
          v_fire[j + 1] = scale_keep * v_fire[j + 1] + scale_mix * f[i][1];
          v_fire[j + 2] = scale_keep * v_fire[j + 2] + scale_mix * f[i][2];
        }
      }

      if (scale_spin > 0.0 && gnorm > mix_tol_spin && unorm > mix_tol_spin) {
        const double scale_keep = 1.0 - mix_alpha;
        const double scale_mix = mix_alpha * unorm / gnorm;
        for (int i = 0; i < nlocal; i++) {
          if (sp[i][3] <= 0.0) continue;
          const int j = 3 * i;
          double ux = scale_keep * u_fire[j + 0] + scale_mix * g_spin[j + 0];
          double uy = scale_keep * u_fire[j + 1] + scale_mix * g_spin[j + 1];
          double uz = scale_keep * u_fire[j + 2] + scale_mix * g_spin[j + 2];
          const double udot = ux * sp[i][0] + uy * sp[i][1] + uz * sp[i][2];
          u_fire[j + 0] = ux - udot * sp[i][0];
          u_fire[j + 1] = uy - udot * sp[i][1];
          u_fire[j + 2] = uz - udot * sp[i][2];
        }
      }

      if (n_pos >= n_accel) {
        dt_fire = MIN(dt_grow * dt_fire, dt_max);
        mix_alpha *= mix_decay;
      }
      n_pos++;
    } else {
      if (nvec) {
        std::memset(v_fire, 0, sizeof(double) * nvec);
        std::memset(u_fire, 0, sizeof(double) * nvec);
      }
      dt_fire *= dt_shrink;
      mix_alpha = mix_init;
      n_pos = 0;
    }

    for (int i = 0; i < nlocal; i++) {
      const int j = 3 * i;

      double drx = 0.0;
      double dry = 0.0;
      double drz = 0.0;
      if (scale_lat > 0.0) {
        drx = dtau_step * v_fire[j + 0];
        dry = dtau_step * v_fire[j + 1];
        drz = dtau_step * v_fire[j + 2];
        const double dr2 = drx * drx + dry * dry + drz * drz;
        if (dr2 > disp_max * disp_max) {
          const double scale = disp_max / sqrt(dr2);
          drx *= scale;
          dry *= scale;
          drz *= scale;
        }
      }
      x[i][0] += drx;
      x[i][1] += dry;
      x[i][2] += drz;

      if (sp[i][3] <= 0.0) {
        u_fire[j + 0] = 0.0;
        u_fire[j + 1] = 0.0;
        u_fire[j + 2] = 0.0;
        continue;
      }

      double dx = 0.0;
      double dy = 0.0;
      double dz = 0.0;
      double delta2 = 0.0;
      if (scale_spin > 0.0) {
        dx = dtau_step * u_fire[j + 0];
        dy = dtau_step * u_fire[j + 1];
        dz = dtau_step * u_fire[j + 2];
        delta2 = dx * dx + dy * dy + dz * dz;
        if (delta2 > angle_max * angle_max) {
          const double scale = angle_max / sqrt(delta2);
          dx *= scale;
          dy *= scale;
          dz *= scale;
          delta2 = dx * dx + dy * dy + dz * dz;
        }
      }

      const double sx = sp[i][0];
      const double sy = sp[i][1];
      const double sz = sp[i][2];
      double enx, eny, enz;
      if (delta2 > eps_theta * eps_theta) {
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

      sp[i][0] = enx;
      sp[i][1] = eny;
      sp[i][2] = enz;

      const double udot_new = u_fire[j + 0] * enx + u_fire[j + 1] * eny + u_fire[j + 2] * enz;
      u_fire[j + 0] -= udot_new * enx;
      u_fire[j + 1] -= udot_new * eny;
      u_fire[j + 2] -= udot_new * enz;
    }

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    if (neval >= update->max_eval) return MAXEVAL;

    nlocal = atom->nlocal;
    compute_projected_spin_gradient();
    double local_fmax_new = 0.0;
    double local_gmax_new = 0.0;
    for (int i = 0; i < nlocal; i++) {
      const int j = 3 * i;
      const double f2 = fvec[j + 0] * fvec[j + 0] + fvec[j + 1] * fvec[j + 1] + fvec[j + 2] * fvec[j + 2];
      const double g2 = g_spin[j + 0] * g_spin[j + 0] + g_spin[j + 1] * g_spin[j + 1] + g_spin[j + 2] * g_spin[j + 2];
      local_fmax_new = MAX(local_fmax_new, f2);
      local_gmax_new = MAX(local_gmax_new, g2);
    }

    double local_conv[2] = {local_fmax_new, local_gmax_new};
    double global_conv[2] = {0.0, 0.0};
    MPI_Allreduce(local_conv, global_conv, 2, MPI_DOUBLE, MPI_MAX, world);

    const double delta_e = fabs(ecurrent - eprevious);
    const double fmax = sqrt(global_conv[0]);
    const double gmax = sqrt(global_conv[1]);
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
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}

int MinTSPINFire::iterate_variable(int maxiter)
{
  int nlocal = atom->nlocal;

  if (nvec) {
    std::memset(v_fire, 0, sizeof(double) * nvec);
    std::memset(u_fire, 0, sizeof(double) * nvec);
    std::memset(g_spin, 0, sizeof(double) * nvec);
    std::memset(w_fire, 0, sizeof(double) * nvec);
    std::memset(g_rho, 0, sizeof(double) * nvec);
  }

  dt_fire = dt_init;
  mix_alpha = mix_init;
  n_pos = 0;
  p_smooth_bar = 0.0;

  for (int iter = 0; iter < maxiter; iter++) {
    if (timer->check_timeout(niter)) return TIMEOUT;

    const bigint ntimestep = ++update->ntimestep;
    const double dtau_step = dt_fire;
    niter++;
    nlocal = atom->nlocal;

    double **x = atom->x;
    double **f = atom->f;
    double **sp = atom->sp;
    double **fm = atom->fm;

    for (int i = 0; i < nlocal; i++) {
      const int j = 3 * i;
      const double sx = sp[i][0];
      const double sy = sp[i][1];
      const double sz = sp[i][2];
      const double rho = sp[i][3];
      const double hpar = fm[i][0] * sx + fm[i][1] * sy + fm[i][2] * sz;
      const double hx = fm[i][0] - hpar * sx;
      const double hy = fm[i][1] - hpar * sy;
      const double hz = fm[i][2] - hpar * sz;
      const double gx = rho * hx;
      const double gy = rho * hy;
      const double gz = rho * hz;

      if (scale_lat > 0.0) {
        v_fire[j + 0] += dtau_step * scale_lat * f[i][0];
        v_fire[j + 1] += dtau_step * scale_lat * f[i][1];
        v_fire[j + 2] += dtau_step * scale_lat * f[i][2];
      } else {
        v_fire[j + 0] = 0.0;
        v_fire[j + 1] = 0.0;
        v_fire[j + 2] = 0.0;
      }

      if (rho <= 0.0) {
        g_rho[j + 0] = 0.0;
        g_rho[j + 1] = 0.0;
        g_rho[j + 2] = 0.0;
        g_spin[j + 0] = 0.0;
        g_spin[j + 1] = 0.0;
        g_spin[j + 2] = 0.0;
        w_fire[j + 0] = 0.0;
        w_fire[j + 1] = 0.0;
        w_fire[j + 2] = 0.0;
        u_fire[j + 0] = 0.0;
        u_fire[j + 1] = 0.0;
        u_fire[j + 2] = 0.0;
        continue;
      }

      g_rho[j + 0] = hpar;
      g_rho[j + 1] = 0.0;
      g_rho[j + 2] = 0.0;
      g_spin[j + 0] = gx;
      g_spin[j + 1] = gy;
      g_spin[j + 2] = gz;

      if (scale_mag > 0.0) w_fire[j + 0] += dtau_step * scale_mag * hpar;
      else w_fire[j + 0] = 0.0;
      w_fire[j + 1] = 0.0;
      w_fire[j + 2] = 0.0;

      if (scale_spin > 0.0) {
        double ux = u_fire[j + 0] + dtau_step * scale_spin * gx;
        double uy = u_fire[j + 1] + dtau_step * scale_spin * gy;
        double uz = u_fire[j + 2] + dtau_step * scale_spin * gz;
        const double udot = ux * sx + uy * sy + uz * sz;
        u_fire[j + 0] = ux - udot * sx;
        u_fire[j + 1] = uy - udot * sy;
        u_fire[j + 2] = uz - udot * sz;
      } else {
        u_fire[j + 0] = 0.0;
        u_fire[j + 1] = 0.0;
        u_fire[j + 2] = 0.0;
      }
    }

    double local_f2 = 0.0, local_v2 = 0.0, local_rho2 = 0.0, local_w2 = 0.0, local_g2 = 0.0, local_u2 = 0.0;
    double local_fdotv = 0.0, local_rhodotw = 0.0, local_gdotu = 0.0;
    double local_fmax2 = 0.0, local_rhomax2 = 0.0, local_gmax2 = 0.0;

    for (int i = 0; i < nlocal; i++) {
      const int j = 3 * i;
      if (scale_lat > 0.0) {
        const double fi2 = f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2];
        const double vi2 = v_fire[j + 0] * v_fire[j + 0] + v_fire[j + 1] * v_fire[j + 1] +
          v_fire[j + 2] * v_fire[j + 2];
        local_f2 += fi2;
        local_v2 += vi2;
        local_fdotv += f[i][0] * v_fire[j + 0] + f[i][1] * v_fire[j + 1] + f[i][2] * v_fire[j + 2];
        local_fmax2 = MAX(local_fmax2, fi2);
      }

      if (scale_mag > 0.0) {
        const double gi = g_rho[j + 0];
        const double wi = w_fire[j + 0];
        local_rho2 += gi * gi;
        local_w2 += wi * wi;
        local_rhodotw += gi * wi;
        local_rhomax2 = MAX(local_rhomax2, gi * gi);
      }

      if (scale_spin > 0.0) {
        const double gi2 = g_spin[j + 0] * g_spin[j + 0] + g_spin[j + 1] * g_spin[j + 1] +
          g_spin[j + 2] * g_spin[j + 2];
        const double ui2 = u_fire[j + 0] * u_fire[j + 0] + u_fire[j + 1] * u_fire[j + 1] +
          u_fire[j + 2] * u_fire[j + 2];
        local_g2 += gi2;
        local_u2 += ui2;
        local_gdotu += g_spin[j + 0] * u_fire[j + 0] + g_spin[j + 1] * u_fire[j + 1] +
          g_spin[j + 2] * u_fire[j + 2];
        local_gmax2 = MAX(local_gmax2, gi2);
      }
    }

    double local_sum[9] = {local_f2, local_v2, local_rho2, local_w2, local_g2,
                           local_u2, local_fdotv, local_rhodotw, local_gdotu};
    double global_sum[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(local_sum, global_sum, 9, MPI_DOUBLE, MPI_SUM, world);

    double local_max[3] = {local_fmax2, local_rhomax2, local_gmax2};
    double global_max[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local_max, global_max, 3, MPI_DOUBLE, MPI_MAX, world);

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

      if (scale_lat > 0.0 && fnorm > mix_tol_lat && vnorm > mix_tol_lat) {
        const double scale_mix = mix_alpha * vnorm / fnorm;
        for (int i = 0; i < nlocal; i++) {
          const int j = 3 * i;
          v_fire[j + 0] = keep * v_fire[j + 0] + scale_mix * f[i][0];
          v_fire[j + 1] = keep * v_fire[j + 1] + scale_mix * f[i][1];
          v_fire[j + 2] = keep * v_fire[j + 2] + scale_mix * f[i][2];
        }
      }

      if (scale_mag > 0.0 && rhonorm > mix_tol_mag && wnorm > mix_tol_mag) {
        const double scale_mix = mix_alpha * wnorm / rhonorm;
        for (int i = 0; i < nlocal; i++) {
          if (sp[i][3] <= 0.0) continue;
          const int j = 3 * i;
          w_fire[j + 0] = keep * w_fire[j + 0] + scale_mix * g_rho[j + 0];
        }
      }

      if (scale_spin > 0.0 && gnorm > mix_tol_spin && unorm > mix_tol_spin) {
        const double scale_mix = mix_alpha * unorm / gnorm;
        for (int i = 0; i < nlocal; i++) {
          if (sp[i][3] <= 0.0) continue;
          const int j = 3 * i;
          double ux = keep * u_fire[j + 0] + scale_mix * g_spin[j + 0];
          double uy = keep * u_fire[j + 1] + scale_mix * g_spin[j + 1];
          double uz = keep * u_fire[j + 2] + scale_mix * g_spin[j + 2];
          const double udot = ux * sp[i][0] + uy * sp[i][1] + uz * sp[i][2];
          u_fire[j + 0] = ux - udot * sp[i][0];
          u_fire[j + 1] = uy - udot * sp[i][1];
          u_fire[j + 2] = uz - udot * sp[i][2];
        }
      }

      if (n_pos >= n_accel) {
        dt_fire = MIN(dt_grow * dt_fire, dt_max);
        mix_alpha *= mix_decay;
      }
      n_pos++;
    } else {
      if (nvec) {
        std::memset(v_fire, 0, sizeof(double) * nvec);
        std::memset(u_fire, 0, sizeof(double) * nvec);
        std::memset(w_fire, 0, sizeof(double) * nvec);
      }
      dt_fire *= dt_shrink;
      mix_alpha = mix_init;
      n_pos = 0;
    }

    for (int i = 0; i < nlocal; i++) {
      const int j = 3 * i;

      double drx = 0.0;
      double dry = 0.0;
      double drz = 0.0;
      if (scale_lat > 0.0) {
        drx = dtau_step * v_fire[j + 0];
        dry = dtau_step * v_fire[j + 1];
        drz = dtau_step * v_fire[j + 2];
        const double dr2 = drx * drx + dry * dry + drz * drz;
        if (dr2 > disp_max * disp_max) {
          const double scale = disp_max / sqrt(dr2);
          drx *= scale;
          dry *= scale;
          drz *= scale;
        }
      }
      x[i][0] += drx;
      x[i][1] += dry;
      x[i][2] += drz;

      if (sp[i][3] <= 0.0) {
        w_fire[j + 0] = 0.0;
        w_fire[j + 1] = 0.0;
        w_fire[j + 2] = 0.0;
        u_fire[j + 0] = 0.0;
        u_fire[j + 1] = 0.0;
        u_fire[j + 2] = 0.0;
        continue;
      }

      double drho = 0.0;
      if (scale_mag > 0.0) {
        drho = dtau_step * w_fire[j + 0];
        if (fabs(drho) > mag_step) drho = (drho > 0.0 ? mag_step : -mag_step);
      }

      double rho_new = sp[i][3] + drho;
      if (rho_new < mag_min) {
        rho_new = mag_min;
        w_fire[j + 0] = 0.0;
      }

      double dx = 0.0;
      double dy = 0.0;
      double dz = 0.0;
      double delta2 = 0.0;
      if (scale_spin > 0.0) {
        dx = dtau_step * u_fire[j + 0];
        dy = dtau_step * u_fire[j + 1];
        dz = dtau_step * u_fire[j + 2];
        delta2 = dx * dx + dy * dy + dz * dz;
        if (delta2 > angle_max * angle_max) {
          const double scale = angle_max / sqrt(delta2);
          dx *= scale;
          dy *= scale;
          dz *= scale;
          delta2 = dx * dx + dy * dy + dz * dz;
        }
      }

      const double sx = sp[i][0];
      const double sy = sp[i][1];
      const double sz = sp[i][2];
      double enx, eny, enz;
      if (delta2 > eps_theta * eps_theta) {
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

      sp[i][0] = enx;
      sp[i][1] = eny;
      sp[i][2] = enz;
      sp[i][3] = rho_new;

      const double udot_new = u_fire[j + 0] * enx + u_fire[j + 1] * eny + u_fire[j + 2] * enz;
      u_fire[j + 0] -= udot_new * enx;
      u_fire[j + 1] -= udot_new * eny;
      u_fire[j + 2] -= udot_new * enz;
    }

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    if (neval >= update->max_eval) return MAXEVAL;

    nlocal = atom->nlocal;
    compute_variable_spin_gradients();

    double local_fmax_new = 0.0;
    double local_rhomax_new = 0.0;
    double local_gmax_new = 0.0;
    for (int i = 0; i < nlocal; i++) {
      const int j = 3 * i;
      const double f2 = fvec[j + 0] * fvec[j + 0] + fvec[j + 1] * fvec[j + 1] + fvec[j + 2] * fvec[j + 2];
      const double rho2 = g_rho[j + 0] * g_rho[j + 0];
      const double g2 = g_spin[j + 0] * g_spin[j + 0] + g_spin[j + 1] * g_spin[j + 1] + g_spin[j + 2] * g_spin[j + 2];
      local_fmax_new = MAX(local_fmax_new, f2);
      local_rhomax_new = MAX(local_rhomax_new, rho2);
      local_gmax_new = MAX(local_gmax_new, g2);
    }

    double local_conv[3] = {local_fmax_new, local_rhomax_new, local_gmax_new};
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
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}
