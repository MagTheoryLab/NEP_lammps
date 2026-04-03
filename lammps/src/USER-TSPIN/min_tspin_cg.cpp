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

/*
  USER-TSPIN: joint lattice + spin conjugate-gradient minimizer.

  The implementation follows the standard CG + line search structure used in
  LAMMPS (min_style cg) but introduces an additional per-atom, 3-component
  "spin displacement" coordinate d with length dimension, mapped to the spin
  direction (and optionally magnitude).

  This provides a unit-consistent way to combine lattice forces (eV/length)
  and spin "forces" derived from the effective field H = -dE/dM (eV/muB)
  used by USER-TSPIN / NEP(SPIN).
*/
// clang-format on

#include "min_tspin_cg.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "fix_minimize.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "output.h"
#include "thermo.h"
#include "timer.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr double EPS_ENERGY = 1.0e-8;
static constexpr double ALPHA_MAX = 1.0;
static constexpr double ALPHA_REDUCE = 0.5;
static constexpr double BACKTRACK_SLOPE = 0.4;
static constexpr double QUADRATIC_TOL = 0.1;
static constexpr double EMACH = 1.0e-8;
static constexpr double EPS_QUAD = 1.0e-28;

static const char cite_minstyle_tspin_cg[] =
  "min_style tspin/cg command:\n\n"
  "This minimizer combines lattice + spin degrees of freedom using a\n"
  "conjugate-gradient method with line search, inspired by spin minimizers in\n"
  "the LAMMPS SPIN package.\n\n";

MinTSPINCG::MinTSPINCG(LAMMPS *lmp) :
  Min(lmp),
  linemin(nullptr),
  x0(nullptr),
  g(nullptr),
  h(nullptr),
  d0(nullptr),
  gd(nullptr),
  hd(nullptr),
  gextra(nullptr),
  hextra(nullptr),
  nlocal_max(0),
  fspin(nullptr),
  dtrial(nullptr)
{
  if (lmp->citeme) lmp->citeme->add(cite_minstyle_tspin_cg);

  searchflag = 1;

  eta_zeta = 0.1;        // length/muB (paper notation)
  eta_auto = 0;
  eta_auto_weight = 0.05;
  eta_auto_min = 1.0e-4;
  eta_auto_max = 10.0;
  spin_dmax = 0.1;       // length units
  vary_mag = 0;
}

MinTSPINCG::~MinTSPINCG()
{
  delete[] gextra;
  delete[] hextra;
  memory->destroy(fspin);
  memory->destroy(dtrial);
}

void MinTSPINCG::init()
{
  Min::init();

  if (linestyle == BACKTRACK) linemin = &MinTSPINCG::linemin_backtrack;
  else if (linestyle == QUADRATIC) linemin = &MinTSPINCG::linemin_quadratic;
  else error->all(FLERR, "min tspin/cg supports only min_modify line backtrack or quadratic");

  delete[] gextra;
  delete[] hextra;
  gextra = hextra = nullptr;
}

void MinTSPINCG::setup_style()
{
  // this minimizer does not (yet) support third-party extra per-atom dof
  // via Min::request(), since it uses its own per-atom spin dof.
  if (nextra_atom)
    error->all(FLERR, "min tspin/cg does not support additional extra per-atom DOFs");

  if (!atom->sp_flag)
    error->all(FLERR, "min tspin/cg requires atom/spin style");

  if (eta_zeta <= 0.0) error->all(FLERR, "min tspin/cg requires eta_zeta > 0");
  if (spin_dmax <= 0.0)
    error->all(FLERR, "min tspin/cg requires spin_dmax > 0");
  if (eta_auto_weight < 0.0 || eta_auto_weight > 1.0)
    error->all(FLERR, "min tspin/cg requires eta_auto_weight in [0,1]");
  if (eta_auto_min <= 0.0) error->all(FLERR, "min tspin/cg requires eta_auto_min > 0");
  if (eta_auto_max <= 0.0) error->all(FLERR, "min tspin/cg requires eta_auto_max > 0");
  if (eta_auto_max < eta_auto_min)
    error->all(FLERR, "min tspin/cg requires eta_auto_max >= eta_auto_min");

  // allocate per-atom vectors in FixMinimize
  // atomic x0,g,h (like MinLineSearch)
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);

  // spin d0,gd,hd (same per-atom size as coordinates)
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);
  fix_minimize->add_vector(3);

  // extra global dof from fixes, fix stores x0
  if (nextra_global) {
    gextra = new double[nextra_global];
    hextra = new double[nextra_global];
  }
}

int MinTSPINCG::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0], "eta_zeta") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eta_zeta = utils::numeric(FLERR, arg[1], false, lmp);
    if (eta_zeta <= 0.0) error->all(FLERR, "min tspin/cg requires eta_zeta > 0");
    return 2;
  }
  if (strcmp(arg[0], "eta_auto") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eta_auto = utils::logical(FLERR, arg[1], false, lmp);
    return 2;
  }
  if (strcmp(arg[0], "eta_auto_weight") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eta_auto_weight = utils::numeric(FLERR, arg[1], false, lmp);
    if (eta_auto_weight < 0.0 || eta_auto_weight > 1.0)
      error->all(FLERR, "min tspin/cg requires eta_auto_weight in [0,1]");
    return 2;
  }
  if (strcmp(arg[0], "eta_auto_min") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eta_auto_min = utils::numeric(FLERR, arg[1], false, lmp);
    if (eta_auto_min <= 0.0) error->all(FLERR, "min tspin/cg requires eta_auto_min > 0");
    return 2;
  }
  if (strcmp(arg[0], "eta_auto_max") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eta_auto_max = utils::numeric(FLERR, arg[1], false, lmp);
    if (eta_auto_max <= 0.0) error->all(FLERR, "min tspin/cg requires eta_auto_max > 0");
    return 2;
  }
  if (strcmp(arg[0], "spin_dmax") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    spin_dmax = utils::numeric(FLERR, arg[1], false, lmp);
    if (spin_dmax <= 0.0) error->all(FLERR, "min tspin/cg requires spin_dmax > 0");
    return 2;
  }
  if (strcmp(arg[0], "vary_mag") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    vary_mag = utils::logical(FLERR, arg[1], false, lmp);
    return 2;
  }
  return 0;
}

void MinTSPINCG::reset_vectors()
{
  nvec = 3 * atom->nlocal;
  if (nvec) xvec = atom->x[0];
  if (nvec) fvec = atom->f[0];

  x0 = fix_minimize->request_vector(0);
  g = fix_minimize->request_vector(1);
  h = fix_minimize->request_vector(2);
  d0 = fix_minimize->request_vector(3);
  gd = fix_minimize->request_vector(4);
  hd = fix_minimize->request_vector(5);

  if (nlocal_max < atom->nlocal) {
    nlocal_max = atom->nlocal;
    memory->destroy(fspin);
    memory->create(fspin, 3 * nlocal_max, "min/tspin/cg:fspin");
    memory->destroy(dtrial);
    memory->create(dtrial, 3 * nlocal_max, "min/tspin/cg:dtrial");
  }
}

void MinTSPINCG::compute_d_from_sp(double *dvec) const
{
  const int nlocal = atom->nlocal;
  double **sp = atom->sp;
  const double scale = eta_zeta;

  for (int i = 0; i < nlocal; i++) {
    const double mag = sp[i][3];
    if (mag == 0.0) {
      dvec[3 * i + 0] = 0.0;
      dvec[3 * i + 1] = 0.0;
      dvec[3 * i + 2] = 0.0;
      continue;
    }
    dvec[3 * i + 0] = sp[i][0] * mag * scale;
    dvec[3 * i + 1] = sp[i][1] * mag * scale;
    dvec[3 * i + 2] = sp[i][2] * mag * scale;
  }
}

void MinTSPINCG::set_sp_from_d(const double *dvec)
{
  const int nlocal = atom->nlocal;
  double **sp = atom->sp;
  const double mag_scale = 1.0 / eta_zeta;

  for (int i = 0; i < nlocal; i++) {
    if (sp[i][3] == 0.0) continue;

    const double dx = dvec[3 * i + 0];
    const double dy = dvec[3 * i + 1];
    const double dz = dvec[3 * i + 2];
    const double dn = sqrt(dx * dx + dy * dy + dz * dz);
    if (dn < 1.0e-40) continue;

    sp[i][0] = dx / dn;
    sp[i][1] = dy / dn;
    sp[i][2] = dz / dn;
    if (vary_mag) sp[i][3] = mag_scale * dn;
  }
}

void MinTSPINCG::compute_fspin()
{
  const int nlocal = atom->nlocal;
  double **sp = atom->sp;
  double **fm = atom->fm;    // USER-TSPIN convention: H = -dE/dM (eV/muB)
  const double scale = 1.0 / eta_zeta;    // converts eV/muB -> eV/length

  for (int i = 0; i < nlocal; i++) {
    if (sp[i][3] == 0.0) {
      fspin[3 * i + 0] = 0.0;
      fspin[3 * i + 1] = 0.0;
      fspin[3 * i + 2] = 0.0;
      continue;
    }

    if (vary_mag) {
      fspin[3 * i + 0] = scale * fm[i][0];
      fspin[3 * i + 1] = scale * fm[i][1];
      fspin[3 * i + 2] = scale * fm[i][2];
    } else {
      const double sx = sp[i][0];
      const double sy = sp[i][1];
      const double sz = sp[i][2];
      const double hdot = fm[i][0] * sx + fm[i][1] * sy + fm[i][2] * sz;
      fspin[3 * i + 0] = scale * (fm[i][0] - hdot * sx);
      fspin[3 * i + 1] = scale * (fm[i][1] - hdot * sy);
      fspin[3 * i + 2] = scale * (fm[i][2] - hdot * sz);
    }
  }
}

double MinTSPINCG::estimate_eta_zeta() const
{
  const int nlocal = atom->nlocal;
  double **f = atom->f;      // eV/length
  double **sp = atom->sp;
  double **fm = atom->fm;    // USER-TSPIN convention: H = -dE/dM (eV/muB)

  double local_f2 = 0.0;
  double local_h2 = 0.0;

  for (int i = 0; i < nlocal; i++) {
    local_f2 += f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2];

    if (sp[i][3] == 0.0) continue;

    double hx = fm[i][0];
    double hy = fm[i][1];
    double hz = fm[i][2];

    if (!vary_mag) {
      const double sx = sp[i][0];
      const double sy = sp[i][1];
      const double sz = sp[i][2];
      const double hdot = hx * sx + hy * sy + hz * sz;
      hx -= hdot * sx;
      hy -= hdot * sy;
      hz -= hdot * sz;
    }

    local_h2 += hx * hx + hy * hy + hz * hz;
  }

  double glob[2] = {local_f2, local_h2};
  MPI_Allreduce(MPI_IN_PLACE, glob, 2, MPI_DOUBLE, MPI_SUM, world);

  const double f2 = glob[0];
  const double h2 = glob[1];
  if (f2 <= 0.0 || h2 <= 0.0) return eta_zeta;

  const double eta_est = sqrt(h2 / f2);
  if (!std::isfinite(eta_est) || eta_est <= 0.0) return eta_zeta;
  return eta_est;
}

void MinTSPINCG::update_eta_zeta()
{
  if (!eta_auto) return;

  double eta_est = estimate_eta_zeta();

  const double lo = eta_auto_min;
  const double hi = eta_auto_max;
  if (eta_est < lo) eta_est = lo;
  if (eta_est > hi) eta_est = hi;

  eta_zeta = (1.0 - eta_auto_weight) * eta_zeta + eta_auto_weight * eta_est;
  if (eta_zeta < lo) eta_zeta = lo;
  if (eta_zeta > hi) eta_zeta = hi;
}

double MinTSPINCG::fnorm_sqr()
{
  compute_fspin();

  double local_norm2_sqr = 0.0;
  for (int i = 0; i < nvec; i++) local_norm2_sqr += fvec[i] * fvec[i];
  for (int i = 0; i < 3 * atom->nlocal; i++) local_norm2_sqr += fspin[i] * fspin[i];

  double norm2_sqr = 0.0;
  norm2_sqr = local_norm2_sqr;
  MPI_Allreduce(MPI_IN_PLACE, &norm2_sqr, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm2_sqr += fextra[i] * fextra[i];

  return norm2_sqr;
}

double MinTSPINCG::fnorm_inf()
{
  compute_fspin();

  double local_norm_inf = 0.0;
  for (int i = 0; i < nvec; i++) local_norm_inf = MAX(fvec[i] * fvec[i], local_norm_inf);
  for (int i = 0; i < 3 * atom->nlocal; i++)
    local_norm_inf = MAX(fspin[i] * fspin[i], local_norm_inf);

  double norm_inf = 0.0;
  MPI_Allreduce(&local_norm_inf, &norm_inf, 1, MPI_DOUBLE, MPI_MAX, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm_inf = MAX(fextra[i] * fextra[i], norm_inf);

  return norm_inf;
}

double MinTSPINCG::fnorm_max()
{
  compute_fspin();

  const int nlocal = atom->nlocal;
  double local_norm_max = 0.0;
  for (int i = 0; i < nvec; i += 3) {
    const double v = fvec[i] * fvec[i] + fvec[i + 1] * fvec[i + 1] + fvec[i + 2] * fvec[i + 2];
    local_norm_max = MAX(v, local_norm_max);
  }
  for (int i = 0; i < 3 * nlocal; i += 3) {
    const double v =
      fspin[i] * fspin[i] + fspin[i + 1] * fspin[i + 1] + fspin[i + 2] * fspin[i + 2];
    local_norm_max = MAX(v, local_norm_max);
  }

  double norm_max = 0.0;
  MPI_Allreduce(&local_norm_max, &norm_max, 1, MPI_DOUBLE, MPI_MAX, world);
  if (nextra_global) {
    for (int i = 0; i < nextra_global; i++) norm_max = MAX(fextra[i] * fextra[i], norm_max);
  }
  return norm_max;
}

int MinTSPINCG::iterate(int maxiter)
{
  int i;
  double beta, gg, dot[2], dotall[2], fdotf;

  // nlimit = max # of CG iterations before restarting
  // set to ndoftotal unless too big

  int nlimit = static_cast<int>(MIN(MAXSMALLINT, ndoftotal));
  if (nlimit < 1) nlimit = 1;

  // initialize working vectors at current configuration

  compute_d_from_sp(d0);
  compute_fspin();

  for (i = 0; i < nvec; i++) h[i] = g[i] = fvec[i];
  for (i = 0; i < nvec; i++) hd[i] = gd[i] = fspin[i];
  if (nextra_global) {
    for (i = 0; i < nextra_global; i++) hextra[i] = gextra[i] = fextra[i];
  }

  gg = fnorm_sqr();

  for (int iter = 0; iter < maxiter; iter++) {
    if (timer->check_timeout(niter)) return TIMEOUT;

    const bigint ntimestep = ++update->ntimestep;
    niter++;

    // line minimization along direction h/hd from current x/sp

    eprevious = ecurrent;
    const int fail = (this->*linemin)(ecurrent, alpha_final);
    if (fail) return fail;

    if (neval >= update->max_eval) return MAXEVAL;

    if (fabs(ecurrent - eprevious) <
        update->etol * 0.5 * (fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
      return ETOL;

    const double eta_prev = eta_zeta;
    if (eta_auto) update_eta_zeta();
    const double eta_rel = (eta_prev > 0.0) ? fabs(eta_zeta - eta_prev) / eta_prev : 0.0;

    // If eta_zeta changes too much, restart CG to avoid mixing incompatible scalings.
    if (eta_rel > 0.5) {
      compute_fspin();
      for (i = 0; i < nvec; i++) h[i] = g[i] = fvec[i];
      for (i = 0; i < nvec; i++) hd[i] = gd[i] = fspin[i];
      if (nextra_global)
        for (i = 0; i < nextra_global; i++) hextra[i] = gextra[i] = fextra[i];
      gg = fnorm_sqr();
    }

    // force tolerance criterion (combined lattice + spin + extra global)

    compute_fspin();
    dot[0] = dot[1] = 0.0;
    for (i = 0; i < nvec; i++) {
      dot[0] += fvec[i] * fvec[i];
      dot[1] += fvec[i] * g[i];
    }
    for (i = 0; i < nvec; i++) {
      dot[0] += fspin[i] * fspin[i];
      dot[1] += fspin[i] * gd[i];
    }
    dotall[0] = dot[0];
    dotall[1] = dot[1];
    MPI_Allreduce(MPI_IN_PLACE, dotall, 2, MPI_DOUBLE, MPI_SUM, world);
    if (nextra_global)
      for (i = 0; i < nextra_global; i++) {
        dotall[0] += fextra[i] * fextra[i];
        dotall[1] += fextra[i] * gextra[i];
      }

    fdotf = 0.0;
    if (update->ftol > 0.0) {
      if (normstyle == MAX) fdotf = fnorm_max();
      else if (normstyle == INF) fdotf = fnorm_inf();
      else if (normstyle == TWO) fdotf = dotall[0];
      else error->all(FLERR, "Illegal min_modify command");
      if (fdotf < update->ftol * update->ftol) return FTOL;
    }

    // Polak-Ribiere beta, then update g/h for both dof blocks

    beta = MAX(0.0, (dotall[0] - dotall[1]) / gg);
    if ((iter + 1) % nlimit == 0) beta = 0.0;
    gg = dotall[0];

    for (i = 0; i < nvec; i++) {
      g[i] = fvec[i];
      h[i] = g[i] + beta * h[i];
    }
    for (i = 0; i < nvec; i++) {
      gd[i] = fspin[i];
      hd[i] = gd[i] + beta * hd[i];
    }
    if (nextra_global)
      for (i = 0; i < nextra_global; i++) {
        gextra[i] = fextra[i];
        hextra[i] = gextra[i] + beta * hextra[i];
      }

    // reinitialize if not downhill

    double downhill_local = 0.0;
    for (i = 0; i < nvec; i++) downhill_local += g[i] * h[i];
    for (i = 0; i < nvec; i++) downhill_local += gd[i] * hd[i];
    MPI_Allreduce(MPI_IN_PLACE, &downhill_local, 1, MPI_DOUBLE, MPI_SUM, world);
    if (nextra_global)
      for (i = 0; i < nextra_global; i++) downhill_local += gextra[i] * hextra[i];

    if (downhill_local <= 0.0) {
      for (i = 0; i < nvec; i++) h[i] = g[i];
      for (i = 0; i < nvec; i++) hd[i] = gd[i];
      if (nextra_global)
        for (i = 0; i < nextra_global; i++) hextra[i] = gextra[i];
    }

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}

int MinTSPINCG::linemin_backtrack(double eoriginal, double &alpha)
{
  int i;
  double fdothall, fdothme, hme, hmaxall, de_ideal, de;
  double hmax_spin;

  compute_fspin();

  fdothme = 0.0;
  for (i = 0; i < nvec; i++) fdothme += fvec[i] * h[i];
  for (i = 0; i < nvec; i++) fdothme += fspin[i] * hd[i];
  fdothall = fdothme;
  MPI_Allreduce(MPI_IN_PLACE, &fdothall, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (i = 0; i < nextra_global; i++) fdothall += fextra[i] * hextra[i];
  if (output->thermo->normflag) fdothall /= atom->natoms;
  if (fdothall <= 0.0) return DOWNHILL;

  // initial alpha based on max step in either block, also cap by ALPHA_MAX

  hme = 0.0;
  for (i = 0; i < nvec; i++) hme = MAX(hme, fabs(h[i]));
  MPI_Allreduce(&hme, &hmaxall, 1, MPI_DOUBLE, MPI_MAX, world);
  alpha = MIN(ALPHA_MAX, dmax / hmaxall);

  hme = 0.0;
  for (i = 0; i < nvec; i++) hme = MAX(hme, fabs(hd[i]));
  MPI_Allreduce(&hme, &hmax_spin, 1, MPI_DOUBLE, MPI_MAX, world);
  if (hmax_spin > 0.0) alpha = MIN(alpha, spin_dmax / hmax_spin);
  hmaxall = MAX(hmaxall, hmax_spin);

  if (nextra_global) {
    const double alpha_extra = modify->max_alpha(hextra);
    alpha = MIN(alpha, alpha_extra);
    for (i = 0; i < nextra_global; i++) hmaxall = MAX(hmaxall, fabs(hextra[i]));
  }
  if (hmaxall == 0.0) return ZEROFORCE;

  fix_minimize->store_box();
  for (i = 0; i < nvec; i++) x0[i] = xvec[i];
  compute_d_from_sp(d0);
  if (nextra_global) modify->min_store();

  while (true) {
    ecurrent = alpha_step(alpha, 1);
    de_ideal = -BACKTRACK_SLOPE * alpha * fdothall;
    de = ecurrent - eoriginal;

    if (de <= de_ideal) {
      if (nextra_global) {
        const int itmp = modify->min_reset_ref();
        if (itmp) ecurrent = energy_force(1);
      }
      return 0;
    }

    alpha *= ALPHA_REDUCE;

    if (alpha <= 0.0 || de_ideal >= -EMACH) {
      ecurrent = alpha_step(0.0, 0);
      if (de < 0.0) return ETOL;
      return ZEROALPHA;
    }
  }
}

int MinTSPINCG::linemin_quadratic(double eoriginal, double &alpha)
{
  int i;
  double fdothall, fdothme, hme, hmaxall, de_ideal, de;
  double delfh, engprev, relerr, alphaprev, fhprev, fh, alpha0;
  double dot, dotall;
  double alphamax;
  double hmax_spin;

  compute_fspin();

  fdothme = 0.0;
  for (i = 0; i < nvec; i++) fdothme += fvec[i] * h[i];
  for (i = 0; i < nvec; i++) fdothme += fspin[i] * hd[i];
  fdothall = fdothme;
  MPI_Allreduce(MPI_IN_PLACE, &fdothall, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (i = 0; i < nextra_global; i++) fdothall += fextra[i] * hextra[i];
  if (output->thermo->normflag) fdothall /= atom->natoms;
  if (fdothall <= 0.0) return DOWNHILL;

  hme = 0.0;
  for (i = 0; i < nvec; i++) hme = MAX(hme, fabs(h[i]));
  MPI_Allreduce(&hme, &hmaxall, 1, MPI_DOUBLE, MPI_MAX, world);
  alphamax = MIN(ALPHA_MAX, dmax / hmaxall);

  hme = 0.0;
  for (i = 0; i < nvec; i++) hme = MAX(hme, fabs(hd[i]));
  MPI_Allreduce(&hme, &hmax_spin, 1, MPI_DOUBLE, MPI_MAX, world);
  if (hmax_spin > 0.0) alphamax = MIN(alphamax, spin_dmax / hmax_spin);
  hmaxall = MAX(hmaxall, hmax_spin);

  if (nextra_global) {
    const double alpha_extra = modify->max_alpha(hextra);
    alphamax = MIN(alphamax, alpha_extra);
    for (i = 0; i < nextra_global; i++) hmaxall = MAX(hmaxall, fabs(hextra[i]));
  }
  if (hmaxall == 0.0) return ZEROFORCE;

  fix_minimize->store_box();
  for (i = 0; i < nvec; i++) x0[i] = xvec[i];
  compute_d_from_sp(d0);
  if (nextra_global) modify->min_store();

  alpha = alphamax;
  fhprev = fdothall;
  engprev = eoriginal;
  alphaprev = 0.0;

  while (true) {
    ecurrent = alpha_step(alpha, 1);

    // compute fh = f . h for updated config
    compute_fspin();
    dot = 0.0;
    for (i = 0; i < nvec; i++) dot += fvec[i] * h[i];
    for (i = 0; i < nvec; i++) dot += fspin[i] * hd[i];
    dotall = dot;
    MPI_Allreduce(MPI_IN_PLACE, &dotall, 1, MPI_DOUBLE, MPI_SUM, world);
    if (nextra_global)
      for (i = 0; i < nextra_global; i++) dotall += fextra[i] * hextra[i];
    fh = dotall;
    if (output->thermo->normflag) fh /= atom->natoms;

    delfh = fh - fhprev;
    if (fabs(fh) < EPS_QUAD || fabs(delfh) < EPS_QUAD) {
      ecurrent = alpha_step(0.0, 0);
      return ZEROQUAD;
    }

    relerr = fabs(1.0 - (0.5 * (alpha - alphaprev) * (fh + fhprev) + ecurrent) / engprev);
    alpha0 = alpha - (alpha - alphaprev) * fh / delfh;

    if (relerr <= QUADRATIC_TOL && alpha0 > 0.0 && alpha0 < alphamax) {
      ecurrent = alpha_step(alpha0, 1);
      if (ecurrent - eoriginal < EMACH) {
        if (nextra_global) {
          const int itmp = modify->min_reset_ref();
          if (itmp) ecurrent = energy_force(1);
        }
        return 0;
      }
    }

    de_ideal = -BACKTRACK_SLOPE * alpha * fdothall;
    de = ecurrent - eoriginal;
    if (de <= de_ideal) {
      if (nextra_global) {
        const int itmp = modify->min_reset_ref();
        if (itmp) ecurrent = energy_force(1);
      }
      return 0;
    }

    fhprev = fh;
    engprev = ecurrent;
    alphaprev = alpha;

    alpha *= ALPHA_REDUCE;
    if (alpha <= 0.0 || de_ideal >= -EMACH) {
      ecurrent = alpha_step(0.0, 0);
      return ZEROALPHA;
    }
  }
}

double MinTSPINCG::alpha_step(double alpha, int resetflag)
{
  const int nlocal = atom->nlocal;

  if (nextra_global) modify->min_step(0.0, hextra);

  // reset to starting point
  for (int i = 0; i < nvec; i++) xvec[i] = x0[i];
  set_sp_from_d(d0);

  // step forward along search directions
  if (alpha > 0.0) {
    if (nextra_global) modify->min_step(alpha, hextra);

    for (int i = 0; i < nvec; i++) xvec[i] += alpha * h[i];

    for (int i = 0; i < 3 * nlocal; i++) dtrial[i] = d0[i] + alpha * hd[i];
    set_sp_from_d(dtrial);
  }

  neval++;
  return energy_force(resetflag);
}
