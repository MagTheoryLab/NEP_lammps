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

#include "min_tspin_cg_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "fix_minimize_kokkos.h"
#include "math_const.h"
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

namespace {
struct s_double2 {
  double d0, d1;
  KOKKOS_INLINE_FUNCTION
  s_double2() : d0(0.0), d1(0.0) {}
  KOKKOS_INLINE_FUNCTION
  s_double2 &operator+=(const s_double2 &rhs)
  {
    d0 += rhs.d0;
    d1 += rhs.d1;
    return *this;
  }
};
}    // namespace

MinTSPINCGKokkos::MinTSPINCGKokkos(LAMMPS *lmp) : MinKokkos(lmp)
{
  kokkosable = 1;
  searchflag = 1;

  gextra = hextra = nullptr;

  eta_zeta = 0.1;     // length/muB
  spin_dmax = 0.1;    // length units
  spin_ftol = 0.0;
  vary_mag = 0;

  eta_auto = 0;
  eta_auto_weight = 0.05;
  eta_auto_min = 1.0e-4;
  eta_auto_max = 10.0;
}

MinTSPINCGKokkos::~MinTSPINCGKokkos()
{
  delete[] gextra;
  delete[] hextra;
}

void MinTSPINCGKokkos::init()
{
  MinKokkos::init();

  if (linestyle == BACKTRACK) linemin = &MinTSPINCGKokkos::linemin_backtrack;
  else if (linestyle == QUADRATIC) linemin = &MinTSPINCGKokkos::linemin_quadratic;
  else error->all(FLERR, "min tspin/cg/kk supports only min_modify line backtrack or quadratic");

  delete[] gextra;
  delete[] hextra;
  gextra = hextra = nullptr;
}

void MinTSPINCGKokkos::setup_style()
{
  if (nextra_atom)
    error->all(FLERR, "min tspin/cg/kk does not support additional extra per-atom DOFs");

  if (!atom->sp_flag)
    error->all(FLERR, "min tspin/cg/kk requires atom/spin style");

  if (eta_zeta <= 0.0) error->all(FLERR, "min tspin/cg/kk requires eta_zeta > 0");
  if (spin_dmax <= 0.0) error->all(FLERR, "min tspin/cg/kk requires spin_dmax > 0");

  // per-atom vectors in FixMinimizeKokkos (always 3 elements per atom)
  fix_minimize_kk->add_vector_kokkos();    // x0
  fix_minimize_kk->add_vector_kokkos();    // g
  fix_minimize_kk->add_vector_kokkos();    // h
  fix_minimize_kk->add_vector_kokkos();    // d0
  fix_minimize_kk->add_vector_kokkos();    // gd
  fix_minimize_kk->add_vector_kokkos();    // hd

  if (nextra_global) {
    gextra = new double[nextra_global];
    hextra = new double[nextra_global];
  }
}

int MinTSPINCGKokkos::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0], "eta_zeta") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eta_zeta = utils::numeric(FLERR, arg[1], false, lmp);
    if (eta_zeta <= 0.0) error->all(FLERR, "min tspin/cg/kk requires eta_zeta > 0");
    return 2;
  }
  if (strcmp(arg[0], "spin_ftol") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    spin_ftol = utils::numeric(FLERR, arg[1], false, lmp);
    if (spin_ftol < 0.0) error->all(FLERR, "min tspin/cg/kk requires spin_ftol >= 0");
    return 2;
  }
  if (strcmp(arg[0], "spin_dmax") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    spin_dmax = utils::numeric(FLERR, arg[1], false, lmp);
    if (spin_dmax <= 0.0) error->all(FLERR, "min tspin/cg/kk requires spin_dmax > 0");
    return 2;
  }
  if (strcmp(arg[0], "vary_mag") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    vary_mag = utils::logical(FLERR, arg[1], false, lmp);
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
      error->all(FLERR, "min tspin/cg/kk requires eta_auto_weight in [0,1]");
    return 2;
  }
  if (strcmp(arg[0], "eta_auto_min") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eta_auto_min = utils::numeric(FLERR, arg[1], false, lmp);
    if (eta_auto_min <= 0.0) error->all(FLERR, "min tspin/cg/kk requires eta_auto_min > 0");
    return 2;
  }
  if (strcmp(arg[0], "eta_auto_max") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal min_modify command");
    eta_auto_max = utils::numeric(FLERR, arg[1], false, lmp);
    if (eta_auto_max <= 0.0) error->all(FLERR, "min tspin/cg/kk requires eta_auto_max > 0");
    return 2;
  }
  return 0;
}

void MinTSPINCGKokkos::reset_vectors()
{
  nvec = 3 * atom->nlocal;

  atomKK->sync(Device, F_MASK | X_MASK);
  auto d_x = atomKK->k_x.d_view;
  auto d_f = atomKK->k_f.d_view;

  if (nvec) xvec = DAT::t_ffloat_1d(d_x.data(), d_x.size());
  if (nvec) fvec = DAT::t_ffloat_1d(d_f.data(), d_f.size());

  x0 = fix_minimize_kk->request_vector_kokkos(0);
  g = fix_minimize_kk->request_vector_kokkos(1);
  h = fix_minimize_kk->request_vector_kokkos(2);
  d0 = fix_minimize_kk->request_vector_kokkos(3);
  gd = fix_minimize_kk->request_vector_kokkos(4);
  hd = fix_minimize_kk->request_vector_kokkos(5);

  // We update these vectors on Device in Kokkos kernels; mark them modified so
  // host-side classic communication (when enabled) will sync correctly.
  fix_minimize_kk->k_vectors.modify<LMPDeviceType>();

  const int need = 3 * atom->nmax;
  if ((int) fspin.extent(0) < need) fspin = DAT::t_ffloat_1d("min/tspin/cg/kk:fspin", need);
  if ((int) dtrial.extent(0) < need) dtrial = DAT::t_ffloat_1d("min/tspin/cg/kk:dtrial", need);
}

void MinTSPINCGKokkos::compute_d_from_sp(DAT::t_ffloat_1d dvec) const
{
  const int nlocal = atom->nlocal;
  atomKK->sync(Device, SP_MASK);
  auto l_sp = atomKK->k_sp.d_view;
  const double scale = eta_zeta;

  Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
    const double mag = l_sp(i, 3);
    const int j = 3 * i;
    if (mag == 0.0) {
      dvec(j + 0) = 0.0;
      dvec(j + 1) = 0.0;
      dvec(j + 2) = 0.0;
    } else {
      dvec(j + 0) = scale * mag * l_sp(i, 0);
      dvec(j + 1) = scale * mag * l_sp(i, 1);
      dvec(j + 2) = scale * mag * l_sp(i, 2);
    }
  });
}

void MinTSPINCGKokkos::set_sp_from_d(const DAT::t_ffloat_1d &dvec)
{
  const int nlocal = atom->nlocal;
  auto l_sp = atomKK->k_sp.d_view;
  const double mag_scale = 1.0 / eta_zeta;
  const int vary = vary_mag;

  Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
    if (l_sp(i, 3) == 0.0) return;

    const int j = 3 * i;
    const double dx = dvec(j + 0);
    const double dy = dvec(j + 1);
    const double dz = dvec(j + 2);
    const double dn = sqrt(dx * dx + dy * dy + dz * dz);
    if (dn < 1.0e-40) return;

    l_sp(i, 0) = dx / dn;
    l_sp(i, 1) = dy / dn;
    l_sp(i, 2) = dz / dn;
    if (vary) l_sp(i, 3) = mag_scale * dn;
  });

  atomKK->modified(Device, SP_MASK);
}

void MinTSPINCGKokkos::compute_fspin()
{
  const int nlocal = atom->nlocal;
  atomKK->sync(Device, SP_MASK | FM_MASK);
  auto l_sp = atomKK->k_sp.d_view;
  auto l_fm = atomKK->k_fm.d_view;

  auto l_fspin = fspin;
  const double scale = 1.0 / eta_zeta;
  const int vary = vary_mag;

  Kokkos::parallel_for(nlocal, LAMMPS_LAMBDA(const int &i) {
    const int j = 3 * i;
    if (l_sp(i, 3) == 0.0) {
      l_fspin(j + 0) = 0.0;
      l_fspin(j + 1) = 0.0;
      l_fspin(j + 2) = 0.0;
      return;
    }

    if (vary) {
      l_fspin(j + 0) = scale * l_fm(i, 0);
      l_fspin(j + 1) = scale * l_fm(i, 1);
      l_fspin(j + 2) = scale * l_fm(i, 2);
      return;
    }

    const double sx = l_sp(i, 0);
    const double sy = l_sp(i, 1);
    const double sz = l_sp(i, 2);
    const double hdot = l_fm(i, 0) * sx + l_fm(i, 1) * sy + l_fm(i, 2) * sz;
    l_fspin(j + 0) = scale * (l_fm(i, 0) - hdot * sx);
    l_fspin(j + 1) = scale * (l_fm(i, 1) - hdot * sy);
    l_fspin(j + 2) = scale * (l_fm(i, 2) - hdot * sz);
  });
}

double MinTSPINCGKokkos::estimate_eta_zeta()
{
  const int nlocal = atom->nlocal;
  atomKK->sync(Device, F_MASK | FM_MASK | SP_MASK);
  auto l_f = atomKK->k_f.d_view;
  auto l_sp = atomKK->k_sp.d_view;
  auto l_fm = atomKK->k_fm.d_view;

  const int vary = vary_mag;

  s_double2 sums;
  Kokkos::parallel_reduce(
    nlocal,
    LAMMPS_LAMBDA(const int &i, s_double2 &sums) {
      const double fx = l_f(i, 0);
      const double fy = l_f(i, 1);
      const double fz = l_f(i, 2);
      sums.d0 += fx * fx + fy * fy + fz * fz;

      if (l_sp(i, 3) == 0.0) return;

      double hx = l_fm(i, 0);
      double hy = l_fm(i, 1);
      double hz = l_fm(i, 2);

      if (!vary) {
        const double sx = l_sp(i, 0);
        const double sy = l_sp(i, 1);
        const double sz = l_sp(i, 2);
        const double hdot = hx * sx + hy * sy + hz * sz;
        hx -= hdot * sx;
        hy -= hdot * sy;
        hz -= hdot * sz;
      }

      sums.d1 += hx * hx + hy * hy + hz * hz;
    },
    sums);

  double local[2] = {sums.d0, sums.d1};
  double all[2] = {0.0, 0.0};
  MPI_Allreduce(local, all, 2, MPI_DOUBLE, MPI_SUM, world);

  if (all[0] <= 0.0 || all[1] <= 0.0) return eta_zeta;
  return sqrt(all[1] / all[0]);
}

void MinTSPINCGKokkos::update_eta_zeta()
{
  if (!eta_auto) return;

  double eta_est = estimate_eta_zeta();
  if (!(eta_est > 0.0) || std::isnan(eta_est) || std::isinf(eta_est)) return;

  double lo = eta_auto_min;
  double hi = eta_auto_max;
  if (lo > hi) {
    const double tmp = lo;
    lo = hi;
    hi = tmp;
  }

  eta_est = MIN(hi, MAX(lo, eta_est));
  eta_zeta = (1.0 - eta_auto_weight) * eta_zeta + eta_auto_weight * eta_est;
  eta_zeta = MIN(hi, MAX(lo, eta_zeta));
}

double MinTSPINCGKokkos::fnorm_sqr()
{
  atomKK->sync(Device, F_MASK);
  compute_fspin();

  double local_norm2_sqr = 0.0;
  {
    auto l_fvec = fvec;
    auto l_fspin = fspin;

    Kokkos::parallel_reduce(
      nvec, LAMMPS_LAMBDA(const int &i, double &local_norm2_sqr) {
        local_norm2_sqr += l_fvec(i) * l_fvec(i) + l_fspin(i) * l_fspin(i);
      },
      local_norm2_sqr);
  }

  double norm2_sqr = 0.0;
  norm2_sqr = local_norm2_sqr;
  MPI_Allreduce(MPI_IN_PLACE, &norm2_sqr, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm2_sqr += fextra[i] * fextra[i];

  return norm2_sqr;
}

double MinTSPINCGKokkos::fnorm_inf()
{
  atomKK->sync(Device, F_MASK);
  compute_fspin();

  double local_norm_inf = 0.0;
  {
    auto l_fvec = fvec;
    auto l_fspin = fspin;

    Kokkos::parallel_reduce(
      nvec,
      LAMMPS_LAMBDA(const int &i, double &local_norm_inf) {
        const double v = MAX(l_fvec(i) * l_fvec(i), l_fspin(i) * l_fspin(i));
        local_norm_inf = MAX(local_norm_inf, v);
      },
      Kokkos::Max<double>(local_norm_inf));
  }

  double norm_inf = 0.0;
  MPI_Allreduce(&local_norm_inf, &norm_inf, 1, MPI_DOUBLE, MPI_MAX, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm_inf = MAX(norm_inf, fextra[i] * fextra[i]);

  return norm_inf;
}

double MinTSPINCGKokkos::fnorm_max()
{
  atomKK->sync(Device, F_MASK);
  compute_fspin();

  const int nlocal = atom->nlocal;
  double local_norm_max = 0.0;
  {
    auto l_fvec = fvec;
    auto l_fspin = fspin;

    Kokkos::parallel_reduce(
      nlocal,
      LAMMPS_LAMBDA(const int &i, double &local_norm_max) {
        const int j = 3 * i;
        const double fv = l_fvec(j) * l_fvec(j) + l_fvec(j + 1) * l_fvec(j + 1) +
          l_fvec(j + 2) * l_fvec(j + 2);
        const double sv = l_fspin(j) * l_fspin(j) + l_fspin(j + 1) * l_fspin(j + 1) +
          l_fspin(j + 2) * l_fspin(j + 2);
        local_norm_max = MAX(local_norm_max, MAX(fv, sv));
      },
      Kokkos::Max<double>(local_norm_max));
  }

  double norm_max = 0.0;
  MPI_Allreduce(&local_norm_max, &norm_max, 1, MPI_DOUBLE, MPI_MAX, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) norm_max = MAX(norm_max, fextra[i] * fextra[i]);

  return norm_max;
}

int MinTSPINCGKokkos::iterate(int maxiter)
{
  double beta, gg, dot[2], dotall[2], fdotf;

  fix_minimize_kk->k_vectors.sync<LMPDeviceType>();
  fix_minimize_kk->k_vectors.modify<LMPDeviceType>();

  atomKK->sync(Device, F_MASK);

  int nlimit = static_cast<int>(MIN(MAXSMALLINT, ndoftotal));
  if (nlimit < 1) nlimit = 1;

  compute_d_from_sp(d0);
  compute_fspin();

  {
    auto l_h = h;
    auto l_g = g;
    auto l_hd = hd;
    auto l_gd = gd;
    auto l_fvec = fvec;
    auto l_fspin = fspin;

    Kokkos::parallel_for(nvec, LAMMPS_LAMBDA(const int &i) {
      l_h(i) = l_g(i) = l_fvec(i);
      l_hd(i) = l_gd(i) = l_fspin(i);
    });
  }

  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) hextra[i] = gextra[i] = fextra[i];

  gg = fnorm_sqr();

  for (int iter = 0; iter < maxiter; iter++) {
    if (timer->check_timeout(niter)) return TIMEOUT;

    const bigint ntimestep = ++update->ntimestep;
    niter++;

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
      {
        auto l_h = h;
        auto l_g = g;
        auto l_hd = hd;
        auto l_gd = gd;
        auto l_fvec = fvec;
        auto l_fspin = fspin;

        Kokkos::parallel_for(nvec, LAMMPS_LAMBDA(const int &i) {
          l_h(i) = l_g(i) = l_fvec(i);
          l_hd(i) = l_gd(i) = l_fspin(i);
        });
      }
      if (nextra_global)
        for (int i = 0; i < nextra_global; i++) hextra[i] = gextra[i] = fextra[i];
      gg = fnorm_sqr();
    }

    compute_fspin();

    s_double2 sdot;
    {
      auto l_fvec = fvec;
      auto l_fspin = fspin;
      auto l_g = g;
      auto l_gd = gd;

      Kokkos::parallel_reduce(
        nvec,
        LAMMPS_LAMBDA(const int &i, s_double2 &sdot) {
          sdot.d0 += l_fvec(i) * l_fvec(i) + l_fspin(i) * l_fspin(i);
          sdot.d1 += l_fvec(i) * l_g(i) + l_fspin(i) * l_gd(i);
        },
        sdot);
    }
    dot[0] = sdot.d0;
    dot[1] = sdot.d1;

    dotall[0] = dot[0];
    dotall[1] = dot[1];
    MPI_Allreduce(MPI_IN_PLACE, dotall, 2, MPI_DOUBLE, MPI_SUM, world);
    if (nextra_global)
      for (int i = 0; i < nextra_global; i++) {
        dotall[0] += fextra[i] * fextra[i];
        dotall[1] += fextra[i] * gextra[i];
      }

    fdotf = 0.0;
    int spin_ok = 1;
    double spin_fdotf = 0.0;

    // Lattice+spin tolerance (standard Min behavior, but with spin included).
    if (update->ftol > 0.0) {
      if (normstyle == MAX) fdotf = fnorm_max();
      else if (normstyle == INF) fdotf = fnorm_inf();
      else if (normstyle == TWO) fdotf = dotall[0];
      else error->all(FLERR, "Illegal min_modify command");
    }

    // Optional spin-only tolerance, to prevent lattice forces from "dominating"
    // the convergence check. Uses the effective spin driving force (fspin),
    // which is already projected transverse to S when vary_mag = 0.
    if (spin_ftol > 0.0) {
      if (normstyle == TWO) {
        double local_spin2 = 0.0;
        {
          auto l_fspin = fspin;
          Kokkos::parallel_reduce(
            nvec,
            LAMMPS_LAMBDA(const int &i, double &local_spin2) { local_spin2 += l_fspin(i) * l_fspin(i); },
            local_spin2);
        }
        spin_fdotf = local_spin2;
        MPI_Allreduce(MPI_IN_PLACE, &spin_fdotf, 1, MPI_DOUBLE, MPI_SUM, world);
      } else if (normstyle == INF) {
        double local_spin_inf = 0.0;
        {
          auto l_fspin = fspin;
          Kokkos::parallel_reduce(
            nvec,
            LAMMPS_LAMBDA(const int &i, double &local_spin_inf) {
              const double v = l_fspin(i) * l_fspin(i);
              local_spin_inf = MAX(local_spin_inf, v);
            },
            Kokkos::Max<double>(local_spin_inf));
        }
        MPI_Allreduce(&local_spin_inf, &spin_fdotf, 1, MPI_DOUBLE, MPI_MAX, world);
      } else if (normstyle == MAX) {
        const int nlocal = atom->nlocal;
        double local_spin_max = 0.0;
        {
          auto l_fspin = fspin;
          Kokkos::parallel_reduce(
            nlocal,
            LAMMPS_LAMBDA(const int &i, double &local_spin_max) {
              const int j = 3 * i;
              const double sv = l_fspin(j) * l_fspin(j) + l_fspin(j + 1) * l_fspin(j + 1) +
                l_fspin(j + 2) * l_fspin(j + 2);
              local_spin_max = MAX(local_spin_max, sv);
            },
            Kokkos::Max<double>(local_spin_max));
        }
        MPI_Allreduce(&local_spin_max, &spin_fdotf, 1, MPI_DOUBLE, MPI_MAX, world);
      } else {
        error->all(FLERR, "Illegal min_modify command");
      }

      spin_ok = (spin_fdotf < spin_ftol * spin_ftol);
    }

    const int ftol_ok = (update->ftol <= 0.0) || (fdotf < update->ftol * update->ftol);
    if ((update->ftol > 0.0 || spin_ftol > 0.0) && ftol_ok && spin_ok) return FTOL;

    beta = MAX(0.0, (dotall[0] - dotall[1]) / gg);
    if ((iter + 1) % nlimit == 0) beta = 0.0;
    gg = dotall[0];

    {
      auto l_h = h;
      auto l_g = g;
      auto l_hd = hd;
      auto l_gd = gd;
      auto l_fvec = fvec;
      auto l_fspin = fspin;

      Kokkos::parallel_for(nvec, LAMMPS_LAMBDA(const int &i) {
        l_g(i) = l_fvec(i);
        l_h(i) = l_g(i) + beta * l_h(i);
        l_gd(i) = l_fspin(i);
        l_hd(i) = l_gd(i) + beta * l_hd(i);
      });
    }

    if (nextra_global)
      for (int i = 0; i < nextra_global; i++) {
        gextra[i] = fextra[i];
        hextra[i] = gextra[i] + beta * hextra[i];
      }

    double downhill_local = 0.0;
    {
      auto l_h = h;
      auto l_g = g;
      auto l_hd = hd;
      auto l_gd = gd;

      Kokkos::parallel_reduce(
        nvec,
        LAMMPS_LAMBDA(const int &i, double &downhill_local) {
          downhill_local += l_g(i) * l_h(i) + l_gd(i) * l_hd(i);
        },
        downhill_local);
    }
    MPI_Allreduce(MPI_IN_PLACE, &downhill_local, 1, MPI_DOUBLE, MPI_SUM, world);
    if (nextra_global)
      for (int i = 0; i < nextra_global; i++) downhill_local += gextra[i] * hextra[i];

    if (downhill_local <= 0.0) {
      auto l_h = h;
      auto l_g = g;
      auto l_hd = hd;
      auto l_gd = gd;

      Kokkos::parallel_for(nvec, LAMMPS_LAMBDA(const int &i) {
        l_h(i) = l_g(i);
        l_hd(i) = l_gd(i);
      });

      if (nextra_global)
        for (int i = 0; i < nextra_global; i++) hextra[i] = gextra[i];
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

int MinTSPINCGKokkos::linemin_backtrack(double eoriginal, double &alpha)
{
  double fdothme = 0.0, fdothall, hme, hmaxall, de_ideal, de;
  double hmax_spin;

  compute_fspin();

  {
    auto l_fvec = fvec;
    auto l_h = h;
    auto l_fspin = fspin;
    auto l_hd = hd;

    Kokkos::parallel_reduce(
      nvec,
      LAMMPS_LAMBDA(const int &i, double &fdothme) {
        fdothme += l_fvec(i) * l_h(i) + l_fspin(i) * l_hd(i);
      },
      fdothme);
  }

  fdothall = fdothme;
  MPI_Allreduce(MPI_IN_PLACE, &fdothall, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) fdothall += fextra[i] * hextra[i];
  if (output->thermo->normflag) fdothall /= atom->natoms;
  if (fdothall <= 0.0) return DOWNHILL;

  hme = 0.0;
  {
    auto l_h = h;
    Kokkos::parallel_reduce(
      nvec,
      LAMMPS_LAMBDA(const int &i, double &hme) { hme = MAX(hme, fabs(l_h(i))); },
      Kokkos::Max<double>(hme));
  }
  MPI_Allreduce(&hme, &hmaxall, 1, MPI_DOUBLE, MPI_MAX, world);
  alpha = MIN(ALPHA_MAX, dmax / hmaxall);

  hme = 0.0;
  {
    auto l_hd = hd;
    Kokkos::parallel_reduce(
      nvec,
      LAMMPS_LAMBDA(const int &i, double &hme) { hme = MAX(hme, fabs(l_hd(i))); },
      Kokkos::Max<double>(hme));
  }
  MPI_Allreduce(&hme, &hmax_spin, 1, MPI_DOUBLE, MPI_MAX, world);
  if (hmax_spin > 0.0) alpha = MIN(alpha, spin_dmax / hmax_spin);
  hmaxall = MAX(hmaxall, hmax_spin);

  if (nextra_global) {
    const double alpha_extra = modify->max_alpha(hextra);
    alpha = MIN(alpha, alpha_extra);
    for (int i = 0; i < nextra_global; i++) hmaxall = MAX(hmaxall, fabs(hextra[i]));
  }
  if (hmaxall == 0.0) return ZEROFORCE;

  fix_minimize_kk->store_box();
  auto l_xvec = xvec;
  auto l_x0 = x0;
  Kokkos::parallel_for(nvec, LAMMPS_LAMBDA(const int &i) { l_x0(i) = l_xvec(i); });
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

int MinTSPINCGKokkos::linemin_quadratic(double eoriginal, double &alpha)
{
  double fdothme = 0.0, fdothall, hme, hmaxall, de_ideal, de;
  double delfh, engprev, relerr, alphaprev, fhprev, fh, alpha0;
  double dot, dotall;
  double alphamax;
  double hmax_spin;

  compute_fspin();

  {
    auto l_fvec = fvec;
    auto l_h = h;
    auto l_fspin = fspin;
    auto l_hd = hd;

    Kokkos::parallel_reduce(
      nvec,
      LAMMPS_LAMBDA(const int &i, double &fdothme) {
        fdothme += l_fvec(i) * l_h(i) + l_fspin(i) * l_hd(i);
      },
      fdothme);
  }

  fdothall = fdothme;
  MPI_Allreduce(MPI_IN_PLACE, &fdothall, 1, MPI_DOUBLE, MPI_SUM, world);
  if (nextra_global)
    for (int i = 0; i < nextra_global; i++) fdothall += fextra[i] * hextra[i];
  if (output->thermo->normflag) fdothall /= atom->natoms;
  if (fdothall <= 0.0) return DOWNHILL;

  hme = 0.0;
  {
    auto l_h = h;
    Kokkos::parallel_reduce(
      nvec,
      LAMMPS_LAMBDA(const int &i, double &hme) { hme = MAX(hme, fabs(l_h(i))); },
      Kokkos::Max<double>(hme));
  }
  MPI_Allreduce(&hme, &hmaxall, 1, MPI_DOUBLE, MPI_MAX, world);
  alphamax = MIN(ALPHA_MAX, dmax / hmaxall);

  hme = 0.0;
  {
    auto l_hd = hd;
    Kokkos::parallel_reduce(
      nvec,
      LAMMPS_LAMBDA(const int &i, double &hme) { hme = MAX(hme, fabs(l_hd(i))); },
      Kokkos::Max<double>(hme));
  }
  MPI_Allreduce(&hme, &hmax_spin, 1, MPI_DOUBLE, MPI_MAX, world);
  if (hmax_spin > 0.0) alphamax = MIN(alphamax, spin_dmax / hmax_spin);
  hmaxall = MAX(hmaxall, hmax_spin);

  if (nextra_global) {
    const double alpha_extra = modify->max_alpha(hextra);
    alphamax = MIN(alphamax, alpha_extra);
    for (int i = 0; i < nextra_global; i++) hmaxall = MAX(hmaxall, fabs(hextra[i]));
  }
  if (hmaxall == 0.0) return ZEROFORCE;

  fix_minimize_kk->store_box();
  auto l_xvec = xvec;
  auto l_x0 = x0;
  Kokkos::parallel_for(nvec, LAMMPS_LAMBDA(const int &i) { l_x0(i) = l_xvec(i); });
  compute_d_from_sp(d0);
  if (nextra_global) modify->min_store();

  alpha = alphamax;
  fhprev = fdothall;
  engprev = eoriginal;
  alphaprev = 0.0;

  while (true) {
    ecurrent = alpha_step(alpha, 1);

    atomKK->sync(Device, F_MASK);
    compute_fspin();

    dot = 0.0;
    {
      auto l_fvec = fvec;
      auto l_h = h;
      auto l_fspin = fspin;
      auto l_hd = hd;

    Kokkos::parallel_reduce(
      nvec,
      LAMMPS_LAMBDA(const int &i, double &dot) {
          dot += l_fvec(i) * l_h(i) + l_fspin(i) * l_hd(i);
        },
      dot);
    }

    dotall = dot;
    MPI_Allreduce(MPI_IN_PLACE, &dotall, 1, MPI_DOUBLE, MPI_SUM, world);
    if (nextra_global)
      for (int i = 0; i < nextra_global; i++) dotall += fextra[i] * hextra[i];
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

double MinTSPINCGKokkos::alpha_step(double alpha, int resetflag)
{
  if (nextra_global) modify->min_step(0.0, hextra);

  atomKK->k_x.clear_sync_state();
  atomKK->k_sp.clear_sync_state();

  auto l_xvec = xvec;
  auto l_x0 = x0;
  Kokkos::parallel_for(nvec, LAMMPS_LAMBDA(const int &i) { l_xvec(i) = l_x0(i); });
  atomKK->modified(Device, X_MASK);

  set_sp_from_d(d0);

  if (alpha > 0.0) {
    if (nextra_global) modify->min_step(alpha, hextra);

    atomKK->sync(Device, X_MASK);    // positions can be modified by fix box/relax

    auto l_h = h;
    auto l_d0 = d0;
    auto l_hd = hd;
    auto l_dtrial = dtrial;

    Kokkos::parallel_for(nvec, LAMMPS_LAMBDA(const int &i) {
      l_xvec(i) += alpha * l_h(i);
      l_dtrial(i) = l_d0(i) + alpha * l_hd(i);
    });

    atomKK->modified(Device, X_MASK);
    set_sp_from_d(dtrial);
  }

  neval++;
  return energy_force(resetflag);
}

#endif
