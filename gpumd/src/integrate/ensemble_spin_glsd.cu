/*
    GLSD spin integrator implementation.
    Two half GLSD drift+noise steps with an EB2B/Boris precession in between.
*/

#include "ensemble_spin_glsd.cuh"

#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <vector>

namespace
{
static __global__ void initialize_rng_states(int N, int seed, curandState* states)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    curand_init(seed, i, 0, &states[i]);
  }
}

static __global__ void glsd_half_step(
  int N,
  double drift_coeff,
  double sigma_half,
  const double* g_mforce,
  double* g_sx,
  double* g_sy,
  double* g_sz,
  curandState* rng_states,
  const int* mask)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (mask && mask[i] == 0) {
      return;
    }
    curandState local = rng_states[i];
    double nx = curand_normal_double(&local);
    double ny = curand_normal_double(&local);
    double nz = curand_normal_double(&local);

    g_sx[i] += drift_coeff * g_mforce[i] + sigma_half * nx;
    g_sy[i] += drift_coeff * g_mforce[i + N] + sigma_half * ny;
    g_sz[i] += drift_coeff * g_mforce[i + 2 * N] + sigma_half * nz;

    rng_states[i] = local;
  }
}

static __global__ void glsd_boris_precession(
  int N,
  double omega_coeff,
  double dt,
  const double* g_mforce,
  double* g_sx,
  double* g_sy,
  double* g_sz,
  int preserve_magnitude,
  const int* mask)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (mask && mask[i] == 0) {
      return;
    }
    double hx = g_mforce[i];
    double hy = g_mforce[i + N];
    double hz = g_mforce[i + 2 * N];

    // Omega = omega_coeff * H_R; Boris update for dS/dt = Omega x S
    double tx = 0.5 * dt * omega_coeff * hx;
    double ty = 0.5 * dt * omega_coeff * hy;
    double tz = 0.5 * dt * omega_coeff * hz;

    double t2 = tx * tx + ty * ty + tz * tz;
    double s = 2.0 / (1.0 + t2);

    double sx = g_sx[i];
    double sy = g_sy[i];
    double sz = g_sz[i];

    double norm_before = 0.0;
    if (preserve_magnitude) {
      norm_before = sqrt(sx * sx + sy * sy + sz * sz);
    }

    // v' = v + v x t
    double vpx = sx + (sy * tz - sz * ty);
    double vpy = sy + (sz * tx - sx * tz);
    double vpz = sz + (sx * ty - sy * tx);

    // svec = s * t
    double stx = s * tx;
    double sty = s * ty;
    double stz = s * tz;

    // v+ = v + v' x svec (standard Boris rotation)
    double sx_new = sx + (vpy * stz - vpz * sty);
    double sy_new = sy + (vpz * stx - vpx * stz);
    double sz_new = sz + (vpx * sty - vpy * stx);

    if (preserve_magnitude) {
      double norm_after = sqrt(sx_new * sx_new + sy_new * sy_new + sz_new * sz_new);
      if (norm_after > 0.0 && norm_before > 0.0) {
        double scale = norm_before / norm_after;
        sx_new *= scale;
        sy_new *= scale;
        sz_new *= scale;
      }
    }

    g_sx[i] = sx_new;
    g_sy[i] = sy_new;
    g_sz[i] = sz_new;
  }
}
} // namespace

void Spin_Ensemble_GLSD::initialize(int natoms, int seed)
{
  if (natoms <= 0) {
    PRINT_INPUT_ERROR("Spin_Ensemble_GLSD requires natoms > 0.\n");
  }
  rng_states.resize(natoms);
  initialize_rng_states<<<(natoms - 1) / 128 + 1, 128>>>(natoms, seed, rng_states.data());
  GPU_CHECK_KERNEL
}

void Spin_Ensemble_GLSD::compute(
  const double time_step,
  const double step_over_number_of_steps,
  Atom& atom,
  GPU_Vector<double>& mforce,
  double lattice_temperature,
  const GPU_Vector<int>* spin_freeze_mask)
{
  if (type == 0) {
    return;
  }

  const int N = atom.number_of_atoms;
  if (N <= 0) {
    return;
  }
  if (atom.spin.size() < static_cast<size_t>(3 * N)) {
    PRINT_INPUT_ERROR("Spin vector is not allocated for GLSD integrator.\n");
  }
  if (mforce.size() < static_cast<size_t>(3 * N)) {
    PRINT_INPUT_ERROR("mforce is not allocated for GLSD integrator.\n");
  }

  double T_spin = use_lattice_T ? lattice_temperature : (T1 + (T2 - T1) * step_over_number_of_steps);
  if (T_spin < 0.0) {
    T_spin = 0.0;
  }
  double mu_sprime = 2.0 * gamma_sprime * K_B * T_spin;
  if (mu_sprime < 0.0) {
    mu_sprime = 0.0;
  }
  double sigma_half = (gamma_sprime == 0.0) ? 0.0 : sqrt(0.5 * mu_sprime * time_step);
  double drift_coeff = gamma_sprime * 0.5 * time_step;
  double omega_coeff = -gamma / HBAR;

  double* sx = atom.spin.data();
  double* sy = atom.spin.data() + N;
  double* sz = atom.spin.data() + 2 * N;
  int preserve_magnitude = (gamma_sprime == 0.0) ? 1 : 0;
  const int* mask_ptr =
    (spin_freeze_mask && spin_freeze_mask->size() == static_cast<size_t>(N))
      ? spin_freeze_mask->data()
      : nullptr;

  static int print_counter = 0;
  ++print_counter;
  if (print_counter <= 10 || (print_counter % 1000 == 0)) {
    std::vector<double> h_spin(static_cast<size_t>(3 * N));
    CHECK(gpuMemcpy(
      h_spin.data(), atom.spin.data(), sizeof(double) * static_cast<size_t>(3 * N), gpuMemcpyDeviceToHost));

    double mmin = 1.0e30;
    double mmax = 0.0;
    for (int i = 0; i < N; ++i) {
      double sxi = h_spin[static_cast<size_t>(i)];
      double syi = h_spin[static_cast<size_t>(i + N)];
      double szi = h_spin[static_cast<size_t>(i + 2 * N)];
      double m = std::sqrt(sxi * sxi + syi * syi + szi * szi);
      if (m < mmin) {
        mmin = m;
      }
      if (m > mmax) {
        mmax = m;
      }
    }
    printf(
      "[GLSD] step_frac=%g T_spin=%g gamma=%g gamma_sprime=%g "
      "mu_sprime=%g sigma_half=%g drift_coeff=%g |S|_min=%g |S|_max=%g\n",
      step_over_number_of_steps,
      T_spin,
      gamma,
      gamma_sprime,
      mu_sprime,
      sigma_half,
      drift_coeff,
      mmin,
      mmax);
    fflush(stdout);
  }

  const int grid = (N - 1) / 128 + 1;
  if (gamma_sprime != 0.0) {
    glsd_half_step<<<grid, 128>>>(
      N,
      drift_coeff,
      sigma_half,
      mforce.data(),
      sx,
      sy,
      sz,
      rng_states.data(),
      mask_ptr);
    GPU_CHECK_KERNEL
  }

  glsd_boris_precession<<<grid, 128>>>(
    N,
    omega_coeff,
    time_step,
    mforce.data(),
    sx,
    sy,
    sz,
    preserve_magnitude,
    mask_ptr);
  GPU_CHECK_KERNEL

  if (gamma_sprime != 0.0) {
    glsd_half_step<<<grid, 128>>>(
      N,
      drift_coeff,
      sigma_half,
      mforce.data(),
      sx,
      sy,
      sz,
      rng_states.data(),
      mask_ptr);
    GPU_CHECK_KERNEL
  }
}
