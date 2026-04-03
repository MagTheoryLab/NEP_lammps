/*
    Implementation of SpinIntegrate and ensemble_spin parsing.
*/

#include "integrate_spin.cuh"

#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

void SpinIntegrate::initialize(Atom& atom, int& total_steps)
{
  if (type == 0 || !spin_ensemble) {
    return;
  }

  const int N = atom.number_of_atoms;
  if (atom.spin.size() < static_cast<size_t>(3 * N)) {
    atom.spin.resize(static_cast<size_t>(3 * N), 0.0);
    if (atom.cpu_spin.size() == static_cast<size_t>(3 * N)) {
      atom.spin.copy_from_host(atom.cpu_spin.data());
    }
  }

  spin_ensemble->initialize(N, seed_);
}

void SpinIntegrate::finalize()
{
  spin_ensemble.reset();
  type = 0;
  T1 = 0.0;
  T2 = 0.0;
  use_lattice_T = true;
  seed_ = 0;
}

void SpinIntegrate::parse_ensemble_spin(const char** param, int num_param)
{
  if (num_param < 2) {
    PRINT_INPUT_ERROR("ensemble_spin should have at least one parameter.\n");
  }

  if (strcmp(param[1], "glsd") == 0) {
    // ensemble_spin glsd  T1  T2  gamma  gamma_sprime  Tmode  seed
    if (num_param != 8) {
      PRINT_INPUT_ERROR("ensemble_spin glsd should have 6 parameters.\n");
    }
    type = 1;
    auto glsd = std::make_unique<Spin_Ensemble_GLSD>();
    glsd->type = type;

    double gamma = 0.0;
    double gamma_sprime = 0.0;

    if (!is_valid_real(param[2], &T1)) {
      PRINT_INPUT_ERROR("T1 for ensemble_spin should be a real number.\n");
    }
    if (!is_valid_real(param[3], &T2)) {
      PRINT_INPUT_ERROR("T2 for ensemble_spin should be a real number.\n");
    }
    if (!is_valid_real(param[4], &gamma)) {
      PRINT_INPUT_ERROR("gamma for ensemble_spin glsd should be a real number.\n");
    }
    if (!is_valid_real(param[5], &gamma_sprime)) {
      PRINT_INPUT_ERROR("gamma_sprime for ensemble_spin glsd should be a real number.\n");
    }

    if (strcmp(param[6], "lattice") == 0) {
      use_lattice_T = true;
    } else if (strcmp(param[6], "fixed") == 0) {
      use_lattice_T = false;
    } else {
      PRINT_INPUT_ERROR("Tmode for ensemble_spin should be lattice or fixed.\n");
    }

    if (!is_valid_int(param[7], &seed_)) {
      PRINT_INPUT_ERROR("seed for ensemble_spin glsd should be an integer.\n");
    }

    glsd->T1 = T1;
    glsd->T2 = T2;
    glsd->use_lattice_T = use_lattice_T;
    glsd->gamma = gamma;
    glsd->gamma_sprime = gamma_sprime;

    printf(
      "Use spin ensemble GLSD: T1=%g T2=%g gamma=%g gamma_sprime=%g Tmode=%s seed=%d\n",
      glsd->T1,
      glsd->T2,
      glsd->gamma,
      glsd->gamma_sprime,
      use_lattice_T ? "lattice" : "fixed",
      seed_);

    spin_ensemble = std::move(glsd);
  } else {
    PRINT_INPUT_ERROR("Unknown spin ensemble integrator.\n");
  }
}

void SpinIntegrate::compute(
  const double time_step,
  const double step_over_number_of_steps,
  Atom& atom,
  GPU_Vector<double>& mforce,
  double lattice_temperature,
  const GPU_Vector<int>* spin_freeze_mask)
{
  if (type == 0 || !spin_ensemble) {
    return;
  }
  spin_ensemble->compute(
    time_step, step_over_number_of_steps, atom, mforce, lattice_temperature, spin_freeze_mask);
}
