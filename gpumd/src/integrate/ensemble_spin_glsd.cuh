/*
    GLSD spin integrator with EB2B (Boris) precession.
*/

#pragma once

#include "ensemble_spin.cuh"
#include "utilities/gpu_vector.cuh"
#include <curand_kernel.h>

class Spin_Ensemble_GLSD : public Spin_Ensemble
{
public:
  double gamma = 0.0;        // gyromagnetic ratio (natural units)
  double gamma_sprime = 0.0; // GLSD drift / damping coefficient

  GPU_Vector<curandState> rng_states; // one RNG state per atom

  void initialize(int natoms, int seed);
  void compute(
    const double time_step,
    const double step_over_number_of_steps,
    Atom& atom,
    GPU_Vector<double>& mforce,
    double lattice_temperature,
    const GPU_Vector<int>* spin_freeze_mask = nullptr) override;
};
