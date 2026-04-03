/*
    Spin ensemble base class for spin-only integrators.
    Manages temperature scheduling parameters shared by all spin integrators.
*/

#pragma once

#include "model/atom.cuh"
#include "utilities/gpu_vector.cuh"

class Spin_Ensemble
{
public:
  Spin_Ensemble() = default;
  virtual ~Spin_Ensemble() = default;

  // Optional per-run initialization (e.g., RNG setup); default is no-op.
  virtual void initialize(int, int) {}

  // time_step: current MD time step (natural units)
  // step_over_number_of_steps: normalized step index within the current run segment
  // lattice_temperature: current lattice temperature passed from the lattice ensemble
  virtual void compute(
    const double time_step,
    const double step_over_number_of_steps,
    Atom& atom,
    GPU_Vector<double>& mforce,
    double lattice_temperature,
    const GPU_Vector<int>* spin_freeze_mask = nullptr) = 0;

  int type = 0;            // 0: off, 1: glsd, ...
  double T1 = 0.0;         // spin temperature schedule start
  double T2 = 0.0;         // spin temperature schedule end
  bool use_lattice_T = true;
};
