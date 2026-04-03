/*
    Driver for spin-only integrators (ensemble_spin keyword).
*/

#pragma once

#include "ensemble_spin.cuh"
#include "ensemble_spin_glsd.cuh"
#include "model/atom.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>

class SpinIntegrate
{
public:
  SpinIntegrate() = default;

  void initialize(Atom& atom, int& total_steps);
  void finalize();

  void parse_ensemble_spin(const char** param, int num_param);

  void compute(
    const double time_step,
    const double step_over_number_of_steps,
    Atom& atom,
    GPU_Vector<double>& mforce,
    double lattice_temperature,
    const GPU_Vector<int>* spin_freeze_mask = nullptr);

  int type = 0; // 0: off, 1: glsd, 2: llg, ...
  double T1 = 0.0;
  double T2 = 0.0;
  bool use_lattice_T = true;

private:
  std::unique_ptr<Spin_Ensemble> spin_ensemble;
  int seed_ = 0;
};
