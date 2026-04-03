/*
    Frozen-lattice ensemble integrator (no atomic motion).
*/

#pragma once

#include "ensemble.cuh"

class Ensemble_Frozen : public Ensemble
{
public:
  explicit Ensemble_Frozen(int t) { type = t; }
  ~Ensemble_Frozen(void) override = default;

  void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;
};

