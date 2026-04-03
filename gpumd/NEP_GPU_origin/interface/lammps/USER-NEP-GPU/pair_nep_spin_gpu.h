/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This file is part of a user-contributed package (USER-NEP-GPU) and is
   distributed under the same GPL terms as LAMMPS.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   GPU NEP spin pair style (classic, non-Kokkos).
   Implements pair_style nep/spin/gpu using the GPUMD-based CUDA backend.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(nep/spin/gpu,PairNEPSpinGPU);
// clang-format on
#else

#ifndef LMP_PAIR_NEP_SPIN_GPU_H
#define LMP_PAIR_NEP_SPIN_GPU_H

#include "pair.h"

#include <string>
#include <vector>

class NepGpuLammpsModel;

namespace LAMMPS_NS {

class PairNEPSpinGPU : public Pair {
 public:
  PairNEPSpinGPU(class LAMMPS *);
  ~PairNEPSpinGPU() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
 void init_style() override;

 private:
  // The backend returns mforce = -dE/dM in eV/mu_B and we expose that directly.
  double cutoff{0.0};
  double cutoffsq_r_{0.0};
  double cutoffsq_a_{0.0};

  int* type_map{nullptr}; // LAMMPS type -> NEP type
  NepGpuLammpsModel* nep_model_spin_lmp{nullptr};
  std::string model_filename_;

  std::vector<int> type_host_;
  std::vector<double> xyz_host_;
  std::vector<double> sp4_host_;
  std::vector<double> f_host_;
  std::vector<double> fm_host_;
  std::vector<double> eatom_host_;
  std::vector<double> vatom_host_;

  // Neighbor lists: compact stride = nlocal, index = i + nlocal*slot
  std::vector<int> nn_radial_;
  std::vector<int> nn_angular_;
  std::vector<int> nl_radial_;
  std::vector<int> nl_angular_;

  void allocate();
};

} // namespace LAMMPS_NS

#endif
#endif
