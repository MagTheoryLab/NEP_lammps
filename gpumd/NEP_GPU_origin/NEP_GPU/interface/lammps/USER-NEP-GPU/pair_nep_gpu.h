/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   GPU NEP pair style (experimental).
   This implements pair_style nep/gpu which offloads NEP evaluation to a
   separate GPU library based on GPUMD's NEP implementation.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nep/gpu, PairNEPGPU)

#else

#ifndef LMP_PAIR_NEP_GPU_H
#define LMP_PAIR_NEP_GPU_H

#include "pair.h"
#include <string>
#include <vector>

// Forward declaration; the implementation is provided by the external
// NEP_GPU library (see NEP_GPU/src/nep_gpu_model.cuh in the GPUMD tree).
class NepGpuModel;

namespace LAMMPS_NS {

class PairNEPGPU : public Pair {
public:
  double cutoff;
  int *type_map;

  PairNEPGPU(class LAMMPS *);
  ~PairNEPGPU() override;

  void coeff(int, char **) override;
  void settings(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;
  void compute(int, int) override;

protected:
  bool inited;
  std::string model_filename;
  double cutoffsq;
  void allocate();

  NepGpuModel *nep_model;

  // Host-side scratch buffers used to build inputs/outputs for NepGpuModel.
  std::vector<int> type_host;
  std::vector<double> xyz_host;
  std::vector<double> f_host;
  std::vector<double> eatom_host;
  std::vector<double> vatom_host;
};

} // namespace LAMMPS_NS

#endif
#endif

