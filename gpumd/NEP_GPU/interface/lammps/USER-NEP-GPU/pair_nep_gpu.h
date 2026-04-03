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

class NepGpuLammpsModel;

namespace LAMMPS_NS {

class PairNEPGPU : public Pair {
public:
  double cutoff;
  double cutoff_radial;
  double cutoff_angular;
  double cutoff_zbl_outer;
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
  bool cache_nl_ = false;          // cache packed NL/NN until LAMMPS rebuilds neighbor list
  bool use_skin_neighbors_ = false; // pack full LAMMPS neighbor list (no cutoff filter)
  bool nl_valid_ = false;
  int cached_natoms_total_ = 0;
  int cached_nlocal_ = 0;
  int cached_mn_r_ = 0;
  int cached_mn_a_ = 0;
  std::string model_filename;
  double cutoffsq;
  double cutoffsq_radial;
  double cutoffsq_angular;
  double cutoffsq_zbl_;
  std::vector<double> rc_radial_by_type_;
  std::vector<double> rc_angular_by_type_;
  void allocate();

  NepGpuLammpsModel *nep_model_lmp;

  // Host-side scratch buffers used to build inputs/outputs for NepGpuModelLmp.
  std::vector<int> type_host;
  std::vector<double> xyz_host;
  std::vector<double> f_host;
  std::vector<double> eatom_host;
  std::vector<double> vatom_host;

  // Neighbor buffers for LMP-direct path (compact stride = natoms_total).
  std::vector<int> nn_radial_host;
  std::vector<int> nl_radial_host;
  std::vector<int> nn_angular_host;
  std::vector<int> nl_angular_host;
};

} // namespace LAMMPS_NS

#endif
#endif
