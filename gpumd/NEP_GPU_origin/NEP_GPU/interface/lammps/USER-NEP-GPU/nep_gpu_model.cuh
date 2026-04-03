/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <vector>

// Lightweight system description for external NEP-GPU callers (e.g. LAMMPS).
// Types must be 0-based and consistent with the ordering in the NEP potential file.
// Positions are stored as [x0,y0,z0,x1,y1,z1,...] on the host side and will be
// internally converted to the GPUMD layout [x(0..N-1), y(0..N-1), z(0..N-1)].
struct NepGpuSystem
{
  int natoms = 0;
  const int* type = nullptr;     // length natoms, 0-based NEP types
  const double* xyz = nullptr;   // length 3*natoms, [x0,y0,z0,x1,y1,z1,...]

  // Box matrix h (row-major): h[d1*3 + d2] as in Box::cpu_h[0..8].
  double h[9] = {0.0};
  int pbc_x = 1;
  int pbc_y = 1;
  int pbc_z = 1;
};

// Result container for NEP-GPU evaluation.
// Forces and virials follow the same layout as xyz:
//   f[3*i+0/1/2] = Fx/Fy/Fz of atom i.
// Per-atom virial is stored as 6 components per atom: [xx,yy,zz,xy,yz,zx].
struct NepGpuResult
{
  double eng = 0.0;       // total potential energy (this rank)
  double virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // xx,yy,zz,xy,yz,zx

  double* f = nullptr;     // length 3*natoms, optional
  double* eatom = nullptr; // length natoms, optional
  double* vatom = nullptr; // length 6*natoms, optional
};

// Thin wrapper around src/force/nep.cu for external GPU NEP evaluation.
// It reuses the production MD implementation (neighbor list, descriptors, ANN, ZBL)
// and only handles host<->device data movement and simple reductions.
//
// IMPORTANT:
//  - This header is intentionally kept free of CUDA-specific types and
//    GPUMD's GPU headers so that it can be included by plain C++ compilers
//    (e.g. LAMMPS built with mpicxx).
//  - All GPU resources and GPUMD-specific types are hidden behind a
//    pimpl (opaque implementation) pointer and are only visible in the
//    NEP_GPU/src/nep_gpu_model.cu translation unit compiled by nvcc.
class NepGpuModel
{
public:
  NepGpuModel(const char* file_potential, int max_atoms);
  ~NepGpuModel();

  // Select GPU device (wrapper over gpuSetDevice).
  void set_device(int dev_id);

  // Radial cutoff from the underlying NEP model.
  float cutoff() const;

  // Number of atom types supported by the NEP model.
  int num_types() const;

  // Main entry: compute energy, forces, and virial for the given system.
  // The system types must already be mapped to NEP's internal type indices.
  void compute(const NepGpuSystem& sys, NepGpuResult& res);

private:
  struct Impl;
  Impl* impl_;
};

