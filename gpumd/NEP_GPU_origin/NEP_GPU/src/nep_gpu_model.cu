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

/*----------------------------------------------------------------------------80
Thin NEP-GPU wrapper intended for external callers (e.g. a LAMMPS pair style).
It reuses the production NEP implementation in src/force/nep.cu and only
handles host<->device copies and simple reductions.
------------------------------------------------------------------------------*/

#include "nep_gpu_model.cuh"

#include "force/nep.cuh"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

#include <cstdio>

struct NepGpuModel::Impl
{
  NEP* nep = nullptr;
  int max_atoms = 0;
  int current_natoms = 0;

  Box box;
  GPU_Vector<int> type_gpu;
  GPU_Vector<double> pos_gpu;    // [x(0..N-1), y(0..N-1), z(0..N-1)]
  GPU_Vector<double> pe_gpu;     // per-atom potential
  GPU_Vector<double> force_gpu;  // [Fx(0..N-1), Fy(0..N-1), Fz(0..N-1)]
  GPU_Vector<double> virial_gpu; // 9 components per atom

  std::vector<double> pos_host;     // host buffer for position (GPUMD layout)
  std::vector<double> pe_host;      // host buffer for per-atom potential
  std::vector<double> force_host;   // host buffer for per-atom force (GPUMD layout)
  std::vector<double> virial_host;  // host buffer for per-atom virial (9 components)
};

NepGpuModel::NepGpuModel(const char* file_potential, int max_atoms)
{
  if (max_atoms <= 0) {
    PRINT_INPUT_ERROR("NepGpuModel requires max_atoms > 0.\n");
  }

  impl_ = new Impl;
  impl_->max_atoms = max_atoms;
  impl_->nep = new NEP(file_potential, impl_->max_atoms);

  // By default, use all atoms in [0,max_atoms) as the compute window.
  impl_->nep->N1 = 0;
  impl_->nep->N2 = impl_->max_atoms;

  // Allocate host buffers to the maximum size; GPU buffers are resized on demand.
  impl_->pos_host.resize(static_cast<size_t>(impl_->max_atoms) * 3);
  impl_->pe_host.resize(static_cast<size_t>(impl_->max_atoms));
  impl_->force_host.resize(static_cast<size_t>(impl_->max_atoms) * 3);
  impl_->virial_host.resize(static_cast<size_t>(impl_->max_atoms) * 9);
}

NepGpuModel::~NepGpuModel()
{
  if (impl_) {
    delete impl_->nep;
    impl_->nep = nullptr;
    delete impl_;
    impl_ = nullptr;
  }
}

void NepGpuModel::set_device(int dev_id)
{
  CHECK(gpuSetDevice(dev_id));
}

float NepGpuModel::cutoff() const
{
  return (impl_ && impl_->nep) ? impl_->nep->get_rc_radial() : 0.0f;
}

int NepGpuModel::num_types() const
{
  return (impl_ && impl_->nep) ? impl_->nep->get_num_types() : 0;
}

void NepGpuModel::compute(const NepGpuSystem& sys, NepGpuResult& res)
{
  if (!impl_ || !impl_->nep) {
    PRINT_INPUT_ERROR("NepGpuModel: internal NEP pointer is null.\n");
  }

  const int N = sys.natoms;
  if (N <= 0 || sys.type == nullptr || sys.xyz == nullptr) {
    res.eng = 0.0;
    for (int k = 0; k < 6; ++k) {
      res.virial[k] = 0.0;
    }
    return;
  }

  if (N > impl_->max_atoms) {
    PRINT_INPUT_ERROR("NepGpuModel: natoms exceeds max_atoms specified at construction.\n");
  }

  if (N != impl_->current_natoms) {
    impl_->current_natoms = N;

    impl_->type_gpu.resize(N);
    impl_->pos_gpu.resize(static_cast<size_t>(N) * 3);
    impl_->pe_gpu.resize(N);
    impl_->force_gpu.resize(static_cast<size_t>(N) * 3);
    impl_->virial_gpu.resize(static_cast<size_t>(N) * 9);

    impl_->pos_host.resize(static_cast<size_t>(N) * 3);
    impl_->pe_host.resize(N);
    impl_->force_host.resize(static_cast<size_t>(N) * 3);
    impl_->virial_host.resize(static_cast<size_t>(N) * 9);
  }

  // Set up box (h matrix) and its inverse.
  for (int i = 0; i < 9; ++i) {
    impl_->box.cpu_h[i] = sys.h[i];
  }
  impl_->box.pbc_x = sys.pbc_x;
  impl_->box.pbc_y = sys.pbc_y;
  impl_->box.pbc_z = sys.pbc_z;
  impl_->box.get_inverse();

  // Copy types to GPU; types must already be mapped to NEP's internal ordering.
  impl_->type_gpu.copy_from_host(sys.type);

  // Re-pack positions from [x0,y0,z0,...] to [x(0..N-1), y(0..N-1), z(0..N-1)].
  for (int i = 0; i < N; ++i) {
    const double x = sys.xyz[3 * i + 0];
    const double y = sys.xyz[3 * i + 1];
    const double z = sys.xyz[3 * i + 2];
    impl_->pos_host[i] = x;
    impl_->pos_host[i + N] = y;
    impl_->pos_host[i + 2 * N] = z;
  }
  impl_->pos_gpu.copy_from_host(impl_->pos_host.data());

  // Initialize output buffers on GPU to zero; NEP kernels accumulate into them.
  impl_->pe_gpu.fill(0.0);
  impl_->force_gpu.fill(0.0);
  impl_->virial_gpu.fill(0.0);

  // Restrict NEP's compute window to the current number of atoms.
  impl_->nep->N1 = 0;
  impl_->nep->N2 = N;

  impl_->nep->compute(
    impl_->box,
    impl_->type_gpu,
    impl_->pos_gpu,
    impl_->pe_gpu,
    impl_->force_gpu,
    impl_->virial_gpu);

  // Bring back per-atom properties to host.
  impl_->pe_gpu.copy_to_host(impl_->pe_host.data());
  impl_->force_gpu.copy_to_host(impl_->force_host.data());
  impl_->virial_gpu.copy_to_host(impl_->virial_host.data());

  // Total energy.
  double total_eng = 0.0;
  for (int i = 0; i < N; ++i) {
    total_eng += impl_->pe_host[i];
  }
  res.eng = total_eng;

  // Total virial (symmetric) and optional per-atom virial.
  double vxx = 0.0, vyy = 0.0, vzz = 0.0;
  double vxy = 0.0, vyz = 0.0, vzx = 0.0;

  for (int i = 0; i < N; ++i) {
    const double sxx = impl_->virial_host[i + 0 * N];
    const double syy = impl_->virial_host[i + 1 * N];
    const double szz = impl_->virial_host[i + 2 * N];
    const double sxy = impl_->virial_host[i + 3 * N];
    const double sxz = impl_->virial_host[i + 4 * N];
    const double syz = impl_->virial_host[i + 5 * N];
    const double syx = impl_->virial_host[i + 6 * N];
    const double szx = impl_->virial_host[i + 7 * N];
    const double szy = impl_->virial_host[i + 8 * N];

    const double vxy_i = 0.5 * (sxy + syx);
    const double vyz_i = 0.5 * (syz + szy);
    const double vzx_i = 0.5 * (sxz + szx);

    vxx += sxx;
    vyy += syy;
    vzz += szz;
    vxy += sxy; // xy
    vxz += sxz; // xz
    vyz += syz; // yz


    if (res.vatom) {
      const int idx = 6 * i;
      // LAMMPS per-atom virial order: xx, yy, zz, xy, xz, yz
      res.vatom[idx + 0] = sxx;
      res.vatom[idx + 1] = syy;
      res.vatom[idx + 2] = szz;
      res.vatom[idx + 3] = sxy; // xy
      res.vatom[idx + 4] = sxz; // xz
      res.vatom[idx + 5] = syz; // yz
    }
  }

  // Match the NEP_CPU / LAMMPS convention for total virial.
  // LAMMPS order is: xx, yy, zz, xy, xz, yz.
  res.virial[0] = vxx;
  res.virial[1] = vyy;
  res.virial[2] = vzz;
  res.virial[3] = vxy; // xy
  res.virial[4] = vxz; // xz
  res.virial[5] = vyz; // yz

  // Optional per-atom energy.
  if (res.eatom) {
    for (int i = 0; i < N; ++i) {
      res.eatom[i] = impl_->pe_host[i];
    }
  }

  // Optional per-atom force, packed as [Fx,Fy,Fz] per atom.
  if (res.f) {
    for (int i = 0; i < N; ++i) {
      res.f[3 * i + 0] = impl_->force_host[i];
      res.f[3 * i + 1] = impl_->force_host[i + N];
      res.f[3 * i + 2] = impl_->force_host[i + 2 * N];
    }
  }
}
