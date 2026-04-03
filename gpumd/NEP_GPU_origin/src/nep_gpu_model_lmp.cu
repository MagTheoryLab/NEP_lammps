#include "nep_gpu_model_lmp.cuh"
#include "nep_lmp_bridge.cuh"
#include "model/box.cuh"

void nep_gpu_set_device(int dev_id)
{
  CHECK(gpuSetDevice(dev_id));
}

NepGpuModelLmp::NepGpuModelLmp(const char* file_potential, int max_atoms)
{
  if (max_atoms <= 0) max_atoms = 1;
  cap_atoms_ = max_atoms;
  // Single GPU by construction; partition dir not used in bridge.
  nep_ = new NEP(file_potential, max_atoms);
}

NepGpuModelLmp::~NepGpuModelLmp()
{
  delete nep_;
  nep_ = nullptr;
}

void NepGpuModelLmp::set_device(int dev_id)
{
  CHECK(gpuSetDevice(dev_id));
}

float NepGpuModelLmp::cutoff() const { return nep_ ? nep_->get_rc_radial() : 0.0f; }
float NepGpuModelLmp::cutoff_angular() const { return nep_ ? nep_->get_rc_angular() : 0.0f; }
float NepGpuModelLmp::cutoff_zbl_outer() const { return nep_ ? nep_->get_zbl_rc_outer_max() : 0.0f; }
int NepGpuModelLmp::num_types() const { return nep_ ? nep_->get_num_types() : 0; }
int NepGpuModelLmp::mn_radial() const { return nep_ ? nep_->get_MN_radial() : 0; }
int NepGpuModelLmp::mn_angular() const { return nep_ ? nep_->get_MN_angular() : 0; }

void NepGpuModelLmp::compute_with_neighbors(
  const NepGpuSystem& sys,
  int nlocal,
  const NepGpuLmpNeighbors& nb,
  NepGpuResult& res)
{
  if (!nep_) {
    PRINT_INPUT_ERROR("NepGpuModelLmp: internal NEP not initialized.\n");
  }
  const int N = sys.natoms;
  if (N <= 0) {
    res.eng = 0.0;
    for (int k = 0; k < 6; ++k) res.virial[k] = 0.0;
    return;
  }
  if (!sys.type || !sys.xyz || !nb.NN_radial || !nb.NL_radial || !nb.NN_angular || !nb.NL_angular) {
    PRINT_INPUT_ERROR("NepGpuModelLmp: null pointer in inputs.\n");
  }

  // Repack AoS -> SoA for positions (persistent buffer).
  pos_soa_.resize(static_cast<size_t>(N) * 3);
  for (int i = 0; i < N; ++i) {
    pos_soa_[i] = sys.xyz[3 * i + 0];
    pos_soa_[i + N] = sys.xyz[3 * i + 1];
    pos_soa_[i + 2 * N] = sys.xyz[3 * i + 2];
  }

  // If neither per-atom potential nor per-atom virial are requested, assume the
  // caller does not need energy/virial totals either (LAMMPS force-only steps).
  const bool need_energy = (res.eatom != nullptr);
  const bool need_virial = (res.vatom != nullptr);
  const bool need_totals = need_energy || need_virial;
  double* potential_out = res.eatom;
  double* virial_out = res.vatom;

  // Forces are typically required by LAMMPS, but keep a scratch for safety.
  double* force_out = res.f;
  if (!force_out) {
    force_scratch_.resize(static_cast<size_t>(N) * 3);
    force_out = force_scratch_.data();
  }

  Box box;
  for (int i = 0; i < 9; ++i) box.cpu_h[i] = sys.h[i];
  box.pbc_x = sys.pbc_x;
  box.pbc_y = sys.pbc_y;
  box.pbc_z = sys.pbc_z;
  box.get_inverse();

  nep_->compute_with_neighbors(
    box,
    nlocal,
    N,
    sys.type,
    pos_soa_.data(),
    nb.NN_radial,
    nb.NL_radial,
    nb.NN_angular,
    nb.NL_angular,
    need_energy ? potential_out : nullptr,
    force_out,
    need_virial ? virial_out : nullptr);

  if (!need_totals) {
    res.eng = 0.0;
    for (int k = 0; k < 6; ++k) res.virial[k] = 0.0;
    return;
  }

  // Reduce totals over local atoms
  double eng = 0.0;
  double vxx = 0.0, vyy = 0.0, vzz = 0.0, vxy = 0.0, vxz = 0.0, vyz = 0.0;
  for (int i = 0; i < nlocal; ++i) {
    if (need_energy) eng += potential_out[i];
    if (need_virial) {
      vxx += virial_out[9 * i + 0];
      vyy += virial_out[9 * i + 1];
      vzz += virial_out[9 * i + 2];
      vxy += virial_out[9 * i + 3];
      vxz += virial_out[9 * i + 4];
      vyz += virial_out[9 * i + 5];
    }
  }
  res.eng = need_energy ? eng : 0.0;
  res.virial[0] = need_virial ? vxx : 0.0;
  res.virial[1] = need_virial ? vyy : 0.0;
  res.virial[2] = need_virial ? vzz : 0.0;
  res.virial[3] = need_virial ? vxy : 0.0;
  res.virial[4] = need_virial ? vxz : 0.0;
  res.virial[5] = need_virial ? vyz : 0.0;
}

void NepGpuModelLmp::compute_with_neighbors_device(
  const NepGpuDeviceSystem& sys,
  int nlocal,
  const NepGpuLmpNeighborsDevice& nb,
  NepGpuDeviceResult& res,
  bool need_energy,
  bool need_virial)
{
  if (!nep_) {
    PRINT_INPUT_ERROR("NepGpuModelLmp: internal NEP not initialized.\n");
  }
  const int N = sys.natoms;
  if (N <= 0 || nlocal <= 0) {
    res.eng = 0.0;
    for (int k = 0; k < 6; ++k) res.virial[k] = 0.0;
    return;
  }
  if (!sys.type || !sys.xyz || !nb.NN_radial || !nb.NL_radial || !nb.NN_angular || !nb.NL_angular) {
    PRINT_INPUT_ERROR("NepGpuModelLmp: null pointer in device inputs.\n");
  }

  Box box;
  for (int i = 0; i < 9; ++i) box.cpu_h[i] = sys.h[i];
  box.pbc_x = sys.pbc_x;
  box.pbc_y = sys.pbc_y;
  box.pbc_z = sys.pbc_z;
  box.get_inverse();

  double eng = 0.0;
  double vir6[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  nep_->compute_with_neighbors_device(
    box,
    nlocal,
    N,
    sys.type,
    sys.xyz,
    sys.owner,
    sys.stream ? reinterpret_cast<gpuStream_t>(sys.stream) : (gpuStream_t)0,
    nb.NN_radial,
    nb.NL_radial,
    nb.NN_angular,
    nb.NL_angular,
    res.f,
    res.eatom,
    res.vatom,
    need_energy,
    need_virial,
    eng,
    vir6);

  res.eng = eng;
  for (int k = 0; k < 6; ++k) res.virial[k] = vir6[k];
}
