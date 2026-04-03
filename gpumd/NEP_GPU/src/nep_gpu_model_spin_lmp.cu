#include "nep_gpu_model_spin_lmp.cuh"
#include "nep_spin_lmp_bridge.cuh"
#include "model/box.cuh"

void nep_spin_gpu_set_device(int dev_id)
{
  CHECK(gpuSetDevice(dev_id));
}

NepGpuModelSpinLmp::NepGpuModelSpinLmp(const char* file_potential, int max_atoms)
{
  if (max_atoms <= 0) max_atoms = 1;
  cap_atoms_ = max_atoms;
  nep_ = new NEP_Spin_LMP(file_potential, max_atoms);
}

NepGpuModelSpinLmp::~NepGpuModelSpinLmp()
{
  if (d_type_) CHECK(gpuFree(d_type_));
  if (d_xyz_) CHECK(gpuFree(d_xyz_));
  if (d_sp4_) CHECK(gpuFree(d_sp4_));
  if (d_f_) CHECK(gpuFree(d_f_));
  if (d_fm_) CHECK(gpuFree(d_fm_));
  if (d_fm_left_iface_) CHECK(gpuFree(d_fm_left_iface_));
  if (d_nn_radial_) CHECK(gpuFree(d_nn_radial_));
  if (d_nn_angular_) CHECK(gpuFree(d_nn_angular_));
  if (d_nl_radial_) CHECK(gpuFree(d_nl_radial_));
  if (d_nl_angular_) CHECK(gpuFree(d_nl_angular_));
  if (d_eatom_) CHECK(gpuFree(d_eatom_));
  if (d_vatom_) CHECK(gpuFree(d_vatom_));

  d_type_ = nullptr;
  d_xyz_ = nullptr;
  d_sp4_ = nullptr;
  d_f_ = nullptr;
  d_fm_ = nullptr;
  d_fm_left_iface_ = nullptr;
  d_nn_radial_ = nullptr;
  d_nn_angular_ = nullptr;
  d_nl_radial_ = nullptr;
  d_nl_angular_ = nullptr;
  d_eatom_ = nullptr;
  d_vatom_ = nullptr;
  staged_natoms_ = 0;
  staged_nlocal_ = 0;
  staged_mn_radial_ = 0;
  staged_mn_angular_ = 0;
  staged_have_eatom_ = false;
  staged_have_vatom_ = false;

  delete nep_;
  nep_ = nullptr;
}

void NepGpuModelSpinLmp::set_device(int dev_id)
{
  CHECK(gpuSetDevice(dev_id));
}

float NepGpuModelSpinLmp::cutoff() const { return nep_ ? nep_->get_rc_radial() : 0.0f; }
float NepGpuModelSpinLmp::cutoff_angular() const { return nep_ ? nep_->get_rc_angular() : 0.0f; }
int NepGpuModelSpinLmp::num_types() const { return nep_ ? nep_->get_num_types() : 0; }
int NepGpuModelSpinLmp::mn_radial() const { return nep_ ? nep_->get_MN_radial() : 0; }
int NepGpuModelSpinLmp::mn_angular() const { return nep_ ? nep_->get_MN_angular() : 0; }

void NepGpuModelSpinLmp::ensure_staging(int natoms, int nlocal, int mn_radial, int mn_angular, bool need_eatom, bool need_vatom)
{
  if (natoms <= 0) natoms = 1;
  if (nlocal < 0) nlocal = 0;
  if (mn_radial < 0) mn_radial = 0;
  if (mn_angular < 0) mn_angular = 0;

  const bool need_realloc_atoms = (staged_natoms_ != natoms) || (!d_type_) || (!d_xyz_) || (!d_sp4_) || (!d_f_) || (!d_fm_) || (!d_fm_left_iface_);
  if (need_realloc_atoms) {
    if (d_type_) CHECK(gpuFree(d_type_));
    if (d_xyz_) CHECK(gpuFree(d_xyz_));
    if (d_sp4_) CHECK(gpuFree(d_sp4_));
    if (d_f_) CHECK(gpuFree(d_f_));
    if (d_fm_) CHECK(gpuFree(d_fm_));
    if (d_fm_left_iface_) CHECK(gpuFree(d_fm_left_iface_));
    if (d_nn_radial_) CHECK(gpuFree(d_nn_radial_));
    if (d_nn_angular_) CHECK(gpuFree(d_nn_angular_));
    if (d_nl_radial_) CHECK(gpuFree(d_nl_radial_));
    if (d_nl_angular_) CHECK(gpuFree(d_nl_angular_));
    d_nl_radial_ = nullptr;
    d_nl_angular_ = nullptr;
    staged_mn_radial_ = 0;
    staged_mn_angular_ = 0;

    CHECK(gpuMalloc((void**)&d_type_, sizeof(int) * natoms));
    CHECK(gpuMalloc((void**)&d_xyz_, sizeof(double) * 3 * natoms));
    CHECK(gpuMalloc((void**)&d_sp4_, sizeof(double) * 4 * natoms));
    CHECK(gpuMalloc((void**)&d_f_, sizeof(double) * 3 * natoms));
    CHECK(gpuMalloc((void**)&d_fm_, sizeof(double) * 3 * natoms));
    CHECK(gpuMalloc((void**)&d_fm_left_iface_, sizeof(double) * 3 * natoms));
    CHECK(gpuMalloc((void**)&d_nn_radial_, sizeof(int) * natoms));
    CHECK(gpuMalloc((void**)&d_nn_angular_, sizeof(int) * natoms));

    staged_natoms_ = natoms;
  }

  const bool need_realloc_mn = (staged_mn_radial_ != mn_radial) || (staged_mn_angular_ != mn_angular) ||
    (!d_nl_radial_) || (!d_nl_angular_);
  if (need_realloc_mn) {
    if (d_nl_radial_) CHECK(gpuFree(d_nl_radial_));
    if (d_nl_angular_) CHECK(gpuFree(d_nl_angular_));
    const size_t nr = static_cast<size_t>(natoms) * static_cast<size_t>(mn_radial > 0 ? mn_radial : 1);
    const size_t na = static_cast<size_t>(natoms) * static_cast<size_t>(mn_angular > 0 ? mn_angular : 1);
    CHECK(gpuMalloc((void**)&d_nl_radial_, sizeof(int) * nr));
    CHECK(gpuMalloc((void**)&d_nl_angular_, sizeof(int) * na));
    staged_mn_radial_ = mn_radial;
    staged_mn_angular_ = mn_angular;
  }

  const bool need_realloc_locals = (staged_nlocal_ != nlocal);
  if (need_realloc_locals) {
    // If sizes change, discard optional buffers and reallocate on demand below.
    if (d_eatom_) { CHECK(gpuFree(d_eatom_)); d_eatom_ = nullptr; }
    if (d_vatom_) { CHECK(gpuFree(d_vatom_)); d_vatom_ = nullptr; }
    staged_have_eatom_ = false;
    staged_have_vatom_ = false;
    staged_nlocal_ = nlocal;
  }

  if (need_eatom && !staged_have_eatom_) {
    if (d_eatom_) CHECK(gpuFree(d_eatom_));
    CHECK(gpuMalloc((void**)&d_eatom_, sizeof(double) * (nlocal > 0 ? nlocal : 1)));
    staged_have_eatom_ = true;
  }

  if (need_vatom && !staged_have_vatom_) {
    if (d_vatom_) CHECK(gpuFree(d_vatom_));
    CHECK(gpuMalloc((void**)&d_vatom_, sizeof(double) * 9 * (nlocal > 0 ? nlocal : 1)));
    staged_have_vatom_ = true;
  }
}

void NepGpuModelSpinLmp::compute_with_neighbors(
  const NepGpuSpinSystem& sys,
  int nlocal,
  const NepGpuSpinLmpNeighbors& nb,
  NepGpuSpinResult& res,
  bool need_energy,
  bool need_virial)
{
  if (!nep_) {
    PRINT_INPUT_ERROR("NepGpuModelSpinLmp: internal NEP_Spin_LMP not initialized.\n");
  }

  const int N = sys.natoms;
  if (N <= 0 || nlocal <= 0) {
    res.eng = 0.0;
    for (int k = 0; k < 6; ++k) res.virial[k] = 0.0;
    if (res.want_virial_raw9) {
      for (int k = 0; k < 9; ++k) res.virial_raw9[k] = 0.0;
    }
    return;
  }

  if (!sys.type || !sys.xyz || !sys.sp4 ||
      !nb.NN_radial || !nb.NL_radial || !nb.NN_angular || !nb.NL_angular) {
    PRINT_INPUT_ERROR("NepGpuModelSpinLmp: null pointer in host inputs.\n");
  }
  if (!res.f || !res.fm) {
    PRINT_INPUT_ERROR("NepGpuModelSpinLmp: null pointer in host outputs (f/fm).\n");
  }

  const bool want_eatom = (res.eatom != nullptr);
  const bool want_vatom = (res.vatom != nullptr);
  const int mn_r = mn_radial();
  const int mn_a = mn_angular();
  ensure_staging(N, nlocal, mn_r, mn_a, want_eatom, want_vatom);

  CHECK(gpuMemcpy(d_type_, sys.type, sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_xyz_, sys.xyz, sizeof(double) * 3 * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_sp4_, sys.sp4, sizeof(double) * 4 * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nn_radial_, nb.NN_radial, sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nn_angular_, nb.NN_angular, sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nl_radial_, nb.NL_radial, sizeof(int) * static_cast<size_t>(N) * static_cast<size_t>(mn_r), gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nl_angular_, nb.NL_angular, sizeof(int) * static_cast<size_t>(N) * static_cast<size_t>(mn_a), gpuMemcpyHostToDevice));

  CHECK(gpuMemset(d_f_, 0, sizeof(double) * 3 * N));
  CHECK(gpuMemset(d_fm_, 0, sizeof(double) * 3 * N));
  CHECK(gpuMemset(d_fm_left_iface_, 0, sizeof(double) * 3 * N));
  if (want_eatom) CHECK(gpuMemset(d_eatom_, 0, sizeof(double) * nlocal));
  if (want_vatom) CHECK(gpuMemset(d_vatom_, 0, sizeof(double) * 9 * nlocal));

  NepGpuSpinDeviceSystem dsys;
  dsys.natoms = N;
  dsys.type = d_type_;
  dsys.xyz = d_xyz_;
  dsys.sp4 = d_sp4_;
  dsys.stream = nullptr;
  for (int i = 0; i < 9; ++i) dsys.h[i] = sys.h[i];
  dsys.pbc_x = sys.pbc_x;
  dsys.pbc_y = sys.pbc_y;
  dsys.pbc_z = sys.pbc_z;

  NepGpuSpinLmpNeighborsDevice dnb;
  dnb.NN_radial = d_nn_radial_;
  dnb.NL_radial = d_nl_radial_;
  dnb.NN_angular = d_nn_angular_;
  dnb.NL_angular = d_nl_angular_;

  NepGpuSpinDeviceResult dres;
  dres.f = d_f_;
  dres.fm = d_fm_;
  dres.fm_left_iface = res.fm_left_iface ? d_fm_left_iface_ : nullptr;
  dres.iface_x = res.iface_x;
  dres.iface_half_width = res.iface_half_width;
  dres.inv_hbar = res.inv_hbar;
  dres.want_virial_raw9 = res.want_virial_raw9;
  dres.eatom = want_eatom ? d_eatom_ : nullptr;
  dres.vatom = want_vatom ? d_vatom_ : nullptr;

  compute_with_neighbors_device(dsys, nlocal, dnb, dres, need_energy, need_virial);

  CHECK(gpuMemcpy(res.f, d_f_, sizeof(double) * 3 * N, gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(res.fm, d_fm_, sizeof(double) * 3 * N, gpuMemcpyDeviceToHost));
  if (res.fm_left_iface) {
    CHECK(gpuMemcpy(res.fm_left_iface, d_fm_left_iface_, sizeof(double) * 3 * N, gpuMemcpyDeviceToHost));
  }
  if (want_eatom) CHECK(gpuMemcpy(res.eatom, d_eatom_, sizeof(double) * nlocal, gpuMemcpyDeviceToHost));
  if (want_vatom) CHECK(gpuMemcpy(res.vatom, d_vatom_, sizeof(double) * 9 * nlocal, gpuMemcpyDeviceToHost));

  res.eng = dres.eng;
  for (int k = 0; k < 6; ++k) res.virial[k] = dres.virial[k];
  if (res.want_virial_raw9) {
    for (int k = 0; k < 9; ++k) res.virial_raw9[k] = dres.virial_raw9[k];
  }
}

void NepGpuModelSpinLmp::compute_with_neighbors_device(
  const NepGpuSpinDeviceSystem& sys,
  int nlocal,
  const NepGpuSpinLmpNeighborsDevice& nb,
  NepGpuSpinDeviceResult& res,
  bool need_energy,
  bool need_virial)
{
  if (!nep_) {
    PRINT_INPUT_ERROR("NepGpuModelSpinLmp: internal NEP_Spin_LMP not initialized.\n");
  }
  const int N = sys.natoms;
  if (N <= 0 || nlocal <= 0) {
    res.eng = 0.0;
    for (int k = 0; k < 6; ++k) res.virial[k] = 0.0;
    return;
  }
  if (!sys.type || !sys.xyz || !sys.sp4 ||
      !nb.NN_radial || !nb.NL_radial || !nb.NN_angular || !nb.NL_angular) {
    PRINT_INPUT_ERROR("NepGpuModelSpinLmp: null pointer in device inputs.\n");
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
    sys.sp4,
    sys.stream,
    nb.NN_radial,
    nb.NL_radial,
    nb.NN_angular,
    nb.NL_angular,
    res.f,
    res.fm,
    res.fm_left_iface,
    res.iface_x,
    res.iface_half_width,
    res.inv_hbar,
    res.eatom,
    res.vatom,
    need_energy,
    need_virial,
    eng,
    vir6,
    res.want_virial_raw9 ? res.virial_raw9 : nullptr);

  res.eng = eng;
  for (int k = 0; k < 6; ++k) res.virial[k] = vir6[k];
}
