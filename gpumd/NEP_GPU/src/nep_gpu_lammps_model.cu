#include "nep_gpu_lammps_model.h"

#include "model/box.cuh"
#include "nep_lmp_bridge.cuh"
#include "nep_spin_lmp_bridge.cuh"

#include <fstream>
#include <cctype>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<std::string> read_tokens_line(std::ifstream& input)
{
  return get_tokens(input);
}

bool starts_with(const std::string& value, const char* prefix)
{
  return value.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& value, const char* suffix)
{
  const std::size_t suffix_len = std::char_traits<char>::length(suffix);
  return value.size() >= suffix_len &&
         value.compare(value.size() - suffix_len, suffix_len, suffix) == 0;
}

NepGpuModelKind parse_kind_from_tag(const std::string& tag, bool& has_zbl, bool& needs_spin)
{
  has_zbl = (tag.find("_zbl") != std::string::npos);
  needs_spin = ends_with(tag, "_spin");
  if (needs_spin) return NepGpuModelKind::spin;
  if (tag.find("_dipole") != std::string::npos) return NepGpuModelKind::dipole;
  if (tag.find("_polarizability") != std::string::npos) return NepGpuModelKind::polarizability;
  if (tag.find("_temperature") != std::string::npos) return NepGpuModelKind::temperature;
  if (tag.find("_charge") != std::string::npos) {
    throw std::runtime_error("NEP_GPU: charge models are recognized but not implemented by this GPU backend.");
  }
  return NepGpuModelKind::potential;
}

int parse_version_from_tag(const std::string& tag)
{
  if (!starts_with(tag, "nep")) {
    throw std::runtime_error("NEP_GPU: invalid potential tag in model header.");
  }
  std::size_t pos = 3;
  std::size_t end = pos;
  while (end < tag.size() && std::isdigit(static_cast<unsigned char>(tag[end]))) ++end;
  if (end == pos) {
    throw std::runtime_error("NEP_GPU: failed to parse model version from potential tag.");
  }
  return std::stoi(tag.substr(pos, end - pos));
}

NepGpuModelInfo parse_model_info(const char* file_potential)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    throw std::runtime_error("NEP_GPU: failed to open potential file.");
  }

  NepGpuModelInfo info;

  auto tokens = read_tokens_line(input);
  if (tokens.size() < 3) {
    throw std::runtime_error("NEP_GPU: malformed potential header.");
  }
  info.version = parse_version_from_tag(tokens[0]);
  info.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  info.elements.reserve(info.num_types);
  for (int i = 0; i < info.num_types; ++i) {
    info.elements.push_back(tokens[2 + i]);
  }
  info.kind = parse_kind_from_tag(tokens[0], info.has_zbl, info.needs_spin);

  if (info.needs_spin) {
    tokens = read_tokens_line(input);
    if (tokens.size() != 3 || tokens[0] != "spin_mode") {
      throw std::runtime_error("NEP_GPU: spin v2 potential must contain 'spin_mode <mode> 3' on the second line.");
    }
    if (get_int_from_token(tokens[2], __FILE__, __LINE__) != 3) {
      throw std::runtime_error("NEP_GPU: only spin v2 headers with 3 spin header lines are supported.");
    }

    tokens = read_tokens_line(input);
    const bool has_single_mref = (tokens.size() == 4);
    const bool has_per_type_mref = (static_cast<int>(tokens.size()) == 3 + info.num_types);
    if ((!has_single_mref && !has_per_type_mref) || tokens[0] != "spin_onsite") {
      throw std::runtime_error("NEP_GPU: spin v2 potential must contain 'spin_onsite'. Old spin_feature headers are not supported.");
    }
    tokens = read_tokens_line(input);
    if (tokens.size() != 3 || tokens[0] != "spin_2body") {
      throw std::runtime_error("NEP_GPU: spin v2 potential must contain 'spin_2body'.");
    }
    tokens = read_tokens_line(input);
    if (tokens.size() != 4 || tokens[0] != "spin_3body") {
      throw std::runtime_error("NEP_GPU: spin v2 potential must contain 'spin_3body'.");
    }
  }

  if (info.has_zbl && !info.needs_spin) {
    tokens = read_tokens_line(input);
    if (tokens.empty() || tokens[0] != "zbl") {
      throw std::runtime_error("NEP_GPU: expected zbl line in potential file.");
    }
    if (tokens.size() >= 3) {
      info.zbl_outer_max = get_double_from_token(tokens[2], __FILE__, __LINE__);
    }
  }

  tokens = read_tokens_line(input);
  if (tokens.empty() || tokens[0] != "cutoff") {
    throw std::runtime_error("NEP_GPU: expected cutoff line in potential file.");
  }

  if (info.needs_spin) {
    if (tokens.size() != 5 && tokens.size() != 8) {
      throw std::runtime_error("NEP_GPU: invalid spin cutoff line.");
    }
    const double rc_r = get_double_from_token(tokens[1], __FILE__, __LINE__);
    const double rc_a = get_double_from_token(tokens[2], __FILE__, __LINE__);
    info.rc_radial_by_type.assign(info.num_types, rc_r);
    info.rc_angular_by_type.assign(info.num_types, rc_a);
    info.rc_radial_max = rc_r;
    info.rc_angular_max = rc_a;
    info.mn_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
    info.mn_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
    return info;
  }

  const int per_type_tokens = 2 * info.num_types + 3;
  const int per_type_with_factors = per_type_tokens + 3;
  if (static_cast<int>(tokens.size()) != 5 &&
      static_cast<int>(tokens.size()) != 8 &&
      static_cast<int>(tokens.size()) != per_type_tokens &&
      static_cast<int>(tokens.size()) != per_type_with_factors) {
    throw std::runtime_error("NEP_GPU: invalid cutoff line for non-spin model.");
  }

  const bool has_per_type_cutoff =
    (static_cast<int>(tokens.size()) == per_type_tokens || static_cast<int>(tokens.size()) == per_type_with_factors);
  info.rc_radial_by_type.resize(info.num_types);
  info.rc_angular_by_type.resize(info.num_types);
  if (has_per_type_cutoff) {
    for (int i = 0; i < info.num_types; ++i) {
      info.rc_radial_by_type[i] = get_double_from_token(tokens[1 + 2 * i], __FILE__, __LINE__);
      info.rc_angular_by_type[i] = get_double_from_token(tokens[2 + 2 * i], __FILE__, __LINE__);
      if (info.rc_radial_by_type[i] > info.rc_radial_max) info.rc_radial_max = info.rc_radial_by_type[i];
      if (info.rc_angular_by_type[i] > info.rc_angular_max) info.rc_angular_max = info.rc_angular_by_type[i];
    }
    info.mn_radial = get_int_from_token(tokens[1 + 2 * info.num_types], __FILE__, __LINE__);
    info.mn_angular = get_int_from_token(tokens[2 + 2 * info.num_types], __FILE__, __LINE__);
  } else {
    const double rc_r = get_double_from_token(tokens[1], __FILE__, __LINE__);
    const double rc_a = get_double_from_token(tokens[2], __FILE__, __LINE__);
    info.rc_radial_by_type.assign(info.num_types, rc_r);
    info.rc_angular_by_type.assign(info.num_types, rc_a);
    info.rc_radial_max = rc_r;
    info.rc_angular_max = rc_a;
    info.mn_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
    info.mn_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
  }
  return info;
}

void fill_box(const double h[9], int pbc_x, int pbc_y, int pbc_z, Box& box)
{
  for (int i = 0; i < 9; ++i) box.cpu_h[i] = h[i];
  box.pbc_x = pbc_x;
  box.pbc_y = pbc_y;
  box.pbc_z = pbc_z;
  box.get_inverse();
}

} // namespace

struct NepGpuLammpsModel::Impl {
  NepGpuModelInfo info;
  NEP* nonspin = nullptr;
  NEP_Spin_LMP* spin = nullptr;

  std::vector<double> pos_soa;
  std::vector<double> force_scratch;
  std::vector<double> virial_scratch;

  int* d_type = nullptr;
  double* d_xyz = nullptr;
  double* d_sp4 = nullptr;
  double* d_f = nullptr;
  double* d_fm = nullptr;
  double* d_fm_left_iface = nullptr;
  int* d_nn_radial = nullptr;
  int* d_nn_angular = nullptr;
  int* d_nl_radial = nullptr;
  int* d_nl_angular = nullptr;
  double* d_eatom = nullptr;
  double* d_vatom = nullptr;
  int staged_natoms = 0;
  int staged_nlocal = 0;
  bool staged_need_fm = false;

  ~Impl()
  {
    delete nonspin;
    delete spin;
    if (d_type) CHECK(gpuFree(d_type));
    if (d_xyz) CHECK(gpuFree(d_xyz));
    if (d_sp4) CHECK(gpuFree(d_sp4));
    if (d_f) CHECK(gpuFree(d_f));
    if (d_fm) CHECK(gpuFree(d_fm));
    if (d_fm_left_iface) CHECK(gpuFree(d_fm_left_iface));
    if (d_nn_radial) CHECK(gpuFree(d_nn_radial));
    if (d_nn_angular) CHECK(gpuFree(d_nn_angular));
    if (d_nl_radial) CHECK(gpuFree(d_nl_radial));
    if (d_nl_angular) CHECK(gpuFree(d_nl_angular));
    if (d_eatom) CHECK(gpuFree(d_eatom));
    if (d_vatom) CHECK(gpuFree(d_vatom));
  }
};

void nep_gpu_lammps_set_device(int dev_id)
{
  CHECK(gpuSetDevice(dev_id));
}

NepGpuLammpsModel::NepGpuLammpsModel(const char* file_potential, int max_atoms)
{
  info_ = parse_model_info(file_potential);
  impl_ = new Impl();
  impl_->info = info_;
  if (max_atoms <= 0) max_atoms = 1;
  if (info_.needs_spin) {
    impl_->spin = new NEP_Spin_LMP(file_potential, max_atoms);
  } else {
    impl_->nonspin = new NEP(file_potential, max_atoms);
  }
}

NepGpuLammpsModel::~NepGpuLammpsModel()
{
  delete impl_;
}

void NepGpuLammpsModel::compute_host(
  const NepGpuLammpsSystemHost& sys,
  int nlocal,
  const NepGpuLammpsNeighborsHost& nb,
  NepGpuLammpsResultHost& res,
  bool need_energy,
  bool need_virial)
{
  if (!impl_) {
    PRINT_INPUT_ERROR("NepGpuLammpsModel: model is not initialized.\n");
  }
  if (info_.needs_spin) {
    if (!sys.sp4) {
      PRINT_INPUT_ERROR("NepGpuLammpsModel: spin model requires sp4 host input.\n");
    }
    if (!res.fm) {
      PRINT_INPUT_ERROR("NepGpuLammpsModel: spin model requires fm host output.\n");
    }

    const int natoms = sys.natoms;
    const int mn_r = info_.mn_radial;
    const int mn_a = info_.mn_angular;
    if (impl_->staged_natoms != natoms) {
      if (impl_->d_type) CHECK(gpuFree(impl_->d_type));
      if (impl_->d_xyz) CHECK(gpuFree(impl_->d_xyz));
      if (impl_->d_sp4) CHECK(gpuFree(impl_->d_sp4));
      if (impl_->d_f) CHECK(gpuFree(impl_->d_f));
      if (impl_->d_fm) CHECK(gpuFree(impl_->d_fm));
      if (impl_->d_fm_left_iface) CHECK(gpuFree(impl_->d_fm_left_iface));
      if (impl_->d_nn_radial) CHECK(gpuFree(impl_->d_nn_radial));
      if (impl_->d_nn_angular) CHECK(gpuFree(impl_->d_nn_angular));
      if (impl_->d_nl_radial) CHECK(gpuFree(impl_->d_nl_radial));
      if (impl_->d_nl_angular) CHECK(gpuFree(impl_->d_nl_angular));
      impl_->d_type = nullptr;
      impl_->d_xyz = nullptr;
      impl_->d_sp4 = nullptr;
      impl_->d_f = nullptr;
      impl_->d_fm = nullptr;
      impl_->d_fm_left_iface = nullptr;
      impl_->d_nn_radial = nullptr;
      impl_->d_nn_angular = nullptr;
      impl_->d_nl_radial = nullptr;
      impl_->d_nl_angular = nullptr;
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_type), sizeof(int) * natoms));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_xyz), sizeof(double) * 3 * natoms));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_sp4), sizeof(double) * 4 * natoms));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_f), sizeof(double) * 3 * natoms));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_fm), sizeof(double) * 3 * natoms));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_fm_left_iface), sizeof(double) * 3 * natoms));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_nn_radial), sizeof(int) * (nlocal > 0 ? nlocal : 1)));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_nn_angular), sizeof(int) * (nlocal > 0 ? nlocal : 1)));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_nl_radial), sizeof(int) * static_cast<size_t>(nlocal > 0 ? nlocal : 1) * (mn_r > 0 ? mn_r : 1)));
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_nl_angular), sizeof(int) * static_cast<size_t>(nlocal > 0 ? nlocal : 1) * (mn_a > 0 ? mn_a : 1)));
      impl_->staged_natoms = natoms;
      impl_->staged_nlocal = 0;
    }
    if (impl_->staged_nlocal != nlocal) {
      if (impl_->d_eatom) CHECK(gpuFree(impl_->d_eatom));
      if (impl_->d_vatom) CHECK(gpuFree(impl_->d_vatom));
      impl_->d_eatom = nullptr;
      impl_->d_vatom = nullptr;
      impl_->staged_nlocal = nlocal;
    }
    if (res.eatom && !impl_->d_eatom) {
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_eatom), sizeof(double) * (nlocal > 0 ? nlocal : 1)));
    }
    if (res.vatom && !impl_->d_vatom) {
      CHECK(gpuMalloc(reinterpret_cast<void**>(&impl_->d_vatom), sizeof(double) * 9 * (nlocal > 0 ? nlocal : 1)));
    }

    CHECK(gpuMemcpy(impl_->d_type, sys.type, sizeof(int) * natoms, gpuMemcpyHostToDevice));
    CHECK(gpuMemcpy(impl_->d_xyz, sys.xyz, sizeof(double) * 3 * natoms, gpuMemcpyHostToDevice));
    CHECK(gpuMemcpy(impl_->d_sp4, sys.sp4, sizeof(double) * 4 * natoms, gpuMemcpyHostToDevice));
    CHECK(gpuMemcpy(impl_->d_nn_radial, nb.NN_radial, sizeof(int) * nlocal, gpuMemcpyHostToDevice));
    CHECK(gpuMemcpy(impl_->d_nn_angular, nb.NN_angular, sizeof(int) * nlocal, gpuMemcpyHostToDevice));
    CHECK(gpuMemcpy(impl_->d_nl_radial, nb.NL_radial, sizeof(int) * static_cast<size_t>(nlocal) * mn_r, gpuMemcpyHostToDevice));
    CHECK(gpuMemcpy(impl_->d_nl_angular, nb.NL_angular, sizeof(int) * static_cast<size_t>(nlocal) * mn_a, gpuMemcpyHostToDevice));
    CHECK(gpuMemset(impl_->d_f, 0, sizeof(double) * 3 * natoms));
    CHECK(gpuMemset(impl_->d_fm, 0, sizeof(double) * 3 * natoms));
    CHECK(gpuMemset(impl_->d_fm_left_iface, 0, sizeof(double) * 3 * natoms));
    if (res.eatom) CHECK(gpuMemset(impl_->d_eatom, 0, sizeof(double) * nlocal));
    if (res.vatom) CHECK(gpuMemset(impl_->d_vatom, 0, sizeof(double) * 9 * nlocal));

    NepGpuLammpsSystemDevice dsys;
    dsys.natoms = natoms;
    dsys.type = impl_->d_type;
    dsys.xyz = impl_->d_xyz;
    dsys.sp4 = impl_->d_sp4;
    for (int i = 0; i < 9; ++i) dsys.h[i] = sys.h[i];
    dsys.pbc_x = sys.pbc_x;
    dsys.pbc_y = sys.pbc_y;
    dsys.pbc_z = sys.pbc_z;

    NepGpuLammpsNeighborsDevice dnb;
    dnb.NN_radial = impl_->d_nn_radial;
    dnb.NL_radial = impl_->d_nl_radial;
    dnb.NN_angular = impl_->d_nn_angular;
    dnb.NL_angular = impl_->d_nl_angular;

    NepGpuLammpsResultDevice dres;
    dres.f = impl_->d_f;
    dres.fm = impl_->d_fm;
    dres.fm_left_iface = res.fm_left_iface ? impl_->d_fm_left_iface : nullptr;
    dres.iface_x = res.iface_x;
    dres.iface_half_width = res.iface_half_width;
    dres.inv_hbar = res.inv_hbar;
    dres.eatom = res.eatom ? impl_->d_eatom : nullptr;
    dres.vatom = res.vatom ? impl_->d_vatom : nullptr;
    dres.want_virial_raw9 = res.want_virial_raw9;

    compute_device(dsys, nlocal, dnb, dres, need_energy, need_virial);

    CHECK(gpuMemcpy(res.f, impl_->d_f, sizeof(double) * 3 * natoms, gpuMemcpyDeviceToHost));
    CHECK(gpuMemcpy(res.fm, impl_->d_fm, sizeof(double) * 3 * natoms, gpuMemcpyDeviceToHost));
    if (res.fm_left_iface) {
      CHECK(gpuMemcpy(res.fm_left_iface, impl_->d_fm_left_iface, sizeof(double) * 3 * natoms, gpuMemcpyDeviceToHost));
    }
    if (res.eatom) CHECK(gpuMemcpy(res.eatom, impl_->d_eatom, sizeof(double) * nlocal, gpuMemcpyDeviceToHost));
    if (res.vatom) CHECK(gpuMemcpy(res.vatom, impl_->d_vatom, sizeof(double) * 9 * nlocal, gpuMemcpyDeviceToHost));
    res.eng = dres.eng;
    for (int k = 0; k < 6; ++k) res.virial[k] = dres.virial[k];
    if (res.want_virial_raw9) {
      for (int k = 0; k < 9; ++k) res.virial_raw9[k] = dres.virial_raw9[k];
    }
    return;
  }

  if (!impl_->nonspin) {
    PRINT_INPUT_ERROR("NepGpuLammpsModel: non-spin backend is not initialized.\n");
  }
  if (!sys.type || !sys.xyz || !nb.NN_radial || !nb.NL_radial || !nb.NN_angular || !nb.NL_angular) {
    PRINT_INPUT_ERROR("NepGpuLammpsModel: null pointer in host inputs.\n");
  }
  const int natoms = sys.natoms;
  impl_->pos_soa.resize(static_cast<std::size_t>(natoms) * 3);
  for (int i = 0; i < natoms; ++i) {
    impl_->pos_soa[i] = sys.xyz[3 * i + 0];
    impl_->pos_soa[i + natoms] = sys.xyz[3 * i + 1];
    impl_->pos_soa[i + natoms * 2] = sys.xyz[3 * i + 2];
  }
  double* force_out = res.f;
  if (!force_out) {
    impl_->force_scratch.resize(static_cast<std::size_t>(natoms) * 3);
    force_out = impl_->force_scratch.data();
  }
  Box box;
  fill_box(sys.h, sys.pbc_x, sys.pbc_y, sys.pbc_z, box);
  impl_->nonspin->compute_with_neighbors(
    box,
    nlocal,
    natoms,
    sys.type,
    impl_->pos_soa.data(),
    nb.NN_radial,
    nb.NL_radial,
    nb.NN_angular,
    nb.NL_angular,
    res.eatom,
    force_out,
    res.vatom);
  res.eng = 0.0;
  for (double& v : res.virial) v = 0.0;
  if (need_energy && res.eatom) {
    for (int i = 0; i < nlocal; ++i) res.eng += res.eatom[i];
  }
  if (need_virial && res.vatom) {
    for (int i = 0; i < nlocal; ++i) {
      res.virial[0] += res.vatom[9 * i + 0];
      res.virial[1] += res.vatom[9 * i + 1];
      res.virial[2] += res.vatom[9 * i + 2];
      res.virial[3] += res.vatom[9 * i + 3];
      res.virial[4] += res.vatom[9 * i + 4];
      res.virial[5] += res.vatom[9 * i + 5];
    }
  }
}

void NepGpuLammpsModel::compute_device(
  const NepGpuLammpsSystemDevice& sys,
  int nlocal,
  const NepGpuLammpsNeighborsDevice& nb,
  NepGpuLammpsResultDevice& res,
  bool need_energy,
  bool need_virial)
{
  if (!impl_) {
    PRINT_INPUT_ERROR("NepGpuLammpsModel: model is not initialized.\n");
  }
  if (info_.needs_spin) {
    if (!impl_->spin) {
      PRINT_INPUT_ERROR("NepGpuLammpsModel: spin backend is not initialized.\n");
    }
    if (!sys.sp4) {
      PRINT_INPUT_ERROR("NepGpuLammpsModel: spin model requires sp4 device input.\n");
    }
    Box box;
    fill_box(sys.h, sys.pbc_x, sys.pbc_y, sys.pbc_z, box);
    double vir6[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    impl_->spin->compute_with_neighbors_device(
      box,
      nlocal,
      sys.natoms,
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
      res.eng,
      vir6,
      res.want_virial_raw9 ? res.virial_raw9 : nullptr);
    for (int k = 0; k < 6; ++k) res.virial[k] = vir6[k];
    return;
  }

  if (!impl_->nonspin) {
    PRINT_INPUT_ERROR("NepGpuLammpsModel: non-spin backend is not initialized.\n");
  }
  Box box;
  fill_box(sys.h, sys.pbc_x, sys.pbc_y, sys.pbc_z, box);
  double vir6[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  impl_->nonspin->compute_with_neighbors_device(
    box,
    nlocal,
    sys.natoms,
    sys.type,
    sys.xyz,
    sys.owner,
    sys.stream ? reinterpret_cast<gpuStream_t>(sys.stream) : static_cast<gpuStream_t>(0),
    nb.NN_radial,
    nb.NL_radial,
    nb.NN_angular,
    nb.NL_angular,
    res.f,
    res.eatom,
    res.vatom,
    need_energy,
    need_virial,
    res.eng,
    vir6);
  for (int k = 0; k < 6; ++k) res.virial[k] = vir6[k];
}

bool NepGpuLammpsModel::debug_copy_last_spin_descriptors_host(std::vector<float>& out)
{
  if (!impl_ || !impl_->spin) return false;
  if (!impl_->spin->has_last_descriptors()) return false;
  const int natoms = impl_->spin->get_current_natoms();
  const int dim = impl_->spin->get_descriptor_dim();
  if (natoms <= 0 || dim <= 0) {
    out.clear();
    return true;
  }
  out.resize(static_cast<size_t>(natoms) * static_cast<size_t>(dim));
  impl_->spin->copy_last_descriptors_to_host(out.data(), out.size());
  return true;
}

bool NepGpuLammpsModel::debug_copy_last_spin_fp_host(std::vector<float>& out)
{
  if (!impl_ || !impl_->spin) return false;
  const int natoms = impl_->spin->get_current_natoms();
  const int dim = impl_->spin->get_descriptor_dim();
  if (natoms <= 0 || dim <= 0) {
    out.clear();
    return true;
  }
  out.resize(static_cast<size_t>(natoms) * static_cast<size_t>(dim));
  impl_->spin->copy_last_Fp_to_host(out.data(), out.size());
  return true;
}

bool NepGpuLammpsModel::debug_copy_spin_q_scaler_host(std::vector<float>& out)
{
  if (!impl_ || !impl_->spin) return false;
  const int dim = impl_->spin->get_descriptor_dim();
  if (dim <= 0) {
    out.clear();
    return true;
  }
  out.resize(static_cast<size_t>(dim));
  impl_->spin->copy_q_scaler_to_host(out.data(), out.size());
  return true;
}
