#include "nep_gpu_lammps_model.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

namespace {

constexpr int kNumAtoms = 2;
constexpr int kOnsiteDescriptorIndex = 1;
constexpr int kExpectedDescriptorDim = 11;
constexpr double kDescriptorTol = 1.0e-6;
constexpr double kForceTol = 1.0e-6;

void set_env_var(const char* name, const char* value)
{
#ifdef _WIN32
  if (value) {
    _putenv_s(name, value);
  } else {
    _putenv_s(name, "");
  }
#else
  if (value) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

struct EnvGuard {
  explicit EnvGuard(const char* key, const char* value) : name(key)
  {
    if (const char* current = std::getenv(key)) {
      had_value = true;
      old_value = current;
    }
    set_env_var(key, value);
  }

  ~EnvGuard()
  {
    set_env_var(name.c_str(), had_value ? old_value.c_str() : nullptr);
  }

  std::string name;
  std::string old_value;
  bool had_value = false;
};

struct RunOutput {
  std::vector<double> force;
  std::vector<double> mforce;
  std::vector<float> descriptors;
};

std::filesystem::path make_temp_root()
{
  const std::filesystem::path root =
    std::filesystem::temp_directory_path() / "nep_gpu_spinv2_mref_check";
  std::filesystem::create_directories(root);
  return root;
}

std::filesystem::path write_spin_model(
  const std::filesystem::path& root,
  const std::string& stem,
  const std::vector<double>& mrefs)
{
  const std::filesystem::path path = root / (stem + ".txt");
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to create temporary potential file.");
  }

  out << "nep4_spin 2 Fe Co\n";
  out << "spin_mode 1 3\n";
  out << "spin_onsite 1 1";
  for (double value : mrefs) {
    out << ' ' << value;
  }
  out << "\n";
  out << "spin_2body 0 0\n";
  out << "spin_3body 0 1 0\n";
  out << "cutoff 0.5 0.5 1 1\n";
  out << "n_max 0 0\n";
  out << "basis_size 0 0\n";
  out << "l_max 0 0 0\n";
  out << "ANN 1 0\n";

  std::vector<float> parameters;
  parameters.reserve(43);
  for (int type = 0; type < 2; ++type) {
    for (int d = 0; d < kExpectedDescriptorDim; ++d) {
      parameters.push_back(d == kOnsiteDescriptorIndex ? 1.0f : 0.0f);
    }
    parameters.push_back(0.0f); // b0
    parameters.push_back(1.0f); // w1
  }
  parameters.push_back(0.0f); // b1
  for (int i = 0; i < 16; ++i) {
    parameters.push_back(0.0f);
  }
  if (parameters.size() != 43) {
    throw std::runtime_error("Unexpected parameter count while building test potential.");
  }
  for (float value : parameters) {
    out << value << "\n";
  }

  for (int d = 0; d < kExpectedDescriptorDim; ++d) {
    out << 1.0 << "\n";
  }

  return path;
}

RunOutput run_model(NepGpuLammpsModel& model, const bool shadow)
{
  EnvGuard export_descriptors("NEP_SPIN_GPU_LMP_EXPORT_DESCRIPTORS", "1");
  EnvGuard runtime_path("NEP_SPIN_GPU_LMP_V2_PATH", shadow ? "shadow" : "reference");

  std::vector<int> type = {0, 1};
  std::vector<double> xyz = {
    0.0, 0.0, 0.0,
    3.0, 0.0, 0.0,
  };
  std::vector<double> sp4 = {
    1.0, 0.0, 0.0, 1.0,
    1.0, 0.0, 0.0, 1.0,
  };
  std::vector<int> nn_radial(kNumAtoms, 0);
  std::vector<int> nn_angular(kNumAtoms, 0);
  std::vector<int> nl_radial(static_cast<std::size_t>(kNumAtoms) * model.info().mn_radial, 0);
  std::vector<int> nl_angular(static_cast<std::size_t>(kNumAtoms) * model.info().mn_angular, 0);
  std::vector<double> force(static_cast<std::size_t>(kNumAtoms) * 3, 0.0);
  std::vector<double> mforce(static_cast<std::size_t>(kNumAtoms) * 3, 0.0);

  NepGpuLammpsSystemHost sys;
  sys.natoms = kNumAtoms;
  sys.type = type.data();
  sys.xyz = xyz.data();
  sys.sp4 = sp4.data();
  sys.h[0] = 10.0;
  sys.h[4] = 10.0;
  sys.h[8] = 10.0;

  NepGpuLammpsNeighborsHost nb;
  nb.NN_radial = nn_radial.data();
  nb.NL_radial = nl_radial.data();
  nb.NN_angular = nn_angular.data();
  nb.NL_angular = nl_angular.data();

  NepGpuLammpsResultHost res;
  res.f = force.data();
  res.fm = mforce.data();
  res.inv_hbar = 1.0;

  model.compute_host(sys, kNumAtoms, nb, res, false, false);

  std::vector<float> descriptors;
  if (!model.debug_copy_last_spin_descriptors_host(descriptors)) {
    throw std::runtime_error("Failed to export spin descriptors from test model.");
  }
  if (static_cast<int>(descriptors.size()) != kExpectedDescriptorDim * kNumAtoms) {
    throw std::runtime_error("Unexpected descriptor size from test model.");
  }

  RunOutput out;
  out.force = std::move(force);
  out.mforce = std::move(mforce);
  out.descriptors = std::move(descriptors);
  return out;
}

void expect_true(const bool condition, const char* message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_close(const double lhs, const double rhs, const double tol, const char* message)
{
  if (std::abs(lhs - rhs) > tol) {
    throw std::runtime_error(message);
  }
}

void expect_vectors_close(
  const std::vector<float>& lhs,
  const std::vector<float>& rhs,
  const double tol,
  const char* message)
{
  expect_true(lhs.size() == rhs.size(), message);
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])) > tol) {
      throw std::runtime_error(message);
    }
  }
}

void expect_vectors_close(
  const std::vector<double>& lhs,
  const std::vector<double>& rhs,
  const double tol,
  const char* message)
{
  expect_true(lhs.size() == rhs.size(), message);
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (std::abs(lhs[i] - rhs[i]) > tol) {
      throw std::runtime_error(message);
    }
  }
}

void expect_constructor_failure(
  const std::filesystem::path& exe_path,
  const std::filesystem::path& model_path,
  const char* message)
{
  int rc = 0;
#ifdef _WIN32
  const std::string exe = exe_path.string();
  const std::string model = model_path.string();
  const char* args[] = {
    exe.c_str(),
    "--expect-fail-model",
    model.c_str(),
    nullptr,
  };
  rc = _spawnv(_P_WAIT, exe.c_str(), args);
#else
  const std::string command =
    "\"" + exe_path.string() + "\" --expect-fail-model \"" + model_path.string() + "\"";
  rc = std::system(command.c_str());
#endif
  expect_true(rc != 0, message);
}

} // namespace

int main(int argc, char** argv)
{
  try {
    if (argc == 3 && std::string(argv[1]) == "--expect-fail-model") {
      NepGpuLammpsModel model(argv[2], kNumAtoms);
      (void)model;
      std::cerr << "Expected constructor failure but model loaded successfully.\n";
      return 0;
    }

    const std::filesystem::path temp_root = make_temp_root();

    const std::filesystem::path per_type_path = write_spin_model(temp_root, "per_type", {1.0, 2.0});
    const std::filesystem::path broadcast_path = write_spin_model(temp_root, "broadcast", {1.0});
    const std::filesystem::path broadcast_expanded_path =
      write_spin_model(temp_root, "broadcast_expanded", {1.0, 1.0});
    const std::filesystem::path bad_count_path = write_spin_model(temp_root, "bad_count", {1.0, 2.0, 3.0});
    const std::filesystem::path bad_value_path = write_spin_model(temp_root, "bad_value", {1.0, 0.0});

    NepGpuLammpsModel per_type_model(per_type_path.string().c_str(), kNumAtoms);
    const RunOutput per_type_reference = run_model(per_type_model, false);

    const double onsite_atom0 =
      per_type_reference.descriptors[0 + kOnsiteDescriptorIndex * kNumAtoms];
    const double onsite_atom1 =
      per_type_reference.descriptors[1 + kOnsiteDescriptorIndex * kNumAtoms];
    expect_close(onsite_atom0, 0.0, kDescriptorTol, "Type-0 onsite descriptor did not match expected broadcast reference.");
    expect_close(onsite_atom1, -0.6, kDescriptorTol, "Type-1 onsite descriptor did not use its per-type mref.");
    expect_true(
      std::abs(per_type_reference.mforce[0] - per_type_reference.mforce[3]) > 1.0e-3,
      "Per-type mref should change onsite magnetic force between atom types.");

    const RunOutput per_type_shadow = run_model(per_type_model, true);
    expect_vectors_close(
      per_type_shadow.descriptors,
      per_type_reference.descriptors,
      kDescriptorTol,
      "Shadow spin-v2 path changed exported descriptors for per-type mref.");
    expect_vectors_close(
      per_type_shadow.mforce,
      per_type_reference.mforce,
      5.0e-6,
      "Shadow spin-v2 path changed mforce for per-type mref.");

    NepGpuLammpsModel broadcast_model(broadcast_path.string().c_str(), kNumAtoms);
    NepGpuLammpsModel broadcast_expanded_model(broadcast_expanded_path.string().c_str(), kNumAtoms);
    const RunOutput broadcast = run_model(broadcast_model, false);
    const RunOutput broadcast_expanded = run_model(broadcast_expanded_model, false);
    expect_vectors_close(
      broadcast.descriptors,
      broadcast_expanded.descriptors,
      kDescriptorTol,
      "Single-value mref should match explicit per-type broadcast in descriptors.");
    expect_vectors_close(
      broadcast.mforce,
      broadcast_expanded.mforce,
      kForceTol,
      "Single-value mref should match explicit per-type broadcast in mforce.");

    expect_constructor_failure(
      std::filesystem::path(argv[0]),
      bad_count_path,
      "spin_onsite with an invalid number of mref values should be rejected.");
    expect_constructor_failure(
      std::filesystem::path(argv[0]),
      bad_value_path,
      "spin_onsite with non-positive mref should be rejected.");

    std::cout << "spinv2_mref_check passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "spinv2_mref_check error: " << e.what() << "\n";
    return 1;
  }
}
