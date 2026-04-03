#include "nep_gpu_model_lmp.cuh"
#include "utilities/error.cuh"
#include "../../NEP_CPU/src/nep.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static inline double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static inline void cross3(const double a[3], const double b[3], double c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
static inline double norm3(const double a[3]) { return std::sqrt(dot3(a, a)); }

static void compute_thickness(const double box9[9], double thickness[3])
{
  const double a[3] = {box9[0], box9[3], box9[6]};
  const double b[3] = {box9[1], box9[4], box9[7]};
  const double c[3] = {box9[2], box9[5], box9[8]};
  double bxc[3], cxa[3], axb[3];
  cross3(b, c, bxc);
  cross3(c, a, cxa);
  cross3(a, b, axb);
  const double volume = std::abs(dot3(a, bxc));
  thickness[0] = volume / norm3(bxc);
  thickness[1] = volume / norm3(cxa);
  thickness[2] = volume / norm3(axb);
}

static bool parse_lattice(const std::string& line, double* lattice9)
{
  const std::string key = "Lattice=\"";
  const std::size_t pos0 = line.find(key);
  if (pos0 == std::string::npos) return false;
  const std::size_t pos = pos0 + key.size();
  const std::size_t end = line.find('"', pos);
  if (end == std::string::npos) return false;
  std::stringstream ss(line.substr(pos, end - pos));
  for (int i = 0; i < 9; ++i) {
    if (!(ss >> lattice9[i])) return false;
  }
  return true;
}

static std::unordered_map<std::string, int> parse_nep_type_map(const std::string& nep_path)
{
  std::ifstream in(nep_path.c_str());
  if (!in.is_open()) {
    std::cerr << "Failed to open potential file: " << nep_path << "\n";
    std::exit(2);
  }
  std::string line;
  std::getline(in, line);
  std::stringstream ss(line);
  std::string model;
  int ntypes = 0;
  ss >> model >> ntypes;
  if (model.find("spin") != std::string::npos) {
    std::cerr << "This verifier targets the non-spin LAMMPS-neighbor-direct NEP GPU path.\n"
              << "Got model header '" << model << "' (spin model). Use the spin verifier instead:\n"
              << "  ./test_nep_spin_md_vs_nep_gpu.exe nep.txt nep_gpu_test.xyz --train_exe ./test_nep_spin_train_eval_from_exyz.exe\n";
    std::exit(2);
  }
  if (ntypes <= 0) {
    std::cerr << "Failed to parse type count from first line of potential: " << line << "\n";
    std::exit(2);
  }
  std::unordered_map<std::string, int> map;
  for (int t = 0; t < ntypes; ++t) {
    std::string elem;
    ss >> elem;
    if (elem.empty()) {
      std::cerr << "Failed to parse element list from first line of potential: " << line << "\n";
      std::exit(2);
    }
    map.emplace(elem, t);
  }
  return map;
}

static void aos_to_soa_positions(const int N, const std::vector<double>& pos_aos, std::vector<double>& pos_soa)
{
  pos_soa.assign(static_cast<std::size_t>(N) * 3, 0.0);
  for (int i = 0; i < N; ++i) {
    pos_soa[i] = pos_aos[3 * i + 0];
    pos_soa[i + N] = pos_aos[3 * i + 1];
    pos_soa[i + 2 * N] = pos_aos[3 * i + 2];
  }
}

static void read_xyz_first_frame(
  const std::string& xyz_path,
  const std::unordered_map<std::string, int>& type_map,
  std::vector<int>& type,
  std::vector<double>& pos_aos,
  double* lattice9_xyz)
{
  std::ifstream in(xyz_path.c_str());
  if (!in.is_open()) {
    std::cerr << "Failed to open xyz file: " << xyz_path << "\n";
    std::exit(2);
  }

  int N = 0;
  if (!(in >> N) || N <= 0) {
    std::cerr << "Failed to read atom count from: " << xyz_path << "\n";
    std::exit(2);
  }
  std::string line;
  std::getline(in, line); // rest of line 1
  std::getline(in, line); // header line
  if (!parse_lattice(line, lattice9_xyz)) {
    std::cerr << "Failed to parse Lattice=... from xyz header\n";
    std::exit(2);
  }

  type.assign(N, 0);
  pos_aos.assign(static_cast<std::size_t>(N) * 3, 0.0);

  for (int i = 0; i < N; ++i) {
    std::string species;
    double x = 0.0, y = 0.0, z = 0.0;
    if (!(in >> species >> x >> y >> z)) {
      std::cerr << "Failed to read atom line " << i << " from " << xyz_path << "\n";
      std::exit(2);
    }
    const auto it = type_map.find(species);
    if (it == type_map.end()) {
      std::cerr << "Unknown element '" << species << "' in xyz (not in potential type map)\n";
      std::exit(2);
    }
    type[i] = it->second;
    pos_aos[3 * i + 0] = x;
    pos_aos[3 * i + 1] = y;
    pos_aos[3 * i + 2] = z;

    // Skip remaining per-atom properties (force/spin/etc) on this line, if present.
    // We only need species+pos for the NEP non-spin evaluation.
    std::getline(in, line);
  }
}

static void compute_cpu_reference_nep3(
  const std::string& nep_path,
  const int N,
  const std::vector<int>& type,
  const double box9[9],
  const std::vector<double>& pos_aos,
  double& energy_per_atom,
  std::vector<double>& force_aos)
{
  NEP3 nep(nep_path);

  std::vector<double> box(9, 0.0);
  for (int i = 0; i < 9; ++i) box[i] = box9[i];

  std::vector<double> pos_soa;
  aos_to_soa_positions(N, pos_aos, pos_soa);

  std::vector<double> potential(static_cast<std::size_t>(N), 0.0);
  std::vector<double> force_soa(static_cast<std::size_t>(N) * 3, 0.0);
  std::vector<double> virial(static_cast<std::size_t>(N) * 9, 0.0);

  nep.compute(type, box, pos_soa, potential, force_soa, virial);

  double etot = 0.0;
  for (int i = 0; i < N; ++i) etot += potential[i];
  energy_per_atom = etot / static_cast<double>(N);

  force_aos.assign(static_cast<std::size_t>(N) * 3, 0.0);
  for (int i = 0; i < N; ++i) {
    force_aos[3 * i + 0] = force_soa[i];
    force_aos[3 * i + 1] = force_soa[i + N];
    force_aos[3 * i + 2] = force_soa[i + 2 * N];
  }
}

static void build_neighbors_min_image(
  const int N,
  const double* box9,
  const std::vector<double>& pos_aos,
  const double rc_rad,
  const double rc_ang,
  const int mn_r,
  const int mn_a,
  std::vector<int>& NN_r,
  std::vector<int>& NL_r,
  std::vector<int>& NN_a,
  std::vector<int>& NL_a)
{
  // Construct inverse of the 3x3 box matrix (row-major).
  const double a00 = box9[0], a01 = box9[1], a02 = box9[2];
  const double a10 = box9[3], a11 = box9[4], a12 = box9[5];
  const double a20 = box9[6], a21 = box9[7], a22 = box9[8];
  const double c00 = a11 * a22 - a12 * a21;
  const double c01 = -(a10 * a22 - a12 * a20);
  const double c02 = a10 * a21 - a11 * a20;
  const double c10 = -(a01 * a22 - a02 * a21);
  const double c11 = a00 * a22 - a02 * a20;
  const double c12 = -(a00 * a21 - a01 * a20);
  const double c20 = a01 * a12 - a02 * a11;
  const double c21 = -(a00 * a12 - a02 * a10);
  const double c22 = a00 * a11 - a01 * a10;
  const double det = a00 * c00 + a01 * c01 + a02 * c02;
  if (std::abs(det) < 1.0e-16) {
    std::cerr << "Box matrix is singular.\n";
    std::exit(2);
  }
  const double inv_det = 1.0 / det;
  const double inv00 = c00 * inv_det, inv01 = c10 * inv_det, inv02 = c20 * inv_det;
  const double inv10 = c01 * inv_det, inv11 = c11 * inv_det, inv12 = c21 * inv_det;
  const double inv20 = c02 * inv_det, inv21 = c12 * inv_det, inv22 = c22 * inv_det;

  const double rc_rad2 = rc_rad * rc_rad;
  const double rc_ang2 = rc_ang * rc_ang;

  std::vector<std::vector<int>> neigh_r(N), neigh_a(N);
  for (int i = 0; i < N; ++i) {
    neigh_r[i].reserve(128);
    neigh_a[i].reserve(128);
  }

  for (int i = 0; i < N; ++i) {
    const double xi = pos_aos[3 * i + 0];
    const double yi = pos_aos[3 * i + 1];
    const double zi = pos_aos[3 * i + 2];
    for (int j = 0; j < N; ++j) {
      if (j == i) continue;
      double dx = pos_aos[3 * j + 0] - xi;
      double dy = pos_aos[3 * j + 1] - yi;
      double dz = pos_aos[3 * j + 2] - zi;

      // Minimum image using fractional coords s = inv(A) * r; wrap to [-0.5,0.5).
      double s0 = inv00 * dx + inv01 * dy + inv02 * dz;
      double s1 = inv10 * dx + inv11 * dy + inv12 * dz;
      double s2 = inv20 * dx + inv21 * dy + inv22 * dz;
      s0 -= std::nearbyint(s0);
      s1 -= std::nearbyint(s1);
      s2 -= std::nearbyint(s2);
      dx = a00 * s0 + a01 * s1 + a02 * s2;
      dy = a10 * s0 + a11 * s1 + a12 * s2;
      dz = a20 * s0 + a21 * s1 + a22 * s2;

      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < rc_rad2) neigh_r[i].push_back(j);
      if (r2 < rc_ang2) neigh_a[i].push_back(j);
    }
  }

  for (int i = 0; i < N; ++i) {
    std::sort(neigh_r[i].begin(), neigh_r[i].end());
    std::sort(neigh_a[i].begin(), neigh_a[i].end());
  }

  NN_r.assign(N, 0);
  NN_a.assign(N, 0);
  NL_r.assign(static_cast<std::size_t>(N) * mn_r, 0);
  NL_a.assign(static_cast<std::size_t>(N) * mn_a, 0);

  for (int i = 0; i < N; ++i) {
    const int nr = static_cast<int>(neigh_r[i].size());
    const int na = static_cast<int>(neigh_a[i].size());
    if (nr > mn_r) {
      std::cerr << "Radial neighbor overflow for atom " << i << ": " << nr << " > MN_radial=" << mn_r << "\n";
      std::exit(2);
    }
    if (na > mn_a) {
      std::cerr << "Angular neighbor overflow for atom " << i << ": " << na << " > MN_angular=" << mn_a << "\n";
      std::exit(2);
    }
    NN_r[i] = nr;
    NN_a[i] = na;
    for (int s = 0; s < nr; ++s) {
      NL_r[i + N * s] = neigh_r[i][s];
    }
    for (int s = 0; s < na; ++s) {
      NL_a[i + N * s] = neigh_a[i][s];
    }
  }
}

static void replicate_supercell_if_needed(
  const double rc_max,
  std::vector<int>& type,
  std::vector<double>& pos_aos,
  double box9[9],
  int& nx,
  int& ny,
  int& nz)
{
  double thickness[3] = {0.0, 0.0, 0.0};
  compute_thickness(box9, thickness);

  nx = std::max(1, static_cast<int>(std::ceil((2.0 * rc_max) / thickness[0])));
  ny = std::max(1, static_cast<int>(std::ceil((2.0 * rc_max) / thickness[1])));
  nz = std::max(1, static_cast<int>(std::ceil((2.0 * rc_max) / thickness[2])));

  if (nx == 1 && ny == 1 && nz == 1) return;

  const int N0 = static_cast<int>(type.size());
  const double a[3] = {box9[0], box9[3], box9[6]};
  const double b[3] = {box9[1], box9[4], box9[7]};
  const double c[3] = {box9[2], box9[5], box9[8]};

  std::vector<int> type_out;
  std::vector<double> pos_out;
  type_out.reserve(static_cast<std::size_t>(N0) * nx * ny * nz);
  pos_out.reserve(static_cast<std::size_t>(N0) * nx * ny * nz * 3);

  for (int ia = 0; ia < nx; ++ia) {
    for (int ib = 0; ib < ny; ++ib) {
      for (int ic = 0; ic < nz; ++ic) {
        const double tx = ia * a[0] + ib * b[0] + ic * c[0];
        const double ty = ia * a[1] + ib * b[1] + ic * c[1];
        const double tz = ia * a[2] + ib * b[2] + ic * c[2];
        for (int i = 0; i < N0; ++i) {
          type_out.push_back(type[i]);
          pos_out.push_back(pos_aos[3 * i + 0] + tx);
          pos_out.push_back(pos_aos[3 * i + 1] + ty);
          pos_out.push_back(pos_aos[3 * i + 2] + tz);
        }
      }
    }
  }

  type.swap(type_out);
  pos_aos.swap(pos_out);

  // Scale the box vectors to match the replication.
  box9[0] *= nx;
  box9[3] *= nx;
  box9[6] *= nx;
  box9[1] *= ny;
  box9[4] *= ny;
  box9[7] *= ny;
  box9[2] *= nz;
  box9[5] *= nz;
  box9[8] *= nz;
}

static void report_diff(const char* label, const std::vector<double>& ref, const std::vector<double>& pred)
{
  if (ref.size() != pred.size()) {
    std::cerr << label << ": size mismatch ref=" << ref.size() << " pred=" << pred.size() << "\n";
    return;
  }
  double max_abs = 0.0;
  double rms = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double diff = pred[i] - ref[i];
    max_abs = std::max(max_abs, std::abs(diff));
    rms += diff * diff;
  }
  rms = std::sqrt(rms / static_cast<double>(ref.size()));
  std::cout << label << " max_abs=" << max_abs << " rms=" << rms << "\n";
}

int main(int argc, char** argv)
{
  std::string nep_path = "NEP_GPU/test_nep/nep.txt";
  std::string xyz_path = "NEP_GPU/test_nep/train.xyz";
  int dev_id = 0;
  if (argc > 1) nep_path = argv[1];
  if (argc > 2) xyz_path = argv[2];
  if (argc > 3) dev_id = std::atoi(argv[3]);

  const auto type_map = parse_nep_type_map(nep_path);

  std::vector<int> type;
  std::vector<double> pos_aos;
  double lattice9_xyz[9];
  read_xyz_first_frame(xyz_path, type_map, type, pos_aos, lattice9_xyz);
  const int N0 = static_cast<int>(type.size());

  // XYZ Lattice attribute is in row-vector order: ax ay az bx by bz cx cy cz.
  // GPUMD/NEP box uses column-vector order (stored row-major): ax bx cx ay by cy az bz cz.
  double box9[9];
  box9[0] = lattice9_xyz[0];
  box9[3] = lattice9_xyz[1];
  box9[6] = lattice9_xyz[2];
  box9[1] = lattice9_xyz[3];
  box9[4] = lattice9_xyz[4];
  box9[7] = lattice9_xyz[5];
  box9[2] = lattice9_xyz[6];
  box9[5] = lattice9_xyz[7];
  box9[8] = lattice9_xyz[8];

  // Query cutoffs/MN from a small temporary model; we may replicate the system below.
  NepGpuModelLmp model0(nep_path.c_str(), N0);
  model0.set_device(dev_id);

  const double rc_rad = static_cast<double>(model0.cutoff());
  const double rc_ang = static_cast<double>(model0.cutoff_angular());
  const double rc_zbl = static_cast<double>(model0.cutoff_zbl_outer());
  const double rc_ang_filter = (rc_zbl > rc_ang) ? rc_zbl : rc_ang;
  const double rc_max = (rc_rad > rc_ang_filter) ? rc_rad : rc_ang_filter;

  const int mn_r = model0.mn_radial();
  const int mn_a = model0.mn_angular();

  int nx = 1, ny = 1, nz = 1;
  replicate_supercell_if_needed(rc_max, type, pos_aos, box9, nx, ny, nz);
  const int N = static_cast<int>(type.size());
  if (N != N0) {
    std::cout << "Replicated supercell: " << nx << "x" << ny << "x" << nz << " (N " << N0 << " -> " << N
              << ") to satisfy cutoff <= half thickness.\n";
  }

  NepGpuModelLmp model(nep_path.c_str(), N);
  model.set_device(dev_id);

  std::vector<int> NN_r, NL_r, NN_a, NL_a;
  build_neighbors_min_image(N, box9, pos_aos, rc_rad, rc_ang_filter, mn_r, mn_a, NN_r, NL_r, NN_a, NL_a);

  double energy_expected = 0.0;
  std::vector<double> force_expected_aos;
  compute_cpu_reference_nep3(nep_path, N, type, box9, pos_aos, energy_expected, force_expected_aos);

  // Upload inputs to GPU.
  int* d_type = nullptr;
  double* d_xyz = nullptr;
  int* d_nn_r = nullptr;
  int* d_nl_r = nullptr;
  int* d_nn_a = nullptr;
  int* d_nl_a = nullptr;
  double* d_force = nullptr;
  double* d_eatom = nullptr;

  CHECK(gpuMalloc((void**)&d_type, sizeof(int) * N));
  CHECK(gpuMalloc((void**)&d_xyz, sizeof(double) * 3 * N));
  CHECK(gpuMalloc((void**)&d_nn_r, sizeof(int) * N));
  CHECK(gpuMalloc((void**)&d_nl_r, sizeof(int) * static_cast<size_t>(N) * mn_r));
  CHECK(gpuMalloc((void**)&d_nn_a, sizeof(int) * N));
  CHECK(gpuMalloc((void**)&d_nl_a, sizeof(int) * static_cast<size_t>(N) * mn_a));
  CHECK(gpuMalloc((void**)&d_force, sizeof(double) * 3 * N));
  CHECK(gpuMalloc((void**)&d_eatom, sizeof(double) * N));

  CHECK(gpuMemcpy(d_type, type.data(), sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_xyz, pos_aos.data(), sizeof(double) * 3 * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nn_r, NN_r.data(), sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nl_r, NL_r.data(), sizeof(int) * static_cast<size_t>(N) * mn_r, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nn_a, NN_a.data(), sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nl_a, NL_a.data(), sizeof(int) * static_cast<size_t>(N) * mn_a, gpuMemcpyHostToDevice));
  CHECK(gpuMemset(d_force, 0, sizeof(double) * 3 * N));
  CHECK(gpuMemset(d_eatom, 0, sizeof(double) * N));

  NepGpuDeviceSystem sys;
  sys.natoms = N;
  sys.type = d_type;
  sys.xyz = d_xyz;
  sys.owner = nullptr;
  sys.stream = nullptr;
  for (int i = 0; i < 9; ++i) sys.h[i] = box9[i];
  sys.pbc_x = 1;
  sys.pbc_y = 1;
  sys.pbc_z = 1;

  NepGpuLmpNeighborsDevice nb;
  nb.NN_radial = d_nn_r;
  nb.NL_radial = d_nl_r;
  nb.NN_angular = d_nn_a;
  nb.NL_angular = d_nl_a;

  NepGpuDeviceResult res;
  res.f = d_force;
  res.eatom = d_eatom;
  res.vatom = nullptr;

  model.compute_with_neighbors_device(sys, N, nb, res, /*need_energy=*/true, /*need_virial=*/false);

  std::vector<double> force_pred_aos(static_cast<std::size_t>(N) * 3, 0.0);
  CHECK(gpuMemcpy(force_pred_aos.data(), d_force, sizeof(double) * 3 * N, gpuMemcpyDeviceToHost));

  const double epa = res.eng / static_cast<double>(N);
  std::cout << "Energy/atom predicted: " << epa << " expected: " << energy_expected
            << " diff=" << (epa - energy_expected) << "\n";
  report_diff("Force", force_expected_aos, force_pred_aos);

  // Basic pass/fail thresholds (tune as needed).
  const double e_tol = 1.0e-6;
  const double f_tol = 1.0e-5;
  double f_max = 0.0;
  for (std::size_t i = 0; i < force_expected_aos.size(); ++i) {
    f_max = std::max(f_max, std::abs(force_pred_aos[i] - force_expected_aos[i]));
  }
  const bool ok = (std::abs(epa - energy_expected) <= e_tol) && (f_max <= f_tol);

  CHECK(gpuFree(d_type));
  CHECK(gpuFree(d_xyz));
  CHECK(gpuFree(d_nn_r));
  CHECK(gpuFree(d_nl_r));
  CHECK(gpuFree(d_nn_a));
  CHECK(gpuFree(d_nl_a));
  CHECK(gpuFree(d_force));
  CHECK(gpuFree(d_eatom));

  return ok ? 0 : 1;
}
