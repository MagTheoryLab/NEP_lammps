#include "nep_gpu_model_spin_lmp.cuh"
#include "utilities/error.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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
  if (model.find("spin") == std::string::npos) {
    std::cerr << "This verifier targets spin NEP models (nep*_spin).\n"
              << "Got model header '" << model << "' (non-spin model).\n";
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

static void read_xyz_first_frame_spin(
  const std::string& xyz_path,
  const std::unordered_map<std::string, int>& type_map,
  std::vector<int>& type,
  std::vector<double>& pos_aos,
  std::vector<double>& spin_aos3,
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
  spin_aos3.assign(static_cast<std::size_t>(N) * 3, 0.0);

  for (int i = 0; i < N; ++i) {
    std::string species;
    double x = 0.0, y = 0.0, z = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    double fmx = 0.0, fmy = 0.0, fmz = 0.0;
    double sx = 0.0, sy = 0.0, sz = 0.0;
    if (!(in >> species >> x >> y >> z >> fx >> fy >> fz >> fmx >> fmy >> fmz >> sx >> sy >> sz)) {
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
    spin_aos3[3 * i + 0] = sx;
    spin_aos3[3 * i + 1] = sy;
    spin_aos3[3 * i + 2] = sz;
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
    for (int s = 0; s < nr; ++s) NL_r[i + N * s] = neigh_r[i][s];
    for (int s = 0; s < na; ++s) NL_a[i + N * s] = neigh_a[i][s];
  }
}

static void replicate_supercell_if_needed_spin(
  const double rc_max,
  std::vector<int>& type,
  std::vector<double>& pos_aos,
  std::vector<double>& spin_aos3,
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
  std::vector<double> spin_out;
  type_out.reserve(static_cast<std::size_t>(N0) * nx * ny * nz);
  pos_out.reserve(static_cast<std::size_t>(N0) * nx * ny * nz * 3);
  spin_out.reserve(static_cast<std::size_t>(N0) * nx * ny * nz * 3);

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
          spin_out.push_back(spin_aos3[3 * i + 0]);
          spin_out.push_back(spin_aos3[3 * i + 1]);
          spin_out.push_back(spin_aos3[3 * i + 2]);
        }
      }
    }
  }

  type.swap(type_out);
  pos_aos.swap(pos_out);
  spin_aos3.swap(spin_out);

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

static void read_expected_energy(const std::string& path, double& e_pred)
{
  std::ifstream in(path.c_str());
  if (!in.is_open()) {
    std::cerr << "Failed to open expected energy file: " << path << "\n";
    std::exit(2);
  }
  double e_ref = 0.0;
  if (!(in >> e_pred >> e_ref)) {
    std::cerr << "Failed to parse energy file: " << path << "\n";
    std::exit(2);
  }
}

static void read_expected_vec3x2(const std::string& path, const int N0, std::vector<double>& pred_aos3)
{
  std::ifstream in(path.c_str());
  if (!in.is_open()) {
    std::cerr << "Failed to open expected file: " << path << "\n";
    std::exit(2);
  }
  pred_aos3.assign(static_cast<std::size_t>(N0) * 3, 0.0);
  for (int i = 0; i < N0; ++i) {
    double px = 0.0, py = 0.0, pz = 0.0;
    double rx = 0.0, ry = 0.0, rz = 0.0;
    if (!(in >> px >> py >> pz >> rx >> ry >> rz)) {
      std::cerr << "Failed to parse " << path << " at line " << (i + 1) << "\n";
      std::exit(2);
    }
    pred_aos3[3 * i + 0] = px;
    pred_aos3[3 * i + 1] = py;
    pred_aos3[3 * i + 2] = pz;
  }
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

// -----------------------------------------------------------------------------
// Binary dump (written by tests/test_nep_spin_md_vs_nep_gpu.cu training-side eval)
// -----------------------------------------------------------------------------

struct NepSpinDumpHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t natoms;
  double energy_total;
};

static bool read_nep_spin_dump(
  const char* filename,
  int& natoms,
  double& energy_total,
  std::vector<double>& force_aos,
  std::vector<double>& mforce_aos,
  std::vector<double>& virial_aos6)
{
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) return false;

  NepSpinDumpHeader hdr;
  in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
  if (!in.good()) return false;
  const bool v1 = (std::memcmp(hdr.magic, "NEPSPIN1", 8) == 0 && hdr.version == 1u);
  const bool v2 = (std::memcmp(hdr.magic, "NEPSPIN2", 8) == 0 && hdr.version == 2u);
  if (!v1 && !v2) return false;

  natoms = static_cast<int>(hdr.natoms);
  energy_total = hdr.energy_total;

  force_aos.assign(static_cast<std::size_t>(natoms) * 3, 0.0);
  mforce_aos.assign(static_cast<std::size_t>(natoms) * 3, 0.0);
  virial_aos6.assign(static_cast<std::size_t>(natoms) * 6, 0.0);

  in.read(reinterpret_cast<char*>(force_aos.data()), sizeof(double) * force_aos.size());
  in.read(reinterpret_cast<char*>(mforce_aos.data()), sizeof(double) * mforce_aos.size());
  if (v2) {
    in.read(reinterpret_cast<char*>(virial_aos6.data()), sizeof(double) * virial_aos6.size());
  } else {
    virial_aos6.clear();
  }
  return in.good();
}

static bool file_exists(const std::string& path)
{
  std::ifstream f(path.c_str(), std::ios::binary);
  return f.good();
}

static std::string to_cmd_path(std::string path)
{
  for (char& c : path) {
    if (c == '/') c = '\\';
  }
  return path;
}

static void read_expected_virial6x2(const std::string& path, double virial6_pred[6])
{
  std::ifstream in(path.c_str());
  if (!in.is_open()) {
    std::cerr << "Failed to open expected file: " << path << "\n";
    std::exit(2);
  }
  // virial_train.out: 12 columns, first 6 are predicted in GPUMD order:
  // (xx,yy,zz,xy,yz,zx) in eV/atom.
  for (int k = 0; k < 6; ++k) virial6_pred[k] = 0.0;
  for (int k = 0; k < 6; ++k) {
    if (!(in >> virial6_pred[k])) {
      std::cerr << "Failed to parse virial file: " << path << "\n";
      std::exit(2);
    }
  }
}

static void gpumd6_to_lammps6(const double gpumd6[6], double lmp6[6])
{
  // GPUMD virial/stress order: xx yy zz xy yz zx
  // LAMMPS virial order:       xx yy zz xy xz yz
  lmp6[0] = gpumd6[0];
  lmp6[1] = gpumd6[1];
  lmp6[2] = gpumd6[2];
  lmp6[3] = gpumd6[3];
  lmp6[4] = gpumd6[5]; // xz <- zx
  lmp6[5] = gpumd6[4]; // yz
}

int main(int argc, char** argv)
{
  std::string nep_path = "NEP_GPU/test_nep_spin/nep.txt";
  std::string xyz_path = "NEP_GPU/test_nep_spin/train.xyz";
  std::string energy_path = "NEP_GPU/test_nep_spin/energy_train.out";
  std::string force_path = "NEP_GPU/test_nep_spin/force_train.out";
  std::string mforce_path = "NEP_GPU/test_nep_spin/mforce_train.out";
  std::string virial_path = "NEP_GPU/test_nep_spin/virial_train.out";
  std::string train_exe = "test_nep_spin_train_eval_from_exyz.exe";
  std::string train_dump = "NEP_GPU/test_nep_spin/_train_eval_dump.bin";
  int dev_id = 0;

  // Backward-compat positional args:
  //   verify_nep_spin_gpu.exe [nep.txt] [train.xyz] [dev_id]
  //
  // Options:
  //   --train_exe  <path>   run training-side evaluator to create dump
  //   --train_dump <path>   read an existing dump (or create it if --train_exe is used)
  int next_pos = 1;
  if (argc > next_pos) nep_path = argv[next_pos++];
  if (argc > next_pos) xyz_path = argv[next_pos++];
  if (argc > next_pos) dev_id = std::atoi(argv[next_pos++]);
  for (int i = next_pos; i < argc; ++i) {
    if (std::strcmp(argv[i], "--train_exe") == 0 && i + 1 < argc) {
      train_exe = argv[++i];
    } else if (std::strcmp(argv[i], "--train_dump") == 0 && i + 1 < argc) {
      train_dump = argv[++i];
    } else {
      std::cerr << "Unknown arg: " << argv[i] << "\n";
      return 2;
    }
  }

  const auto type_map = parse_nep_type_map(nep_path);

  std::vector<int> type;
  std::vector<double> pos_aos;
  std::vector<double> spin_aos3;
  double lattice9_xyz[9];
  read_xyz_first_frame_spin(xyz_path, type_map, type, pos_aos, spin_aos3, lattice9_xyz);
  const int N0 = static_cast<int>(type.size());

  double e_expected = 0.0;
  std::vector<double> f_expected, mf_expected;
  double v_expected_lmp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  bool have_expected = false;
  if (file_exists(train_dump)) {
    int dump_natoms = 0;
    double dump_eng = 0.0;
    std::vector<double> dump_force, dump_mforce, dump_virial;
    if (!read_nep_spin_dump(train_dump.c_str(), dump_natoms, dump_eng, dump_force, dump_mforce, dump_virial)) {
      std::cerr << "Failed to read train dump: " << train_dump << "\n";
      return 2;
    }
    if (dump_natoms != N0) {
      std::cerr << "Train dump natoms mismatch: " << dump_natoms << " (dump) vs " << N0 << " (xyz)\n";
      return 2;
    }
    e_expected = dump_eng / static_cast<double>(N0);
    f_expected = std::move(dump_force);
    mf_expected = std::move(dump_mforce);
    if (dump_virial.size() == static_cast<std::size_t>(N0) * 6) {
      double sum_gpumd[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int i = 0; i < N0; ++i) {
        for (int d = 0; d < 6; ++d) sum_gpumd[d] += dump_virial[6 * i + d];
      }
      double vpa_gpumd[6];
      for (int d = 0; d < 6; ++d) vpa_gpumd[d] = sum_gpumd[d] / static_cast<double>(N0);
      gpumd6_to_lammps6(vpa_gpumd, v_expected_lmp);
    }
    have_expected = true;
  } else if (file_exists(train_exe)) {
    std::string exe_cmd = to_cmd_path(train_exe);
    if (exe_cmd.find('\\') == std::string::npos && exe_cmd.find(':') == std::string::npos) {
      exe_cmd = ".\\" + exe_cmd;
    }
    const std::string nep_cmd = to_cmd_path(nep_path);
    const std::string xyz_cmd = to_cmd_path(xyz_path);
    const std::string dump_cmd = to_cmd_path(train_dump);

    std::ostringstream cmd;
    // cmd.exe parsing rule: cmd /C ""exe" "arg1" "arg2" --dump "arg3""
    cmd << "cmd /C \"\""
        << exe_cmd << "\""
        << " \"" << nep_cmd << "\""
        << " \"" << xyz_cmd << "\""
        << " --dump \"" << dump_cmd << "\"\"";

    std::cout << "Running training-side evaluator to generate dump:\n  " << cmd.str() << "\n";
    const int rc = std::system(cmd.str().c_str());
    if (rc != 0 || !file_exists(train_dump)) {
      std::cerr << "Training-side evaluator failed (rc=" << rc << ") or dump not created: " << train_dump << "\n";
      return 2;
    }
    int dump_natoms = 0;
    double dump_eng = 0.0;
    std::vector<double> dump_force, dump_mforce, dump_virial;
    if (!read_nep_spin_dump(train_dump.c_str(), dump_natoms, dump_eng, dump_force, dump_mforce, dump_virial)) {
      std::cerr << "Failed to read train dump: " << train_dump << "\n";
      return 2;
    }
    if (dump_natoms != N0) {
      std::cerr << "Train dump natoms mismatch: " << dump_natoms << " (dump) vs " << N0 << " (xyz)\n";
      return 2;
    }
    e_expected = dump_eng / static_cast<double>(N0);
    f_expected = std::move(dump_force);
    mf_expected = std::move(dump_mforce);
    if (dump_virial.size() == static_cast<std::size_t>(N0) * 6) {
      double sum_gpumd[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int i = 0; i < N0; ++i) {
        for (int d = 0; d < 6; ++d) sum_gpumd[d] += dump_virial[6 * i + d];
      }
      double vpa_gpumd[6];
      for (int d = 0; d < 6; ++d) vpa_gpumd[d] = sum_gpumd[d] / static_cast<double>(N0);
      gpumd6_to_lammps6(vpa_gpumd, v_expected_lmp);
    }
    have_expected = true;
  }

  if (!have_expected) {
    std::cout << "No train dump available; fallback to *_train.out for expected values.\n";
    read_expected_energy(energy_path, e_expected);
    read_expected_vec3x2(force_path, N0, f_expected);
    read_expected_vec3x2(mforce_path, N0, mf_expected);
    double v_gpumd[6];
    read_expected_virial6x2(virial_path, v_gpumd);
    gpumd6_to_lammps6(v_gpumd, v_expected_lmp);
  }

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
  NepGpuModelSpinLmp model0(nep_path.c_str(), N0);
  model0.set_device(dev_id);

  const double rc_rad = static_cast<double>(model0.cutoff());
  const double rc_ang = static_cast<double>(model0.cutoff_angular());
  const double rc_max = (rc_rad > rc_ang) ? rc_rad : rc_ang;

  const int mn_r = model0.mn_radial();
  const int mn_a = model0.mn_angular();

  int nx = 1, ny = 1, nz = 1;
  replicate_supercell_if_needed_spin(rc_max, type, pos_aos, spin_aos3, box9, nx, ny, nz);
  const int N = static_cast<int>(type.size());
  if (N != N0) {
    std::cout << "Replicated supercell: " << nx << "x" << ny << "x" << nz << " (N " << N0 << " -> " << N
              << ") to satisfy cutoff <= half thickness.\n";
  }

  NepGpuModelSpinLmp model(nep_path.c_str(), N);
  model.set_device(dev_id);

  std::vector<int> NN_r, NL_r, NN_a, NL_a;
  build_neighbors_min_image(N, box9, pos_aos, rc_rad, rc_ang, mn_r, mn_a, NN_r, NL_r, NN_a, NL_a);

  // Convert spin vector S (from xyz) -> LAMMPS sp(4): unit direction + magnitude.
  std::vector<double> sp4_aos(static_cast<std::size_t>(N) * 4, 0.0);
  for (int i = 0; i < N; ++i) {
    const double sx = spin_aos3[3 * i + 0];
    const double sy = spin_aos3[3 * i + 1];
    const double sz = spin_aos3[3 * i + 2];
    const double m = std::sqrt(sx * sx + sy * sy + sz * sz);
    double ux = 1.0, uy = 0.0, uz = 0.0;
    if (m > 0.0) {
      ux = sx / m;
      uy = sy / m;
      uz = sz / m;
    }
    sp4_aos[4 * i + 0] = ux;
    sp4_aos[4 * i + 1] = uy;
    sp4_aos[4 * i + 2] = uz;
    sp4_aos[4 * i + 3] = m;
  }

  int* d_type = nullptr;
  double* d_xyz = nullptr;
  double* d_sp4 = nullptr;
  int* d_nn_r = nullptr;
  int* d_nl_r = nullptr;
  int* d_nn_a = nullptr;
  int* d_nl_a = nullptr;
  double* d_force = nullptr;
  double* d_fm = nullptr;
  double* d_eatom = nullptr;

  CHECK(gpuMalloc((void**)&d_type, sizeof(int) * N));
  CHECK(gpuMalloc((void**)&d_xyz, sizeof(double) * 3 * N));
  CHECK(gpuMalloc((void**)&d_sp4, sizeof(double) * 4 * N));
  CHECK(gpuMalloc((void**)&d_nn_r, sizeof(int) * N));
  CHECK(gpuMalloc((void**)&d_nl_r, sizeof(int) * static_cast<size_t>(N) * mn_r));
  CHECK(gpuMalloc((void**)&d_nn_a, sizeof(int) * N));
  CHECK(gpuMalloc((void**)&d_nl_a, sizeof(int) * static_cast<size_t>(N) * mn_a));
  CHECK(gpuMalloc((void**)&d_force, sizeof(double) * 3 * N));
  CHECK(gpuMalloc((void**)&d_fm, sizeof(double) * 3 * N));
  CHECK(gpuMalloc((void**)&d_eatom, sizeof(double) * N));

  CHECK(gpuMemcpy(d_type, type.data(), sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_xyz, pos_aos.data(), sizeof(double) * 3 * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_sp4, sp4_aos.data(), sizeof(double) * 4 * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nn_r, NN_r.data(), sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nl_r, NL_r.data(), sizeof(int) * static_cast<size_t>(N) * mn_r, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nn_a, NN_a.data(), sizeof(int) * N, gpuMemcpyHostToDevice));
  CHECK(gpuMemcpy(d_nl_a, NL_a.data(), sizeof(int) * static_cast<size_t>(N) * mn_a, gpuMemcpyHostToDevice));
  CHECK(gpuMemset(d_force, 0, sizeof(double) * 3 * N));
  CHECK(gpuMemset(d_fm, 0, sizeof(double) * 3 * N));
  CHECK(gpuMemset(d_eatom, 0, sizeof(double) * N));

  NepGpuSpinDeviceSystem sys;
  sys.natoms = N;
  sys.type = d_type;
  sys.xyz = d_xyz;
  sys.sp4 = d_sp4;
  sys.stream = nullptr;
  for (int i = 0; i < 9; ++i) sys.h[i] = box9[i];
  sys.pbc_x = 1;
  sys.pbc_y = 1;
  sys.pbc_z = 1;

  NepGpuSpinLmpNeighborsDevice nb;
  nb.NN_radial = d_nn_r;
  nb.NL_radial = d_nl_r;
  nb.NN_angular = d_nn_a;
  nb.NL_angular = d_nl_a;

  NepGpuSpinDeviceResult res;
  res.f = d_force;
  res.fm = d_fm;
  // For verification we want mforce in energy units; set inv_hbar=1 so fm == mforce.
  res.inv_hbar = 1.0;
  res.eatom = d_eatom;
  res.vatom = nullptr;

  model.compute_with_neighbors_device(sys, N, nb, res, /*need_energy=*/true, /*need_virial=*/true);

  std::vector<double> f_pred_aos(static_cast<std::size_t>(N) * 3, 0.0);
  std::vector<double> mf_pred_aos(static_cast<std::size_t>(N) * 3, 0.0);
  CHECK(gpuMemcpy(f_pred_aos.data(), d_force, sizeof(double) * 3 * N, gpuMemcpyDeviceToHost));
  CHECK(gpuMemcpy(mf_pred_aos.data(), d_fm, sizeof(double) * 3 * N, gpuMemcpyDeviceToHost));

  const double epa = res.eng / static_cast<double>(N);
  std::cout << "Energy/atom predicted: " << epa << " expected: " << e_expected
            << " diff=" << (epa - e_expected) << "\n";
  const double vpa[6] = {
    res.virial[0] / static_cast<double>(N),
    res.virial[1] / static_cast<double>(N),
    res.virial[2] / static_cast<double>(N),
    res.virial[3] / static_cast<double>(N),
    res.virial[4] / static_cast<double>(N),
    res.virial[5] / static_cast<double>(N),
  };
  std::cout << "Virial/atom (LAMMPS order xx yy zz xy xz yz): predicted="
            << vpa[0] << " " << vpa[1] << " " << vpa[2] << " "
            << vpa[3] << " " << vpa[4] << " " << vpa[5]
            << " expected=" << v_expected_lmp[0] << " " << v_expected_lmp[1] << " " << v_expected_lmp[2] << " "
            << v_expected_lmp[3] << " " << v_expected_lmp[4] << " " << v_expected_lmp[5] << "\n";

  // Compare only the original-cell atoms (first N0) if we replicated.
  std::vector<double> f_pred0(static_cast<std::size_t>(N0) * 3, 0.0);
  std::vector<double> mf_pred0(static_cast<std::size_t>(N0) * 3, 0.0);
  for (int i = 0; i < N0; ++i) {
    f_pred0[3 * i + 0] = f_pred_aos[3 * i + 0];
    f_pred0[3 * i + 1] = f_pred_aos[3 * i + 1];
    f_pred0[3 * i + 2] = f_pred_aos[3 * i + 2];
    mf_pred0[3 * i + 0] = mf_pred_aos[3 * i + 0];
    mf_pred0[3 * i + 1] = mf_pred_aos[3 * i + 1];
    mf_pred0[3 * i + 2] = mf_pred_aos[3 * i + 2];
  }

  report_diff("Force", f_expected, f_pred0);
  report_diff("MForce", mf_expected, mf_pred0);

  // Basic pass/fail thresholds (tune as needed).
  const double e_tol = 2.0e-6;
  const double f_tol = 5.0e-5;
  const double mf_tol = 5.0e-5;

  double f_max = 0.0;
  for (std::size_t i = 0; i < f_expected.size(); ++i) {
    f_max = std::max(f_max, std::abs(f_pred0[i] - f_expected[i]));
  }
  double mf_max = 0.0;
  for (std::size_t i = 0; i < mf_expected.size(); ++i) {
    mf_max = std::max(mf_max, std::abs(mf_pred0[i] - mf_expected[i]));
  }

  double v_max = 0.0;
  for (int d = 0; d < 6; ++d) {
    v_max = std::max(v_max, std::abs(vpa[d] - v_expected_lmp[d]));
  }
  const double v_tol = 2.0e-6;

  const bool ok =
    (std::abs(epa - e_expected) <= e_tol) &&
    (f_max <= f_tol) &&
    (mf_max <= mf_tol) &&
    (v_max <= v_tol);

  CHECK(gpuFree(d_type));
  CHECK(gpuFree(d_xyz));
  CHECK(gpuFree(d_sp4));
  CHECK(gpuFree(d_nn_r));
  CHECK(gpuFree(d_nl_r));
  CHECK(gpuFree(d_nn_a));
  CHECK(gpuFree(d_nl_a));
  CHECK(gpuFree(d_force));
  CHECK(gpuFree(d_fm));
  CHECK(gpuFree(d_eatom));

  std::cout << (ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
