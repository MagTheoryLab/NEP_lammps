/*
    Spin NEP CPU bench: compare against GPUMD dump.xyz.

    Compile (g++):
        g++ -O3 -std=c++11 main.cpp ../src/nep.cpp ../src/neighbor_nep.cpp ../src/ewald.cpp
    Compile (MSVC):
        cl /O2 /openmp /EHsc /std:c++17 main.cpp ..\\src\\nep.cpp ..\\src\\neighbor_nep.cpp ..\\src\\ewald.cpp
    Run:
        ./a.out ../../tests/gpumd/Fe/nep.txt ../../tests/gpumd/Fe/dump.xyz --repeat 100 --lammps
*/

#include "../src/nep.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Property {
  std::string name;
  char type;
  int count;
};

static bool parse_properties(const std::string& props, std::vector<Property>& out)
{
  out.clear();
  std::vector<std::string> parts;
  std::string token;
  std::stringstream ss(props);
  while (std::getline(ss, token, ':')) {
    if (!token.empty()) {
      parts.push_back(token);
    }
  }
  if (parts.size() % 3 != 0) {
    return false;
  }
  for (std::size_t i = 0; i < parts.size(); i += 3) {
    Property p;
    p.name = parts[i];
    p.type = parts[i + 1].empty() ? 'R' : parts[i + 1][0];
    p.count = std::atoi(parts[i + 2].c_str());
    out.push_back(p);
  }
  return true;
}

static bool parse_lattice(const std::string& line, std::vector<double>& lattice)
{
  const std::string key = "Lattice=\"";
  std::size_t pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  pos += key.size();
  std::size_t end = line.find('"', pos);
  if (end == std::string::npos) {
    return false;
  }
  std::stringstream ss(line.substr(pos, end - pos));
  lattice.assign(9, 0.0);
  for (int i = 0; i < 9; ++i) {
    if (!(ss >> lattice[i])) {
      return false;
    }
  }
  return true;
}

static bool parse_properties_token(const std::string& line, std::string& props)
{
  const std::string key = "Properties=";
  std::size_t pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  pos += key.size();
  std::size_t end = line.find(' ', pos);
  if (end == std::string::npos) {
    end = line.size();
  }
  props = line.substr(pos, end - pos);
  return true;
}

static bool parse_energy_total(const std::string& line, double& energy_total)
{
  const std::string key = "energy=";
  std::size_t pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  pos += key.size();
  std::size_t end = line.find(' ', pos);
  std::stringstream ss(line.substr(pos, end - pos));
  return static_cast<bool>(ss >> energy_total);
}

static bool parse_virial_total(const std::string& line, std::vector<double>& virial)
{
  const std::string key = "virial=\"";
  std::size_t pos = line.find(key);
  if (pos == std::string::npos) {
    return false;
  }
  pos += key.size();
  std::size_t end = line.find('"', pos);
  if (end == std::string::npos) {
    return false;
  }
  std::stringstream ss(line.substr(pos, end - pos));
  virial.assign(9, 0.0);
  for (int i = 0; i < 9; ++i) {
    if (!(ss >> virial[i])) {
      return false;
    }
  }
  return true;
}

static void map_virial_gpumd_6(const std::vector<double>& in9, std::vector<double>& out6)
{
  out6.assign(6, 0.0);
  if (in9.size() != 9) {
    return;
  }
  out6[0] = in9[0]; // xx
  out6[1] = in9[4]; // yy
  out6[2] = in9[8]; // zz
  out6[3] = in9[1]; // xy
  out6[4] = in9[5]; // yz
  out6[5] = in9[6]; // zx
}

static void report_diff(const char* label, const std::vector<double>& ref, const std::vector<double>& pred)
{
  if (ref.empty() || pred.empty() || ref.size() != pred.size()) {
    return;
  }
  double max_abs = 0.0;
  double rms = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    double diff = pred[i] - ref[i];
    max_abs = std::max(max_abs, std::abs(diff));
    rms += diff * diff;
  }
  rms = std::sqrt(rms / ref.size());
  std::cout << "    " << label << " max_abs=" << std::setprecision(10) << max_abs
            << " rms=" << std::setprecision(10) << rms << "\n";
}

static void invert_3x3(const double* m, double* inv)
{
  const double a00 = m[0], a01 = m[1], a02 = m[2];
  const double a10 = m[3], a11 = m[4], a12 = m[5];
  const double a20 = m[6], a21 = m[7], a22 = m[8];

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
    std::exit(1);
  }
  const double inv_det = 1.0 / det;

  inv[0] = c00 * inv_det;
  inv[1] = c10 * inv_det;
  inv[2] = c20 * inv_det;
  inv[3] = c01 * inv_det;
  inv[4] = c11 * inv_det;
  inv[5] = c21 * inv_det;
  inv[6] = c02 * inv_det;
  inv[7] = c12 * inv_det;
  inv[8] = c22 * inv_det;
}

static void apply_mic(const double* box, const double* inv_box, double dx, double dy, double dz, double* out)
{
  double s0 = inv_box[0] * dx + inv_box[1] * dy + inv_box[2] * dz;
  double s1 = inv_box[3] * dx + inv_box[4] * dy + inv_box[5] * dz;
  double s2 = inv_box[6] * dx + inv_box[7] * dy + inv_box[8] * dz;

  s0 -= std::nearbyint(s0);
  s1 -= std::nearbyint(s1);
  s2 -= std::nearbyint(s2);

  out[0] = box[0] * s0 + box[1] * s1 + box[2] * s2;
  out[1] = box[3] * s0 + box[4] * s1 + box[5] * s2;
  out[2] = box[6] * s0 + box[7] * s1 + box[8] * s2;
}

int main(int argc, char* argv[])
{
  std::string nep_path = "nep.txt";
  std::string xyz_path = "dump.xyz";
  int repeat = 0;
  bool run_lammps = false;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--repeat" && i + 1 < argc) {
      repeat = std::max(0, std::atoi(argv[++i]));
    } else if (arg.rfind("--repeat=", 0) == 0) {
      repeat = std::max(0, std::atoi(arg.substr(9).c_str()));
    } else if (arg == "--lammps") {
      run_lammps = true;
    } else {
      positional.push_back(arg);
    }
  }
  if (!positional.empty()) {
    nep_path = positional[0];
  }
  if (positional.size() > 1) {
    xyz_path = positional[1];
  }

  std::ifstream in(xyz_path.c_str());
  if (!in.is_open()) {
    std::cerr << "Failed to open " << xyz_path << "\n";
    return 1;
  }

  int N = 0;
  if (!(in >> N)) {
    std::cerr << "Failed to read atom count from " << xyz_path << "\n";
    return 1;
  }
  std::string line;
  std::getline(in, line); // rest of line 1
  std::getline(in, line); // header line

  std::vector<double> lattice;
  if (!parse_lattice(line, lattice)) {
    std::cerr << "Failed to parse Lattice from header.\n";
    return 1;
  }

  std::string props_str;
  if (!parse_properties_token(line, props_str)) {
    std::cerr << "Failed to parse Properties from header.\n";
    return 1;
  }

  std::vector<Property> props;
  if (!parse_properties(props_str, props)) {
    std::cerr << "Failed to parse Properties list.\n";
    return 1;
  }

  double energy_total_ref = 0.0;
  const bool has_energy_total = parse_energy_total(line, energy_total_ref);
  std::vector<double> virial_total_ref;
  const bool has_virial_total = parse_virial_total(line, virial_total_ref);

  NEP nep(nep_path);

  std::unordered_map<std::string, int> type_map;
  for (std::size_t i = 0; i < nep.element_list.size(); ++i) {
    type_map[nep.element_list[i]] = static_cast<int>(i);
  }

  std::vector<int> type(N, 0);
  std::vector<double> position(N * 3, 0.0);
  std::vector<double> spin(N * 3, 0.0);
  std::vector<double> force_ref;
  std::vector<double> mforce_ref;
  std::vector<double> energy_atom_ref;

  bool has_force = false;
  bool has_mforce = false;
  bool has_spin = false;
  bool has_energy_atom = false;

  for (int n = 0; n < N; ++n) {
    for (std::size_t p = 0; p < props.size(); ++p) {
      const Property& prop = props[p];
      if (prop.type == 'S') {
        for (int c = 0; c < prop.count; ++c) {
          std::string sval;
          in >> sval;
          if (prop.name == "species" && c == 0) {
            auto it = type_map.find(sval);
            if (it == type_map.end()) {
              std::cerr << "Unknown element " << sval << "\n";
              return 1;
            }
            type[n] = it->second;
          }
        }
      } else {
        for (int c = 0; c < prop.count; ++c) {
          double v = 0.0;
          in >> v;
          if (prop.name == "pos") {
            position[n + c * N] = v;
          } else if (prop.name == "force" || prop.name == "forces") {
            if (!has_force) {
              force_ref.assign(N * 3, 0.0);
              has_force = true;
            }
            force_ref[n + c * N] = v;
          } else if (prop.name == "spin" || prop.name == "spins") {
            if (!has_spin) {
              has_spin = true;
            }
            spin[n + c * N] = v;
          } else if (prop.name == "mforce") {
            if (!has_mforce) {
              mforce_ref.assign(N * 3, 0.0);
              has_mforce = true;
            }
            mforce_ref[n + c * N] = v;
          } else if (prop.name == "energy_atom") {
            if (!has_energy_atom) {
              energy_atom_ref.assign(N, 0.0);
              has_energy_atom = true;
            }
            if (c == 0) {
              energy_atom_ref[n] = v;
            }
          }
        }
      }
    }
  }

  if (!has_spin) {
    std::cerr << "Spin property not found in the input file.\n";
    return 1;
  }

  std::vector<double> box(9, 0.0);
  box[0] = lattice[0];
  box[3] = lattice[1];
  box[6] = lattice[2];
  box[1] = lattice[3];
  box[4] = lattice[4];
  box[7] = lattice[5];
  box[2] = lattice[6];
  box[5] = lattice[7];
  box[8] = lattice[8];

  std::vector<double> potential(N, 0.0);
  std::vector<double> force(N * 3, 0.0);
  std::vector<double> virial(N * 9, 0.0);
  std::vector<double> mforce(N * 3, 0.0);

  const std::vector<double>& spin_const = spin;
  nep.compute(type, box, position, spin_const, potential, force, virial, mforce);

  double energy_total = 0.0;
  for (int n = 0; n < N; ++n) {
    energy_total += potential[n];
  }

  std::cout << "Computed total energy: " << std::setprecision(12) << energy_total << "\n";
  if (has_energy_total) {
    std::cout << "Reference total energy: " << std::setprecision(12) << energy_total_ref
              << " (diff=" << (energy_total - energy_total_ref) << ")\n";
  }

  if (has_energy_atom) {
    report_diff("energy_atom", energy_atom_ref, potential);
  }
  if (has_force) {
    report_diff("force", force_ref, force);
  }
  if (has_mforce) {
    report_diff("mforce", mforce_ref, mforce);
  }
  std::vector<double> virial_total9(9, 0.0);
  for (int d = 0; d < 9; ++d) {
    double sum = 0.0;
    for (int n = 0; n < N; ++n) {
      sum += virial[d * N + n];
    }
    virial_total9[d] = sum;
  }
  std::vector<double> virial_pred6;
  map_virial_gpumd_6(virial_total9, virial_pred6);
  std::cout << "NEP total virial (xx yy zz xy yz zx): " << std::setprecision(12)
            << virial_pred6[0] << " " << virial_pred6[1] << " " << virial_pred6[2] << " "
            << virial_pred6[3] << " " << virial_pred6[4] << " " << virial_pred6[5] << "\n";
  std::cout << "NEP total virial (LAMMPS order xx yy zz xy xz yz): " << std::setprecision(12)
            << virial_total9[0] << " " << virial_total9[4] << " " << virial_total9[8] << " "
            << virial_total9[1] << " " << virial_total9[2] << " " << virial_total9[5] << "\n";
  if (has_virial_total) {
    std::vector<double> virial_ref6;
    map_virial_gpumd_6(virial_total_ref, virial_ref6);
    report_diff("virial_total_gpumd", virial_ref6, virial_pred6);
  }

  if (run_lammps) {
    std::vector<int> ilist(N, 0);
    std::vector<int> numneigh(N, 0);
    std::vector<std::vector<int>> neigh(N);
    std::vector<int*> firstneigh(N, nullptr);

    const double rc_max = std::max(nep.paramb.rc_radial_max, nep.paramb.rc_angular_max);
    const double rc_max_sq = rc_max * rc_max;
    const double len_a = std::sqrt(box[0] * box[0] + box[1] * box[1] + box[2] * box[2]);
    const double len_b = std::sqrt(box[3] * box[3] + box[4] * box[4] + box[5] * box[5]);
    const double len_c = std::sqrt(box[6] * box[6] + box[7] * box[7] + box[8] * box[8]);
    double min_len = std::min(len_a, std::min(len_b, len_c));
    if (min_len <= 0.0) {
      min_len = 1.0;
    }
    const int nmax = static_cast<int>(std::ceil(rc_max / min_len)) + 1;
    struct Shift {
      int ix;
      int iy;
      int iz;
      double dx;
      double dy;
      double dz;
    };
    std::vector<Shift> shifts;
    shifts.reserve((2 * nmax + 1) * (2 * nmax + 1) * (2 * nmax + 1));
    for (int ix = -nmax; ix <= nmax; ++ix) {
      for (int iy = -nmax; iy <= nmax; ++iy) {
        for (int iz = -nmax; iz <= nmax; ++iz) {
          Shift s;
          s.ix = ix;
          s.iy = iy;
          s.iz = iz;
          s.dx = box[0] * ix + box[1] * iy + box[2] * iz;
          s.dy = box[3] * ix + box[4] * iy + box[5] * iz;
          s.dz = box[6] * ix + box[7] * iy + box[8] * iz;
          shifts.push_back(s);
        }
      }
    }

    const int num_types = static_cast<int>(nep.element_list.size());
    std::vector<int> type_map_lmp(num_types + 1, 0);
    for (int t = 0; t < num_types; ++t) {
      type_map_lmp[t + 1] = t;
    }
    std::vector<int> type_lmp(N, 0);
    for (int n = 0; n < N; ++n) {
      type_lmp[n] = type[n] + 1;
    }

    std::vector<double> pos_aos(N * 3, 0.0);
    std::vector<double> sp_aos(N * 4, 0.0);
    for (int n = 0; n < N; ++n) {
      pos_aos[n * 3 + 0] = position[n];
      pos_aos[n * 3 + 1] = position[n + N];
      pos_aos[n * 3 + 2] = position[n + 2 * N];
      const double sx = spin[n];
      const double sy = spin[n + N];
      const double sz = spin[n + 2 * N];
      const double mag = std::sqrt(sx * sx + sy * sy + sz * sz);
      double ux = 0.0, uy = 0.0, uz = 0.0;
      if (mag > 0.0) {
        const double inv = 1.0 / mag;
        ux = sx * inv;
        uy = sy * inv;
        uz = sz * inv;
      }
      sp_aos[n * 4 + 0] = ux;
      sp_aos[n * 4 + 1] = uy;
      sp_aos[n * 4 + 2] = uz;
      sp_aos[n * 4 + 3] = mag;
    }

    struct GhostKey {
      int atom;
      int ix;
      int iy;
      int iz;
    };
    struct GhostKeyLess {
      bool operator()(const GhostKey& a, const GhostKey& b) const
      {
        if (a.atom != b.atom) return a.atom < b.atom;
        if (a.ix != b.ix) return a.ix < b.ix;
        if (a.iy != b.iy) return a.iy < b.iy;
        return a.iz < b.iz;
      }
    };

    std::map<GhostKey, int, GhostKeyLess> ghost_map;
    std::vector<int> ghost_base;
    ghost_base.reserve(N);
    for (int n = 0; n < N; ++n) {
      ghost_base.push_back(-1);
    }

    for (int n1 = 0; n1 < N; ++n1) {
      ilist[n1] = n1;
      for (int n2 = 0; n2 < N; ++n2) {
        const double dx0 = position[n2] - position[n1];
        const double dy0 = position[n2 + N] - position[n1 + N];
        const double dz0 = position[n2 + 2 * N] - position[n1 + 2 * N];
        for (const auto& s : shifts) {
          if (n1 == n2 && s.ix == 0 && s.iy == 0 && s.iz == 0) {
            continue;
          }
          const double dx = dx0 + s.dx;
          const double dy = dy0 + s.dy;
          const double dz = dz0 + s.dz;
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= rc_max_sq) {
            continue;
          }
          int neigh_index = n2;
          if (s.ix != 0 || s.iy != 0 || s.iz != 0) {
            GhostKey key{n2, s.ix, s.iy, s.iz};
            auto it = ghost_map.find(key);
            if (it != ghost_map.end()) {
              neigh_index = it->second;
            } else {
              neigh_index = static_cast<int>(pos_aos.size() / 3);
              pos_aos.push_back(pos_aos[n2 * 3 + 0] + s.dx);
              pos_aos.push_back(pos_aos[n2 * 3 + 1] + s.dy);
              pos_aos.push_back(pos_aos[n2 * 3 + 2] + s.dz);
              sp_aos.push_back(sp_aos[n2 * 4 + 0]);
              sp_aos.push_back(sp_aos[n2 * 4 + 1]);
              sp_aos.push_back(sp_aos[n2 * 4 + 2]);
              sp_aos.push_back(sp_aos[n2 * 4 + 3]);
              type_lmp.push_back(type_lmp[n2]);
              ghost_base.push_back(n2);
              ghost_map[key] = neigh_index;
            }
          }
          neigh[n1].push_back(neigh_index);
        }
      }
      numneigh[n1] = static_cast<int>(neigh[n1].size());
    }

    for (int n = 0; n < N; ++n) {
      firstneigh[n] = neigh[n].data();
    }

    const int nall = static_cast<int>(pos_aos.size() / 3);
    std::vector<double*> pos_ptrs(nall, nullptr);
    std::vector<double*> sp_ptrs(nall, nullptr);
    for (int n = 0; n < nall; ++n) {
      pos_ptrs[n] = &pos_aos[n * 3];
      sp_ptrs[n] = &sp_aos[n * 4];
    }

    std::vector<double> force_lmp_aos(nall * 3, 0.0);
    std::vector<double*> force_lmp_ptrs(nall, nullptr);
    std::vector<double> fm_aos(nall * 3, 0.0);
    std::vector<double*> fm_ptrs(nall, nullptr);
    std::vector<double> virial_lmp_aos(nall * 9, 0.0);
    std::vector<double*> virial_lmp_ptrs(nall, nullptr);
    for (int n = 0; n < nall; ++n) {
      force_lmp_ptrs[n] = &force_lmp_aos[n * 3];
      fm_ptrs[n] = &fm_aos[n * 3];
      virial_lmp_ptrs[n] = &virial_lmp_aos[n * 9];
    }

    std::vector<double> potential_lmp(N, 0.0);
    double total_potential_lmp = 0.0;
    double total_virial_lmp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    nep.compute_for_lammps(
      N, N, ilist.data(), numneigh.data(), firstneigh.data(), type_lmp.data(), type_map_lmp.data(),
      pos_ptrs.data(), sp_ptrs.data(), total_potential_lmp, total_virial_lmp, potential_lmp.data(),
      force_lmp_ptrs.data(), fm_ptrs.data(), virial_lmp_ptrs.data());

    for (int idx = N; idx < nall; ++idx) {
      int base = ghost_base[idx];
      if (base < 0 || base >= N) {
        continue;
      }
      force_lmp_aos[base * 3 + 0] += force_lmp_aos[idx * 3 + 0];
      force_lmp_aos[base * 3 + 1] += force_lmp_aos[idx * 3 + 1];
      force_lmp_aos[base * 3 + 2] += force_lmp_aos[idx * 3 + 2];
      fm_aos[base * 3 + 0] += fm_aos[idx * 3 + 0];
      fm_aos[base * 3 + 1] += fm_aos[idx * 3 + 1];
      fm_aos[base * 3 + 2] += fm_aos[idx * 3 + 2];
    }

    std::cout << "LAMMPS total energy: " << std::setprecision(12) << total_potential_lmp << "\n";
    std::cout << "LAMMPS total virial (xx yy zz xy xz yz): " << std::setprecision(12)
              << total_virial_lmp[0] << " " << total_virial_lmp[1] << " " << total_virial_lmp[2]
              << " " << total_virial_lmp[3] << " " << total_virial_lmp[4] << " "
              << total_virial_lmp[5] << "\n";

    report_diff("lammps_potential", potential, potential_lmp);

    std::vector<double> force_lmp_soa(N * 3, 0.0);
    std::vector<double> fm_lmp_soa(N * 3, 0.0);
    for (int n = 0; n < N; ++n) {
      force_lmp_soa[n] = force_lmp_aos[n * 3 + 0];
      force_lmp_soa[n + N] = force_lmp_aos[n * 3 + 1];
      force_lmp_soa[n + 2 * N] = force_lmp_aos[n * 3 + 2];
      fm_lmp_soa[n] = fm_aos[n * 3 + 0];
      fm_lmp_soa[n + N] = fm_aos[n * 3 + 1];
      fm_lmp_soa[n + 2 * N] = fm_aos[n * 3 + 2];
    }
    report_diff("lammps_force", force, force_lmp_soa);
    report_diff("lammps_mforce", mforce, fm_lmp_soa);

    std::vector<double> virial_ref6(6, 0.0);
    for (int n = 0; n < N; ++n) {
      virial_ref6[0] += virial[0 * N + n]; // xx
      virial_ref6[1] += virial[4 * N + n]; // yy
      virial_ref6[2] += virial[8 * N + n]; // zz
      virial_ref6[3] += virial[1 * N + n]; // xy
      virial_ref6[4] += virial[2 * N + n]; // xz
      virial_ref6[5] += virial[5 * N + n]; // yz
    }
    std::vector<double> virial_lmp6(6, 0.0);
    for (int d = 0; d < 6; ++d) {
      virial_lmp6[d] = total_virial_lmp[d];
    }
    report_diff("lammps_virial_total", virial_ref6, virial_lmp6);
  }

  if (repeat > 0) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeat; ++r) {
      nep.compute(type, box, position, spin_const, potential, force, virial, mforce);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = t1 - t0;
    double avg_ms = (elapsed.count() * 1000.0) / static_cast<double>(repeat);
    std::cout << "Timing: repeat=" << repeat << " total_s=" << std::setprecision(6)
              << elapsed.count() << " avg_ms=" << std::setprecision(6) << avg_ms << "\n";
  }

  return 0;
}
