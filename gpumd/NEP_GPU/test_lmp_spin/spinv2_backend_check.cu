#include "nep_gpu_lammps_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct AtomData {
  int type = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double spx = 0.0;
  double spy = 0.0;
  double spz = 0.0;
  double spmag = 0.0;
};

struct ExpandedAtom {
  int type = 0;
  int source = -1;
  int ix = 0;
  int iy = 0;
  int iz = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double spx = 0.0;
  double spy = 0.0;
  double spz = 0.0;
  double spmag = 0.0;
};

struct DataFile {
  int natoms = 0;
  int num_types = 0;
  double xlo = 0.0;
  double xhi = 0.0;
  double ylo = 0.0;
  double yhi = 0.0;
  double zlo = 0.0;
  double zhi = 0.0;
  double xy = 0.0;
  double xz = 0.0;
  double yz = 0.0;
  std::vector<AtomData> atoms;
};

std::string trim(const std::string& s)
{
  const std::size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return std::string();
  const std::size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::vector<double> load_scalar_file(const std::string& path, int cols)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open " + path);
  std::vector<double> out;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;
    std::istringstream iss(line);
    for (int i = 0; i < cols; ++i) {
      double v = 0.0;
      if (!(iss >> v)) {
        throw std::runtime_error("Failed to parse numeric row in " + path);
      }
      out.push_back(v);
    }
  }
  return out;
}

DataFile load_lammps_data(const std::string& path)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open " + path);

  DataFile data;
  std::string line;
  bool in_atoms = false;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;
    if (!in_atoms) {
      if (line.find("atoms") != std::string::npos && data.natoms == 0) {
        std::istringstream iss(line);
        iss >> data.natoms;
        continue;
      }
      if (line.find("atom types") != std::string::npos) {
        std::istringstream iss(line);
        iss >> data.num_types;
        continue;
      }
      if (line.find("xlo xhi") != std::string::npos) {
        std::istringstream iss(line);
        iss >> data.xlo >> data.xhi;
        continue;
      }
      if (line.find("ylo yhi") != std::string::npos) {
        std::istringstream iss(line);
        iss >> data.ylo >> data.yhi;
        continue;
      }
      if (line.find("zlo zhi") != std::string::npos) {
        std::istringstream iss(line);
        iss >> data.zlo >> data.zhi;
        continue;
      }
      if (line.find("xy xz yz") != std::string::npos) {
        std::istringstream iss(line);
        iss >> data.xy >> data.xz >> data.yz;
        continue;
      }
      if (line.rfind("Atoms", 0) == 0) {
        in_atoms = true;
        data.atoms.resize(static_cast<std::size_t>(data.natoms));
        continue;
      }
      continue;
    }

    std::istringstream iss(line);
    int id = 0;
    AtomData atom;
    if (!(iss >> id >> atom.type >> atom.x >> atom.y >> atom.z >> atom.spx >> atom.spy >> atom.spz >> atom.spmag)) {
      continue;
    }
    if (id < 1 || id > data.natoms) {
      throw std::runtime_error("Atom id out of range in " + path);
    }
    data.atoms[static_cast<std::size_t>(id - 1)] = atom;
  }

  if (static_cast<int>(data.atoms.size()) != data.natoms) {
    throw std::runtime_error("Incomplete atom section in " + path);
  }
  return data;
}

void invert3x3(const double h[9], double hinv[9])
{
  const double det =
    h[0] * (h[4] * h[8] - h[5] * h[7]) -
    h[1] * (h[3] * h[8] - h[5] * h[6]) +
    h[2] * (h[3] * h[7] - h[4] * h[6]);
  if (std::abs(det) < 1.0e-20) {
    throw std::runtime_error("Singular box matrix");
  }
  const double inv_det = 1.0 / det;
  hinv[0] =  (h[4] * h[8] - h[5] * h[7]) * inv_det;
  hinv[1] = -(h[1] * h[8] - h[2] * h[7]) * inv_det;
  hinv[2] =  (h[1] * h[5] - h[2] * h[4]) * inv_det;
  hinv[3] = -(h[3] * h[8] - h[5] * h[6]) * inv_det;
  hinv[4] =  (h[0] * h[8] - h[2] * h[6]) * inv_det;
  hinv[5] = -(h[0] * h[5] - h[2] * h[3]) * inv_det;
  hinv[6] =  (h[3] * h[7] - h[4] * h[6]) * inv_det;
  hinv[7] = -(h[0] * h[7] - h[1] * h[6]) * inv_det;
  hinv[8] =  (h[0] * h[4] - h[1] * h[3]) * inv_det;
}

void apply_mic(const double h[9], const double hinv[9], double& dx, double& dy, double& dz)
{
  double sx = hinv[0] * dx + hinv[1] * dy + hinv[2] * dz;
  double sy = hinv[3] * dx + hinv[4] * dy + hinv[5] * dz;
  double sz = hinv[6] * dx + hinv[7] * dy + hinv[8] * dz;
  sx -= std::floor(sx + 0.5);
  sy -= std::floor(sy + 0.5);
  sz -= std::floor(sz + 0.5);
  const double ndx = h[0] * sx + h[1] * sy + h[2] * sz;
  const double ndy = h[3] * sx + h[4] * sy + h[5] * sz;
  const double ndz = h[6] * sx + h[7] * sy + h[8] * sz;
  dx = ndx;
  dy = ndy;
  dz = ndz;
}

double vec_norm(double x, double y, double z)
{
  return std::sqrt(x * x + y * y + z * z);
}

struct CompareStats {
  double max_abs = 0.0;
  double rmse = 0.0;
};

CompareStats compare_vectors(const std::vector<double>& a, const std::vector<double>& b)
{
  if (a.size() != b.size()) {
    throw std::runtime_error("Vector size mismatch");
  }
  CompareStats stats;
  if (a.empty()) return stats;
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = a[i] - b[i];
    stats.max_abs = std::max(stats.max_abs, std::abs(diff));
    sum_sq += diff * diff;
  }
  stats.rmse = std::sqrt(sum_sq / static_cast<double>(a.size()));
  return stats;
}

} // namespace

int main(int argc, char** argv)
{
  try {
#ifdef _WIN32
    if (std::getenv("NEP_SPIN_GPU_LMP_EXPORT_DESCRIPTORS") == nullptr) {
      _putenv_s("NEP_SPIN_GPU_LMP_EXPORT_DESCRIPTORS", "1");
    }
#else
    if (std::getenv("NEP_SPIN_GPU_LMP_EXPORT_DESCRIPTORS") == nullptr) {
      setenv("NEP_SPIN_GPU_LMP_EXPORT_DESCRIPTORS", "1", 0);
    }
#endif
    std::string bench_root = (argc >= 2) ? argv[1] : "D:\\Desktop\\benchmark\\SpinV2";
    const std::string nep_path = bench_root + "\\nep.txt";
    const std::string data_path = bench_root + "\\single.dat";
    const std::string pred_dir = bench_root + "\\pred";

    nep_gpu_lammps_set_device(0);
    NepGpuLammpsModel model(nep_path.c_str(), 256);
    const NepGpuModelInfo& info = model.info();
    if (!info.needs_spin) {
      throw std::runtime_error("Benchmark potential is not parsed as a spin model");
    }

    const DataFile data = load_lammps_data(data_path);
    const int nlocal = data.natoms;

    double h[9] = {
      data.xhi - data.xlo, data.xy, data.xz,
      0.0,                data.yhi - data.ylo, data.yz,
      0.0,                0.0,                 data.zhi - data.zlo};
    double hinv[9];
    invert3x3(h, hinv);

    const bool no_ghosts = (std::getenv("SPINV2_CHECK_NO_GHOSTS") != nullptr);
    std::vector<ExpandedAtom> expanded;
    if (no_ghosts) {
      expanded.reserve(static_cast<std::size_t>(nlocal));
      for (int i = 0; i < nlocal; ++i) {
        const AtomData& atom = data.atoms[static_cast<std::size_t>(i)];
        ExpandedAtom ex;
        ex.type = atom.type - 1;
        ex.source = i;
        ex.ix = 0;
        ex.iy = 0;
        ex.iz = 0;
        ex.x = atom.x;
        ex.y = atom.y;
        ex.z = atom.z;
        ex.spx = atom.spx;
        ex.spy = atom.spy;
        ex.spz = atom.spz;
        ex.spmag = atom.spmag;
        expanded.push_back(ex);
      }
    } else {
      const double a_norm = vec_norm(h[0], h[3], h[6]);
      const double b_norm = vec_norm(h[1], h[4], h[7]);
      const double c_norm = vec_norm(h[2], h[5], h[8]);
      const double rc_max = std::max(info.rc_radial_max, info.rc_angular_max);
      const int rep_x = std::max(1, static_cast<int>(std::ceil(rc_max / a_norm)));
      const int rep_y = std::max(1, static_cast<int>(std::ceil(rc_max / b_norm)));
      const int rep_z = std::max(1, static_cast<int>(std::ceil(rc_max / c_norm)));

      expanded.reserve(static_cast<std::size_t>(nlocal) * static_cast<std::size_t>((2 * rep_x + 1) * (2 * rep_y + 1) * (2 * rep_z + 1)));
      for (int i = 0; i < nlocal; ++i) {
        const AtomData& atom = data.atoms[static_cast<std::size_t>(i)];
        for (int ix = -rep_x; ix <= rep_x; ++ix) {
          for (int iy = -rep_y; iy <= rep_y; ++iy) {
            for (int iz = -rep_z; iz <= rep_z; ++iz) {
              ExpandedAtom ex;
              ex.type = atom.type - 1;
              ex.source = i;
              ex.ix = ix;
              ex.iy = iy;
              ex.iz = iz;
              ex.x = atom.x + ix * h[0] + iy * h[1] + iz * h[2];
              ex.y = atom.y + ix * h[3] + iy * h[4] + iz * h[5];
              ex.z = atom.z + ix * h[6] + iy * h[7] + iz * h[8];
              ex.spx = atom.spx;
              ex.spy = atom.spy;
              ex.spz = atom.spz;
              ex.spmag = atom.spmag;
              expanded.push_back(ex);
            }
          }
        }
      }

      std::stable_sort(expanded.begin(), expanded.end(), [](const ExpandedAtom& a, const ExpandedAtom& b) {
        const bool a_local = (a.ix == 0 && a.iy == 0 && a.iz == 0);
        const bool b_local = (b.ix == 0 && b.iy == 0 && b.iz == 0);
        if (a_local != b_local) return a_local > b_local;
        if (a.source != b.source) return a.source < b.source;
        if (a.ix != b.ix) return a.ix < b.ix;
        if (a.iy != b.iy) return a.iy < b.iy;
        return a.iz < b.iz;
      });
    }

    const int natoms = static_cast<int>(expanded.size());
    std::vector<int> types(static_cast<std::size_t>(natoms), 0);
    std::vector<double> xyz(static_cast<std::size_t>(natoms) * 3);
    std::vector<double> sp4(static_cast<std::size_t>(natoms) * 4);
    const bool sp4_mag_first = (std::getenv("SPINV2_CHECK_SP4_MAG_FIRST") != nullptr);
    for (int i = 0; i < natoms; ++i) {
      const ExpandedAtom& atom = expanded[static_cast<std::size_t>(i)];
      types[static_cast<std::size_t>(i)] = atom.type;
      xyz[static_cast<std::size_t>(3 * i + 0)] = atom.x;
      xyz[static_cast<std::size_t>(3 * i + 1)] = atom.y;
      xyz[static_cast<std::size_t>(3 * i + 2)] = atom.z;
      if (sp4_mag_first) {
        sp4[static_cast<std::size_t>(4 * i + 0)] = atom.spmag;
        sp4[static_cast<std::size_t>(4 * i + 1)] = atom.spx;
        sp4[static_cast<std::size_t>(4 * i + 2)] = atom.spy;
        sp4[static_cast<std::size_t>(4 * i + 3)] = atom.spz;
      } else {
        sp4[static_cast<std::size_t>(4 * i + 0)] = atom.spx;
        sp4[static_cast<std::size_t>(4 * i + 1)] = atom.spy;
        sp4[static_cast<std::size_t>(4 * i + 2)] = atom.spz;
        sp4[static_cast<std::size_t>(4 * i + 3)] = atom.spmag;
      }
    }

    const int mn_r = info.mn_radial;
    const int mn_a = info.mn_angular;
    std::vector<int> nn_r(static_cast<std::size_t>(nlocal), 0);
    std::vector<int> nn_a(static_cast<std::size_t>(nlocal), 0);
    std::vector<int> nl_r(static_cast<std::size_t>(nlocal) * mn_r, -1);
    std::vector<int> nl_a(static_cast<std::size_t>(nlocal) * mn_a, -1);

    for (int i = 0; i < nlocal; ++i) {
      int nr = 0;
      int na = 0;
      const int ti = types[static_cast<std::size_t>(i)];
      for (int j = 0; j < natoms; ++j) {
        if (expanded[static_cast<std::size_t>(j)].source == i &&
            expanded[static_cast<std::size_t>(j)].ix == 0 &&
            expanded[static_cast<std::size_t>(j)].iy == 0 &&
            expanded[static_cast<std::size_t>(j)].iz == 0) {
          continue;
        }
        double dx = xyz[static_cast<std::size_t>(3 * j + 0)] - xyz[static_cast<std::size_t>(3 * i + 0)];
        double dy = xyz[static_cast<std::size_t>(3 * j + 1)] - xyz[static_cast<std::size_t>(3 * i + 1)];
        double dz = xyz[static_cast<std::size_t>(3 * j + 2)] - xyz[static_cast<std::size_t>(3 * i + 2)];
        if (no_ghosts) {
          apply_mic(h, hinv, dx, dy, dz);
        }
        const double rsq = dx * dx + dy * dy + dz * dz;
        const int tj = types[static_cast<std::size_t>(j)];
        const double rc_r = 0.5 * (info.rc_radial_by_type[static_cast<std::size_t>(ti)] + info.rc_radial_by_type[static_cast<std::size_t>(tj)]);
        const double rc_a = 0.5 * (info.rc_angular_by_type[static_cast<std::size_t>(ti)] + info.rc_angular_by_type[static_cast<std::size_t>(tj)]);
        if (rsq < rc_r * rc_r) {
          if (nr >= mn_r) throw std::runtime_error("Radial neighbor overflow while building benchmark NN/NL");
          nl_r[static_cast<std::size_t>(i) + static_cast<std::size_t>(nlocal) * nr] = j;
          ++nr;
        }
        if (rsq < rc_a * rc_a) {
          if (na >= mn_a) throw std::runtime_error("Angular neighbor overflow while building benchmark NN/NL");
          nl_a[static_cast<std::size_t>(i) + static_cast<std::size_t>(nlocal) * na] = j;
          ++na;
        }
      }
      nn_r[static_cast<std::size_t>(i)] = nr;
      nn_a[static_cast<std::size_t>(i)] = na;
    }

    std::vector<double> f(static_cast<std::size_t>(natoms) * 3, 0.0);
    std::vector<double> fm(static_cast<std::size_t>(natoms) * 3, 0.0);
    std::vector<double> eatom(static_cast<std::size_t>(nlocal), 0.0);

    NepGpuLammpsSystemHost sys;
    sys.natoms = natoms;
    sys.type = types.data();
    sys.xyz = xyz.data();
    sys.sp4 = sp4.data();
    for (int i = 0; i < 9; ++i) sys.h[i] = h[i];
    sys.pbc_x = 1;
    sys.pbc_y = 1;
    sys.pbc_z = 1;

    NepGpuLammpsNeighborsHost nb;
    nb.NN_radial = nn_r.data();
    nb.NL_radial = nl_r.data();
    nb.NN_angular = nn_a.data();
    nb.NL_angular = nl_a.data();

    NepGpuLammpsResultHost res;
    res.f = f.data();
    res.fm = fm.data();
    res.eatom = eatom.data();
    res.inv_hbar = 1.0;

    model.compute_host(sys, nlocal, nb, res, true, true);

    const std::vector<double> f_before_fold = f;
    const std::vector<double> fm_before_fold = fm;

    if (!no_ghosts) {
      // The bridge accumulates f and fm for ALL natoms (local + ghost).
      // Ghost entries [nlocal, natoms) hold "mj" contributions from local atoms
      // that processed ghost copies of unit-cell atoms as neighbors.
      // In the LAMMPS pair style these are folded back via comm->reverse_comm.
      for (int ghost_i = nlocal; ghost_i < natoms; ++ghost_i) {
        const int src = expanded[static_cast<std::size_t>(ghost_i)].source;
        f[static_cast<std::size_t>(src) * 3 + 0] += f[static_cast<std::size_t>(ghost_i) * 3 + 0];
        f[static_cast<std::size_t>(src) * 3 + 1] += f[static_cast<std::size_t>(ghost_i) * 3 + 1];
        f[static_cast<std::size_t>(src) * 3 + 2] += f[static_cast<std::size_t>(ghost_i) * 3 + 2];
        fm[static_cast<std::size_t>(src) * 3 + 0] += fm[static_cast<std::size_t>(ghost_i) * 3 + 0];
        fm[static_cast<std::size_t>(src) * 3 + 1] += fm[static_cast<std::size_t>(ghost_i) * 3 + 1];
        fm[static_cast<std::size_t>(src) * 3 + 2] += fm[static_cast<std::size_t>(ghost_i) * 3 + 2];
      }
    }

    double energy_sum = 0.0;
    for (double e : eatom) energy_sum += e;

    const std::vector<double> ref_energy = load_scalar_file(pred_dir + "\\energy_train.out", 1);
    const std::vector<double> ref_force = load_scalar_file(pred_dir + "\\force_train.out", 3);
    const std::vector<double> ref_mforce = load_scalar_file(pred_dir + "\\mforce_train.out", 3);
    const std::vector<double> ref_descriptor = load_scalar_file(pred_dir + "\\descriptor.out", 126);

    std::vector<double> f_local(static_cast<std::size_t>(nlocal) * 3);
    std::vector<double> fm_local(static_cast<std::size_t>(nlocal) * 3);
    std::vector<double> fm_local_before_fold(static_cast<std::size_t>(nlocal) * 3);
    std::vector<double> fm_ghost_only(static_cast<std::size_t>(nlocal) * 3, 0.0);
    std::copy(f.begin(), f.begin() + static_cast<std::ptrdiff_t>(f_local.size()), f_local.begin());
    std::copy(fm.begin(), fm.begin() + static_cast<std::ptrdiff_t>(fm_local.size()), fm_local.begin());
    std::copy(
      fm_before_fold.begin(),
      fm_before_fold.begin() + static_cast<std::ptrdiff_t>(fm_local_before_fold.size()),
      fm_local_before_fold.begin());
    if (!no_ghosts) {
      for (int atom = 0; atom < nlocal; ++atom) {
        const std::size_t base = static_cast<std::size_t>(3 * atom);
        fm_ghost_only[base + 0] = fm_local[base + 0] - fm_local_before_fold[base + 0];
        fm_ghost_only[base + 1] = fm_local[base + 1] - fm_local_before_fold[base + 1];
        fm_ghost_only[base + 2] = fm_local[base + 2] - fm_local_before_fold[base + 2];
      }
    }

    std::vector<float> descriptor_soa;
    if (!model.debug_copy_last_spin_descriptors_host(descriptor_soa)) {
      throw std::runtime_error("Failed to export last spin descriptors from backend");
    }
    if (descriptor_soa.empty()) {
      throw std::runtime_error("Backend returned empty descriptor buffer");
    }
    const std::size_t descriptor_dim = ref_descriptor.size();
    if (descriptor_soa.size() < descriptor_dim * static_cast<std::size_t>(natoms)) {
      throw std::runtime_error("Backend descriptor buffer is smaller than expected");
    }
    std::vector<float> q_scaler;
    if (!model.debug_copy_spin_q_scaler_host(q_scaler)) {
      throw std::runtime_error("Failed to export q_scaler from backend");
    }
    if (q_scaler.size() != descriptor_dim) {
      throw std::runtime_error("Backend q_scaler size does not match descriptor size");
    }

    std::vector<double> descriptor_structure_avg(descriptor_dim, 0.0);
    for (std::size_t d = 0; d < descriptor_dim; ++d) {
      double sum = 0.0;
      for (int atom = 0; atom < nlocal; ++atom) {
        sum += static_cast<double>(descriptor_soa[static_cast<std::size_t>(natoms) * d + static_cast<std::size_t>(atom)]) *
               static_cast<double>(q_scaler[d]);
      }
      descriptor_structure_avg[d] = sum / static_cast<double>(nlocal);
    }

    std::vector<double> descriptor_atom0_scaled(descriptor_dim, 0.0);
    for (std::size_t d = 0; d < descriptor_dim; ++d) {
      descriptor_atom0_scaled[d] =
        static_cast<double>(descriptor_soa[static_cast<std::size_t>(natoms) * d + 0]) * static_cast<double>(q_scaler[d]);
    }
    int best_descriptor_atom = -1;
    CompareStats best_descriptor_stats;
    best_descriptor_stats.max_abs = std::numeric_limits<double>::infinity();
    best_descriptor_stats.rmse = std::numeric_limits<double>::infinity();
    for (int atom = 0; atom < nlocal; ++atom) {
      std::vector<double> descriptor_atom(descriptor_dim, 0.0);
      for (std::size_t d = 0; d < descriptor_dim; ++d) {
        descriptor_atom[d] = static_cast<double>(descriptor_soa[static_cast<std::size_t>(natoms) * d + static_cast<std::size_t>(atom)]) *
                             static_cast<double>(q_scaler[d]);
      }
      const CompareStats stats = compare_vectors(descriptor_atom, ref_descriptor);
      if (stats.rmse < best_descriptor_stats.rmse) {
        best_descriptor_stats = stats;
        best_descriptor_atom = atom;
      }
    }

    const CompareStats force_stats = compare_vectors(f_local, ref_force);
    const CompareStats mforce_stats = compare_vectors(fm_local, ref_mforce);
    const CompareStats mforce_nofold_stats = compare_vectors(fm_local_before_fold, ref_mforce);
    const CompareStats mforce_ghost_only_stats = compare_vectors(fm_ghost_only, std::vector<double>(fm_ghost_only.size(), 0.0));
    const CompareStats descriptor_atom0_stats = compare_vectors(descriptor_atom0_scaled, ref_descriptor);
    const CompareStats descriptor_structure_stats = compare_vectors(descriptor_structure_avg, ref_descriptor);

    std::cout << std::setprecision(16);
    std::cout << "energy_result_compute_host " << res.eng << "\n";
    std::cout << "energy_result_sum_eatom " << energy_sum << "\n";
    std::cout << "energy_reference " << ref_energy.at(0) << "\n";
    std::cout << "energy_abs_diff_compute_host " << std::abs(res.eng - ref_energy.at(0)) << "\n";
    std::cout << "energy_abs_diff_sum_eatom " << std::abs(energy_sum - ref_energy.at(0)) << "\n";
    std::cout << "force_max_abs " << force_stats.max_abs << "\n";
    std::cout << "force_rmse " << force_stats.rmse << "\n";
    std::cout << "mforce_nofold_max_abs " << mforce_nofold_stats.max_abs << "\n";
    std::cout << "mforce_nofold_rmse " << mforce_nofold_stats.rmse << "\n";
    std::cout << "mforce_max_abs " << mforce_stats.max_abs << "\n";
    std::cout << "mforce_rmse " << mforce_stats.rmse << "\n";
    std::cout << "mforce_ghost_only_max_abs " << mforce_ghost_only_stats.max_abs << "\n";
    std::cout << "mforce_ghost_only_rmse " << mforce_ghost_only_stats.rmse << "\n";
    std::cout << "descriptor_dim " << descriptor_dim << "\n";
    std::cout << "descriptor_atom0_max_abs " << descriptor_atom0_stats.max_abs << "\n";
    std::cout << "descriptor_atom0_rmse " << descriptor_atom0_stats.rmse << "\n";
    std::cout << "descriptor_structure_avg_max_abs " << descriptor_structure_stats.max_abs << "\n";
    std::cout << "descriptor_structure_avg_rmse " << descriptor_structure_stats.rmse << "\n";
    std::cout << "descriptor_best_atom " << (best_descriptor_atom + 1) << "\n";
    std::cout << "descriptor_best_atom_max_abs " << best_descriptor_stats.max_abs << "\n";
    std::cout << "descriptor_best_atom_rmse " << best_descriptor_stats.rmse << "\n";
    std::cout << "sp4_mag_first " << (sp4_mag_first ? 1 : 0) << "\n";
    std::cout << "no_ghosts " << (no_ghosts ? 1 : 0) << "\n";
    std::cout << "natoms_total " << natoms << "\n";
    std::cout << "ghost_atoms " << (natoms - nlocal) << "\n";

    for (int i = 0; i < std::min(natoms, 5); ++i) {
      std::cout << "atom " << (i + 1)
                << " ref_mforce " << ref_mforce[static_cast<std::size_t>(3 * i + 0)] << " "
                << ref_mforce[static_cast<std::size_t>(3 * i + 1)] << " "
                << ref_mforce[static_cast<std::size_t>(3 * i + 2)]
                << " got " << fm_local[static_cast<std::size_t>(3 * i + 0)] << " "
                << fm_local[static_cast<std::size_t>(3 * i + 1)] << " "
                << fm_local[static_cast<std::size_t>(3 * i + 2)] << "\n";
    }
    for (int d = 0; d < std::min<int>(static_cast<int>(descriptor_dim), 8); ++d) {
      std::cout << "descriptor atom0 d" << d
                << " ref " << ref_descriptor[static_cast<std::size_t>(d)]
                << " got " << descriptor_atom0_scaled[static_cast<std::size_t>(d)] << "\n";
    }
    for (int d = 0; d < std::min<int>(static_cast<int>(descriptor_dim), 8); ++d) {
      std::cout << "descriptor structure_avg d" << d
                << " ref " << ref_descriptor[static_cast<std::size_t>(d)]
                << " got " << descriptor_structure_avg[static_cast<std::size_t>(d)] << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "spinv2_backend_check error: " << e.what() << "\n";
    return 1;
  }
}
