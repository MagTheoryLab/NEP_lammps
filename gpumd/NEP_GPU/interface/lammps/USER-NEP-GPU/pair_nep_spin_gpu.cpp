/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   GPU NEP spin pair style (classic, non-Kokkos).
   Offloads NEP spin evaluation to a GPU implementation based on GPUMD.
------------------------------------------------------------------------- */

#include "pair_nep_spin_gpu.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include "mpi.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// Unified LAMMPS-facing NEP GPU model
#include "nep_gpu_lammps_model.h"

#ifdef LMP_KOKKOS
#include "kokkos.h"
#endif

using namespace LAMMPS_NS;
using namespace MathConst;

namespace {
inline double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline void cross3(const double a[3], const double b[3], double c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

inline void compute_box_thickness(const double h[9], double thickness[3])
{
  // h is row-major of column vectors: [ax bx cx ay by cy az bz cz]
  const double a[3] = {h[0], h[3], h[6]};
  const double b[3] = {h[1], h[4], h[7]};
  const double c[3] = {h[2], h[5], h[8]};
  double bxc[3], cxa[3], axb[3];
  cross3(b, c, bxc);
  cross3(c, a, cxa);
  cross3(a, b, axb);
  const double volume = std::abs(dot3(a, bxc));
  thickness[0] = volume / std::sqrt(dot3(bxc, bxc));
  thickness[1] = volume / std::sqrt(dot3(cxa, cxa));
  thickness[2] = volume / std::sqrt(dot3(axb, axb));
}

inline bool try_parse_int_env(const char* name, int& value_out)
{
  const char* s = std::getenv(name);
  if (!s) return false;
  while (*s && std::isspace(static_cast<unsigned char>(*s))) ++s;
  if (!*s) return false;
  char* end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s) return false;
  value_out = static_cast<int>(v);
  return true;
}

inline int count_csv_tokens(const char* s)
{
  if (!s) return 0;
  int count = 0;
  const char* p = s;
  while (*p) {
    while (*p == ',' || std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (!*p) break;
    ++count;
    while (*p && *p != ',') ++p;
  }
  return count;
}

inline int visible_gpu_count_from_env()
{
  int n = count_csv_tokens(std::getenv("CUDA_VISIBLE_DEVICES"));
  if (n > 0) return n;
  n = count_csv_tokens(std::getenv("SLURM_STEP_GPUS"));
  if (n > 0) return n;
  n = count_csv_tokens(std::getenv("SLURM_JOB_GPUS"));
  if (n > 0) return n;
  int m = 0;
  if (try_parse_int_env("SLURM_GPUS_ON_NODE", m) && m > 0) return m;
  return 0;
}

inline int local_rank_from_env_or_mpi(MPI_Comm world)
{
  int r = 0;
  if (try_parse_int_env("OMPI_COMM_WORLD_LOCAL_RANK", r)) return r;
  if (try_parse_int_env("MV2_COMM_WORLD_LOCAL_RANK", r)) return r;
  if (try_parse_int_env("MPT_LRANK", r)) return r;
  if (try_parse_int_env("PMI_LOCAL_RANK", r)) return r;
  if (try_parse_int_env("PALS_LOCAL_RANKID", r)) return r;
  if (try_parse_int_env("SLURM_LOCALID", r)) return r;
  if (try_parse_int_env("FLUX_TASK_LOCAL_ID", r)) return r;

#if defined(MPI_VERSION) && (MPI_VERSION >= 3)
  MPI_Comm local = MPI_COMM_NULL;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
  if (local != MPI_COMM_NULL) {
    MPI_Comm_rank(local, &r);
    MPI_Comm_free(&local);
    return r;
  }
#endif

  MPI_Comm_rank(world, &r);
  return r;
}

inline long long env_ll(const char* name, long long def)
{
  const char* s = std::getenv(name);
  if (!s || !*s) return def;
  char* end = nullptr;
  const long long v = std::strtoll(s, &end, 10);
  if (end == s) return def;
  return v;
}

inline std::uint64_t splitmix64(std::uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline std::uint64_t hash_u64(std::uint64_t h, std::uint64_t x)
{
  // XOR-folded splitmix for cheap, order-dependent hashing.
  return h ^ splitmix64(x + 0x9e3779b97f4a7c15ULL + (h << 1));
}

inline bool env_is_true(const char* name)
{
  const char* s = std::getenv(name);
  if (!s) return false;
  if (!*s) return true;
  if (std::strcmp(s, "1") == 0) return true;
  if (std::strcmp(s, "true") == 0) return true;
  if (std::strcmp(s, "TRUE") == 0) return true;
  if (std::strcmp(s, "yes") == 0) return true;
  if (std::strcmp(s, "YES") == 0) return true;
  return false;
}
} // namespace

PairNEPSpinGPU::PairNEPSpinGPU(LAMMPS *lmp) : Pair(lmp)
{
  allocated = 0;
  restartinfo = 0;
  manybody_flag = 1;
  one_coeff = 1;
  single_enable = 0;
}

PairNEPSpinGPU::~PairNEPSpinGPU()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] type_map;
    type_map = nullptr;
  }

  delete nep_model_spin_lmp;
  nep_model_spin_lmp = nullptr;
}

void PairNEPSpinGPU::allocate()
{
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  map = new int[n + 1];
  if (type_map) delete [] type_map;
  type_map = new int[n + 1];
  for (int i = 0; i <= n; ++i) type_map[i] = 0;
  allocated = 1;
}

void PairNEPSpinGPU::settings(int narg, char **arg)
{
  if (narg == 0) return;
  error->all(FLERR, "nep/spin/gpu: pair_style does not accept settings; atom->fm always stores H = -dE/dM in eV/μB");
}

void PairNEPSpinGPU::coeff(int narg, char **arg)
{
  if (narg < 4) error->all(FLERR, "Incorrect args for pair_style nep/spin/gpu");
  if (!allocated) allocate();

  model_filename_ = arg[2];
  std::string model_path_from_lammps = utils::get_potential_file_path(model_filename_);

  std::ifstream input(model_path_from_lammps.c_str());
  if (!input.is_open()) error->all(FLERR, "NEP_SPIN_GPU: Failed to open potential file");

  std::string line;
  std::getline(input, line);
  input.close();

  std::istringstream iss(line);
  std::string header;
  int num_nep_types = 0;
  iss >> header >> num_nep_types;
  if (num_nep_types <= 0) error->all(FLERR, "NEP_SPIN_GPU: invalid number of atom types in potential file");

  std::vector<std::string> nep_elements(num_nep_types);
  for (int n = 0; n < num_nep_types; ++n) {
    if (!(iss >> nep_elements[n])) error->all(FLERR, "NEP_SPIN_GPU: failed to read element symbols from potential file");
  }

  const int ntype = atom->ntypes;
  if (narg != 3 + ntype) {
    error->all(FLERR, "NEP_SPIN_GPU: pair_coeff must supply one element symbol per atom type");
  }

  for (int i = 1; i <= ntype; ++i) {
    const char *elem = arg[2 + i];
    int nep_index = -1;
    for (int n = 0; n < num_nep_types; ++n) {
      if (elem && nep_elements[n] == elem) { nep_index = n; break; }
    }
    if (nep_index < 0) {
      error->all(FLERR, "NEP_SPIN_GPU: pair_coeff element not present in potential file");
    }
    type_map[i] = nep_index;
  }

  const int max_atoms = (atom->nmax > 0) ? atom->nmax : ((atom->natoms > 0) ? atom->natoms : 1);

  // Select GPU device for the NEP spin backend *before* constructing the model,
  // because the constructor may allocate device buffers.
  //
  // Priority:
  //  1) Explicit env override: NEP_SPIN_GPU_DEVICE (or NEP_GPU_DEVICE) (index within CUDA_VISIBLE_DEVICES)
  //  2) If Kokkos GPU mode is active, trust Kokkos' device selection for this rank
  //  3) Otherwise, map per node-local MPI rank onto visible devices
  int requested_dev = -1;
  if (!try_parse_int_env("NEP_SPIN_GPU_DEVICE", requested_dev)) {
    (void) try_parse_int_env("NEP_GPU_DEVICE", requested_dev);
  }

  bool kokkos_gpu_active = false;
#ifdef LMP_KOKKOS
  if (lmp->kokkos && lmp->kokkos->kokkos_exists && (lmp->kokkos->ngpus > 0)) kokkos_gpu_active = true;
#endif

  if (requested_dev >= 0) {
    nep_gpu_lammps_set_device(requested_dev);
  } else if (!kokkos_gpu_active) {
    int nvis = visible_gpu_count_from_env();
    if (nvis <= 0) nvis = 1;
    const int local_rank = local_rank_from_env_or_mpi(lmp->world);
    const int dev = local_rank % nvis;
    nep_gpu_lammps_set_device(dev);
  }

  delete nep_model_spin_lmp;
  nep_model_spin_lmp = new NepGpuLammpsModel(model_path_from_lammps.c_str(), max_atoms);
  const NepGpuModelInfo& info = nep_model_spin_lmp->info();
  if (!info.needs_spin) {
    error->all(FLERR, "NEP_SPIN_GPU: non-spin model cannot be used with pair_style nep/spin/gpu.");
  }

  const double rc_r = info.rc_radial_max;
  const double rc_a = info.rc_angular_max;
  cutoff = std::max(rc_r, rc_a);
  cutoffsq_r_ = rc_r * rc_r;
  cutoffsq_a_ = rc_a * rc_a;

  for (int i = 1; i <= ntype; i++)
    for (int j = 1; j <= ntype; j++)
      cutsq[i][j] = cutoff * cutoff;
}

double PairNEPSpinGPU::init_one(int /*i*/, int /*j*/)
{
  return cutoff;
}

void PairNEPSpinGPU::init_style()
{
  if (!atom->sp || !atom->fm) error->all(FLERR, "Pair style nep/spin/gpu requires atom_style spin (sp/fm arrays missing)");
  // `error->all()` must be called collectively; do not gate on rank 0, or MPI may hang.
  if (force->newton_pair == 0) error->all(FLERR, "Pair style nep/spin/gpu requires newton pair on");

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

void PairNEPSpinGPU::compute(int eflag_in, int vflag_in)
{
  if (!nep_model_spin_lmp) error->all(FLERR, "NEP_SPIN_GPU: model not initialized; check pair_coeff");

  if (eflag_in || vflag_in) ev_setup(eflag_in, vflag_in);
  else ev_init(0, 0, 0);

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int natoms_total = nlocal + nghost;
  if (natoms_total <= 0) return;

  const bool no_time_integration = (modify->n_initial_integrate == 0 && modify->n_final_integrate == 0);

  // The backend adds into f/fm. LAMMPS does not reliably clear spin-force
  // buffers across all execution paths, so clear them explicitly here before
  // accumulating this step's contributions.
  if (atom->fm) {
    for (int i = 0; i < natoms_total; ++i) {
      atom->fm[i][0] = 0.0;
      atom->fm[i][1] = 0.0;
      atom->fm[i][2] = 0.0;
    }
  }
  if (atom->fm_long) {
    for (int i = 0; i < natoms_total; ++i) {
      atom->fm_long[i][0] = 0.0;
      atom->fm_long[i][1] = 0.0;
      atom->fm_long[i][2] = 0.0;
    }
  }
  if (no_time_integration) {
    for (int i = 0; i < natoms_total; ++i) {
      atom->f[i][0] = 0.0;
      atom->f[i][1] = 0.0;
      atom->f[i][2] = 0.0;
    }
  } else if (nghost > 0) {
    for (int i = nlocal; i < natoms_total; ++i) {
      atom->f[i][0] = 0.0;
      atom->f[i][1] = 0.0;
      atom->f[i][2] = 0.0;
    }
  }

  // This rank may have no owned atoms for certain processor grids. In that case
  // LAMMPS may still create ghost atoms, but there is no work to do because pair
  // computations loop over local atoms only. Avoid passing zero-length neighbor
  // buffers (std::vector::data() == nullptr) into the NEP backend.
  if (nlocal <= 0) return;

  const int mn_r = nep_model_spin_lmp->info().mn_radial;
  const int mn_a = nep_model_spin_lmp->info().mn_angular;
  if (mn_r <= 0 || mn_a <= 0) error->one(FLERR, "NEP_SPIN_GPU: invalid MN_* from model");

  static int cfg_inited = 0;
  static int sort_nl = 0;
  static int sort_by_tag = 1;
  if (!cfg_inited) {
    // Optional determinism aid: sort packed neighbor indices per atom.
    // - NEP_SPIN_GPU_LMP_SORT_NL=1 enables sorting (default: by tag if available).
    // - NEP_SPIN_GPU_LMP_SORT_NL=tag|index chooses key explicitly.
    const char* ssort = std::getenv("NEP_SPIN_GPU_LMP_SORT_NL");
    if (ssort) {
      sort_nl = 1;
      if (std::strcmp(ssort, "index") == 0) sort_by_tag = 0;
      if (std::strcmp(ssort, "tag") == 0) sort_by_tag = 1;
    }

    cfg_inited = 1;
  }

  const bool stable_ghost = env_is_true("NEP_SPIN_GPU_LMP_STABLE_GHOST");

  // Construct triclinic box matrix from LAMMPS domain.
  double h[9];
  h[0] = domain->h[0]; h[3] = 0.0;          h[6] = 0.0;
  h[1] = domain->h[5]; h[4] = domain->h[1]; h[7] = 0.0;
  h[2] = domain->h[4]; h[5] = domain->h[3]; h[8] = domain->h[2];

  // Avoid multi-image neighbor ambiguity:
  // neighbor cutoff (= model cutoff + neighbor skin) must not exceed half the periodic
  // box thickness in any periodic dimension. If violated, the neighbor list can include
  // multiple periodic images of the same atom, which will overflow MN_* and (more
  // importantly) produce invalid physics. Use LAMMPS 'replicate' and/or reduce skin.
  double box_thickness[3] = {0.0, 0.0, 0.0};
  bool box_violation = false;
  double rc_neigh = 0.0;
  {
    const bool strict_box = (std::getenv("NEP_GPU_LMP_STRICT_BOX") != nullptr);
    const double rc_model = std::max(nep_model_spin_lmp->info().rc_radial_max, nep_model_spin_lmp->info().rc_angular_max);
    rc_neigh = rc_model + neighbor->skin;
    compute_box_thickness(h, box_thickness);
    box_violation =
      (domain->xperiodic && rc_neigh > 0.5 * box_thickness[0]) ||
      (domain->yperiodic && rc_neigh > 0.5 * box_thickness[1]) ||
      (domain->zperiodic && rc_neigh > 0.5 * box_thickness[2]);
    if (box_violation) {
      if (strict_box) {
        error->all(FLERR, "NEP_SPIN_GPU: cutoff+skin exceeds half periodic box thickness; replicate the cell (LAMMPS 'replicate') and/or reduce neighbor skin.");
      } else if (comm->me == 0 && screen) {
        static int warned_once = 0;
        if (!warned_once) {
          warned_once = 1;
          std::fprintf(screen,
            "WARNING: NEP_SPIN_GPU: cutoff+skin exceeds half periodic box thickness; neighbor list will include multiple periodic images (results likely invalid / MN_* may overflow). "
            "Replicate the cell or reduce neighbor skin. Set NEP_GPU_LMP_STRICT_BOX=1 to make this a fatal error.\n");
        }
      }
    }
  }

  type_host_.resize(natoms_total);
  xyz_host_.resize(3 * natoms_total);
  sp4_host_.resize(4 * natoms_total);
  f_host_.resize(3 * natoms_total);
  fm_host_.resize(3 * natoms_total);
  // Defensive: the NEP spin backend may accumulate into the provided output buffers
  // (e.g., using atomic adds).  When std::vector is resized without reallocation,
  // existing values are preserved, so we must explicitly zero outputs each step.
  // This matters for ghost entries in particular: we accumulate f/fm on local+ghost
  // atoms and rely on LAMMPS reverse_comm to move ghost contributions to owners.
  if (!f_host_.empty()) std::memset(f_host_.data(), 0, sizeof(double) * f_host_.size());
  if (!fm_host_.empty()) std::memset(fm_host_.data(), 0, sizeof(double) * fm_host_.size());

  nn_radial_.assign(nlocal, 0);
  nn_angular_.assign(nlocal, 0);
  nl_radial_.resize(static_cast<size_t>(nlocal) * mn_r);
  nl_angular_.resize(static_cast<size_t>(nlocal) * mn_a);

  const bool force_energy = env_is_true("NEP_SPIN_GPU_LMP_FORCE_ENERGY");
  const bool force_virial = env_is_true("NEP_SPIN_GPU_LMP_FORCE_VIRIAL");
  const bool need_energy = force_energy || (eflag_in != 0);
  const bool need_virial = force_virial || (vflag_in != 0);

  // Optional workaround/debug: use per-atom outputs to form totals.
  // If enabled we must request per-atom buffers even when LAMMPS does not.
  const bool use_eatom_sum = env_is_true("NEP_SPIN_GPU_LMP_USE_EATOM_SUM");
  const bool use_vatom_sum = env_is_true("NEP_SPIN_GPU_LMP_USE_VATOM_SUM");

  // Provide per-atom potential/virial buffers when needed, independent of whether
  // LAMMPS requested per-atom outputs (required for the optional "use per-atom sums
  // for totals" workaround).
  const bool want_eatom = (eflag_atom != 0) || need_energy || use_eatom_sum;
  const bool want_vatom = (vflag_atom != 0) || (cvflag_atom != 0) || need_virial || (eflag_atom != 0) || use_vatom_sum;
  if (want_eatom) {
    eatom_host_.resize(nlocal);
    std::fill(eatom_host_.begin(), eatom_host_.end(), 0.0);
  }
  if (want_vatom) {
    vatom_host_.resize(9 * nlocal);
    std::fill(vatom_host_.begin(), vatom_host_.end(), 0.0);
  }

  // Optional: stabilize ghost ordering by atom tag (keeps local indices unchanged),
  // to reduce sensitivity to border/exchange ordering when the GPU backend uses
  // atomic accumulation.
  std::vector<int> old_to_new;
  std::vector<int> new_to_old;
  if (stable_ghost) {
    old_to_new.resize(natoms_total);
    new_to_old.resize(natoms_total);
    std::iota(old_to_new.begin(), old_to_new.end(), 0);
    std::iota(new_to_old.begin(), new_to_old.end(), 0);

    std::vector<int> ghosts;
    ghosts.reserve(std::max(0, natoms_total - nlocal));
    for (int old = nlocal; old < natoms_total; ++old) ghosts.push_back(old);

    const bool have_tag = atom->tag_enable && atom->tag;
    std::stable_sort(ghosts.begin(), ghosts.end(), [&](int a, int b) {
      if (have_tag) {
        const tagint ta = atom->tag[a];
        const tagint tb = atom->tag[b];
        if (ta != tb) return ta < tb;
      }
      return a < b;
    });

    int next = nlocal;
    for (int old : ghosts) {
      const int neu = next++;
      old_to_new[old] = neu;
      new_to_old[neu] = old;
    }
  }

  // Gather mapped types, positions, and spins (local+ghost) into the internal
  // ordering expected by the backend. When stable_ghost is enabled, ghosts are
  // reordered in our internal buffers only; local atom indices remain unchanged.
  for (int neu = 0; neu < natoms_total; ++neu) {
    const int old = stable_ghost ? new_to_old[neu] : neu;
    const int t_lmp = atom->type[old];
    if (t_lmp < 1 || t_lmp > atom->ntypes) error->one(FLERR, "NEP_SPIN_GPU: invalid atom type index");
    type_host_[neu] = type_map[t_lmp];

    xyz_host_[3 * neu + 0] = atom->x[old][0];
    xyz_host_[3 * neu + 1] = atom->x[old][1];
    xyz_host_[3 * neu + 2] = atom->x[old][2];

    sp4_host_[4 * neu + 0] = atom->sp[old][0];
    sp4_host_[4 * neu + 1] = atom->sp[old][1];
    sp4_host_[4 * neu + 2] = atom->sp[old][2];
    sp4_host_[4 * neu + 3] = atom->sp[old][3];
  }

  // Pack NN/NL from the full neighbor list.
  // IMPORTANT: NEP trusts the LAMMPS full neighbor list and the ghost coordinates
  // that go with it. For periodic boundaries, LAMMPS stores ghost atoms at the
  // correct periodic image, so dx = x[j]-x[i] is already the displacement for the
  // neighbor that is actually in the neighbor list.
  //
  // Do NOT apply an additional minimum-image convention here; it can fold distinct
  // periodic images onto the same displacement and corrupt the cutoff-filtering logic
  // (especially in small boxes / when cutoff+skin approaches half box thickness).
  // We must filter separately for radial vs angular cutoffs to avoid overflowing MN_*.
  if (!list) error->one(FLERR, "NEP_SPIN_GPU: neighbor list is null (init_style not called?)");
  const int inum = list->inum;
  if (inum != nlocal) error->one(FLERR, "NEP_SPIN_GPU: expected inum == nlocal for full neighbor list");

  const int *ilist = list->ilist;
  bool overflow_mn = false;
  int max_nr = 0;
  int max_na = 0;
  int first_i = -1;
  tagint first_tag = 0;
  int first_nr = 0;
  int first_na = 0;
  for (int ii = 0; ii < inum; ++ii) {
    const int i = ilist[ii];
    if (i < 0 || i >= nlocal) error->one(FLERR, "NEP_SPIN_GPU: ilist contained non-local indices (expected 0 <= i < nlocal)");
    const int jnum = list->numneigh[i];
    const int *jlist = list->firstneigh[i];

    int nr = 0;
    int na = 0;
    bool overflow_r = false;
    bool overflow_a = false;

    for (int jj = 0; jj < jnum; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;
      if (j < 0 || j >= natoms_total) continue;

      double dx = atom->x[j][0] - atom->x[i][0];
      double dy = atom->x[j][1] - atom->x[i][1];
      double dz = atom->x[j][2] - atom->x[i][2];
      const double rsq = dx * dx + dy * dy + dz * dz;

      const int j_neu = stable_ghost ? old_to_new[j] : j;
      if (rsq < cutoffsq_r_) {
        if (nr < mn_r) nl_radial_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * nr] = j_neu;
        else overflow_r = true;
        ++nr;
      }
      if (rsq < cutoffsq_a_) {
        if (na < mn_a) nl_angular_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * na] = j_neu;
        else overflow_a = true;
        ++na;
      }

      // Only apply the early-out optimization when we are still within MN_*.
      // If we have overflowed, continue scanning to get accurate counts for diagnostics.
      if (!overflow_r && !overflow_a && nr >= mn_r && na >= mn_a) break;
    }

    nn_radial_[i] = std::min(nr, mn_r);
    nn_angular_[i] = std::min(na, mn_a);
    if (overflow_r || overflow_a) {
      overflow_mn = true;
      max_nr = std::max(max_nr, nr);
      max_na = std::max(max_na, na);
      if (first_i < 0) {
        first_i = i;
        first_nr = nr;
        first_na = na;
        first_tag = (atom->tag_enable && atom->tag) ? atom->tag[i] : 0;
      }
    }
  }

  if (overflow_mn) {
    std::ostringstream msg;
    msg << "NEP_SPIN_GPU: neighbor list exceeds MN_* from model at step " << update->ntimestep
        << " (max radial neighbors " << max_nr << " > MN_radial " << mn_r
        << ", max angular neighbors " << max_na << " > MN_angular " << mn_a << "). "
        << "First offender: i=" << first_i;
    if (first_tag) msg << " tag=" << first_tag;
    msg << " (nr=" << first_nr << ", na=" << first_na << "). "
        << "Model cutoff_r=" << std::sqrt(cutoffsq_r_) << " cutoff_a=" << std::sqrt(cutoffsq_a_)
        << ", neighbor skin=" << neighbor->skin;
    if (rc_neigh > 0.0) msg << " (cutoff_max+skin=" << rc_neigh << ")";
    if (box_violation) {
      msg << ". Also: cutoff+skin exceeds half periodic box thickness (thickness="
          << box_thickness[0] << "," << box_thickness[1] << "," << box_thickness[2]
          << "); replicate the cell or reduce skin.";
    } else {
      msg << ". Increase MN_* in the NEP model or reduce density/cutoffs/skin.";
    }
    error->one(FLERR, msg.str().c_str());
  }

  // Optional determinism aid: sort neighbors per atom (after packing).
  // This can make results more stable across neighbor rebuilds and MPI decompositions
  // if the backend is order-dependent (e.g., uses atomic adds).
  if (sort_nl) {
    const bool have_tag = atom->tag_enable && (atom->tag != nullptr);
    const bool by_tag = sort_by_tag && have_tag;
    auto tag_of = [&](int idx) -> tagint {
      return stable_ghost ? atom->tag[new_to_old[idx]] : atom->tag[idx];
    };
    for (int i = 0; i < nlocal; ++i) {
      const int nr = nn_radial_[i];
      const int na = nn_angular_[i];
      if (nr > 1) {
        if (by_tag) {
          std::vector<std::pair<tagint, int>> tmp;
          tmp.reserve(nr);
          for (int k = 0; k < nr; ++k) {
            const int j = nl_radial_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * k];
            tmp.emplace_back(tag_of(j), j);
          }
          std::stable_sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
          for (int k = 0; k < nr; ++k) nl_radial_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * k] = tmp[k].second;
        } else {
          std::vector<int> tmp(nr);
          for (int k = 0; k < nr; ++k) tmp[k] = nl_radial_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * k];
          std::stable_sort(tmp.begin(), tmp.end());
          for (int k = 0; k < nr; ++k) nl_radial_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * k] = tmp[k];
        }
      }
      if (na > 1) {
        if (by_tag) {
          std::vector<std::pair<tagint, int>> tmp;
          tmp.reserve(na);
          for (int k = 0; k < na; ++k) {
            const int j = nl_angular_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * k];
            tmp.emplace_back(tag_of(j), j);
          }
          std::stable_sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
          for (int k = 0; k < na; ++k) nl_angular_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * k] = tmp[k].second;
        } else {
          std::vector<int> tmp(na);
          for (int k = 0; k < na; ++k) tmp[k] = nl_angular_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * k];
          std::stable_sort(tmp.begin(), tmp.end());
          for (int k = 0; k < na; ++k) nl_angular_[static_cast<size_t>(i) + static_cast<size_t>(nlocal) * k] = tmp[k];
        }
      }
    }
  }

  NepGpuLammpsSystemHost sys;
  sys.natoms = natoms_total;
  sys.type = type_host_.data();
  sys.xyz = xyz_host_.data();
  sys.sp4 = sp4_host_.data();
  for (int k = 0; k < 9; ++k) sys.h[k] = h[k];
  sys.pbc_x = domain->xperiodic ? 1 : 0;
  sys.pbc_y = domain->yperiodic ? 1 : 0;
  sys.pbc_z = domain->zperiodic ? 1 : 0;

  NepGpuLammpsNeighborsHost nb;
  nb.NN_radial = nn_radial_.data();
  nb.NL_radial = nl_radial_.data();
  nb.NN_angular = nn_angular_.data();
  nb.NL_angular = nl_angular_.data();

  NepGpuLammpsResultHost res;
  res.f = f_host_.data();
  res.fm = fm_host_.data();
  res.eatom = want_eatom ? eatom_host_.data() : nullptr;
  res.vatom = want_vatom ? vatom_host_.data() : nullptr;
  res.want_virial_raw9 = false;
  res.eng = 0.0;
  res.virial[0] = 0.0;
  res.virial[1] = 0.0;
  res.virial[2] = 0.0;
  res.virial[3] = 0.0;
  res.virial[4] = 0.0;
  res.virial[5] = 0.0;
  // Always expose field H = -dE/dM in eV/μB.
  res.inv_hbar = 1.0;

  nep_model_spin_lmp->compute_host(sys, nlocal, nb, res, need_energy, need_virial);

  // Optional workaround: use sums of per-atom outputs (local atoms only) for totals.
  // If the backend's `res.eng` / `res.virial` accidentally include ghost contributions,
  // this makes totals invariant under atom migration / domain decomposition.
  double eatom_sum_local = 0.0;
  double vatom_sum_local[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (use_eatom_sum && want_eatom && !eatom_host_.empty()) {
    for (int i = 0; i < nlocal; ++i) eatom_sum_local += eatom_host_[i];
  }
  if (use_vatom_sum && want_vatom && vatom_host_.size() >= static_cast<size_t>(9 * nlocal)) {
    for (int i = 0; i < nlocal; ++i) {
      const int idx = 9 * i;
      vatom_sum_local[0] += vatom_host_[idx + 0];
      vatom_sum_local[1] += vatom_host_[idx + 1];
      vatom_sum_local[2] += vatom_host_[idx + 2];
      vatom_sum_local[3] += vatom_host_[idx + 3];
      vatom_sum_local[4] += vatom_host_[idx + 4];
      vatom_sum_local[5] += vatom_host_[idx + 5];
    }
  }

  // Accumulate forces and fm on local + ghost atoms; newton pair on means
  // LAMMPS will reverse-communicate both f and fm via AtomVecSpin.
  for (int neu = 0; neu < natoms_total; ++neu) {
    const int old = stable_ghost ? new_to_old[neu] : neu;
    atom->f[old][0] += res.f[3 * neu + 0];
    atom->f[old][1] += res.f[3 * neu + 1];
    atom->f[old][2] += res.f[3 * neu + 2];

    atom->fm[old][0] += res.fm[3 * neu + 0];
    atom->fm[old][1] += res.fm[3 * neu + 1];
    atom->fm[old][2] += res.fm[3 * neu + 2];
  }

  if (eflag_in) {
    eng_vdwl += use_eatom_sum ? eatom_sum_local : res.eng;
  }
  if (vflag_in) {
    if (use_vatom_sum) {
      virial[0] += vatom_sum_local[0];
      virial[1] += vatom_sum_local[1];
      virial[2] += vatom_sum_local[2];
      virial[3] += vatom_sum_local[3];
      virial[4] += vatom_sum_local[4];
      virial[5] += vatom_sum_local[5];
    } else {
      virial[0] += res.virial[0];
      virial[1] += res.virial[1];
      virial[2] += res.virial[2];
      virial[3] += res.virial[3];
      virial[4] += res.virial[4];
      virial[5] += res.virial[5];
    }
  }

  if (eflag_atom) {
    for (int i = 0; i < nlocal; ++i) eatom[i] += eatom_host_[i];
  }
  // Per-atom virial outputs:
  // - Standard per-atom virial uses vatom[i][0..5] (xx,yy,zz,xy,xz,yz).
  // - Centroid stress compute uses cvatom[i][0..8] (full 3x3: xx..zy).
  if (vflag_atom && !vatom) error->one(FLERR, "NEP_SPIN_GPU: vflag_atom set but vatom is null (internal allocation mismatch)");
  if (cvflag_atom && !cvatom) error->one(FLERR, "NEP_SPIN_GPU: cvflag_atom set but cvatom is null (internal allocation mismatch)");
  if (vflag_atom && vatom) {
    for (int i = 0; i < nlocal; ++i) {
      const int idx = 9 * i;
      vatom[i][0] += vatom_host_[idx + 0];
      vatom[i][1] += vatom_host_[idx + 1];
      vatom[i][2] += vatom_host_[idx + 2];
      vatom[i][3] += vatom_host_[idx + 3];
      vatom[i][4] += vatom_host_[idx + 4];
      vatom[i][5] += vatom_host_[idx + 5];
    }
  }

  if (cvflag_atom && cvatom) {
    for (int i = 0; i < nlocal; ++i) {
      const int idx = 9 * i;
      for (int k = 0; k < 9; ++k) cvatom[i][k] += vatom_host_[idx + k];
    }
  }

}
