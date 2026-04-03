/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   GPU NEP pair style (experimental).
   Offloads NEP evaluation to a GPU implementation based on GPUMD.
   Contributing author: (your name here)
------------------------------------------------------------------------- */

#include "pair_nep_gpu.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "utils.h"
#include "special.h"
#include "update.h"
#include "mpi.h"

#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Unified LAMMPS-facing NEP GPU model
#include "nep_gpu_lammps_model.h"

#ifdef LMP_KOKKOS
#include "kokkos.h"
#endif

// This macro can be updated based on LAMMPS's version.h (optional).
#define LAMMPS_VERSION_NUMBER 20220324

using namespace LAMMPS_NS;

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
} // namespace

PairNEPGPU::PairNEPGPU(LAMMPS *lmp) : Pair(lmp)
{
#if LAMMPS_VERSION_NUMBER >= 20201130
  centroidstressflag = CENTROID_AVAIL;
#else
  centroidstressflag = 2;
#endif

  restartinfo = 0;
  manybody_flag = 1;

  single_enable = 0;
  one_coeff = 1;

  inited = false;
  cache_nl_ = (std::getenv("NEP_GPU_LMP_CACHE_NL") != nullptr);
  use_skin_neighbors_ = cache_nl_ || (std::getenv("NEP_GPU_LMP_USE_SKIN") != nullptr);
  allocated = 0;
  cutoff = 0.0;
  cutoff_radial = 0.0;
  cutoff_angular = 0.0;
  cutoff_zbl_outer = 0.0;
  cutoffsq = 0.0;
  cutoffsq_radial = 0.0;
  cutoffsq_angular = 0.0;
  cutoffsq_zbl_ = 0.0;
  nep_model_lmp = nullptr;
  type_map = nullptr;
}

PairNEPGPU::~PairNEPGPU()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] type_map;
    type_map = nullptr;
  }

  delete nep_model_lmp;
  nep_model_lmp = nullptr;
}

void PairNEPGPU::allocate()
{
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  map = new int[n + 1];
  type_map = new int[n + 1];
  allocated = 1;
}

void PairNEPGPU::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg < 4) {
    error->all(FLERR, "Incorrect args for pair_style nep/gpu");
  }

  bool is_rank_0 = (comm->me == 0);
  model_filename = arg[2];

  // Map element symbols supplied in pair_coeff to NEP type indices by
  // reading the first line of the NEP file. We do not rely on LAMMPS's
  // internal "elements" array, but instead use the symbols directly
  // from the pair_coeff arguments.
  std::string model_path_from_lammps = utils::get_potential_file_path(model_filename);

  std::ifstream input(model_path_from_lammps.c_str());
  if (!input.is_open()) {
    error->all(FLERR, "NEP_GPU: Failed to open potential file");
  }

  std::string line;
  std::getline(input, line);
  input.close();

  std::istringstream iss(line);
  std::string header;
  int num_nep_types = 0;
  iss >> header >> num_nep_types;
  if (num_nep_types <= 0) {
    error->all(FLERR, "NEP_GPU: invalid number of atom types in NEP file");
  }

  std::vector<std::string> nep_elements(num_nep_types);
  for (int n = 0; n < num_nep_types; ++n) {
    if (!(iss >> nep_elements[n])) {
      error->all(FLERR, "NEP_GPU: failed to read element symbols from NEP file");
    }
  }

  // Ensure the number of element symbols in pair_coeff matches LAMMPS types.
  const int ntype = atom->ntypes;
  if (narg != 3 + ntype) {
    error->all(FLERR, "NEP_GPU: pair_coeff must supply one element symbol per atom type");
  }

  // Build LAMMPS type -> NEP type mapping using the element names provided
  // in the pair_coeff command. For a line like
  //   pair_coeff * * nep_file C C
  // arg[3] corresponds to type 1, arg[4] to type 2, etc.
  for (int i = 1; i <= ntype; ++i) {
    const char *elem = arg[2 + i];
    int nep_index = -1;
    for (int n = 0; n < num_nep_types; ++n) {
      if (elem && nep_elements[n] == elem) {
        nep_index = n;
        break;
      }
    }
    if (nep_index < 0) {
      if (is_rank_0) {
        char msg[256];
        std::snprintf(
          msg,
          sizeof(msg),
          "NEP_GPU: element %s (LAMMPS type %d) not present in NEP file",
          elem ? elem : "NULL",
          i);
        error->all(FLERR, msg);
      } else {
        error->all(FLERR, "NEP_GPU: element not present in NEP file");
      }
    }
    type_map[i] = nep_index;
  }

  // Construct the GPU NEP model. Use atom->nmax as a conservative upper bound
  // for the number of atoms on this rank.
  int max_atoms = atom->nmax;
  if (max_atoms <= 0) max_atoms = atom->natoms;
  if (max_atoms <= 0) max_atoms = 1;

  if (nep_model_lmp) {
    delete nep_model_lmp;
    nep_model_lmp = nullptr;
  }

  // Select GPU device for the NEP backend *before* constructing the model,
  // because the constructor may allocate device buffers.
  //
  // Priority:
  //  1) Explicit env override: NEP_GPU_DEVICE (index within CUDA_VISIBLE_DEVICES)
  //  2) If Kokkos GPU mode is active, trust Kokkos' device selection for this rank
  //  3) Otherwise, map per node-local MPI rank onto visible devices
  int requested_dev = -1;
  (void) try_parse_int_env("NEP_GPU_DEVICE", requested_dev);

  bool kokkos_gpu_active = false;
#ifdef LMP_KOKKOS
  if (lmp->kokkos && lmp->kokkos->kokkos_exists && (lmp->kokkos->ngpus > 0)) kokkos_gpu_active = true;
#endif

  if (requested_dev >= 0) {
    nep_gpu_lammps_set_device(requested_dev);
  } else if (!kokkos_gpu_active) {
    int nvis = visible_gpu_count_from_env();
    if (nvis <= 0) nvis = 1;
    const int local_rank = local_rank_from_env_or_mpi(world);
    const int dev = local_rank % nvis;
    nep_gpu_lammps_set_device(dev);
  }

  nep_model_lmp = new NepGpuLammpsModel(model_path_from_lammps.c_str(), max_atoms);

  const NepGpuModelInfo& info = nep_model_lmp->info();
  if (info.needs_spin) {
    error->all(FLERR, "NEP_GPU: spin model cannot be used with pair_style nep/gpu; use nep/spin/gpu.");
  }
  if (info.kind != NepGpuModelKind::potential && info.kind != NepGpuModelKind::temperature) {
    error->all(FLERR, "NEP_GPU: this pair style only supports potential-like NEP models.");
  }
  rc_radial_by_type_ = info.rc_radial_by_type;
  rc_angular_by_type_ = info.rc_angular_by_type;

  // Get cutoffs from NEP model:
  // - cutoff_radial / cutoff_angular are used to filter packed NL/NN separately.
  // - cutoff (max) is reported to LAMMPS to size neighbor lists.
  cutoff_radial = info.rc_radial_max;
  cutoff_angular = info.rc_angular_max;
  cutoff_zbl_outer = info.zbl_outer_max;
  cutoff = std::max(std::max(cutoff_radial, cutoff_angular), cutoff_zbl_outer);
  cutoffsq = cutoff * cutoff;
  cutoffsq_radial = cutoff_radial * cutoff_radial;
  cutoffsq_angular = cutoff_angular * cutoff_angular;
  cutoffsq_zbl_ = cutoff_zbl_outer * cutoff_zbl_outer;
  int n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      cutsq[i][j] = cutoffsq;
}

void PairNEPGPU::settings(int narg, char **arg)
{
  // For convenience and script compatibility with spin-enabled workflows, we accept:
  //   pair_style nep/gpu fm_units frequency|energy
  // even though non-spin NEP does not produce fm; the option is ignored here.
  if (narg == 0) return;
  if (narg == 2 && std::strcmp(arg[0], "fm_units") == 0) {
    if (std::strcmp(arg[1], "frequency") == 0) return;
    if (std::strcmp(arg[1], "energy") == 0) return;
    error->all(FLERR, "nep/gpu: fm_units must be 'frequency' or 'energy'");
  }
  error->all(FLERR, "Illegal pair_style command; usage: pair_style nep/gpu [fm_units frequency|energy]");
}

void PairNEPGPU::init_style()
{
#if LAMMPS_VERSION_NUMBER >= 20220324
  // Request a full neighbor list to match NEP's manybody nature, even though
  // the GPU NEP implementation currently rebuilds its own neighbor list.
  neighbor->add_request(this, NeighConst::REQ_FULL);
#else
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
#endif
}

double PairNEPGPU::init_one(int i, int j)
{
  return cutoff;
}

void PairNEPGPU::compute(int eflag, int vflag)
{
  if (!nep_model_lmp) {
    error->all(FLERR, "NEP_GPU: model not initialized; check pair_coeff");
  }

  if (eflag || vflag) {
    ev_setup(eflag, vflag);
  }

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int natoms_total = nlocal + nghost;
  if (natoms_total <= 0) return;

  // This rank may have no owned atoms for certain processor grids. In that case
  // LAMMPS may still create ghost atoms, but there is no work to do because pair
  // computations loop over local atoms only. Avoid passing zero-length neighbor
  // buffers (std::vector::data() == nullptr) into the NEP backend.
  if (nlocal <= 0) return;

  type_host.resize(natoms_total);
  xyz_host.resize(3 * natoms_total);
  // Store forces for local + ghost atoms, so that we can accumulate ghost
  // contributions and let LAMMPS reverse_comm move them to owners, matching
  // the CPU NEP semantics.
  f_host.resize(3 * natoms_total);
  eatom_host.resize(nlocal);
  vatom_host.resize(9 * nlocal);

  const int mn_r = nep_model_lmp->info().mn_radial;
  const int mn_a = static_cast<int>(rc_angular_by_type_.empty() ? 0 : nep_model_lmp->info().mn_angular);
  // Only local atoms [0,nlocal) are evaluated on this rank, so keep compact
  // neighbor buffers with stride = nlocal to reduce packing/copy overhead.
  nn_radial_host.resize(nlocal);
  nn_angular_host.resize(nlocal);
  nl_radial_host.resize(static_cast<size_t>(nlocal) * mn_r);
  nl_angular_host.resize(static_cast<size_t>(nlocal) * mn_a);

  const bool map_ghost_to_owner = false; // 非磁版本保留原始 j 索引
  bool use_skin_neighbors_step = use_skin_neighbors_;
  if (use_skin_neighbors_step || cache_nl_) {
    // This pair style uses fixed-size neighbor arrays based on the model MN_*.
    // Packing the full (skin) neighbor list is not safe when the skin neighbor
    // count exceeds MN_*, and caching a cutoff-filtered list is not correct.
    // For correctness, fall back to per-step cutoff filtering.
    if (comm->me == 0 && screen) {
      static int warned_once = 0;
      if (!warned_once) {
        warned_once = 1;
        std::fprintf(screen,
          "WARNING: NEP_GPU: NEP_GPU_LMP_CACHE_NL/NEP_GPU_LMP_USE_SKIN are not compatible with fixed MN_*; falling back to per-step cutoff filtering.\n");
      }
    }
    use_skin_neighbors_step = false;
    cache_nl_ = false;
    nl_valid_ = false;
  }

  // Decide whether to (re)build packed NL/NN this step.
  // - Default: rebuild every step (matches the original implementation).
  // - If NEP_GPU_LMP_CACHE_NL is set: rebuild only when LAMMPS has rebuilt the
  //   neighbor list (neighbor->ago == 0), and pack the full skin neighbors
  //   (no cutoff filter) so the packed NL remains valid between rebuilds.
  bool rebuild_nl = true;
  if (cache_nl_) {
    rebuild_nl = (!nl_valid_) || (neighbor->ago == 0) ||
                 (cached_natoms_total_ != natoms_total) || (cached_nlocal_ != nlocal) ||
                 (cached_mn_r_ != mn_r) || (cached_mn_a_ != mn_a);
  }
  if (rebuild_nl) {
    // Only local atoms are evaluated on this rank (0..nlocal-1). Avoid clearing
    // the full local+ghost neighbor buffers each timestep.
    std::fill(nn_radial_host.begin(), nn_radial_host.end(), 0);
    std::fill(nn_angular_host.begin(), nn_angular_host.end(), 0);
  }

  // Map LAMMPS types to NEP types and gather positions (local + ghost).
  for (int i = 0; i < natoms_total; ++i) {
    int t_lmp = atom->type[i];
    if (t_lmp < 1 || t_lmp > atom->ntypes) {
      error->one(FLERR, "NEP_GPU: invalid atom type index");
    }
    int t_nep = type_map[t_lmp];
    type_host[i] = t_nep;

    xyz_host[3 * i + 0] = atom->x[i][0];
    xyz_host[3 * i + 1] = atom->x[i][1];
    xyz_host[3 * i + 2] = atom->x[i][2];
  }

  // Construct triclinic box matrix from LAMMPS domain, consistent with CPU path.
  double h[9];
  h[0] = domain->h[0]; // a_x = xprd
  h[3] = 0.0;          // a_y
  h[6] = 0.0;          // a_z

  h[1] = domain->h[5]; // b_x = xy
  h[4] = domain->h[1]; // b_y = yprd
  h[7] = 0.0;          // b_z

  h[2] = domain->h[4]; // c_x = xz
  h[5] = domain->h[3]; // c_y = yz
  h[8] = domain->h[2]; // c_z = zprd

  // "Small box" consistency check: if cutoff is larger than half the periodic box
  // thickness, the neighbor list will include multiple periodic images (including
  // self-images). That can overflow MN_* and usually does not match intended NEP usage.
  double box_thickness[3] = {0.0, 0.0, 0.0};
  bool box_violation = false;
  double rc_neigh = 0.0;
  {
    const bool strict_box = (std::getenv("NEP_GPU_LMP_STRICT_BOX") != nullptr);
    const NepGpuModelInfo& info = nep_model_lmp->info();
    const double rc_model = std::max(
      std::max(info.rc_radial_max, info.rc_angular_max),
      info.zbl_outer_max);
    rc_neigh = rc_model + neighbor->skin;
    compute_box_thickness(h, box_thickness);
    box_violation =
      (domain->xperiodic && rc_neigh > 0.5 * box_thickness[0]) ||
      (domain->yperiodic && rc_neigh > 0.5 * box_thickness[1]) ||
      (domain->zperiodic && rc_neigh > 0.5 * box_thickness[2]);
    if (box_violation) {
      if (strict_box) {
        error->all(FLERR, "NEP_GPU: cutoff+skin exceeds half periodic box thickness; replicate the cell (LAMMPS 'replicate') and/or reduce neighbor skin.");
      } else if (comm->me == 0 && screen) {
        static int warned_once = 0;
        if (!warned_once) {
          warned_once = 1;
          std::fprintf(screen,
            "WARNING: NEP_GPU: cutoff+skin exceeds half periodic box thickness; neighbor list will include multiple periodic images (results likely invalid / MN_* may overflow). "
            "Replicate the cell or reduce neighbor skin. Set NEP_GPU_LMP_STRICT_BOX=1 to make this a fatal error.\n");
        }
      }
    }
  }

  if (rebuild_nl) {
    // Build compact neighbor lists from LAMMPS neighbor list (full).
    bool overflow_mn = false;
    int max_nr = 0;
    int max_na = 0;
    int first_i = -1;
    tagint first_tag = 0;
    int first_nr = 0;
    int first_na = 0;
    for (int ii = 0; ii < list->inum; ++ii) {
      int i = list->ilist[ii];
      if (i < 0 || i >= nlocal) error->one(FLERR, "NEP_GPU: expected full neighbor list with ilist in [0,nlocal)");
      int nr = 0;
      int na = 0;
      bool overflow_r = false;
      bool overflow_a = false;
      const int jnum = list->numneigh[i];
      const int* jlist = list->firstneigh[i];
      for (int jj = 0; jj < jnum; ++jj) {
        int jraw = jlist[jj];
        int j = jraw & NEIGHMASK;
        if (j < 0 || j >= natoms_total) continue;

        // Distance check against radial/angular cutoffs (neighbor list has skin).
        // Note: ghost atoms are already shifted by LAMMPS for the active periodic image,
        // so dx = x[j]-x[i] is the correct displacement for this neighbor entry.
        const double dx = atom->x[j][0] - atom->x[i][0];
        const double dy = atom->x[j][1] - atom->x[i][1];
        const double dz = atom->x[j][2] - atom->x[i][2];
        const double rsq = dx * dx + dy * dy + dz * dz;

        int j_store = j;
        if (map_ghost_to_owner && j >= nlocal) {
          const int owner = atom->map(atom->tag[j]);
          if (owner >= 0) j_store = owner;
        }

        const int type_i = type_host[i];
        const int type_j = type_host[j];
        const double rc_radial_ij = 0.5 * (rc_radial_by_type_[type_i] + rc_radial_by_type_[type_j]);
        const double rc_angular_ij = 0.5 * (rc_angular_by_type_[type_i] + rc_angular_by_type_[type_j]);
        if (rsq <= rc_radial_ij * rc_radial_ij) {
          if (nr < mn_r) nl_radial_host[i + nlocal * nr] = j_store;
          else overflow_r = true;
          ++nr;
        }
        if (rsq <= rc_angular_ij * rc_angular_ij) {
          if (na < mn_a) nl_angular_host[i + nlocal * na] = j_store;
          else overflow_a = true;
          ++na;
        }

        // Only apply the early-out optimization when we are still within MN_*.
        // If we have overflowed, continue scanning to get accurate counts for diagnostics.
        if (!overflow_r && !overflow_a && nr >= mn_r && na >= mn_a) break;
      }
      nn_radial_host[i] = std::min(nr, mn_r);
      nn_angular_host[i] = std::min(na, mn_a);
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
      msg << "NEP_GPU: neighbor list exceeds MN_* from model at step " << update->ntimestep
          << " (max radial neighbors " << max_nr << " > MN_radial " << mn_r
          << ", max angular neighbors " << max_na << " > MN_angular " << mn_a << "). "
          << "First offender: i=" << first_i;
      if (first_tag) msg << " tag=" << first_tag;
      msg << " (nr=" << first_nr << ", na=" << first_na << "). "
          << "Model cutoff_r=" << std::sqrt(cutoffsq_radial) << " cutoff_a=" << std::sqrt(cutoffsq_angular)
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

    nl_valid_ = true;
    cached_natoms_total_ = natoms_total;
    cached_nlocal_ = nlocal;
    cached_mn_r_ = mn_r;
    cached_mn_a_ = mn_a;
  }

  // // Sort neighbor lists per-atom for determinism（按 j 升序，便于与参考输出对齐）
  // for (int i = 0; i < nlocal; ++i) {
  //   int nr = nn_radial_host[i];
  //   int na = nn_angular_host[i];
  //   if (nr > 1) {
  //     std::vector<int> tmp(nr);
  //     for (int k = 0; k < nr; ++k) tmp[k] = nl_radial_host[i + nlocal * k];
  //     std::stable_sort(tmp.begin(), tmp.end());
  //     for (int k = 0; k < nr; ++k) nl_radial_host[i + nlocal * k] = tmp[k];
  //   }
  //   if (na > 1) {
  //     std::vector<int> tmp(na);
  //     for (int k = 0; k < na; ++k) tmp[k] = nl_angular_host[i + nlocal * k];
  //     std::stable_sort(tmp.begin(), tmp.end());
  //     for (int k = 0; k < na; ++k) nl_angular_host[i + nlocal * k] = tmp[k];
  //   }
  // }

  NepGpuLammpsSystemHost sys;
  sys.natoms = natoms_total;
  sys.type = type_host.data();
  sys.xyz = xyz_host.data();
  for (int i = 0; i < 9; ++i) sys.h[i] = h[i];
  sys.pbc_x = domain->xperiodic ? 1 : 0;
  sys.pbc_y = domain->yperiodic ? 1 : 0;
  sys.pbc_z = domain->zperiodic ? 1 : 0;

  NepGpuLammpsResultHost res;
  res.f = f_host.data();
  // Provide per-atom potential/virial buffers whenever total energy/virial is requested
  // (LAMMPS eflag/vflag), even if per-atom outputs are not requested.
  res.eatom = (eflag || eflag_atom) ? eatom_host.data() : nullptr;
  res.vatom = (vflag || vflag_atom || cvflag_atom || eflag_atom) ? vatom_host.data() : nullptr;

  NepGpuLammpsNeighborsHost nb;
  nb.NN_radial = nn_radial_host.data();
  nb.NL_radial = nl_radial_host.data();
  nb.NN_angular = nn_angular_host.data();
  nb.NL_angular = nl_angular_host.data();

  nep_model_lmp->compute_host(sys, nlocal, nb, res, eflag || eflag_atom, vflag || vflag_atom || cvflag_atom || eflag_atom);

  // Accumulate forces on local + ghost atoms; LAMMPS will later move ghost
  // contributions back to owning ranks via reverse_comm, just like the CPU
  // NEP implementation.
  for (int i = 0; i < natoms_total; ++i) {
    atom->f[i][0] += res.f[3 * i + 0];
    atom->f[i][1] += res.f[3 * i + 1];
    atom->f[i][2] += res.f[3 * i + 2];
  }

  if (eflag) eng_vdwl += res.eng;

  if (vflag) {
    virial[0] += res.virial[0];
    virial[1] += res.virial[1];
    virial[2] += res.virial[2];
    virial[3] += res.virial[3];
    virial[4] += res.virial[4];
    virial[5] += res.virial[5];
  }

  if (eflag_atom) {
    for (int i = 0; i < nlocal; ++i) {
      eatom[i] += eatom_host[i];
    }
  }

  // Per-atom virial outputs:
  // - Standard per-atom virial uses vatom[i][0..5] (xx,yy,zz,xy,xz,yz).
  // - Centroid stress compute uses cvatom[i][0..8] (full 3x3: xx..zy).
  if (vflag_atom && !vatom) error->one(FLERR, "NEP_GPU: vflag_atom set but vatom is null (internal allocation mismatch)");
  if (cvflag_atom && !cvatom) error->one(FLERR, "NEP_GPU: cvflag_atom set but cvatom is null (internal allocation mismatch)");
  if (vflag_atom && vatom) {
    for (int i = 0; i < nlocal; ++i) {
      const int idx = 9 * i;
      vatom[i][0] += vatom_host[idx + 0];
      vatom[i][1] += vatom_host[idx + 1];
      vatom[i][2] += vatom_host[idx + 2];
      vatom[i][3] += vatom_host[idx + 3];
      vatom[i][4] += vatom_host[idx + 4];
      vatom[i][5] += vatom_host[idx + 5];
    }
  }

  if (cvflag_atom && cvatom) {
    for (int i = 0; i < nlocal; ++i) {
      const int idx = 9 * i;
      for (int k = 0; k < 9; ++k) cvatom[i][k] += vatom_host[idx + k];
    }
  }
}
