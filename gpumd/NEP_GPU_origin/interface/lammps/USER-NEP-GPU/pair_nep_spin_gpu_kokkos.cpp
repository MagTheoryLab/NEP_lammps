/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef LMP_KOKKOS
#include "kokkos_type.h"
#if defined(LMP_KOKKOS_GPU) || defined(KOKKOS_ENABLE_CUDA)

#include "pair_nep_spin_gpu_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "atom_vec.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "modify.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include "nep_gpu_lammps_model.h"
#include "utils/nep_kokkos_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace LAMMPS_NS;

namespace {
using nep_gpu_kokkos_utils::compute_box_thickness;
using nep_gpu_kokkos_utils::invert3x3_rowmajor;
template<class ViewType>
using NepSpinKokkosAosTraits = nep_gpu_kokkos_utils::NepKokkosAosTraits<ViewType>;
template<bool direct_ok, class DeviceType, class ViewType>
using NepSpinXyzPtr = nep_gpu_kokkos_utils::NepXyzPtr<direct_ok, DeviceType, ViewType>;

template<bool direct_ok, class DeviceType, class ViewType>
struct NepSpinSp4Ptr;

template<class DeviceType, class ViewType>
struct NepSpinSp4Ptr<true, DeviceType, ViewType> {
  static const double* get(const typename DeviceType::execution_space& /*exec*/, const ViewType& d_sp,
                           Kokkos::View<double*, DeviceType>& /*d_sp4_aos*/, int /*nall*/, bool& /*did_pack*/,
                           const char* /*label*/)
  {
    return d_sp.data();
  }
};

template<class DeviceType, class ViewType>
struct NepSpinSp4Ptr<false, DeviceType, ViewType> {
  static const double* get(const typename DeviceType::execution_space& exec, const ViewType& d_sp,
                           Kokkos::View<double*, DeviceType>& d_sp4_aos, int nall, bool& did_pack,
                           const char* label)
  {
    if (d_sp4_aos.extent_int(0) != 4 * nall) {
      d_sp4_aos = Kokkos::View<double*, DeviceType>(Kokkos::NoInit(label), 4 * nall);
    }
    auto d_sp4_aos_l = d_sp4_aos;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename DeviceType::execution_space>(exec, 0, nall),
      KOKKOS_LAMBDA(const int i) {
        d_sp4_aos_l(4 * i + 0) = d_sp(i, 0);
        d_sp4_aos_l(4 * i + 1) = d_sp(i, 1);
        d_sp4_aos_l(4 * i + 2) = d_sp(i, 2);
        d_sp4_aos_l(4 * i + 3) = d_sp(i, 3);
      });
    did_pack = true;
    return d_sp4_aos.data();
  }
};

template<class ViewType>
struct NepSpinKokkosForceAosTraits {
  using value_type = typename ViewType::non_const_value_type;
  using layout_type = typename ViewType::array_layout;
  static constexpr bool is_layout_right = std::is_same<layout_type, Kokkos::LayoutRight>::value;
  static constexpr bool is_double = std::is_same<value_type, double>::value;
  static constexpr bool direct_ok = is_layout_right && is_double;
};
} // namespace

template<class DeviceType>
PairNEPSpinGPUKokkos<DeviceType>::PairNEPSpinGPUKokkos(LAMMPS *lmp) : Pair(lmp)
{
#if LAMMPS_VERSION_NUMBER >= 20201130
  centroidstressflag = CENTROID_AVAIL;
#else
  centroidstressflag = 1;
#endif

  kokkosable = 1;
  manybody_flag = 1;
  one_coeff = 1;
  single_enable = 0;
  restartinfo = 0;

  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // Read-only atom data required by the backend (forces are write-only).
  datamask_read = X_MASK | TYPE_MASK | SP_MASK;
  // The backend writes f/fm.  We also keep fm_long zeroed, since AtomVecSpin
  // includes it in reverse comm; leaving stale values can corrupt spin fixes.
  datamask_modify = F_MASK | FM_MASK | FML_MASK;
  // Kokkos `neigh full` requires `newton off`, but the NEP backend accumulates
  // forces/torques onto neighbor atoms (including periodic/MPI ghosts).  Request
  // reverse force communication even when Newton is off.
  // AtomVec reverse comm always packs f[3] plus spin arrays (fm[3], fm_long[3]) => 9.
  comm_reverse_off = 9;

}

template<class DeviceType>
PairNEPSpinGPUKokkos<DeviceType>::~PairNEPSpinGPUKokkos()
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

template<class DeviceType>
void PairNEPSpinGPUKokkos<DeviceType>::allocate()
{
  const int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  map = new int[n + 1];
  allocated = 1;
}

template<class DeviceType>
void PairNEPSpinGPUKokkos<DeviceType>::allocate_type_map()
{
  const int ntype = atom->ntypes;
  if (type_map) delete [] type_map;
  type_map = new int[ntype + 1];
  for (int i = 0; i <= ntype; ++i) type_map[i] = 0;
}

template<class DeviceType>
void PairNEPSpinGPUKokkos<DeviceType>::ensure_device_maps()
{
  if (!type_map) return;
  if (cached_ntypes == atom->ntypes && d_type_map.extent_int(0) == atom->ntypes + 1) return;

  cached_ntypes = atom->ntypes;
  d_type_map = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:type_map"), cached_ntypes + 1);

  auto h = Kokkos::create_mirror_view(d_type_map);
  for (int i = 0; i <= cached_ntypes; ++i) h(i) = type_map[i];
  Kokkos::deep_copy(d_type_map, h);
}

template<class DeviceType>
void PairNEPSpinGPUKokkos<DeviceType>::ensure_device_buffers(int nlocal, int nall, int mn_r, int mn_a)
{
  if (cached_nlocal == nlocal && cached_nall == nall && cached_mn_r == mn_r && cached_mn_a == mn_a &&
      d_type_mapped.extent_int(0) == nall &&
      d_nn_radial.extent_int(0) == nlocal &&
      d_nl_radial.extent_int(0) == nlocal * mn_r &&
      d_nl_angular.extent_int(0) == nlocal * mn_a) {
    return;
  }

  cached_nlocal = nlocal;
  cached_nall = nall;
  cached_mn_r = mn_r;
  cached_mn_a = mn_a;

  d_type_mapped = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:type_mapped"), nall);
  d_nn_radial = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:nn_r"), nlocal);
  d_nn_angular = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:nn_a"), nlocal);
  d_nl_radial = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:nl_r"), nlocal * mn_r);
  d_nl_angular = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:nl_a"), nlocal * mn_a);
  d_overflow = Kokkos::View<int, DeviceType>("nep_spin_gpu:nl_overflow");

  neighbors_packed_ = false;
  packed_list_ptr_ = nullptr;
}

template<class DeviceType>
void PairNEPSpinGPUKokkos<DeviceType>::settings(int narg, char **arg)
{
  if (narg == 0) return;
  error->all(FLERR, "nep/spin/gpu/kk: pair_style does not accept settings; atom->fm always stores H = -dE/dM in eV/μB");
}

template<class DeviceType>
void PairNEPSpinGPUKokkos<DeviceType>::coeff(int narg, char **arg)
{
  if (narg < 4) error->all(FLERR, "Incorrect args for pair_style nep/spin/gpu/kk");

  if (!allocated) allocate();
  allocate_type_map();

  model_filename_ = arg[2];
  std::string model_path_from_lammps = utils::get_potential_file_path(model_filename_);

  // Map element symbols to NEP type indices by reading the first line of the NEP file.
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

  // Ensure the number of element symbols in pair_coeff matches LAMMPS types.
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

  delete nep_model_spin_lmp;
  nep_model_spin_lmp = new NepGpuLammpsModel(model_path_from_lammps.c_str(), max_atoms);
  // Do not override Kokkos' GPU selection here. In MPI multi-GPU runs, Kokkos
  // (or CUDA_VISIBLE_DEVICES) typically makes the intended device appear as
  // device-0 within each rank's process.

  // LAMMPS neighbor lists are built using Pair::cutoff (plus skin).  NEP spin
  // has separate radial and angular cutoffs, so we must advertise the maximum
  // to ensure the full list contains all required neighbors.
  const NepGpuModelInfo& info = nep_model_spin_lmp->info();
  if (!info.needs_spin) {
    error->all(FLERR, "NEP_SPIN_GPU/kk: non-spin model cannot be used with pair_style nep/spin/gpu/kk.");
  }
  const double rc_r = info.rc_radial_max;
  const double rc_a = info.rc_angular_max;
  cutoff = std::max(rc_r, rc_a);
  cutoffsq = cutoff * cutoff;

  for (int i = 1; i <= ntype; i++)
    for (int j = 1; j <= ntype; j++)
      cutsq[i][j] = cutoffsq;
}

template<class DeviceType>
double PairNEPSpinGPUKokkos<DeviceType>::init_one(int i, int j)
{
  return cutoff;
}

template<class DeviceType>
void PairNEPSpinGPUKokkos<DeviceType>::init_style()
{
  if (!lmp->kokkos || !lmp->kokkos->kokkos_exists) {
    error->all(FLERR, "Pair style nep/spin/gpu/kk requires Kokkos enabled at runtime (use: -k on g 1 ... or 'package kokkos').");
  }

  atomKK = (AtomKokkos *) atom;
  if (!atomKK) error->all(FLERR, "Pair style nep/spin/gpu/kk requires AtomKokkos (Kokkos runtime not active?)");
  if (!atom->sp || !atom->fm) error->all(FLERR, "Pair style nep/spin/gpu/kk requires atom_style spin (sp/fm arrays missing)");

  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_host(false);

  if (neighflag != FULL) {
    error->all(FLERR, "Pair style nep/spin/gpu/kk requires full neighbor list style (use: -pk kokkos neigh full)");
  }
}

template<class DeviceType>
void PairNEPSpinGPUKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  static_assert(std::is_same<X_FLOAT, double>::value, "nep/spin/gpu/kk currently requires LAMMPS built with double-precision positions (X_FLOAT=double).");
  static_assert(std::is_same<F_FLOAT, double>::value, "nep/spin/gpu/kk currently requires LAMMPS built with double-precision forces (F_FLOAT=double).");

  eflag = eflag_in;
  vflag = vflag_in;
  ev_init(eflag, vflag, 0);

  if (cvflag_atom) {
    const int nthreads = (comm && comm->nthreads > 0) ? comm->nthreads : 1;
    if (!cvatom || atom->nmax > maxcvatom) {
      maxcvatom = atom->nmax;
      memory->destroy(cvatom);
      memory->create(cvatom, nthreads * maxcvatom, 9, "pair:cvatom");
    }
    if (!cvatom) {
      error->one(FLERR, "NEP_SPIN_GPU/kk: cvatom allocation failed after ev_init");
    }
    int n = atom->nlocal;
    if (force->newton) n += atom->nghost;
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < 9; ++k) cvatom[i][k] = 0.0;
    }
  }

  if (!nep_model_spin_lmp) error->one(FLERR, "NEP_SPIN_GPU/kk: model not initialized (pair_coeff not set?)");

  using ExecSpace = typename DeviceType::execution_space;
  ExecSpace exec;

  atomKK->sync(execution_space, datamask_read);
  Kokkos::fence();

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;

  // Kokkos note: mixing `run_style verlet/kk` with non-Kokkos time integrators
  // (e.g. `fix nve`) can lead to host/device state mismatches (x/v/f) and show
  // up as energy drift. Prefer `fix nve/kk` (and other /kk integrators) when
  // running Kokkos on device.
  if (comm->me == 0 && (screen || logfile)) {
    static bool warned_integrator_once = false;
    if (!warned_integrator_once) {
      const bool strict_integrator = (std::getenv("NEP_SPIN_GPU_LMP_STRICT_INTEGRATOR") != nullptr);
      for (int ifix = 0; ifix < modify->nfix; ++ifix) {
        Fix* fx = modify->fix[ifix];
        if (!fx) continue;
        if (fx->time_integrate && fx->kokkosable == 0) {
          warned_integrator_once = true;
          if (strict_integrator) {
            error->all(FLERR,
                       "NEP_SPIN_GPU/kk: non-Kokkos time integrator fix is not supported with Kokkos device execution (fix id='{}' style='{}'); use the corresponding /kk fix",
                       fx->id ? fx->id : "?",
                       fx->style ? fx->style : "?");
          } else {
            error->warning(FLERR,
              "NEP_SPIN_GPU/kk: detected non-Kokkos time integrator fix id='{}' style='{}' while running Kokkos; use the corresponding /kk fix to avoid host/device drift. Set NEP_SPIN_GPU_LMP_STRICT_INTEGRATOR=1 to make this fatal.",
              fx->id ? fx->id : "?",
              fx->style ? fx->style : "?");
          }
          break;
        }
      }
    }
  }

  // Kokkos neigh full requires newton off, so LAMMPS may not zero ghost f/fm nor reverse-communicate.
  // Our backend accumulates to neighbor atoms (including ghosts), so we must zero ghosts and reverse-comm.
  //
  // When users run without any time integration fixes (LAMMPS warning: "No fixes with time integration,
  // atoms won't move"), some LAMMPS versions may not clear per-step force buffers as usual. Since the
  // NEP backend *adds* into `f`/`fm`, explicitly clear them to avoid accumulation and possible NaNs.
  auto d_f = atomKK->k_f.view<DeviceType>();
  auto d_fm = atomKK->k_fm.view<DeviceType>();
  const bool no_time_integration = (modify->n_initial_integrate == 0 && modify->n_final_integrate == 0);
  // `fm` is not part of LAMMPS' standard force-clear path in all versions /
  // configs (especially with Kokkos); clear it explicitly to avoid unintended
  // timestep-to-timestep accumulation.
  Kokkos::deep_copy(exec, d_fm, 0.0);
  // NEP does not produce long-range spin forces; keep the corresponding buffer zero
  // so reverse comm (which includes fm_long for atom_style spin) cannot accumulate
  // stale values from previous steps.
  if (atomKK->k_fm_long.extent_int(0) > 0) {
    auto d_fm_long = atomKK->k_fm_long.view<DeviceType>();
    Kokkos::deep_copy(exec, d_fm_long, 0.0);
  }
  if (no_time_integration) {
    Kokkos::deep_copy(exec, d_f, 0.0);
  } else if (nall > nlocal) {
    // LAMMPS will clear local forces, but in some comm modes it may not clear ghosts.
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(exec, nlocal, nall),
      KOKKOS_LAMBDA(const int i) {
        d_f(i,0) = 0.0;
        d_f(i,1) = 0.0;
        d_f(i,2) = 0.0;
      });
  }

  ensure_device_maps();

  const NepGpuModelInfo& info = nep_model_spin_lmp->info();
  const int mn_r = info.mn_radial;
  const int mn_a = info.mn_angular;
  ensure_device_buffers(nlocal, nall, mn_r, mn_a);

  // IMPORTANT: NEP trusts the LAMMPS full neighbor list and associated ghost
  // coordinates. For periodic boundaries, LAMMPS stores ghosts at the correct
  // periodic image, so dx = x[j]-x[i] is already the displacement for the neighbor
  // entry we are iterating over.
  //
  // The additional periodic wrapping logic below exists only for special internal
  // consistency tests (e.g. strict small-box checks) and should not be needed for
  // normal NEP runs/debugging.
  const bool strict_box_env = (std::getenv("NEP_GPU_LMP_STRICT_BOX") != nullptr);
  const bool force_mic_with_ghosts =
    strict_box_env ||
    (std::getenv("NEP_SPIN_GPU_LMP_BACKEND_FORCE_MIC") != nullptr) ||
    (std::getenv("NEP_SPIN_GPU_LMP_FORCE_MIC") != nullptr);
  const bool need_mic = (nall == nlocal) || force_mic_with_ghosts;
  double h_box[9];
  h_box[0] = domain->h[0]; h_box[3] = 0.0;          h_box[6] = 0.0;
  h_box[1] = domain->h[5]; h_box[4] = domain->h[1]; h_box[7] = 0.0;
  h_box[2] = domain->h[4]; h_box[5] = domain->h[3]; h_box[8] = domain->h[2];
  double hinv_box[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (need_mic) invert3x3_rowmajor(h_box, hinv_box);
  const int pbc_x = domain->xperiodic ? 1 : 0;
  const int pbc_y = domain->yperiodic ? 1 : 0;
  const int pbc_z = domain->zperiodic ? 1 : 0;

  // Map LAMMPS types -> NEP types on device
  auto d_type = atomKK->k_type.view<DeviceType>();
  auto d_type_map_l = d_type_map;
  auto d_type_mapped_l = d_type_mapped;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<ExecSpace>(exec, 0, nall),
    KOKKOS_LAMBDA(const int i) {
      d_type_mapped_l(i) = d_type_map_l(d_type(i));
    });

  // Pack compact NN/NL from Kokkos neighbor list.
  auto klist = (NeighListKokkos<DeviceType> *) list;
  auto d_neighbors = klist->d_neighbors;
  auto d_numneigh = klist->d_numneigh;
  auto d_ilist = klist->d_ilist;
  const int inum = klist->inum;
  const int neigh_stride = d_neighbors.extent_int(1);
  const int neigh_rows = d_neighbors.extent_int(0);
  const int numneigh_len = d_numneigh.extent_int(0);
  if (inum != nlocal) error->one(FLERR, "NEP_SPIN_GPU/kk: expected inum == nlocal for full neighbor list");

  // Rebuild compact NN/NL every timestep (every compute call).
  //
  // LAMMPS neighbor lists include neighbors out to (cutoff + skin) and are only
  // rebuilt occasionally. If we cache a *distance-filtered* NN/NL list and only
  // refresh it when the neighbor list rebuilds, neighbors that were just outside
  // the true cutoff at build time can move inside the cutoff before the next
  // rebuild, but would never be added to NN/NL. That breaks force/energy
  // consistency and shows up as NVE energy drift.
  neighbors_packed_ = true;
  packed_list_ptr_ = (void *) list;

  auto d_nn_r = d_nn_radial;
  auto d_nn_a = d_nn_angular;
  auto d_nl_r = d_nl_radial;
  auto d_nl_a = d_nl_angular;
  auto d_over = d_overflow;
  auto d_x = atomKK->k_x.view<DeviceType>();

  Kokkos::deep_copy(exec, d_over, 0);
  Kokkos::deep_copy(exec, d_nn_r, 0);
  Kokkos::deep_copy(exec, d_nn_a, 0);

  const int mn_r_l = mn_r;
  const int mn_a_l = mn_a;
  const double rc_r = info.rc_radial_max;
  const double rc_r2 = rc_r * rc_r;
  const double rc_a = info.rc_angular_max;
  const double rc_a2 = rc_a * rc_a;
  const int neigh_stride_l = neigh_stride;
  const int neigh_rows_l = neigh_rows;
  const int numneigh_len_l = numneigh_len;
  const bool need_mic_l = need_mic;
  const double h0 = h_box[0], h1 = h_box[1], h2 = h_box[2], h3 = h_box[3], h4 = h_box[4], h5 = h_box[5], h6 = h_box[6],
               h7 = h_box[7], h8 = h_box[8];
  const double hi0 = hinv_box[0], hi1 = hinv_box[1], hi2 = hinv_box[2], hi3 = hinv_box[3], hi4 = hinv_box[4], hi5 = hinv_box[5],
               hi6 = hinv_box[6], hi7 = hinv_box[7], hi8 = hinv_box[8];
  const int pbc_x_l = pbc_x, pbc_y_l = pbc_y, pbc_z_l = pbc_z;

  Kokkos::parallel_for(
    Kokkos::RangePolicy<ExecSpace>(exec, 0, inum),
    KOKKOS_LAMBDA(const int ii) {
      const int i = d_ilist(ii);
      if (i < 0 || i >= nlocal) { Kokkos::atomic_or(&d_over(), 2); return; }
      if (i >= neigh_rows_l || i >= numneigh_len_l) { Kokkos::atomic_or(&d_over(), 16); return; }

      int jnum = d_numneigh(i);
      if (jnum > neigh_stride_l) { Kokkos::atomic_or(&d_over(), 8); jnum = neigh_stride_l; }

      int nr = 0;
      int na = 0;
      bool overflow_mn = false;

      for (int jj = 0; jj < jnum; ++jj) {
        const int jraw = d_neighbors(i, jj);
        const int j0 = jraw & NEIGHMASK;
        if (j0 < 0 || j0 >= nall) continue;

        double dx = static_cast<double>(d_x(j0, 0) - d_x(i, 0));
        double dy = static_cast<double>(d_x(j0, 1) - d_x(i, 1));
        double dz = static_cast<double>(d_x(j0, 2) - d_x(i, 2));

        if (need_mic_l) {
          // Minimum-image convention via fractional wrapping:
          //   s = H^{-1} * dr; s -= round(s); dr = H * s
          double sx = hi0 * dx + hi1 * dy + hi2 * dz;
          double sy = hi3 * dx + hi4 * dy + hi5 * dz;
          double sz = hi6 * dx + hi7 * dy + hi8 * dz;
          if (pbc_x_l) sx -= floor(sx + 0.5);
          if (pbc_y_l) sy -= floor(sy + 0.5);
          if (pbc_z_l) sz -= floor(sz + 0.5);
          dx = h0 * sx + h1 * sy + h2 * sz;
          dy = h3 * sx + h4 * sy + h5 * sz;
          dz = h6 * sx + h7 * sy + h8 * sz;
        }

        const double rsq = dx * dx + dy * dy + dz * dz;

        if (rsq < rc_r2) {
          if (nr < mn_r_l) d_nl_r(i + nlocal * nr++) = j0;
          else overflow_mn = true;
        }
        if (rsq < rc_a2) {
          if (na < mn_a_l) d_nl_a(i + nlocal * na++) = j0;
          else overflow_mn = true;
        }

        if (nr >= mn_r_l && na >= mn_a_l) break;
      }

      d_nn_r(i) = nr;
      d_nn_a(i) = na;
      if (overflow_mn) Kokkos::atomic_or(&d_over(), 1);
    });

  int overflow = 0;
  // Use the same execution-space instance to avoid cross-stream races.
  Kokkos::deep_copy(exec, overflow, d_overflow);
  if (overflow & 2) error->one(FLERR, "NEP_SPIN_GPU/kk: ilist contained non-local indices; cannot build NN/NL safely");
  if (overflow & 1) error->one(FLERR, "NEP_SPIN_GPU/kk: neighbor list exceeds MN_*; increase MN_* or reduce neighbor skin");
  if (overflow & 8) error->one(FLERR, "NEP_SPIN_GPU/kk: d_numneigh exceeded d_neighbors stride; neighbor view layout mismatch");
  if (overflow & 16) error->one(FLERR, "NEP_SPIN_GPU/kk: d_ilist index exceeded d_neighbors/d_numneigh extents; neighbor view layout mismatch");

  // Call backend with device pointers.
  NepGpuLammpsSystemDevice sys;
  sys.natoms = nall;
  sys.type = d_type_mapped.data();
  auto d_sp = atomKK->k_sp.view<DeviceType>();

  // Use the same CUDA stream as Kokkos to avoid cross-stream races when LAMMPS/Kokkos
  // launches other kernels that read/write the same device buffers.
#if defined(KOKKOS_ENABLE_CUDA)
  sys.stream = (void*) exec.cuda_stream();
#else
  sys.stream = nullptr;
#endif

  // Our CUDA backend expects AoS buffers:
  //   xyz_aos: (x0,y0,z0,x1,y1,z1,...) length 3*nall
  //   sp4_aos: (spx,spy,spz,spmag,...) length 4*nall
  // but LAMMPS/Kokkos may store fixed-size arrays in LayoutLeft (SoA) and/or single precision (X_FLOAT=float)
  // depending on build options. Pack into double AoS buffers when needed.
  constexpr bool x_direct_ok = NepSpinKokkosAosTraits<decltype(d_x)>::direct_ok;
  constexpr bool sp_direct_ok = NepSpinKokkosAosTraits<decltype(d_sp)>::direct_ok;

  bool did_pack_aos = false;
  sys.xyz = NepSpinXyzPtr<x_direct_ok, DeviceType, decltype(d_x)>::get(
    exec, d_x, d_xyz_aos, nall, did_pack_aos, "nep_spin_gpu:xyz_aos");
  sys.sp4 = NepSpinSp4Ptr<sp_direct_ok, DeviceType, decltype(d_sp)>::get(
    exec, d_sp, d_sp4_aos, nall, did_pack_aos, "nep_spin_gpu:sp4_aos");

  sys.h[0] = domain->h[0]; sys.h[3] = 0.0;        sys.h[6] = 0.0;
  sys.h[1] = domain->h[5]; sys.h[4] = domain->h[1]; sys.h[7] = 0.0;
  sys.h[2] = domain->h[4]; sys.h[5] = domain->h[3]; sys.h[8] = domain->h[2];
  sys.pbc_x = domain->xperiodic ? 1 : 0;
  sys.pbc_y = domain->yperiodic ? 1 : 0;
  sys.pbc_z = domain->zperiodic ? 1 : 0;

  // Ensure "small box" vs "replicated big box" consistency:
  // with nearest-image neighbor lists, cutoff must be <= half the periodic thickness.
  {
    const bool strict_box = (std::getenv("NEP_GPU_LMP_STRICT_BOX") != nullptr);
    const double rc_max = std::max(info.rc_radial_max, info.rc_angular_max);
    double thickness[3];
    compute_box_thickness(sys.h, thickness);
    double min_th = 1.0e300;
    if (sys.pbc_x) min_th = std::min(min_th, thickness[0]);
    if (sys.pbc_y) min_th = std::min(min_th, thickness[1]);
    if (sys.pbc_z) min_th = std::min(min_th, thickness[2]);
    if (strict_box && min_th < 1.0e299 && rc_max > 0.5 * min_th) {
      error->all(FLERR,
        "NEP_SPIN_GPU/kk: cutoff exceeds half periodic box thickness; replicate the cell (LAMMPS 'replicate') or reduce cutoff.");
    }
  }

  NepGpuLammpsNeighborsDevice nb;
  nb.NN_radial = d_nn_radial.data();
  nb.NL_radial = d_nl_radial.data();
  nb.NN_angular = d_nn_angular.data();
  nb.NL_angular = d_nl_angular.data();

  NepGpuLammpsResultDevice res;
  // Backend writes AoS (x0,y0,z0,...) forces. If Kokkos stores forces in
  // LayoutLeft (SoA), compute into temporary AoS buffers and scatter-add.
  constexpr bool f_direct_ok = NepSpinKokkosForceAosTraits<decltype(d_f)>::direct_ok;
  constexpr bool fm_direct_ok = NepSpinKokkosForceAosTraits<decltype(d_fm)>::direct_ok;
  constexpr bool need_scatter_f = !f_direct_ok;
  constexpr bool need_scatter_fm = !fm_direct_ok;

  if constexpr (f_direct_ok) {
    res.f = (double *) d_f.data();
  } else {
    if (d_f_aos.extent_int(0) != 3 * nall) {
      d_f_aos = Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:f_aos"), 3 * nall);
    }
    Kokkos::deep_copy(exec, d_f_aos, 0.0);
    res.f = (double *) d_f_aos.data();
  }

  if constexpr (fm_direct_ok) {
    res.fm = (double *) d_fm.data();
  } else {
    if (d_fm_aos.extent_int(0) != 3 * nall) {
      d_fm_aos = Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:fm_aos"), 3 * nall);
    }
    Kokkos::deep_copy(exec, d_fm_aos, 0.0);
    res.fm = (double *) d_fm_aos.data();
  }
  res.want_virial_raw9 = false;
  // Convention: spin magnitude is magnetic moment M in μB, and NEP outputs field = -dE/dM in eV/μB.
  res.inv_hbar = 1.0;

  if (eflag_atom) {
    if (d_eatom.extent_int(0) != nlocal) d_eatom = Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:eatom"), nlocal);
    res.eatom = d_eatom.data();
  }
  const bool want_vatom_buffer = (vflag_atom != 0) || (cvflag_atom != 0);
  if (want_vatom_buffer) {
    if (d_vatom.extent_int(0) != 9 * nlocal) d_vatom = Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_spin_gpu:vatom"), 9 * nlocal);
    res.vatom = d_vatom.data();
  }

  const bool need_energy = (eflag != 0);
  const bool need_virial = (vflag != 0);

  nep_model_spin_lmp->compute_device(sys, nlocal, nb, res, need_energy, need_virial);
  exec.fence();

  if constexpr (need_scatter_f) {
    auto d_f_aos_l = d_f_aos;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(exec, 0, nall),
      KOKKOS_LAMBDA(const int i) {
        d_f(i, 0) += d_f_aos_l(3 * i + 0);
        d_f(i, 1) += d_f_aos_l(3 * i + 1);
        d_f(i, 2) += d_f_aos_l(3 * i + 2);
      });
  }
  if constexpr (need_scatter_fm) {
    auto d_fm_aos_l = d_fm_aos;
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(exec, 0, nall),
      KOKKOS_LAMBDA(const int i) {
        d_fm(i, 0) += d_fm_aos_l(3 * i + 0);
        d_fm(i, 1) += d_fm_aos_l(3 * i + 1);
        d_fm(i, 2) += d_fm_aos_l(3 * i + 2);
      });
  }

  if constexpr (need_scatter_f || need_scatter_fm) {
    exec.fence();
  }

  // Mark device-side force buffers as modified only after all device kernels have
  // finished (backend + optional scatter-add). This ensures any subsequent sync
  // (e.g. for reverse comm) treats device as the authoritative source.
  atomKK->modified(execution_space, datamask_modify);

  if (nall > nlocal) {
    comm->reverse_comm();

    // Some LAMMPS/Kokkos communication paths update host-side force buffers.
    // Sync those MPI-reduced results back to the device before continuing;
    // otherwise a later device->host sync can overwrite communicated forces
    // and make results MPI-rank dependent (showing up as energy drift in NVE).
    atomKK->sync(execution_space, F_MASK | FM_MASK | FML_MASK);
    atomKK->modified(execution_space, F_MASK | FM_MASK | FML_MASK);
    Kokkos::fence();
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
    auto h_e = Kokkos::create_mirror_view(d_eatom);
    Kokkos::deep_copy(h_e, d_eatom);
    for (int i = 0; i < nlocal; ++i) eatom[i] += h_e(i);
  }

  if (want_vatom_buffer) {
    auto h_v = Kokkos::create_mirror_view(d_vatom);
    Kokkos::deep_copy(h_v, d_vatom);

    // Internal buffer stores raw 9-component virial in the order
    // (xx,yy,zz,xy,xz,yz,yx,zx,zy). The 6-component LAMMPS vatom view uses
    // the symmetrized off-diagonals, while cvatom consumes the raw 9 values.
    if (vflag_atom && !vatom && !cvatom) {
      error->one(FLERR, "NEP_SPIN_GPU/kk: per-atom virial requested but neither vatom nor cvatom is allocated");
    }
    if (cvflag_atom && !cvatom && !vatom) {
      error->one(FLERR, "NEP_SPIN_GPU/kk: centroid per-atom virial requested but neither cvatom nor vatom is allocated");
    }
    if (vatom) {
      for (int i = 0; i < nlocal; ++i) {
        const int idx = 9 * i;
        vatom[i][0] += h_v(idx + 0);
        vatom[i][1] += h_v(idx + 1);
        vatom[i][2] += h_v(idx + 2);
        vatom[i][3] += 0.5 * (h_v(idx + 3) + h_v(idx + 6));
        vatom[i][4] += 0.5 * (h_v(idx + 4) + h_v(idx + 7));
        vatom[i][5] += 0.5 * (h_v(idx + 5) + h_v(idx + 8));
      }
    }

    if (cvatom) {
      for (int i = 0; i < nlocal; ++i) {
        const int idx = 9 * i;
        for (int k = 0; k < 9; ++k) cvatom[i][k] += h_v(idx + k);
      }
    }
  }
}

namespace LAMMPS_NS {
template class PairNEPSpinGPUKokkos<LMPDeviceType>;
} // namespace LAMMPS_NS

#endif // LMP_KOKKOS_GPU || KOKKOS_ENABLE_CUDA
#endif // LMP_KOKKOS
