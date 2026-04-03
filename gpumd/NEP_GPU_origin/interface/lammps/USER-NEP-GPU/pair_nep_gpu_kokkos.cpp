/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef LMP_KOKKOS
#include "kokkos_type.h"
#if defined(LMP_KOKKOS_GPU) || defined(KOKKOS_ENABLE_CUDA)

#include "pair_nep_gpu_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "comm_kokkos.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "kokkos.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "nep_gpu_lammps_model.h"
#include "update.h"
#include "mpi.h"
#include "utils/nep_kokkos_utils.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

using namespace LAMMPS_NS;

namespace {
using nep_gpu_kokkos_utils::compute_box_thickness;
using nep_gpu_kokkos_utils::invert3x3_rowmajor;
template<class ViewType>
using NepKokkosAosTraits = nep_gpu_kokkos_utils::NepKokkosAosTraits<ViewType>;
} // namespace

template<class DeviceType>
PairNEPGPUKokkos<DeviceType>::PairNEPGPUKokkos(LAMMPS *lmp) : PairNEPGPU(lmp)
{
  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // Read-only atom data required by the backend (forces are write-only).
  datamask_read = X_MASK | TYPE_MASK;
  datamask_modify = F_MASK;
  // Kokkos `neigh full` requires `newton off`, but the NEP backend accumulates
  // forces onto neighbor atoms (including periodic/MPI ghosts).  Request reverse
  // force communication even when Newton is off (f[3] => 3).
  comm_reverse_off = 3;
}

template<class DeviceType>
PairNEPGPUKokkos<DeviceType>::~PairNEPGPUKokkos()
{
  if (copymode) return;
}

template<class DeviceType>
void PairNEPGPUKokkos<DeviceType>::ensure_device_maps()
{
  if (!type_map) return;
  if (cached_ntypes == atom->ntypes &&
      d_type_map.extent_int(0) == atom->ntypes + 1 &&
      d_rc_radial_by_type.extent_int(0) == atom->ntypes &&
      d_rc_angular_by_type.extent_int(0) == atom->ntypes) return;

  cached_ntypes = atom->ntypes;
  d_type_map = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_gpu:type_map"), cached_ntypes + 1);
  d_rc_radial_by_type =
    Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_gpu:rc_r_by_type"), cached_ntypes);
  d_rc_angular_by_type =
    Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_gpu:rc_a_by_type"), cached_ntypes);

  auto h = Kokkos::create_mirror_view(d_type_map);
  auto h_rr = Kokkos::create_mirror_view(d_rc_radial_by_type);
  auto h_ra = Kokkos::create_mirror_view(d_rc_angular_by_type);
  for (int i = 0; i <= cached_ntypes; ++i) h(i) = type_map[i];
  for (int i = 0; i < cached_ntypes; ++i) {
    h_rr(i) = rc_radial_by_type_[type_map[i + 1]];
    h_ra(i) = rc_angular_by_type_[type_map[i + 1]];
  }
  Kokkos::deep_copy(d_type_map, h);
  Kokkos::deep_copy(d_rc_radial_by_type, h_rr);
  Kokkos::deep_copy(d_rc_angular_by_type, h_ra);
}

template<class DeviceType>
void PairNEPGPUKokkos<DeviceType>::ensure_device_buffers(int nlocal, int nall, int mn_r, int mn_a)
{
  if (cached_nlocal == nlocal && cached_nall == nall && cached_mn_r == mn_r && cached_mn_a == mn_a &&
      d_type_mapped.extent_int(0) == nall &&
      d_nn_radial.extent_int(0) == nlocal &&
      d_nl_radial.extent_int(0) == nlocal * mn_r) {
    return;
  }

  cached_nlocal = nlocal;
  cached_nall = nall;
  cached_mn_r = mn_r;
  cached_mn_a = mn_a;

  d_type_mapped = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_gpu:type_mapped"), nall);
  d_nn_radial = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_gpu:nn_r"), nlocal);
  d_nn_angular = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_gpu:nn_a"), nlocal);
  d_nl_radial = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_gpu:nl_r"), nlocal * mn_r);
  d_nl_angular = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_gpu:nl_a"), nlocal * mn_a);

  d_overflow = Kokkos::View<int, DeviceType>("nep_gpu:nl_overflow");

  neighbors_packed_ = false;
  packed_list_ptr_ = nullptr;
}

template<class DeviceType>
void PairNEPGPUKokkos<DeviceType>::init_style()
{
  if (!lmp->kokkos || !lmp->kokkos->kokkos_exists) {
    error->all(FLERR, "Pair style nep/gpu/kk requires Kokkos to be enabled at runtime (use: -k on g 1 ... or 'package kokkos').");
  }

  atomKK = (AtomKokkos *) atom;
  if (!atomKK) {
    error->all(FLERR, "Pair style nep/gpu/kk requires AtomKokkos (Kokkos runtime not active?)");
  }

  // neighbor list request for KOKKOS (device)
  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_host(false);

  if (neighflag != FULL) {
    error->all(FLERR, "Pair style nep/gpu/kk requires full neighbor list style (use: -pk kokkos neigh full)");
  }
}

template<class DeviceType>
void PairNEPGPUKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  static_assert(std::is_same<X_FLOAT, double>::value, "NEP_GPU/kk currently requires LAMMPS built with double-precision positions (X_FLOAT=double).");
  static_assert(std::is_same<F_FLOAT, double>::value, "NEP_GPU/kk currently requires LAMMPS built with double-precision forces (F_FLOAT=double).");

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
      error->one(FLERR, "NEP_GPU/kk: cvatom allocation failed after ev_init");
    }
    int n = atom->nlocal;
    if (force->newton) n += atom->nghost;
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < 9; ++k) cvatom[i][k] = 0.0;
    }
  }

  // Kokkos-managed atom data on device
  if (!atomKK) error->one(FLERR, "NEP_GPU/kk: AtomKokkos is null (Kokkos runtime not active?)");
  atomKK->sync(execution_space, datamask_read);
  // AtomKokkos::sync() may perform asynchronous host<->device copies using an
  // internal execution-space instance.  Ensure those copies complete before we
  // launch our own device kernels and before the NEP backend reads device data.
  Kokkos::fence();

  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;

  using ExecSpace = typename DeviceType::execution_space;
  ExecSpace exec;

  if (!nep_model_lmp) error->one(FLERR, "NEP_GPU/kk: nep_model_lmp is null (pair_coeff not set?)");

  // Kokkos note: mixing `run_style verlet/kk` with non-Kokkos time integrators
  // (e.g. `fix nve`) can lead to host/device state mismatches (x/v/f) and show
  // up as non-conservation or large energy noise. Prefer `fix nve/kk` (and other
  // /kk integrators) when running Kokkos on device.
  if (comm->me == 0 && (screen || logfile)) {
    static bool warned_integrator_once = false;
    if (!warned_integrator_once) {
      const bool strict_integrator = (std::getenv("NEP_GPU_LMP_STRICT_INTEGRATOR") != nullptr);
      for (int ifix = 0; ifix < modify->nfix; ++ifix) {
        Fix* fx = modify->fix[ifix];
        if (!fx) continue;
        if (fx->time_integrate && fx->kokkosable == 0) {
          warned_integrator_once = true;
          if (strict_integrator) {
            error->all(FLERR,
                       "NEP_GPU/kk: non-Kokkos time integrator fix is not supported with Kokkos device execution (fix id='{}' style='{}'); use the corresponding /kk fix",
                       fx->id ? fx->id : "?",
                       fx->style ? fx->style : "?");
          } else {
            error->warning(FLERR,
              "NEP_GPU/kk: detected non-Kokkos time integrator fix id='{}' style='{}' while running Kokkos; use the corresponding /kk fix to avoid host/device drift. Set NEP_GPU_LMP_STRICT_INTEGRATOR=1 to make this fatal.",
              fx->id ? fx->id : "?",
              fx->style ? fx->style : "?");
          }
          break;
        }
      }
    }
  }

  // Kokkos "neigh full" requires global newton off, so LAMMPS may not zero ghost
  // forces nor reverse-communicate them.
  //
  // Our NEP_GPU backend accumulates forces on neighbor atoms (Newton's 3rd law),
  // including ghosts. For correctness with MPI ranks > 1 we must:
  //  1) explicitly zero ghost forces before compute, and
  //  2) explicitly reverse-communicate forces after compute.
  auto d_f = atomKK->k_f.view<DeviceType>();
  if (nall > nlocal) {
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(exec, nlocal, nall),
      KOKKOS_LAMBDA(const int i) {
        d_f(i, 0) = 0.0;
        d_f(i, 1) = 0.0;
        d_f(i, 2) = 0.0;
      });
  }

  ensure_device_maps();

  const int mn_r = nep_model_lmp->info().mn_radial;
  const int mn_a = nep_model_lmp->info().mn_angular;
  ensure_device_buffers(nlocal, nall, mn_r, mn_a);

  // Map LAMMPS types -> NEP types on device
  auto d_type = atomKK->k_type.view<DeviceType>();
  auto d_type_map_l = d_type_map;
  auto d_type_mapped_l = d_type_mapped;
  Kokkos::parallel_for(
    Kokkos::RangePolicy<ExecSpace>(exec, 0, nall),
    KOKKOS_LAMBDA(const int i) {
      const int t_lmp = d_type(i);
      d_type_mapped_l(i) = d_type_map_l(t_lmp);
    });

  // Build compact NN/NL on device from Kokkos neighbor list.
  // IMPORTANT: this is just a repacking into the NEP library's fixed-stride
  // NN/NL layout. We do not filter by distance here; the NEP kernels will
  // apply the true cutoff. This avoids duplicating minimum-image + cutoff work.
  //
  // Also, we only rebuild when the LAMMPS neighbor list is rebuilt.
  auto klist = (NeighListKokkos<DeviceType> *) list;
  auto d_neighbors = klist->d_neighbors;
  auto d_numneigh = klist->d_numneigh;
  auto d_ilist = klist->d_ilist;
  const int inum = klist->inum;
  const int neigh_stride = d_neighbors.extent_int(1);
  const int neigh_rows = d_neighbors.extent_int(0);
  const int numneigh_len = d_numneigh.extent_int(0);
  if (inum != nlocal) {
    error->one(FLERR, "NEP_GPU/kk: expected Kokkos neighbor list inum == nlocal for full neighbor list.");
  }
  // Rebuild compact NN/NL every timestep (every compute call).
  //
  // LAMMPS neighbor lists include neighbors out to (cutoff + skin) and are only
  // rebuilt occasionally. If we cache a *distance-filtered* NN/NL list and only
  // refresh it when the neighbor list rebuilds, neighbors can cross the true
  // cutoff between rebuilds but would never be added to NN/NL. That breaks
  // force/energy consistency and can show up as NVE energy drift.
  neighbors_packed_ = true;
  packed_list_ptr_ = (void *) list;

  auto d_nn_r = d_nn_radial;
  auto d_nn_a = d_nn_angular;
  auto d_nl_r = d_nl_radial;
  auto d_nl_a = d_nl_angular;
  auto d_over = d_overflow;
  auto d_x = atomKK->k_x.view<DeviceType>();
  auto d_rc_r = d_rc_radial_by_type;
  auto d_rc_a = d_rc_angular_by_type;

  Kokkos::deep_copy(exec, d_over, 0);
  Kokkos::deep_copy(exec, d_nn_r, 0);
  Kokkos::deep_copy(exec, d_nn_a, 0);

  // Optional: for single-rank (no MPI) runs, we can collapse periodic ghost
  // indices back to their owning local atom index and run the NEP backend in
  // "no-ghost" mode (N == Nloc). This enables the gather formulation in the
  // backend, which can improve symmetry and momentum conservation.
  //
  // Enable with: NEP_GPU_LMP_SINGLE_RANK_GATHER=1
  const bool want_owner_map = (comm->nprocs == 1) && (nall > nlocal);
  const bool want_single_rank_gather =
    want_owner_map && (std::getenv("NEP_GPU_LMP_SINGLE_RANK_GATHER") != nullptr);
  bool owner_map_ok = false;
  bool single_rank_gather_ok = false;
  int backend_natoms = nall;
  if (want_owner_map) {
    atomKK->sync(Host, TAG_MASK);
    Kokkos::fence();
    if (d_owner.extent_int(0) != nall) {
      d_owner = Kokkos::View<int*, DeviceType>(Kokkos::NoInit("nep_gpu:owner"), nall);
    }
    auto h_owner = Kokkos::create_mirror_view(d_owner);
    for (int i = 0; i < nlocal; ++i) h_owner(i) = i;

    bool ok = true;
    for (int i = nlocal; i < nall; ++i) {
      const tagint t = atom->tag[i];
      const int owner = atom->map(t);
      if (owner < 0 || owner >= nlocal) {
        ok = false;
        break;
      }
      h_owner(i) = owner;
    }
    if (ok) {
      Kokkos::deep_copy(exec, d_owner, h_owner);
      exec.fence();
      owner_map_ok = true;
      if (want_single_rank_gather) {
        single_rank_gather_ok = true;
        backend_natoms = nlocal;
      }
    }

  }

  // IMPORTANT: NEP trusts the LAMMPS neighbor list and associated ghost coordinates.
  // Normally, dx = x[j]-x[i] is already the correct periodic-image displacement for
  // this neighbor entry.
  //
  // The additional periodic wrapping logic below is only used in special single-rank
  // modes where neighbor indices can be remapped back to local owners (owner-map /
  // gather). Keep it disabled unless you know you need it for a consistency check.
  const bool need_mic = (nall == nlocal) || owner_map_ok || single_rank_gather_ok;
  double h_box[9];
  h_box[0] = domain->h[0]; h_box[3] = 0.0;          h_box[6] = 0.0;
  h_box[1] = domain->h[5]; h_box[4] = domain->h[1]; h_box[7] = 0.0;
  h_box[2] = domain->h[4]; h_box[5] = domain->h[3]; h_box[8] = domain->h[2];
  double hinv_box[9];
  if (need_mic) invert3x3_rowmajor(h_box, hinv_box);
  const int pbc_x = domain->xperiodic ? 1 : 0;
  const int pbc_y = domain->yperiodic ? 1 : 0;
  const int pbc_z = domain->zperiodic ? 1 : 0;

  const int mn_r_l = mn_r;
  const int mn_a_l = mn_a;
  const int neigh_stride_l = neigh_stride;
  const int neigh_rows_l = neigh_rows;
  const int numneigh_len_l = numneigh_len;
  const bool need_mic_l = need_mic;
  const bool use_owner_map_l = single_rank_gather_ok;
  const int backend_natoms_l = backend_natoms;
  auto d_owner_l = d_owner;
  const double h0 = h_box[0], h1 = h_box[1], h2 = h_box[2], h3 = h_box[3], h4 = h_box[4], h5 = h_box[5], h6 = h_box[6],
               h7 = h_box[7], h8 = h_box[8];
  const double hi0 = hinv_box[0], hi1 = hinv_box[1], hi2 = hinv_box[2], hi3 = hinv_box[3], hi4 = hinv_box[4], hi5 = hinv_box[5],
               hi6 = hinv_box[6], hi7 = hinv_box[7], hi8 = hinv_box[8];
  const int pbc_x_l = pbc_x, pbc_y_l = pbc_y, pbc_z_l = pbc_z;

  Kokkos::parallel_for(
    Kokkos::RangePolicy<ExecSpace>(exec, 0, inum),
    KOKKOS_LAMBDA(const int ii) {
      const int i = d_ilist(ii);
      if (i < 0 || i >= nlocal) {
        Kokkos::atomic_or(&d_over(), 2);
        return;
      }
      if (i >= neigh_rows_l || i >= numneigh_len_l) {
        Kokkos::atomic_or(&d_over(), 16);
        return;
      }
      int jnum = d_numneigh(i);
      if (jnum > neigh_stride_l) {
        // Guard against reading past the allocated 2D neighbor view (can happen with some Kokkos neighbor layouts).
        Kokkos::atomic_or(&d_over(), 8);
        jnum = neigh_stride_l;
      }

      int nr = 0;
      int na = 0;
      bool overflow_mn = false;

      for (int jj = 0; jj < jnum; ++jj) {
        const int jraw = d_neighbors(i, jj);
        const int j0 = jraw & NEIGHMASK;
        if (j0 < 0 || j0 >= nall) continue;
        const int j = use_owner_map_l ? d_owner_l(j0) : j0;
        if (j < 0 || j >= backend_natoms_l) continue;
        if (j == i) continue;

          // Filter neighbors by the true NEP cutoffs (LAMMPS list includes cutoff+skin).
          // NOTE: for periodic ghosts, ghost coordinates are already shifted so dx,dy,dz
          // correspond to the correct periodic image for this neighbor entry.
          double dx = static_cast<double>(d_x(j, 0) - d_x(i, 0));
          double dy = static_cast<double>(d_x(j, 1) - d_x(i, 1));
          double dz = static_cast<double>(d_x(j, 2) - d_x(i, 2));

          if (need_mic_l) {
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
          const int ti = d_type_mapped_l(i);
          const int tj = d_type_mapped_l(j);
          const double rc_rad2 = 0.25 * (d_rc_r(ti) + d_rc_r(tj)) * (d_rc_r(ti) + d_rc_r(tj));
          const double rc_ang2 = 0.25 * (d_rc_a(ti) + d_rc_a(tj)) * (d_rc_a(ti) + d_rc_a(tj));

          if (rsq <= rc_rad2) {
            if (nr < mn_r_l) d_nl_r(i + nlocal * nr++) = j;
            else overflow_mn = true;
          }
        if (rsq <= rc_ang2) {
          if (na < mn_a_l) d_nl_a(i + nlocal * na++) = j;
          else overflow_mn = true;
        }
        if (nr >= mn_r_l && na >= mn_a_l) break;
      }

      d_nn_r(i) = nr;
      d_nn_a(i) = na;
      if (overflow_mn) Kokkos::atomic_or(&d_over(), 1);
    });

  int overflow = 0;
  Kokkos::deep_copy(exec, overflow, d_overflow);
  if (overflow & 2) error->one(FLERR, "NEP_GPU/kk: Kokkos ilist contained non-local indices; cannot build NN/NL safely.");
  if (overflow & 1) error->one(FLERR, "NEP_GPU/kk: neighbor list exceeds MN_*; increase MN_* or reduce neighbor skin.");
  if (overflow & 8) error->one(FLERR, "NEP_GPU/kk: d_numneigh(ii) exceeded d_neighbors stride; neighbor view layout mismatch (would have read OOB).");
  if (overflow & 16) error->one(FLERR, "NEP_GPU/kk: d_ilist(ii) index exceeded d_neighbors/d_numneigh extents; neighbor list view layout mismatch.");

  // We call into a non-Kokkos CUDA backend next; fence to guarantee NN/NL and
  // mapped types are ready on device (and to avoid stream ordering surprises).
  exec.fence();

  // Call NEP backend with device pointers (no host staging)
  NepGpuLammpsSystemDevice sys;
  sys.natoms = backend_natoms;
  sys.type = d_type_mapped.data();
  constexpr bool x_direct_ok = NepKokkosAosTraits<decltype(d_x)>::direct_ok;
  bool did_pack_aos = false;
  sys.xyz = nep_gpu_kokkos_utils::NepXyzPtr<x_direct_ok, DeviceType, decltype(d_x)>::get(
    exec, d_x, d_xyz_aos, backend_natoms, did_pack_aos, "nep_gpu:xyz_aos");
  sys.owner = owner_map_ok ? d_owner.data() : nullptr;
#if defined(KOKKOS_ENABLE_CUDA)
  sys.stream = (void*) exec.cuda_stream();
#else
  sys.stream = nullptr;
#endif
  if (did_pack_aos) exec.fence();

  // Construct triclinic box matrix from LAMMPS domain (Voigt ordering: xx,yy,zz,yz,xz,xy)
  sys.h[0] = domain->h[0];
  sys.h[3] = 0.0;
  sys.h[6] = 0.0;
  sys.h[1] = domain->h[5];
  sys.h[4] = domain->h[1];
  sys.h[7] = 0.0;
  sys.h[2] = domain->h[4];
  sys.h[5] = domain->h[3];
  sys.h[8] = domain->h[2];
  sys.pbc_x = domain->xperiodic ? 1 : 0;
  sys.pbc_y = domain->yperiodic ? 1 : 0;
  sys.pbc_z = domain->zperiodic ? 1 : 0;

  // Ensure "small box" vs "replicated big box" consistency:
  // with nearest-image neighbor lists, cutoff must be <= half the periodic thickness.
  {
    const bool strict_box = (std::getenv("NEP_GPU_LMP_STRICT_BOX") != nullptr);
    const NepGpuModelInfo& info = nep_model_lmp->info();
    const double rc_max = std::max(
      std::max(info.rc_radial_max, info.rc_angular_max),
      info.zbl_outer_max);
    double thickness[3];
    compute_box_thickness(sys.h, thickness);
    double min_th = 1.0e300;
    if (sys.pbc_x) min_th = std::min(min_th, thickness[0]);
    if (sys.pbc_y) min_th = std::min(min_th, thickness[1]);
    if (sys.pbc_z) min_th = std::min(min_th, thickness[2]);
    if (strict_box && min_th < 1.0e299 && rc_max > 0.5 * min_th) {
      error->all(FLERR,
        "NEP_GPU/kk: cutoff exceeds half periodic box thickness; replicate the cell (LAMMPS 'replicate') or reduce cutoff.");
    }
  }

  NepGpuLammpsNeighborsDevice nb;
  nb.NN_radial = d_nn_radial.data();
  nb.NL_radial = d_nl_radial.data();
  nb.NN_angular = d_nn_angular.data();
  nb.NL_angular = d_nl_angular.data();

  NepGpuLammpsResultDevice res;
  res.eatom = nullptr;
  res.vatom = nullptr;
  // Backend writes AoS (x0,y0,z0,...) forces. If Kokkos stores forces in
  // LayoutLeft (SoA), compute into a temporary AoS buffer and scatter-add.
  constexpr bool f_direct_ok = NepKokkosAosTraits<decltype(d_f)>::direct_ok;
  constexpr bool need_scatter_f = !f_direct_ok;
  if constexpr (f_direct_ok) {
    res.f = (double *) d_f.data();
  } else {
    if (d_f_aos.extent_int(0) != 3 * nall) {
      d_f_aos = Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_gpu:f_aos"), 3 * nall);
    }
    Kokkos::deep_copy(exec, d_f_aos, 0.0);
    res.f = (double *) d_f_aos.data();
  }

  const bool need_energy = (eflag != 0);
  const bool need_virial = (vflag != 0);

  // Optional: compute total energy by summing per-atom energies (like the non-/kk
  // wrapper does), rather than trusting the backend's device reduction. This can
  // reduce run-to-run noise in Etot/Econserve caused purely by reduction order.
  const bool totals_from_eatom_host = (std::getenv("NEP_GPU_LMP_TOTALS_FROM_EATOM_HOST") != nullptr);
  const bool totals_from_eatom = totals_from_eatom_host || (std::getenv("NEP_GPU_LMP_TOTALS_FROM_EATOM") != nullptr);
  const bool want_eatom_buffer = eflag_atom || totals_from_eatom;
  if (want_eatom_buffer && (need_energy || eflag_atom)) {
    if (d_eatom.extent_int(0) != nlocal) {
      d_eatom = Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_gpu:eatom"), nlocal);
    }
    res.eatom = d_eatom.data();
  }

  // Keep per-atom virial off by default (large write bandwidth). Enable only
  // when LAMMPS explicitly requests per-atom virial output.
  const bool want_vatom_buffer = (vflag_atom != 0) || (cvflag_atom != 0);
  if (want_vatom_buffer) {
    if (d_vatom.extent_int(0) != 9 * nlocal) {
      d_vatom = Kokkos::View<double*, DeviceType>(Kokkos::NoInit("nep_gpu:vatom"), 9 * nlocal);
    }
    res.vatom = d_vatom.data();
  }

  nep_model_lmp->compute_device(sys, nlocal, nb, res, need_energy, need_virial);

  // Ensure forces/outputs are complete before LAMMPS proceeds with more Kokkos work.
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
    exec.fence();
  }

  // Mark device-side force buffers as modified only after all device kernels
  // have finished (backend + optional scatter-add). This ensures any subsequent
  // sync treats device as the authoritative source.
  atomKK->modified(execution_space, datamask_modify);

  // Reverse-communicate ghost forces back to owning ranks (required for MPI correctness).
  if (nall > nlocal) {
    atomKK->modified(execution_space, F_MASK);

    // In single-rank gather mode the backend only writes forces for owned atoms
    // (no ghost contributions), so reverse_comm is unnecessary.
    if (!single_rank_gather_ok) {
      comm->reverse_comm();
    }

    if (!single_rank_gather_ok) {
      // Some LAMMPS/Kokkos communication paths update host-side force buffers.
      // Ensure we sync those MPI-reduced results back to the device before marking
      // device data as modified; otherwise a later device->host sync can overwrite
      // the communicated forces and make results MPI-rank dependent.
      atomKK->sync(execution_space, F_MASK);
      Kokkos::fence();
      atomKK->modified(execution_space, F_MASK);
    }
  }

  if (eflag) {
    double eng_local = res.eng;
    if (totals_from_eatom) {
      double eng_sum = 0.0;
      if (totals_from_eatom_host) {
        auto h_e = Kokkos::create_mirror_view(d_eatom);
        Kokkos::deep_copy(h_e, d_eatom);
        for (int i = 0; i < nlocal; ++i) eng_sum += h_e(i);
      } else {
        auto d_eatom_l = d_eatom;
        Kokkos::parallel_reduce(
          Kokkos::RangePolicy<ExecSpace>(exec, 0, nlocal),
          KOKKOS_LAMBDA(const int i, double& lsum) { lsum += d_eatom_l(i); },
          eng_sum);
        exec.fence();
      }

      eng_local = eng_sum;
    }
    eng_vdwl += eng_local;
  }
  if (vflag) {
    virial[0] += res.virial[0];
    virial[1] += res.virial[1];
    virial[2] += res.virial[2];
    virial[3] += res.virial[3];
    virial[4] += res.virial[4];
    virial[5] += res.virial[5];
  }

  // Optional per-atom outputs: copy device arrays back and accumulate on host
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
      error->one(FLERR, "NEP_GPU/kk: per-atom virial requested but neither vatom nor cvatom is allocated");
    }
    if (cvflag_atom && !cvatom && !vatom) {
      error->one(FLERR, "NEP_GPU/kk: centroid per-atom virial requested but neither cvatom nor vatom is allocated");
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
template class PairNEPGPUKokkos<LMPDeviceType>;
} // namespace LAMMPS_NS

#endif // LMP_KOKKOS_GPU || KOKKOS_ENABLE_CUDA
#endif // LMP_KOKKOS
