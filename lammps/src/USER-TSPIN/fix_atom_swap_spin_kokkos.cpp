/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_atom_swap_spin_kokkos.h"

#ifdef LMP_KOKKOS

#include "angle.h"
#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "random_park.h"
#include "update.h"
#include "utils.h"

#include "mpi.h"

#include <cmath>

using namespace LAMMPS_NS;

template <class DeviceType>
FixAtomSwapSpinKokkos<DeviceType>::FixAtomSwapSpinKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixAtomSwapSpin(lmp, narg, arg),
    atomKK((AtomKokkos *) atom),
    d_sp4("atom/swap/spin/kk:d_sp4"),
    h_sp4("atom/swap/spin/kk:h_sp4")
{
  kokkosable = 1;
  forward_comm_device = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::init()
{
  if (!atomKK) error->all(FLERR, "Fix atom/swap/spin/kk requires a Kokkos-enabled atom style");
  FixAtomSwapSpin::init();
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::read_spin_local_device(int local_index, double sp_out[4])
{
  for (int k = 0; k < 4; k++) sp_out[k] = 0.0;
  if (local_index < 0) return;

  auto sp = atomKK->k_sp.template view<DeviceType>();
  auto d_sp4_ = d_sp4;

  const int idx = local_index;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, 1), LAMMPS_LAMBDA(const int) {
    d_sp4_(0) = static_cast<double>(sp(idx, 0));
    d_sp4_(1) = static_cast<double>(sp(idx, 1));
    d_sp4_(2) = static_cast<double>(sp(idx, 2));
    d_sp4_(3) = static_cast<double>(sp(idx, 3));
  });
  DeviceType().fence();
  Kokkos::deep_copy(h_sp4, d_sp4);
  for (int k = 0; k < 4; k++) sp_out[k] = h_sp4(k);
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::write_spin_local_device(int local_index, const double sp_in[4])
{
  if (local_index < 0) return;

  auto sp = atomKK->k_sp.template view<DeviceType>();
  const int idx = local_index;
  const double s0 = sp_in[0];
  const double s1 = sp_in[1];
  const double s2 = sp_in[2];
  const double s3 = sp_in[3];

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, 1), LAMMPS_LAMBDA(const int) {
    sp(idx, 0) = s0;
    sp(idx, 1) = s1;
    sp(idx, 2) = s2;
    sp(idx, 3) = s3;
  });
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::write_type_local_device(int local_index, int new_type)
{
  if (local_index < 0) return;

  auto type = atomKK->k_type.template view<DeviceType>();
  const int idx = local_index;
  const int t = new_type;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, 1), LAMMPS_LAMBDA(const int) { type(idx) = t; });
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::write_q_local_device(int local_index, double new_q)
{
  if (local_index < 0) return;
  if (!atom->q_flag) return;

  auto q = atomKK->k_q.template view<DeviceType>();
  const int idx = local_index;
  const double v = new_q;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, 1), LAMMPS_LAMBDA(const int) { q(idx) = v; });
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::write_rmass_local_device(int local_index, double new_rmass)
{
  if (local_index < 0) return;
  if (atom->rmass == nullptr) return;

  auto rmass = atomKK->k_rmass.template view<DeviceType>();
  const int idx = local_index;
  const double v = new_rmass;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, 1), LAMMPS_LAMBDA(const int) { rmass(idx) = v; });
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::scale_v_local_device(int local_index, double factor)
{
  if (local_index < 0) return;

  auto v = atomKK->k_v.template view<DeviceType>();
  const int idx = local_index;
  const double f = factor;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, 1), LAMMPS_LAMBDA(const int) {
    v(idx, 0) *= f;
    v(idx, 1) *= f;
    v(idx, 2) *= f;
  });
}

template <class DeviceType>
double FixAtomSwapSpinKokkos<DeviceType>::energy_full_kokkos()
{
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) {
    atomKK->sync(force->pair->execution_space, force->pair->datamask_read);
    force->pair->compute(eflag, vflag);
    atomKK->modified(force->pair->execution_space, force->pair->datamask_modify);
  }

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) {
      atomKK->sync(force->bond->execution_space, force->bond->datamask_read);
      force->bond->compute(eflag, vflag);
      atomKK->modified(force->bond->execution_space, force->bond->datamask_modify);
    }
    if (force->angle) {
      atomKK->sync(force->angle->execution_space, force->angle->datamask_read);
      force->angle->compute(eflag, vflag);
      atomKK->modified(force->angle->execution_space, force->angle->datamask_modify);
    }
    if (force->dihedral) {
      atomKK->sync(force->dihedral->execution_space, force->dihedral->datamask_read);
      force->dihedral->compute(eflag, vflag);
      atomKK->modified(force->dihedral->execution_space, force->dihedral->datamask_modify);
    }
    if (force->improper) {
      atomKK->sync(force->improper->execution_space, force->improper->datamask_read);
      force->improper->compute(eflag, vflag);
      atomKK->modified(force->improper->execution_space, force->improper->datamask_modify);
    }
  }

  if (force->kspace) {
    atomKK->sync(force->kspace->execution_space, force->kspace->datamask_read);
    force->kspace->compute(eflag, vflag);
    atomKK->modified(force->kspace->execution_space, force->kspace->datamask_modify);
  }

  if (modify->n_post_force_any) {
    // MC energy evaluation must not advance spin/lattice integrators.
    // In particular, `fix glsd/*` has POST_FORCE and would evolve spins during the trial.
    // So we replay post_force fixes except for USER-TSPIN integrators.
    for (int ifix = 0; ifix < modify->nfix; ifix++) {
      Fix *f = modify->fix[ifix];
      if (!f) continue;
      if (!(modify->fmask[ifix] & FixConst::POST_FORCE)) continue;
      if (utils::strmatch(f->style, "^glsd/")) continue;
      if (utils::strmatch(f->style, "^tspin/")) continue;
      f->post_force(vflag);
    }
  }

  update->eflag_global = update->ntimestep;
  return c_pe->compute_scalar();
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::pre_exchange()
{
  if (next_reneighbor != update->ntimestep) return;

  // ensure current system is ready to compute energy
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);

  // Ensure required per-atom data is current on the device (selection is done on host separately).
  unsigned int devmask = TYPE_MASK | SP_MASK;
  if (atom->q_flag && !semi_grand_flag) devmask |= Q_MASK;
  if (atom->rmass != nullptr && !semi_grand_flag) devmask |= RMASS_MASK;
  if (ke_flag) devmask |= V_MASK;
  atomKK->sync(execution_space, devmask);

  // Sync only what we need for building candidate lists on host.
  unsigned int hostmask = TYPE_MASK | MASK_MASK;
  if (region) hostmask |= X_MASK;
  atomKK->sync(Host, hostmask);

  energy_stored = energy_full_kokkos();

  int nsuccess = 0;
  if (semi_grand_flag) {
    update_semi_grand_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_semi_grand();
  } else {
    update_swap_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_swap();
  }

  nswap_attempts += ncycles;
  nswap_successes += nsuccess;

  next_reneighbor = update->ntimestep + nevery;
}

template <class DeviceType>
int FixAtomSwapSpinKokkos<DeviceType>::attempt_semi_grand()
{
  if (nswap == 0) return 0;

  const double energy_before = energy_stored;

  int itype = 0;
  int jtype = 0;

  const int i = pick_semi_grand_atom();

  double sp_old[4] = {0.0, 0.0, 0.0, 0.0};
  if (i >= 0) {
    itype = atom->type[i];
    read_spin_local_device(i, sp_old);

    int jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
    jtype = type_list[jswaptype];
    while (itype == jtype) {
      jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
      jtype = type_list[jswaptype];
    }

    // update host-visible type for list maintenance
    atom->type[i] = jtype;

    // compute new spin consistent with new type
    double sp_new[4] = {sp_old[0], sp_old[1], sp_old[2], sp_old[3]};
    const double dx = spdir_type[3 * jtype + 0];
    const double dy = spdir_type[3 * jtype + 1];
    const double dz = spdir_type[3 * jtype + 2];
    const double dnorm2 = dx * dx + dy * dy + dz * dz;
    const double mag_new = spmag_type[jtype];

    if (mag_new <= 0.0 && dnorm2 == 0.0) {
      // no reference: keep old spin
      sp_new[0] = sp_old[0];
      sp_new[1] = sp_old[1];
      sp_new[2] = sp_old[2];
      sp_new[3] = sp_old[3];
    } else {
      if (dnorm2 > 0.0) {
        sp_new[0] = dx;
        sp_new[1] = dy;
        sp_new[2] = dz;
      } else {
        const double onorm = std::sqrt(sp_old[0] * sp_old[0] + sp_old[1] * sp_old[1] +
                                       sp_old[2] * sp_old[2]);
        if (onorm > 0.0) {
          sp_new[0] = sp_old[0] / onorm;
          sp_new[1] = sp_old[1] / onorm;
          sp_new[2] = sp_old[2] / onorm;
        } else {
          sp_new[0] = 1.0;
          sp_new[1] = 0.0;
          sp_new[2] = 0.0;
        }
      }
      sp_new[3] = mag_new;
    }

    write_type_local_device(i, jtype);
    write_spin_local_device(i, sp_new);
    DeviceType().fence();
    atomKK->modified(execution_space, TYPE_MASK | SP_MASK);
  }

  // if unequal_cutoffs, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  if (force->kspace) force->kspace->qsum_qsq();
  const double energy_after = energy_full_kokkos();

  int accept_local = 0;
  if (i >= 0) {
    if (random_unequal->uniform() <
        std::exp(beta * (energy_before - energy_after + mu[jtype] - mu[itype])))
      accept_local = 1;
  }

  int accept_all = 0;
  MPI_Allreduce(&accept_local, &accept_all, 1, MPI_INT, MPI_MAX, world);

  if (accept_all) {
    update_semi_grand_atoms_list();
    energy_stored = energy_after;
    if (ke_flag && i >= 0) {
      scale_v_local_device(i, sqrt_mass_ratio[itype][jtype]);
      DeviceType().fence();
      atomKK->modified(execution_space, V_MASK);
    }
    if (i >= 0) this->mc_sync_atom_custom_after_accept_local(i, itype, jtype);
    if (i >= 0) this->mc_sync_user_tspin_kokkos_local(i);
    return 1;
  }

  // reject: restore local atom (ghosts refreshed on next trial)
  if (i >= 0) {
    atom->type[i] = itype;
    write_type_local_device(i, itype);
    write_spin_local_device(i, sp_old);
    DeviceType().fence();
    atomKK->modified(execution_space, TYPE_MASK | SP_MASK);
  }

  if (force->kspace) force->kspace->qsum_qsq();
  return 0;
}

template <class DeviceType>
int FixAtomSwapSpinKokkos<DeviceType>::attempt_swap()
{
  if ((niswap == 0) || (njswap == 0)) return 0;

  const double energy_before = energy_stored;

  const int i = pick_i_swap_atom();
  const int j = pick_j_swap_atom();
  const int itype = type_list[0];
  const int jtype = type_list[1];

  // gather spin state (needed even when i/j are on different MPI ranks)
  double spi_local[4] = {0.0, 0.0, 0.0, 0.0};
  double spj_local[4] = {0.0, 0.0, 0.0, 0.0};
  if (i >= 0) read_spin_local_device(i, spi_local);
  if (j >= 0) read_spin_local_device(j, spj_local);
  double spi[4], spj[4];
  MPI_Allreduce(spi_local, spi, 4, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(spj_local, spj, 4, MPI_DOUBLE, MPI_SUM, world);

  // proposed swap: types (and q/rmass if present) + exchange spin vectors
  unsigned int modified_mask = TYPE_MASK | SP_MASK;
  if (i >= 0) {
    atom->type[i] = jtype;
    write_type_local_device(i, jtype);
    if (atom->q_flag) {
      atom->q[i] = qtype[1];
      write_q_local_device(i, qtype[1]);
      modified_mask |= Q_MASK;
    }
    if (atom->rmass != nullptr) {
      atom->rmass[i] = mtype[1];
      write_rmass_local_device(i, mtype[1]);
      modified_mask |= RMASS_MASK;
    }
    write_spin_local_device(i, spj);
  }
  if (j >= 0) {
    atom->type[j] = itype;
    write_type_local_device(j, itype);
    if (atom->q_flag) {
      atom->q[j] = qtype[0];
      write_q_local_device(j, qtype[0]);
      modified_mask |= Q_MASK;
    }
    if (atom->rmass != nullptr) {
      atom->rmass[j] = mtype[0];
      write_rmass_local_device(j, mtype[0]);
      modified_mask |= RMASS_MASK;
    }
    write_spin_local_device(j, spi);
  }

  DeviceType().fence();
  atomKK->modified(execution_space, modified_mask);

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  const double energy_after = energy_full_kokkos();

  int accept_local = 0;
  if (random_equal->uniform() < std::exp(beta * (energy_before - energy_after))) accept_local = 1;

  int accept_all = 0;
  MPI_Allreduce(&accept_local, &accept_all, 1, MPI_INT, MPI_MAX, world);

  if (accept_all) {
    update_swap_atoms_list();
    if (ke_flag) {
      if (i >= 0) scale_v_local_device(i, sqrt_mass_ratio[itype][jtype]);
      if (j >= 0) scale_v_local_device(j, sqrt_mass_ratio[jtype][itype]);
      DeviceType().fence();
      atomKK->modified(execution_space, V_MASK);
    }
    energy_stored = energy_after;
    if (i >= 0) this->mc_sync_atom_custom_after_accept_local(i, itype, jtype);
    if (j >= 0) this->mc_sync_atom_custom_after_accept_local(j, jtype, itype);
    if (i >= 0) this->mc_sync_user_tspin_kokkos_local(i);
    if (j >= 0) this->mc_sync_user_tspin_kokkos_local(j);
    return 1;
  }

  // reject: restore local atoms (ghosts refreshed on next trial)
  modified_mask = TYPE_MASK | SP_MASK;
  if (i >= 0) {
    atom->type[i] = itype;
    write_type_local_device(i, itype);
    if (atom->q_flag) {
      atom->q[i] = qtype[0];
      write_q_local_device(i, qtype[0]);
      modified_mask |= Q_MASK;
    }
    if (atom->rmass != nullptr) {
      atom->rmass[i] = mtype[0];
      write_rmass_local_device(i, mtype[0]);
      modified_mask |= RMASS_MASK;
    }
    write_spin_local_device(i, spi);
  }
  if (j >= 0) {
    atom->type[j] = jtype;
    write_type_local_device(j, jtype);
    if (atom->q_flag) {
      atom->q[j] = qtype[1];
      write_q_local_device(j, qtype[1]);
      modified_mask |= Q_MASK;
    }
    if (atom->rmass != nullptr) {
      atom->rmass[j] = mtype[1];
      write_rmass_local_device(j, mtype[1]);
      modified_mask |= RMASS_MASK;
    }
    write_spin_local_device(j, spj);
  }
  DeviceType().fence();
  atomKK->modified(execution_space, modified_mask);

  return 0;
}

template <class DeviceType>
int FixAtomSwapSpinKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist,
                                                               DAT::tdual_xfloat_1d &k_buf, int /*pbc_flag*/,
                                                               int * /*pbc*/)
{
  auto sendlist = k_sendlist.view<DeviceType>();
  auto buf = k_buf.view<DeviceType>();

  auto type = atomKK->k_type.template view<DeviceType>();
  auto sp = atomKK->k_sp.template view<DeviceType>();
  const int pack_q = atom->q_flag ? 1 : 0;
  auto q = atomKK->k_q.template view<DeviceType>();

  const int nsize = comm_forward;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n), LAMMPS_LAMBDA(const int ii) {
    const int j = sendlist(ii);
    int m = ii * nsize;
    buf(m++) = static_cast<double>(type(j));
    if (pack_q) buf(m++) = static_cast<double>(q(j));
    buf(m++) = static_cast<double>(sp(j, 0));
    buf(m++) = static_cast<double>(sp(j, 1));
    buf(m++) = static_cast<double>(sp(j, 2));
    buf(m++) = static_cast<double>(sp(j, 3));
  });

  return n * nsize;
}

template <class DeviceType>
void FixAtomSwapSpinKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first, DAT::tdual_xfloat_1d &k_buf)
{
  auto buf = k_buf.view<DeviceType>();

  auto type = atomKK->k_type.template view<DeviceType>();
  auto sp = atomKK->k_sp.template view<DeviceType>();
  const int unpack_q = atom->q_flag ? 1 : 0;
  auto q = atomKK->k_q.template view<DeviceType>();

  const int nsize = comm_forward;

  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0, n), LAMMPS_LAMBDA(const int ii) {
    const int i = first + ii;
    int m = ii * nsize;
    type(i) = static_cast<int>(buf(m++));
    if (unpack_q) q(i) = static_cast<double>(buf(m++));
    sp(i, 0) = static_cast<double>(buf(m++));
    sp(i, 1) = static_cast<double>(buf(m++));
    sp(i, 2) = static_cast<double>(buf(m++));
    sp(i, 3) = static_cast<double>(buf(m++));
  });

  atomKK->modified(execution_space, TYPE_MASK | (unpack_q ? Q_MASK : EMPTY_MASK) | SP_MASK);
}

namespace LAMMPS_NS {
template class FixAtomSwapSpinKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixAtomSwapSpinKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

#endif
