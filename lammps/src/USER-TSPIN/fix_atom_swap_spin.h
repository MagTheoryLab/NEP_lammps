/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(atom/swap/spin,FixAtomSwapSpin);
// clang-format on
#else

#ifndef LMP_FIX_ATOM_SWAP_SPIN_H
#define LMP_FIX_ATOM_SWAP_SPIN_H

#include "fix.h"

#include <vector>

namespace LAMMPS_NS {

class FixAtomSwapSpin : public Fix {
 public:
  FixAtomSwapSpin(class LAMMPS *, int, char **);
  ~FixAtomSwapSpin() override;

  int setmask() override;
  void init() override;
  void pre_exchange() override;

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  double compute_vector(int) override;
  double memory_usage() override;

 protected:
  int nevery, seed;
  int ke_flag;
  int semi_grand_flag;
  int ncycles;

  int niswap, njswap;                  // # of i,j swap atoms on all procs
  int niswap_local, njswap_local;      // # of swap atoms on this proc
  int niswap_before, njswap_before;    // # of swap atoms on procs < this proc
  int nswap, nswap_local, nswap_before;

  class Region *region;
  char *idregion;

  int nswaptypes;
  int *type_list;
  std::vector<double> mu_values;
  double *mu;    // chemical potentials indexed by atom type
  double *spmag_type;    // reference spin magnitude per type (sp[3])
  double *spdir_type;    // reference spin direction per type (length 3*(ntypes+1))

  double nswap_attempts;
  double nswap_successes;

  bool unequal_cutoffs;

  int atom_swap_nmax;
  double beta;
  double *qtype, *mtype;
  double energy_stored;
  double **sqrt_mass_ratio;
  int *local_swap_iatom_list;
  int *local_swap_jatom_list;
  int *local_swap_atom_list;

  class RanPark *random_equal;
  class RanPark *random_unequal;

  class Compute *c_pe;

  void options(int, char **);
  int attempt_swap();
  int attempt_semi_grand();
  double energy_full();
  int pick_semi_grand_atom();
  int pick_i_swap_atom();
  int pick_j_swap_atom();
  void update_semi_grand_atoms_list();
  void update_swap_atoms_list();

  void gather_spin(int local_index, double sp_out[4]) const;

  // Optional synchronization for USER-TSPIN integrators.
  // If a tspin integrator is active, atom->sp is mirrored in the custom per-atom state:
  //  - tspin_sreal (vector) and tspin_vs (spin "velocity")
  //  - tspin_smass (scalar) and tspin_isspin (int flag)
  // When MC swaps change atom->sp (and possibly type), those must be swapped consistently,
  // otherwise the integrator will overwrite atom->sp from stale tspin_sreal on the next step.
  int idx_tspin_vs = -1;
  int idx_tspin_sreal = -1;
  int idx_tspin_smass = -1;
  int idx_tspin_isspin = -1;

  // Optional synchronization for USER-TSPIN GLSD integrators.
  // GLSD uses per-atom caches (migrate with atoms) for midpoint iterations.
  // If MC swaps change atom->sp after GLSD saved its per-step S0 cache, the midpoint
  // step can become inconsistent unless the cache entries are swapped/updated too.
  int idx_glsd_fm_cache = -1;
  int idx_glsd_s0_cache = -1;

  void gather_custom_darray3(int idx, int local_index, double out[3]) const;
  void gather_custom_dvector(int idx, int local_index, double &out) const;
  void gather_custom_ivector(int idx, int local_index, int &out) const;

  // If running with USER-TSPIN Kokkos fixes, their per-atom state is stored inside the fix
  // (not Atom custom properties). After MC acceptance, ask those fixes to resync their state
  // for the affected local atoms.
  void mc_sync_user_tspin_kokkos_local(int local_index);

  // If USER-TSPIN host fixes are present, they store per-atom state in Atom custom properties.
  // When using the /kk MC fix, keep those properties consistent after an accepted swap.
  void mc_sync_atom_custom_after_accept_local(int local_index, int old_type, int new_type);
};

}    // namespace LAMMPS_NS

#endif
#endif
