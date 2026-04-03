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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// NepGpuModel interface; provided by NEP_GPU/src/nep_gpu_model.cuh
#include "nep_gpu_model.cuh"

// This macro is updated by Install.sh based on LAMMPS's version.h
#define LAMMPS_VERSION_NUMBER 20220324

using namespace LAMMPS_NS;

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
  allocated = 0;
  nep_model = nullptr;
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

  delete nep_model;
  nep_model = nullptr;
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

  // This GPU backend is currently implemented for single-MPI-rank runs only.
  // Abort early if the user attempts to use it with more than one rank.
  if (comm->nprocs != 1) {
    error->all(FLERR, "NEP_GPU: pair_style nep/gpu currently supports only a single MPI rank");
  }

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

  // Construct the GPU NEP model. Start from a simple upper bound based on
  // the current total atom count and let compute() grow the capacity
  // dynamically if needed. We do not account for ghost atoms here.
  int max_atoms = atom->natoms;
  if (max_atoms <= 0) max_atoms = 1;

  if (nep_model) {
    delete nep_model;
    nep_model = nullptr;
  }
  nep_model = new NepGpuModel(model_path_from_lammps.c_str(), max_atoms);
  max_atoms_gpu = max_atoms;
  // For now, always use GPU-0; in multi-GPU environments this can be extended.
  nep_model->set_device(0);

  // get cutoff from NEP model
  cutoff = nep_model->cutoff();
  cutoffsq = cutoff * cutoff;
  int n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      cutsq[i][j] = cutoffsq;
}

void PairNEPGPU::settings(int narg, char **arg)
{
  if (narg > 0) {
    error->all(FLERR, "Illegal pair_style command; nep/gpu doesn't require any parameter");
  }
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
  if (!nep_model) {
    error->all(FLERR, "NEP_GPU: model not initialized; check pair_coeff");
  }

  if (eflag || vflag) {
    ev_setup(eflag, vflag);
  }

  const int nlocal = atom->nlocal;
  // Single-rank mode: only pass owned atoms; the GPU backend handles PBC itself.
  const int nall = nlocal;
  if (nlocal <= 0) return;

  // Ensure backend capacity covers local atoms; resize if needed.
  if (max_atoms_gpu > 0 && nall > max_atoms_gpu) {
    int new_max = std::max(nall * 2, max_atoms_gpu * 2);
    delete nep_model;
    std::string path = utils::get_potential_file_path(model_filename);
    nep_model = new NepGpuModel(path.c_str(), new_max);
    nep_model->set_device(0);
    max_atoms_gpu = new_max;
  }

  // Host buffers hold the atoms we pass to the GPU (owned only here).
  if (type_host.size() < static_cast<size_t>(nall)) type_host.resize(nall);
  if (xyz_host.size() < static_cast<size_t>(3) * nall) xyz_host.resize(static_cast<size_t>(3) * nall);
  if (f_host.size()   < static_cast<size_t>(3) * nlocal) f_host.resize(static_cast<size_t>(3) * nlocal);
  if (eatom_host.size() < static_cast<size_t>(nlocal)) eatom_host.resize(nlocal);
  if (vatom_host.size() < static_cast<size_t>(6) * nlocal) vatom_host.resize(static_cast<size_t>(6) * nlocal);

  // Map LAMMPS types to NEP types and gather positions.
  for (int i = 0; i < nall; ++i) {
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

  // Construct a triclinic box matrix from LAMMPS domain, consistent with
  // the non-spin nep/gpu pair style. LAMMPS stores:
  //   h[0]=xprd, h[1]=yprd, h[2]=zprd, h[3]=yz, h[4]=xz, h[5]=xy
  // and the triclinic vectors are:
  //   a = (xprd, 0,    0   )
  //   b = (xy,   yprd, 0   )
  //   c = (xz,   yz,   zprd)
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

  NepGpuSystem sys;
  sys.natoms = nall;
  sys.nlocal = nlocal;
  sys.type = type_host.data();
  sys.xyz = xyz_host.data();
  for (int i = 0; i < 9; ++i) {
    sys.h[i] = h[i];
  }
  sys.pbc_x = domain->xperiodic ? 1 : 0;
  sys.pbc_y = domain->yperiodic ? 1 : 0;
  sys.pbc_z = domain->zperiodic ? 1 : 0;

  NepGpuResult res;
  res.f = f_host.data();
  res.eatom = eflag_atom ? eatom_host.data() : nullptr;
  res.vatom = (vflag_atom || eflag_atom) ? vatom_host.data() : nullptr;

  nep_model->compute(sys, res);

  // Accumulate forces on local atoms.
  for (int i = 0; i < nlocal; ++i) {
    atom->f[i][0] += res.f[3 * i + 0];
    atom->f[i][1] += res.f[3 * i + 1];
    atom->f[i][2] += res.f[3 * i + 2];
  }

  if (eflag) {
    eng_vdwl += res.eng;
  }

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

  if (vflag_atom) {
    for (int i = 0; i < nlocal; ++i) {
      const int idx = 6 * i;
      cvatom[i][0] += vatom_host[idx + 0]; // xx
      cvatom[i][1] += vatom_host[idx + 1]; // yy
      cvatom[i][2] += vatom_host[idx + 2]; // zz
      cvatom[i][3] += vatom_host[idx + 3]; // xy
      cvatom[i][4] += vatom_host[idx + 4]; // xz
      cvatom[i][5] += vatom_host[idx + 5]; // yz
    }
  }
}
