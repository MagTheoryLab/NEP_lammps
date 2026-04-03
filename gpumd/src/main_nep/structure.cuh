/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "precision.cuh"
#include <vector>

class Parameters;

struct Structure {
  int num_cell[3];
  int num_atom;
  int has_virial;
  int has_atomic_virial;
  int atomic_virial_diag_only;
  int has_spin;
  int has_mforce;
  int has_bec;
  int has_temperature;
  StructReal weight;
  StructReal charge = 0.0f;
  StructReal energy = 0.0f;
  StructReal energy_weight = 1.0f;
  StructReal virial[6];
  StructReal box_original[9];
  StructReal volume;
  StructReal box[18];
  StructReal temperature;
  std::vector<int> type;
  std::vector<StructReal> x;
  std::vector<StructReal> y;
  std::vector<StructReal> z;
  std::vector<StructReal> fx;
  std::vector<StructReal> fy;
  std::vector<StructReal> fz;
  std::vector<StructReal> avirialxx;
  std::vector<StructReal> avirialyy;
  std::vector<StructReal> avirialzz;
  std::vector<StructReal> avirialxy;
  std::vector<StructReal> avirialyz;
  std::vector<StructReal> avirialzx;
  std::vector<StructReal> bec;
  // spin vectors per atom (optional)
  std::vector<SpinReal> sx;
  std::vector<SpinReal> sy;
  std::vector<SpinReal> sz;
  // magnetic force per atom (optional)
  std::vector<SpinReal> mfx;
  std::vector<SpinReal> mfy;
  std::vector<SpinReal> mfz;
};

bool read_structures(bool is_train, Parameters& para, std::vector<Structure>& structures);
