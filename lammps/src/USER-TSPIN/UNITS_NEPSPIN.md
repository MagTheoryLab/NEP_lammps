# USER-TSPIN / NEP(SPIN) unit conventions (self-consistent)

This note documents the internal unit conventions used to keep the NEP spin pair styles and the USER-TSPIN fixes self-consistent.
It intentionally does not aim to be compatible with the LAMMPS SPIN “ecosystem” conventions.

## 1) Core conventions (fixed)

- `atom->sp[0..2]`: spin direction (unit vector)
- `atom->sp[3]`: spin magnitude interpreted as magnetic moment `|M|` in `muB` (Bohr magneton)
- Energy unit: `eV`
- NEP “magnetic force / effective field”:
  - `field = -dE/dM` with units `eV/muB`

LAMMPS does not automatically rescale `atom->sp`; it is just numbers stored in the atom data structure. In this project, we interpret those numbers as `muB` by convention.

## 2) Field <-> frequency conversion

`atom->fm` stores `H = -dE/dM` in `eV/muB` (field mode).

## 3) What the code supports (current)

### 3.1 `pair_style nep/spin/gpu` and `pair_style nep/spin/gpu/kk`

- `fm_units field`: write `H = -dE/dM` (`eV/muB`) into `atom->fm`

There is no `spin_units` option: NEP always receives `sp[3]` as `muB`.

### 3.2 USER-TSPIN fixes (`fix glsd/*`, `fix spin/gsld`, `fix tspin/*`)

- `fix glsd/*` requires `atom->fm` in field mode: `H = -dE/dM` (`eV/muB`).
- There is no `spin_units` option: USER-TSPIN always treats `sp[3]` as `muB`.
- `fix precession/spin` (from SPIN package) is not supported together with USER-TSPIN fixes and will error out at init time.
- External Zeeman-like fields can be added with `fix tspin/precession/spin` (`zeeman` takes `H` directly in `eV/muB` and adds it to `atom->fm`) or `fix setforce/spin` (and are replayed during GLSD midpoint recomputes).

### 3.3 `min_style tspin/cg` (synergistic lattice+spin minimization)

This project includes `min_style tspin/cg`, which follows the “pseudoatom” synergistic optimization idea described in *Phys. Rev. B* **111**, 134412 (2025):

- Paper mapping: `R_ip = R_i + (ηζ) * S_i` with `S_i` in `muB` and `ηζ` in `length/muB`.
- In code, this scaling is exposed directly as `eta_zeta` (`length/muB`).
- The corresponding “pseudoatom force” scaling is `1/eta_zeta` (`muB/length`), so `F_ip = H / eta_zeta` converts `H = -dE/dM` (`eV/muB`) to a force-like quantity (`eV/length`).

Practical usage: set `eta_zeta` from the paper/training (e.g. `0.227` `Å/muB` for their Fe example), then tune `spin_dmax` as a per-iteration step limit for the spin pseudo-coordinate (same unit as `dmax`).

Robust usage (less empirical): enable adaptive scaling with
`min_modify eta_auto yes eta_auto_weight 0.05 eta_auto_min 1e-4 eta_auto_max 10.0`.
With `eta_auto yes`, the provided `eta_zeta` acts as an initial value and is updated only after each successful line-search step (kept fixed during the line search).

Kokkos usage: with the KOKKOS package enabled, use `min_style tspin/cg/kk` (or `tspin/cg/kk/device`, `tspin/cg/kk/host`) to run the same minimizer under Kokkos minimize. The same `min_modify` keywords apply (`eta_zeta`, `eta_auto`, `eta_auto_weight`, `eta_auto_min`, `eta_auto_max`, `spin_dmax`, `vary_mag`).

## 4) USER-TSPIN restart continuity policy

For `fix tspin/nh`, `fix tspin/nh/kk`, `fix glsd/nh`, and `fix glsd/nh/kk`:

- New restart files use a USER-TSPIN versioned global payload (marker + version) in addition to the base `FixNH` state.
- Per-atom USER-TSPIN state is now persisted via per-atom restart callbacks:
  - `tspin/*`: `vs`, `sreal`, `smass`, `isspin`
  - `glsd/*`: `fm_cache`, `s0_cache`
- CPU and Kokkos paths are intended to be feature-parity for restart semantics.

Legacy compatibility:

- Old restart files (without USER-TSPIN marker/payload) are still readable.
- In that case, USER-TSPIN enters documented fallback initialization/rebuild paths and emits one compatibility warning per affected fix instance.
- This fallback is safe-to-run but not guaranteed to be strictly continuation-equivalent to a no-restart trajectory.
