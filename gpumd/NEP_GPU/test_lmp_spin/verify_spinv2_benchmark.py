#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


BAR_PER_EV_A3 = 1602176.6208


@dataclass
class CompareStats:
    max_abs: float
    rmse: float


def read_numeric_rows(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])
    return rows


def load_reference_scalar(path: Path, col: int = 0) -> float:
    rows = read_numeric_rows(path)
    if len(rows) != 1 or len(rows[0]) <= col:
        raise ValueError(f"Unexpected scalar file format: {path}")
    return rows[0][col]


def load_reference_vec3(path: Path, cols: Sequence[int] = (0, 1, 2)) -> List[List[float]]:
    rows = read_numeric_rows(path)
    out: List[List[float]] = []
    for row in rows:
        out.append([row[cols[0]], row[cols[1]], row[cols[2]]])
    return out


def load_reference_virial6(path: Path) -> List[float] | None:
    rows = read_numeric_rows(path)
    if not rows:
        return None
    row = rows[0]
    if len(row) < 6:
        return None
    vals = row[:6]
    if any(v <= -9.0e5 for v in vals):
        return None
    return vals


def resolve_lammps_exe(explicit: str | None) -> str:
    if explicit:
        return explicit
    for name in ("lmp", "lmp.exe", "lmp_serial", "lmp_mpi", "lammps"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(
        "Could not find a LAMMPS executable. Pass --lmp-exe explicitly."
    )


def make_lammps_input(data_file: Path, nep_file: Path) -> str:
    data_str = str(data_file).replace("\\", "/")
    nep_str = str(nep_file).replace("\\", "/")
    return f"""units           metal
dimension       3
boundary        p p p
atom_style      spin
atom_modify     map array sort 0 0

read_data       {data_str}
mass            1 55.845

pair_style      nep/spin/gpu fm_units energy
pair_coeff      * * {nep_str} Fe

neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes

compute spinprop all property/atom sp spx spy spz fmx fmy fmz fx fy fz

thermo          1
thermo_style    custom step pe pxx pyy pzz pxy pxz pyz vol
thermo_modify   line one format float %20.12g

dump            verify all custom 1 dump.verify id type x y z c_spinprop[5] c_spinprop[6] c_spinprop[7] c_spinprop[8] c_spinprop[9] c_spinprop[10]
dump_modify     verify sort id format float %20.12g

run 0

variable epa equal pe/count(all)
variable pxx_now equal pxx
variable pyy_now equal pyy
variable pzz_now equal pzz
variable pxy_now equal pxy
variable pxz_now equal pxz
variable pyz_now equal pyz
variable vol_now equal vol
print "${{epa}}" file result_energy.txt screen no
print "${{pxx_now}} ${{pyy_now}} ${{pzz_now}} ${{pxy_now}} ${{pxz_now}} ${{pyz_now}} ${{vol_now}}" file result_pressure.txt screen no
"""


def parse_dump_vectors(path: Path) -> tuple[List[List[float]], List[List[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_header = "ITEM: ATOMS "
    idx = -1
    for i, line in enumerate(lines):
      if line.startswith(atom_header):
        idx = i
    if idx < 0:
        raise ValueError(f"No ATOMS section found in dump: {path}")
    cols = lines[idx][len(atom_header):].split()
    wanted = ["id", "type", "x", "y", "z", "c_spinprop[5]", "c_spinprop[6]", "c_spinprop[7]", "c_spinprop[8]", "c_spinprop[9]", "c_spinprop[10]"]
    if cols != wanted:
        raise ValueError(f"Unexpected dump columns: {cols}")
    natoms = int(lines[idx - 7].strip())
    rows = lines[idx + 1: idx + 1 + natoms]
    mforce: List[List[float]] = []
    force: List[List[float]] = []
    for row in rows:
        toks = row.split()
        mforce.append([float(toks[5]), float(toks[6]), float(toks[7])])
        force.append([float(toks[8]), float(toks[9]), float(toks[10])])
    return force, mforce


def flatten(rows: Iterable[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for row in rows:
        out.extend(float(x) for x in row)
    return out


def compare_arrays(actual: Sequence[float], reference: Sequence[float]) -> CompareStats:
    if len(actual) != len(reference):
        raise ValueError(f"Length mismatch: {len(actual)} != {len(reference)}")
    if not actual:
        return CompareStats(max_abs=0.0, rmse=0.0)
    diffs = [a - b for a, b in zip(actual, reference)]
    max_abs = max(abs(d) for d in diffs)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return CompareStats(max_abs=max_abs, rmse=rmse)


def load_pressure_and_convert(path: Path, natoms: int) -> dict[str, List[float] | float]:
    rows = read_numeric_rows(path)
    if len(rows) != 1 or len(rows[0]) != 7:
        raise ValueError(f"Unexpected pressure file format: {path}")
    pxx, pyy, pzz, pxy, pxz, pyz, volume = rows[0]
    virial_pos = [pxx, pyy, pzz, pxy, pxz, pyz]
    virial_pos = [v * volume / BAR_PER_EV_A3 / natoms for v in virial_pos]
    virial_neg = [-v for v in virial_pos]
    return {"volume": volume, "virial_pos": virial_pos, "virial_neg": virial_neg}


def run_benchmark(bench_root: Path, lmp_exe: str, keep_run_dir: Path | None) -> dict:
    pred_dir = bench_root / "pred"
    nep_file = bench_root / "nep.txt"
    data_file = bench_root / "single.dat"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Missing pred directory: {pred_dir}")
    if not nep_file.is_file():
        raise FileNotFoundError(f"Missing potential file: {nep_file}")
    if not data_file.is_file():
        raise FileNotFoundError(f"Missing LAMMPS data file: {data_file}")

    ref_energy = load_reference_scalar(pred_dir / "energy_train.out", col=0)
    ref_force = load_reference_vec3(pred_dir / "force_train.out", cols=(0, 1, 2))
    ref_mforce = load_reference_vec3(pred_dir / "mforce_train.out", cols=(0, 1, 2))
    ref_virial = load_reference_virial6(pred_dir / "virial_train.out")
    natoms = len(ref_force)

    ctx = tempfile.TemporaryDirectory(prefix="spinv2_verify_")
    run_dir = Path(ctx.name)
    if keep_run_dir is not None:
        run_dir = keep_run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nep_file, run_dir / "nep.txt")
    shutil.copy2(data_file, run_dir / "single.dat")
    (run_dir / "in.verify").write_text(
        make_lammps_input(run_dir / "single.dat", run_dir / "nep.txt"),
        encoding="utf-8",
        newline="\n",
    )

    proc = subprocess.run(
        [lmp_exe, "-in", "in.verify"],
        cwd=run_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed with exit code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    actual_energy = load_reference_scalar(run_dir / "result_energy.txt", col=0)
    actual_force, actual_mforce = parse_dump_vectors(run_dir / "dump.verify")
    pressure_info = load_pressure_and_convert(run_dir / "result_pressure.txt", natoms)

    energy_stats = compare_arrays([actual_energy], [ref_energy])
    force_stats = compare_arrays(flatten(actual_force), flatten(ref_force))
    mforce_stats = compare_arrays(flatten(actual_mforce), flatten(ref_mforce))

    virial_report = None
    if ref_virial is not None:
        virial_pos_stats = compare_arrays(pressure_info["virial_pos"], ref_virial)
        virial_neg_stats = compare_arrays(pressure_info["virial_neg"], ref_virial)
        use_neg = virial_neg_stats.rmse < virial_pos_stats.rmse
        virial_report = {
            "reference": ref_virial,
            "actual": pressure_info["virial_neg"] if use_neg else pressure_info["virial_pos"],
            "sign_convention": "negative_pressure" if use_neg else "positive_pressure",
            "max_abs": virial_neg_stats.max_abs if use_neg else virial_pos_stats.max_abs,
            "rmse": virial_neg_stats.rmse if use_neg else virial_pos_stats.rmse,
        }

    report = {
        "bench_root": str(bench_root),
        "run_dir": str(run_dir),
        "lmp_exe": lmp_exe,
        "natoms": natoms,
        "energy": {
            "reference": ref_energy,
            "actual": actual_energy,
            "abs_diff": energy_stats.max_abs,
        },
        "force": {
            "max_abs": force_stats.max_abs,
            "rmse": force_stats.rmse,
        },
        "mforce": {
            "max_abs": mforce_stats.max_abs,
            "rmse": mforce_stats.rmse,
        },
        "virial": virial_report,
        "pressure_volume": pressure_info,
    }
    if keep_run_dir is None:
        ctx.cleanup()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single-frame SpinV2 LAMMPS benchmark and compare against pred/*.out.")
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=Path(r"D:\Desktop\benchmark\SpinV2"),
        help="Benchmark root containing nep.txt, single.dat and pred/.",
    )
    parser.add_argument("--lmp-exe", help="LAMMPS executable path. If omitted, the script tries PATH.")
    parser.add_argument("--keep-run-dir", type=Path, help="Keep the generated LAMMPS run directory for inspection.")
    parser.add_argument("--json-out", type=Path, help="Optional path to write the JSON report.")
    args = parser.parse_args(argv)

    lmp_exe = resolve_lammps_exe(args.lmp_exe)
    report = run_benchmark(args.bench_root.resolve(), lmp_exe, args.keep_run_dir)

    print(f"bench_root: {report['bench_root']}")
    print(f"lmp_exe:    {report['lmp_exe']}")
    print(f"natoms:     {report['natoms']}")
    print(f"energy abs diff:   {report['energy']['abs_diff']:.12g}")
    print(f"force  max abs:    {report['force']['max_abs']:.12g}")
    print(f"force  rmse:       {report['force']['rmse']:.12g}")
    print(f"mforce max abs:    {report['mforce']['max_abs']:.12g}")
    print(f"mforce rmse:       {report['mforce']['rmse']:.12g}")
    if report["virial"] is not None:
        print(f"virial sign:       {report['virial']['sign_convention']}")
        print(f"virial max abs:    {report['virial']['max_abs']:.12g}")
        print(f"virial rmse:       {report['virial']['rmse']:.12g}")

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
