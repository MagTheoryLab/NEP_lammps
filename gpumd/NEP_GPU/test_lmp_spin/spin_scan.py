#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class XYZFrame:
    natoms: int
    comment: str
    atom_lines: List[str]


_PROPERTIES_RE = re.compile(r"(?:^|\s)Properties=([^\s]+)")


def _read_all_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def read_xyz_frames(path: Path) -> List[XYZFrame]:
    lines = _read_all_lines(path)
    frames: List[XYZFrame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            natoms = int(lines[i].strip())
        except ValueError as e:
            raise ValueError(f"Invalid XYZ: expected natoms at line {i+1} in {path}") from e
        if i + 1 >= len(lines):
            raise ValueError(f"Invalid XYZ: missing comment line after natoms in {path}")
        comment = lines[i + 1].rstrip("\n")
        start = i + 2
        end = start + natoms
        if end > len(lines):
            raise ValueError(f"Invalid XYZ: truncated atom lines for a frame in {path}")
        atom_lines = [lines[j].rstrip("\n") for j in range(start, end)]
        frames.append(XYZFrame(natoms=natoms, comment=comment, atom_lines=atom_lines))
        i = end
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def _parse_properties(comment: str) -> Optional[List[Tuple[str, str, int]]]:
    m = _PROPERTIES_RE.search(comment)
    if not m:
        return None
    raw = m.group(1)
    parts = raw.split(":")
    if len(parts) % 3 != 0:
        return None
    props: List[Tuple[str, str, int]] = []
    for j in range(0, len(parts), 3):
        name = parts[j]
        typ = parts[j + 1]
        try:
            count = int(parts[j + 2])
        except ValueError:
            return None
        props.append((name, typ, count))
    return props


def _spin_token_indices(comment: str, atom_tokens: Sequence[str]) -> Tuple[int, int, int]:
    props = _parse_properties(comment)
    if props:
        has_species = any(name.lower() == "species" for name, _, _ in props)
        base = 0 if has_species else 1
        col = 0
        for name, _, count in props:
            if name.lower() == "spin":
                i0 = base + col
                if i0 + 2 >= len(atom_tokens):
                    raise ValueError("Spin columns out of range for detected Properties=")
                return (i0, i0 + 1, i0 + 2)
            col += count
    if len(atom_tokens) < 7:
        raise ValueError(
            "Could not detect spin columns from Properties= and line is too short; "
            "expected at least: element x y z sx sy sz"
        )
    return (len(atom_tokens) - 3, len(atom_tokens) - 2, len(atom_tokens) - 1)


def _parse_floats(tokens: Sequence[str], idxs: Sequence[int]) -> List[float]:
    out: List[float] = []
    for idx in idxs:
        try:
            out.append(float(tokens[idx]))
        except ValueError as e:
            raise ValueError(f"Expected float at column {idx+1} in atom line: {' '.join(tokens)}") from e
    return out


def _format_float(x: float, fmt: str) -> str:
    if fmt == "auto":
        return f"{x:.10g}"
    return fmt.format(x)


def _theta_deg_from_xy(sx: float, sy: float) -> float:
    deg = math.degrees(math.atan2(sy, sx))
    if deg < 0:
        deg += 360.0
    return deg


def generate_rotation_scan(
    base: XYZFrame,
    atom_index_1based: int,
    start_deg: float,
    stop_deg: float,
    step_deg: float,
    include_stop: bool,
    spin_mag: Optional[float],
    float_fmt: str,
    output_xyz: Path,
    output_meta_csv: Path,
    output_info_json: Path,
) -> None:
    if atom_index_1based < 1 or atom_index_1based > base.natoms:
        raise ValueError(f"--atom must be in [1,{base.natoms}]")
    atom0 = atom_index_1based - 1
    base_tokens = base.atom_lines[atom0].split()
    sidx = _spin_token_indices(base.comment, base_tokens)
    s0x, s0y, s0z = _parse_floats(base_tokens, sidx)
    if spin_mag is None:
        spin_mag = math.sqrt(s0x * s0x + s0y * s0y + s0z * s0z)
        if spin_mag == 0.0:
            spin_mag = 1.0

    thetas: List[float] = []
    theta = start_deg
    if step_deg <= 0:
        raise ValueError("--step-deg must be > 0")

    eps = 1e-12
    def done(curr: float) -> bool:
        return curr > stop_deg + eps if include_stop else curr >= stop_deg - eps

    while not done(theta):
        thetas.append(theta)
        theta += step_deg
        if len(thetas) > 1000000:
            raise ValueError("Too many scan points; check start/stop/step")

    with output_xyz.open("w", encoding="utf-8", newline="\n") as fxyz, output_meta_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "frame",
                "theta_deg",
                "theta_rad",
                "atom",
                "sx",
                "sy",
                "sz",
                "spin_mag",
            ],
        )
        writer.writeheader()

        for frame, theta_deg in enumerate(thetas):
            theta_rad = math.radians(theta_deg)
            sx = spin_mag * math.cos(theta_rad)
            sy = spin_mag * math.sin(theta_rad)
            sz = 0.0

            comment = (
                f"{base.comment} scan=spin_rotation scan_atom={atom_index_1based} "
                f"theta_deg={theta_deg:.10g}"
            ).strip()

            fxyz.write(f"{base.natoms}\n")
            fxyz.write(f"{comment}\n")

            for i, line in enumerate(base.atom_lines):
                if i != atom0:
                    fxyz.write(line.rstrip("\n") + "\n")
                    continue
                tokens = line.split()
                for k, v in zip(sidx, (sx, sy, sz)):
                    tokens[k] = _format_float(v, float_fmt)
                fxyz.write(" ".join(tokens) + "\n")

            writer.writerow(
                {
                    "frame": frame,
                    "theta_deg": theta_deg,
                    "theta_rad": theta_rad,
                    "atom": atom_index_1based,
                    "sx": sx,
                    "sy": sy,
                    "sz": sz,
                    "spin_mag": spin_mag,
                }
            )

    output_info_json.write_text(
        json.dumps(
            {
                "input_comment": base.comment,
                "natoms": base.natoms,
                "scan": "spin_rotation_xy",
                "atom": atom_index_1based,
                "spin_mag": spin_mag,
                "start_deg": start_deg,
                "stop_deg": stop_deg,
                "step_deg": step_deg,
                "include_stop": include_stop,
                "base_spin": {"sx": s0x, "sy": s0y, "sz": s0z},
                "spin_columns": {"sx": sidx[0], "sy": sidx[1], "sz": sidx[2]},
                "output_xyz": str(output_xyz),
                "output_meta_csv": str(output_meta_csv),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _pos_token_indices(comment: str, atom_tokens: Sequence[str]) -> Tuple[int, int, int]:
    props = _parse_properties(comment)
    if props:
        has_species = any(name.lower() == "species" for name, _, _ in props)
        base = 0 if has_species else 1
        col = 0
        for name, _, count in props:
            if name.lower() == "pos":
                i0 = base + col
                if i0 + 2 >= len(atom_tokens):
                    raise ValueError("Position columns out of range for detected Properties=")
                return (i0, i0 + 1, i0 + 2)
            col += count
    if len(atom_tokens) < 4:
        raise ValueError("Could not detect position columns; expected at least: element x y z")
    return (1, 2, 3)


def generate_distance_scan(
    base: XYZFrame,
    atom_i_1based: int,
    atom_j_1based: int,
    deltas: Sequence[float],
    symmetric: bool,
    float_fmt: str,
    output_xyz: Path,
    output_meta_csv: Path,
    output_info_json: Path,
) -> None:
    if atom_i_1based < 1 or atom_i_1based > base.natoms:
        raise ValueError(f"--atom-i must be in [1,{base.natoms}]")
    if atom_j_1based < 1 or atom_j_1based > base.natoms:
        raise ValueError(f"--atom-j must be in [1,{base.natoms}]")
    if atom_i_1based == atom_j_1based:
        raise ValueError("--atom-i and --atom-j must be different")
    if not deltas:
        raise ValueError("No deltas provided")

    i0 = atom_i_1based - 1
    j0 = atom_j_1based - 1
    ti = base.atom_lines[i0].split()
    tj = base.atom_lines[j0].split()
    pidx_i = _pos_token_indices(base.comment, ti)
    pidx_j = _pos_token_indices(base.comment, tj)
    ri = _parse_floats(ti, pidx_i)
    rj = _parse_floats(tj, pidx_j)
    dx, dy, dz = (rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2])
    dist0 = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist0 == 0.0:
        raise ValueError("Atoms i and j are at the same position; cannot define scan direction")
    ux, uy, uz = (dx / dist0, dy / dist0, dz / dist0)

    with output_xyz.open("w", encoding="utf-8", newline="\n") as fxyz, output_meta_csv.open(
        "w", encoding="utf-8", newline=""
    ) as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "frame",
                "atom_i",
                "atom_j",
                "delta",
                "distance",
            ],
        )
        writer.writeheader()

        for frame, delta in enumerate(deltas):
            if symmetric:
                ri_new = (ri[0] - 0.5 * delta * ux, ri[1] - 0.5 * delta * uy, ri[2] - 0.5 * delta * uz)
                rj_new = (rj[0] + 0.5 * delta * ux, rj[1] + 0.5 * delta * uy, rj[2] + 0.5 * delta * uz)
            else:
                ri_new = (ri[0], ri[1], ri[2])
                rj_new = (rj[0] + delta * ux, rj[1] + delta * uy, rj[2] + delta * uz)
            dist = math.sqrt(
                (rj_new[0] - ri_new[0]) ** 2 + (rj_new[1] - ri_new[1]) ** 2 + (rj_new[2] - ri_new[2]) ** 2
            )

            comment = (
                f"{base.comment} scan=distance scan_atom_i={atom_i_1based} scan_atom_j={atom_j_1based} "
                f"delta={delta:.10g} distance={dist:.10g}"
            ).strip()
            fxyz.write(f"{base.natoms}\n")
            fxyz.write(f"{comment}\n")

            for k, line in enumerate(base.atom_lines):
                if k != i0 and k != j0:
                    fxyz.write(line.rstrip("\n") + "\n")
                    continue
                tokens = line.split()
                if k == i0:
                    for idx, v in zip(pidx_i, ri_new):
                        tokens[idx] = _format_float(v, float_fmt)
                else:
                    for idx, v in zip(pidx_j, rj_new):
                        tokens[idx] = _format_float(v, float_fmt)
                fxyz.write(" ".join(tokens) + "\n")

            writer.writerow(
                {
                    "frame": frame,
                    "atom_i": atom_i_1based,
                    "atom_j": atom_j_1based,
                    "delta": delta,
                    "distance": dist,
                }
            )

    output_info_json.write_text(
        json.dumps(
            {
                "input_comment": base.comment,
                "natoms": base.natoms,
                "scan": "distance_ij",
                "atom_i": atom_i_1based,
                "atom_j": atom_j_1based,
                "symmetric": symmetric,
                "distance0": dist0,
                "direction_unit": {"ux": ux, "uy": uy, "uz": uz},
                "deltas": list(deltas),
                "output_xyz": str(output_xyz),
                "output_meta_csv": str(output_meta_csv),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_numeric_table(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for line in _read_all_lines(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No numeric rows parsed from {path}")
    return rows


def _pick_column(rows: List[List[float]], col: int) -> List[float]:
    out: List[float] = []
    for r in rows:
        if not r:
            raise ValueError("Encountered empty numeric row")
        idx = col if col >= 0 else len(r) + col
        if idx < 0 or idx >= len(r):
            raise ValueError(f"Requested column {col} out of range for row with {len(r)} columns")
        out.append(r[idx])
    return out


def _parse_vec_cols(spec: str) -> Tuple[int, int, int] | str:
    s = spec.strip().lower()
    if s in {"first3", "last3"}:
        return s
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--mforce-cols must be 'first3', 'last3', or 3 comma-separated indices like '0,1,2'")
    try:
        idxs = tuple(int(p) for p in parts)
    except ValueError as e:
        raise ValueError("--mforce-cols indices must be integers") from e
    return idxs  # type: ignore[return-value]


def _pick_vec3(rows: List[List[float]], spec: Tuple[int, int, int] | str) -> List[Tuple[float, float, float]]:
    out: List[Tuple[float, float, float]] = []
    for r in rows:
        if len(r) < 3:
            raise ValueError("Expected at least 3 columns for vector data")
        if spec == "first3":
            out.append((r[0], r[1], r[2]))
        elif spec == "last3":
            out.append((r[-3], r[-2], r[-1]))
        else:
            i0, i1, i2 = spec
            idxs = []
            for idx in (i0, i1, i2):
                j = idx if idx >= 0 else len(r) + idx
                if j < 0 or j >= len(r):
                    raise ValueError(f"--mforce-cols index {idx} out of range for row with {len(r)} columns")
                idxs.append(j)
            out.append((r[idxs[0]], r[idxs[1]], r[idxs[2]]))
    return out


def _select_atom_vectors(
    rows: List[List[float]],
    nframes: int,
    natoms: Optional[int],
    atom_index_1based: int,
    vec_cols: Tuple[int, int, int] | str,
) -> List[Tuple[float, float, float]]:
    vecs = _pick_vec3(rows, vec_cols)
    if len(vecs) == nframes:
        return vecs
    if natoms is None:
        raise ValueError(
            f"Vector file has {len(vecs)} rows, but scan has {nframes} frames; "
            "pass --natoms to interpret per-atom-per-frame output"
        )
    if len(vecs) != nframes * natoms:
        raise ValueError(
            f"Vector file has {len(vecs)} rows; expected {nframes} (per-frame) or {nframes*natoms} "
            f"(per-atom-per-frame with natoms={natoms})"
        )
    if atom_index_1based < 1 or atom_index_1based > natoms:
        raise ValueError(f"--atom must be in [1,{natoms}] for per-atom-per-frame vectors")
    atom0 = atom_index_1based - 1
    return [vecs[frame * natoms + atom0] for frame in range(nframes)]


def _load_scan_meta(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in meta csv: {path}")
    return rows


def _derive_theta_from_scan_xyz(scan_xyz: Path, atom_index_1based: int) -> List[float]:
    frames = read_xyz_frames(scan_xyz)
    atom0 = atom_index_1based - 1
    thetas: List[float] = []
    for fr in frames:
        tokens = fr.atom_lines[atom0].split()
        sidx = _spin_token_indices(fr.comment, tokens)
        sx, sy, _ = _parse_floats(tokens, sidx)
        thetas.append(_theta_deg_from_xy(sx, sy))
    return thetas


def plot_scan(
    meta_csv: Optional[Path],
    scan_xyz: Optional[Path],
    energy_path: Optional[Path],
    mforce_path: Optional[Path],
    atom_index_1based: int,
    natoms: Optional[int],
    energy_col: int,
    mforce_cols: str,
    xcol: str,
    xlabel: str,
    out_png: Path,
    out_csv: Optional[Path],
    show: bool,
) -> None:
    if meta_csv is None and scan_xyz is None:
        raise ValueError("Pass --meta or --scan-xyz to define the scan x-axis")

    if meta_csv is not None:
        meta = _load_scan_meta(meta_csv)
        if xcol not in meta[0]:
            raise ValueError(f"Meta CSV missing column '{xcol}': {meta_csv}")
        xvals = [float(r[xcol]) for r in meta]
    else:
        assert scan_xyz is not None
        xvals = _derive_theta_from_scan_xyz(scan_xyz, atom_index_1based)
        meta = [{"frame": i, xcol: v} for i, v in enumerate(xvals)]

    nframes = len(xvals)

    energy: Optional[List[float]] = None
    if energy_path is not None:
        energy = _pick_column(_load_numeric_table(energy_path), energy_col)
        if len(energy) != nframes:
            raise ValueError(f"Energy rows ({len(energy)}) != scan frames ({nframes})")

    hvec: Optional[List[Tuple[float, float, float]]] = None
    if mforce_path is not None:
        hvec = _select_atom_vectors(
            _load_numeric_table(mforce_path),
            nframes,
            natoms,
            atom_index_1based,
            vec_cols=_parse_vec_cols(mforce_cols),
        )

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            fields = ["frame", xcol]
            if energy is not None:
                fields.append("energy")
            if hvec is not None:
                fields += ["hx", "hy", "hz"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in range(nframes):
                row = {"frame": i, xcol: xvals[i]}
                if energy is not None:
                    row["energy"] = energy[i]
                if hvec is not None:
                    row["hx"], row["hy"], row["hz"] = hvec[i]
                w.writerow(row)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib)") from e

    out_png.parent.mkdir(parents=True, exist_ok=True)
    nrows = (1 if energy is not None else 0) + (1 if hvec is not None else 0)
    if nrows == 0:
        raise ValueError("Nothing to plot: pass --energy and/or --mforce")

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax_i = 0
    if energy is not None:
        ax = axes[ax_i]
        ax.plot(xvals, energy, lw=1.5)
        ax.set_ylabel("E")
        ax.grid(True, alpha=0.3)
        ax_i += 1
    if hvec is not None:
        hx = [v[0] for v in hvec]
        hy = [v[1] for v in hvec]
        hz = [v[2] for v in hvec]
        ax = axes[ax_i]
        ax.plot(xvals, hx, label="Hx", lw=1.2)
        ax.plot(xvals, hy, label="Hy", lw=1.2)
        ax.plot(xvals, hz, label="Hz", lw=1.2)
        ax.set_ylabel(r"$H_\alpha$")
        ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _float_or_auto(s: str) -> Optional[float]:
    if s.lower() == "auto":
        return None
    return float(s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate spin-rotation scan XYZs and plot E(theta)/H(theta) curves."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate-rotation", help="Rotate one atom spin in the xy plane.")
    g.add_argument("--input", type=Path, required=True, help="Input XYZ (single or multi-frame).")
    g.add_argument("--frame", type=int, default=0, help="Frame index in input XYZ (0-based).")
    g.add_argument("--atom", type=int, required=True, help="Atom index to perturb (1-based).")
    g.add_argument("--start-deg", type=float, default=0.0)
    g.add_argument("--stop-deg", type=float, default=360.0)
    g.add_argument("--step-deg", type=float, default=1.0)
    g.add_argument("--no-include-stop", action="store_true", help="Do not include stop angle.")
    g.add_argument(
        "--spin-mag",
        type=_float_or_auto,
        default=None,
        help="Spin magnitude to use; 'auto' uses norm of the selected atom spin.",
    )
    g.add_argument(
        "--float-fmt",
        default="auto",
        help="Float format for updated spin columns: 'auto' or a python format like '{:.8f}'.",
    )
    g.add_argument("--output-xyz", type=Path, default=Path("scan_spin_rotation.xyz"))
    g.add_argument("--output-meta", type=Path, default=Path("scan_spin_rotation_meta.csv"))
    g.add_argument("--output-info", type=Path, default=Path("scan_spin_rotation_info.json"))

    gd = sub.add_parser("generate-distance", help="Scan the distance between 2 atoms by moving along their bond.")
    gd.add_argument("--input", type=Path, required=True, help="Input XYZ (single or multi-frame).")
    gd.add_argument("--frame", type=int, default=0, help="Frame index in input XYZ (0-based).")
    gd.add_argument("--atom-i", type=int, required=True, help="Atom i (1-based).")
    gd.add_argument("--atom-j", type=int, required=True, help="Atom j (1-based).")
    gd.add_argument(
        "--deltas",
        type=str,
        required=True,
        help="Comma-separated delta list in Angstrom, e.g. '-0.05,-0.04,...,0.05'",
    )
    gd.add_argument("--symmetric", action="store_true", help="Move i and j symmetrically (+/-delta/2).")
    gd.add_argument(
        "--float-fmt",
        default="auto",
        help="Float format for updated position columns: 'auto' or a python format like '{:.8f}'.",
    )
    gd.add_argument("--output-xyz", type=Path, default=Path("scan_distance.xyz"))
    gd.add_argument("--output-meta", type=Path, default=Path("scan_distance_meta.csv"))
    gd.add_argument("--output-info", type=Path, default=Path("scan_distance_info.json"))

    pl = sub.add_parser("plot", help="Plot scan results (E and/or H vs theta).")
    pl.add_argument("--meta", type=Path, help="Meta CSV from generate-rotation.")
    pl.add_argument("--scan-xyz", type=Path, help="Scan XYZ to derive theta from spin (fallback).")
    pl.add_argument("--energy", type=Path, help="Energy output file (1 or 2+ cols).")
    pl.add_argument(
        "--energy-col",
        type=int,
        default=0,
        help="Energy column index (0-based). For typical 'pred ref' files: 0=NEP, -1=DFT.",
    )
    pl.add_argument(
        "--mforce",
        type=Path,
        help="Effective field / magnetic force output (3 cols or 6 cols).",
    )
    pl.add_argument(
        "--mforce-cols",
        default="first3",
        help="Which 3 columns to use for vectors: 'first3' (typical NEP) or 'last3' (typical DFT), "
        "or explicit indices like '0,1,2' or '-3,-2,-1'.",
    )
    pl.add_argument("--natoms", type=int, help="Natoms for per-atom-per-frame mforce output.")
    pl.add_argument("--atom", type=int, required=True, help="Atom index to plot (1-based).")
    pl.add_argument("--xcol", default="theta_deg", help="X-axis column name in meta CSV (default: theta_deg).")
    pl.add_argument("--xlabel", default=r"$\theta$ (deg)", help="X-axis label for the plot.")
    pl.add_argument("--out-png", type=Path, default=Path("scan_plot.png"))
    pl.add_argument("--out-csv", type=Path, default=Path("scan_curve.csv"))
    pl.add_argument("--no-out-csv", action="store_true", help="Do not write combined CSV.")
    pl.add_argument("--show", action="store_true", help="Show plot interactively.")

    args = p.parse_args(argv)

    if args.cmd == "generate-rotation":
        frames = read_xyz_frames(args.input)
        if args.frame < 0 or args.frame >= len(frames):
            raise ValueError(f"--frame must be in [0,{len(frames)-1}] for {args.input}")
        generate_rotation_scan(
            base=frames[args.frame],
            atom_index_1based=args.atom,
            start_deg=args.start_deg,
            stop_deg=args.stop_deg,
            step_deg=args.step_deg,
            include_stop=not args.no_include_stop,
            spin_mag=args.spin_mag,
            float_fmt=args.float_fmt,
            output_xyz=args.output_xyz,
            output_meta_csv=args.output_meta,
            output_info_json=args.output_info,
        )
        return 0

    if args.cmd == "generate-distance":
        frames = read_xyz_frames(args.input)
        if args.frame < 0 or args.frame >= len(frames):
            raise ValueError(f"--frame must be in [0,{len(frames)-1}] for {args.input}")
        deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
        generate_distance_scan(
            base=frames[args.frame],
            atom_i_1based=args.atom_i,
            atom_j_1based=args.atom_j,
            deltas=deltas,
            symmetric=args.symmetric,
            float_fmt=args.float_fmt,
            output_xyz=args.output_xyz,
            output_meta_csv=args.output_meta,
            output_info_json=args.output_info,
        )
        return 0

    if args.cmd == "plot":
        plot_scan(
            meta_csv=args.meta,
            scan_xyz=args.scan_xyz,
            energy_path=args.energy,
            mforce_path=args.mforce,
            atom_index_1based=args.atom,
            natoms=args.natoms,
            energy_col=args.energy_col,
            mforce_cols=args.mforce_cols,
            xcol=args.xcol,
            xlabel=args.xlabel,
            out_png=args.out_png,
            out_csv=None if args.no_out_csv else args.out_csv,
            show=args.show,
        )
        return 0

    raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
