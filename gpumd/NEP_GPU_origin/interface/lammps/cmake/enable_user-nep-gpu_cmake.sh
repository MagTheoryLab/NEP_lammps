#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <LAMMPS_ROOT|LAMMPS_SRC> <GPUMD_ROOT>"
  echo "  LAMMPS_ROOT: path to LAMMPS source tree (contains cmake/ and src/)"
  echo "  LAMMPS_SRC : alternatively, the src/ directory inside the LAMMPS tree"
  echo "  GPUMD_ROOT : path to this GPUMD repo (contains NEP_GPU/)"
  exit 2
fi

LAMMPS_IN="$(cd "$1" && pwd)"
GPUMD_ROOT="$(cd "$2" && pwd)"

if [[ -d "$LAMMPS_IN/src" && -d "$LAMMPS_IN/cmake" ]]; then
  LAMMPS_ROOT="$LAMMPS_IN"
elif [[ -d "$LAMMPS_IN/../src" && -d "$LAMMPS_IN/../cmake" ]]; then
  LAMMPS_ROOT="$(cd "$LAMMPS_IN/.." && pwd)"
else
  echo "ERROR: LAMMPS_ROOT must contain src/ and cmake/ (got: $LAMMPS_IN)" >&2
  exit 1
fi
if [[ ! -f "$GPUMD_ROOT/NEP_GPU/CMakeLists.txt" ]]; then
  echo "ERROR: GPUMD_ROOT must contain NEP_GPU/CMakeLists.txt (got: $GPUMD_ROOT)" >&2
  exit 1
fi

copy_tree() {
  local src="$1"
  local dst="$2"

  if command -v rsync >/dev/null 2>&1; then
    mkdir -p "$dst"
    rsync -a --delete "$src/" "$dst/"
    return
  fi

  rm -rf "$dst"
  mkdir -p "$dst"
  cp -a "$src/." "$dst/"
}

echo "[1/4] Installing USER-NEP-GPU sources into LAMMPS src/"
copy_tree \
  "$GPUMD_ROOT/NEP_GPU/interface/lammps/USER-NEP-GPU" \
  "$LAMMPS_ROOT/src/USER-NEP-GPU"

# Spin NEP is supported via `pair_style nep/spin/gpu/kk`.

echo "[2/4] Installing CMake package module"
mkdir -p "$LAMMPS_ROOT/cmake/Modules/Packages"
SRC_PKG_CMAKE="$GPUMD_ROOT/NEP_GPU/interface/lammps/cmake/Packages/USER-NEP-GPU.cmake"
DST_PKG_CMAKE="$LAMMPS_ROOT/cmake/Modules/Packages/USER-NEP-GPU.cmake"

files_identical() {
  local a="$1"
  local b="$2"
  if command -v cmp >/dev/null 2>&1; then
    cmp -s "$a" "$b"
    return
  fi
  if command -v diff >/dev/null 2>&1; then
    diff -q "$a" "$b" >/dev/null 2>&1
    return
  fi
  return 1
}

if [[ -f "$DST_PKG_CMAKE" ]] && files_identical "$SRC_PKG_CMAKE" "$DST_PKG_CMAKE"; then
  echo "  USER-NEP-GPU.cmake already up-to-date; skipping copy."
else
  cp -f "$SRC_PKG_CMAKE" "$DST_PKG_CMAKE"
fi

echo "[3/4] Patching LAMMPS cmake/CMakeLists.txt to register USER-NEP-GPU"
PATCH_FILE="$GPUMD_ROOT/NEP_GPU/interface/lammps/cmake/lammps_cmake_patch_user-nep-gpu.diff"
LMP_CMAKE_LISTS="$LAMMPS_ROOT/cmake/CMakeLists.txt"

has_user_nep_gpu_in_standard_packages() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$LMP_CMAKE_LISTS" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

in_block = False
found = False
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("set(STANDARD_PACKAGES"):
        in_block = True
    if in_block and "USER-NEP-GPU" in line:
        found = True
    if in_block and ")" in line:
        raise SystemExit(0 if found else 1)
raise SystemExit(1)
PY
    return $?
  fi
  if command -v python >/dev/null 2>&1; then
    python - "$LMP_CMAKE_LISTS" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

in_block = False
found = False
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("set(STANDARD_PACKAGES"):
        in_block = True
    if in_block and "USER-NEP-GPU" in line:
        found = True
    if in_block and ")" in line:
        raise SystemExit(0 if found else 1)
raise SystemExit(1)
PY
    return $?
  fi

  # Fallback if python isn't available.
  grep -qE '^[[:space:]]+USER-NEP-GPU[[:space:]]*$' "$LMP_CMAKE_LISTS"
}

has_user_nep_gpu_in_pkg_with_incl() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$LMP_CMAKE_LISTS" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

in_block = False
found = False
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("foreach(PKG_WITH_INCL"):
        in_block = True
    if in_block and "USER-NEP-GPU" in line:
        found = True
    if in_block and ")" in line:
        raise SystemExit(0 if found else 1)
raise SystemExit(1)
PY
    return $?
  fi
  if command -v python >/dev/null 2>&1; then
    python - "$LMP_CMAKE_LISTS" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

in_block = False
found = False
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("foreach(PKG_WITH_INCL"):
        in_block = True
    if in_block and "USER-NEP-GPU" in line:
        found = True
    if in_block and ")" in line:
        raise SystemExit(0 if found else 1)
raise SystemExit(1)
PY
    return $?
  fi

  # Fallback if python isn't available.
  grep -q "USER-NEP-GPU" "$LMP_CMAKE_LISTS"
}

pkg_registered=0
incl_registered=0
if has_user_nep_gpu_in_standard_packages; then
  pkg_registered=1
fi
if has_user_nep_gpu_in_pkg_with_incl; then
  incl_registered=1
fi

if [[ $pkg_registered -eq 1 && $incl_registered -eq 1 ]]; then
  echo "  USER-NEP-GPU already registered; skipping patch."
  if [[ -f "$LAMMPS_ROOT/cmake/CMakeLists.txt.rej" ]]; then
    rm -f "$LAMMPS_ROOT/cmake/CMakeLists.txt.rej" || true
  fi
else
  patch_status=0
  if command -v git >/dev/null 2>&1 && [[ -d "$LAMMPS_ROOT/.git" ]]; then
    (cd "$LAMMPS_ROOT" && git apply --ignore-space-change --ignore-whitespace "$PATCH_FILE") || patch_status=$?
  elif command -v patch >/dev/null 2>&1; then
    patch -p1 -d "$LAMMPS_ROOT" < "$PATCH_FILE" || patch_status=$?
  fi
  if [[ $patch_status -ne 0 ]]; then
    echo "  NOTE: patch hunks did not apply cleanly (LAMMPS version/layout differs); attempting a robust fallback edit."
  fi

  # If patch tools are missing or context differs, fall back to a small text edit.
  pkg_registered=0
  incl_registered=0
  if has_user_nep_gpu_in_standard_packages; then
    pkg_registered=1
  fi
  if has_user_nep_gpu_in_pkg_with_incl; then
    incl_registered=1
  fi

  if [[ $pkg_registered -eq 0 || $incl_registered -eq 0 ]]; then
    PYTHON_BIN=""
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
      PYTHON_BIN="python"
    fi
    if [[ -z "$PYTHON_BIN" ]]; then
      echo "ERROR: failed to apply patch and no python/python3 available for fallback edit." >&2
      exit 1
    fi

    "$PYTHON_BIN" - <<'PY' "$LMP_CMAKE_LISTS"
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="strict").splitlines(True)
orig = "".join(lines)

def _find_block_start(prefixes):
    for i, l in enumerate(lines):
        s = l.lstrip()
        for p in prefixes:
            if s.startswith(p):
                return i
    return None

def _find_block_end(start, max_lookahead=200):
    if start is None:
        return None
    for i in range(start + 1, min(start + max_lookahead, len(lines))):
        if ")" in lines[i]:
            return i
    return None

def patch_package_list():
    # LAMMPS uses a package list to define PKG_* options. Across versions this is
    # typically STANDARD_PACKAGES, but some branches/older forks can differ.
    start = _find_block_start(("set(STANDARD_PACKAGES", "set(PACKAGES", "set(LAMMPS_PACKAGES"))
    end = _find_block_end(start)
    if start is None or end is None:
        return False
    block = "".join(lines[start:end+1])
    if "USER-NEP-GPU" in block:
        return True
    lines.insert(end, "  USER-NEP-GPU\n")
    return True

def patch_includes():
    # Preferred modern pattern: foreach(PKG_WITH_INCL ...) include(Packages/${PKG_WITH_INCL})
    start = _find_block_start(("foreach(PKG_WITH_INCL", "foreach(PKG_WITH_INCLUDES", "foreach(PKG_WITH_MODULES"))
    end = _find_block_end(start, max_lookahead=100)
    if start is not None and end is not None:
        block = "".join(lines[start:end+1])
        if "USER-NEP-GPU" in block:
            return True
        lines[end] = lines[end].replace(")", " USER-NEP-GPU)")
        return True

    # Older pattern: a series of if(PKG_X) include(Packages/X) endif().
    # Insert our include near other includes.
    for i, l in enumerate(lines):
        if "include(Packages/" in l and "endif" not in l:
            insert_at = i
            break
    else:
        insert_at = None
    if insert_at is None:
        # Last resort: append at end of file.
        insert_at = len(lines)
        if insert_at > 0 and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"

    snippet = [
        "\n",
        "if(PKG_USER-NEP-GPU)\n",
        "  include(Packages/USER-NEP-GPU)\n",
        "endif()\n",
    ]
    text = "".join(lines)
    if "include(Packages/USER-NEP-GPU)" in text:
        return True
    lines[insert_at:insert_at] = snippet
    return True

ok1 = patch_package_list()
ok2 = patch_includes()

text = "".join(lines)
if text == orig:
    if "USER-NEP-GPU" in orig:
        print("No changes needed; USER-NEP-GPU already present.")
        raise SystemExit(0)
    raise SystemExit("No changes made; USER-NEP-GPU may already be registered, or file format is unexpected.")
if not (ok1 and ok2):
    raise SystemExit("Failed to patch required CMake sections (could not locate a package list and/or include registration pattern).")
path.write_text(text, encoding="utf-8")
print("Patched:", path)
PY
  fi

  if ! has_user_nep_gpu_in_standard_packages || ! has_user_nep_gpu_in_pkg_with_incl; then
    echo "ERROR: failed to register USER-NEP-GPU in $LMP_CMAKE_LISTS" >&2
    exit 1
  fi

  # Clean up reject file if the initial patch created one but we succeeded via fallback edit.
  if [[ -f "$LAMMPS_ROOT/cmake/CMakeLists.txt.rej" ]]; then
    rm -f "$LAMMPS_ROOT/cmake/CMakeLists.txt.rej" || true
  fi
fi

echo "[4/4] Done."
echo "Next: configure LAMMPS with CMake using:"
echo "  -DPKG_USER-NEP-GPU=ON -DNEP_GPU_SOURCE_DIR=$GPUMD_ROOT"
