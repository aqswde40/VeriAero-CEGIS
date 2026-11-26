#!/usr/bin/env bash
set -euo pipefail

# VeriAero-CEGIS v3.2 — C2-WSL2 Ubuntu 24.04 Idempotent Bootstrap Script
# Run inside WSL2 Ubuntu 24.04 after installing WSL2 & configuring .wslconfig:
#   bash /mnt/c/Users/jyoun/Documents/VeriAero-CEGIS/scripts/bootstrap_wsl2.sh

REPO_URL="https://github.com/aqswde40/VeriAero-CEGIS.git"
WIN_SRC="/mnt/c/Users/jyoun/Documents/VeriAero-CEGIS"
TARGET_DIR="${HOME}/VeriAero-CEGIS"

echo "=== [1/6] Checking WSL2 Environment & Resources ==="
if ! grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
  echo "WARNING: This script is designed for WSL2 Ubuntu 24.04."
fi

CORES=$(nproc)
RAM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
RAM_GB=$(( (RAM_KB + 1048575) / 1048576 ))
echo "Detected CPU logical cores: ${CORES}, RAM: ~${RAM_GB} GB"

if (( RAM_GB < 24 )) || (( CORES < 10 )); then
  echo "WARNING: Recommended WSL2 allocation is >= 24 GB RAM and >= 10 cores."
  echo "Create/update C:\\Users\\<USER>\\.wslconfig with:"
  echo "  [wsl2]"
  echo "  memory=24GB"
  echo "  processors=10"
  echo "  swap=8GB"
  echo "Then run 'wsl --shutdown' in Windows PowerShell and reopen Ubuntu."
fi

echo "=== [2/6] Installing System Toolchains (apt) ==="
sudo apt-get update -y
sudo apt-get install -y \
  build-essential gfortran libx11-dev git curl wget unzip rsync ca-certificates \
  libgl1 libglu1-mesa libxft2 libxrender1 libxcursor1 libxinerama1 libgomp1 python3 python3-venv

echo "=== [3/6] Syncing ~/VeriAero-CEGIS Workspace & Git Remote ==="
mkdir -p "${TARGET_DIR}"
if [ -d "${WIN_SRC}" ] && [ "${PWD}" != "${TARGET_DIR}" ]; then
  rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
    --exclude 'bin/xfoil_patched' --exclude 'bin/SU2_CFD' --exclude 'bin/osmapDP.dat' \
    "${WIN_SRC}/" "${TARGET_DIR}/"
fi
cd "${TARGET_DIR}"
mkdir -p bin

if [ ! -d ".git" ]; then
  git init -b main
  git remote add origin "${REPO_URL}" || true
else
  git remote set-url origin "${REPO_URL}" || git remote add origin "${REPO_URL}" || true
fi

echo "=== [4/6] Provisioning uv + Python 3.12 + CUDA PyTorch Environment ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
uv python install 3.12
OPENBLAS_NUM_THREADS=1 uv sync --python 3.12

uv run python -c "
import torch
print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

echo "=== [5/6] Building Patched XFoil 6.99 -> bin/xfoil_patched ==="
if [ ! -x "${TARGET_DIR}/bin/xfoil_patched" ] || [ ! -f "${TARGET_DIR}/bin/osmapDP.dat" ]; then
  WORK_XFOIL=$(mktemp -d /tmp/xfoil_build.XXXXXX)
  trap 'rm -rf "${WORK_XFOIL}"' EXIT
  curl -fL "https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz" -o "${WORK_XFOIL}/xfoil6.99.tgz"
  tar -xzf "${WORK_XFOIL}/xfoil6.99.tgz" -C "${WORK_XFOIL}"
  XDIR="${WORK_XFOIL}/Xfoil"
  OSMAP_DEST="${TARGET_DIR}/bin/osmapDP.dat"

  # Patch orrs/src/osmap.f so OSFILE points persistently to ~/VeriAero-CEGIS/bin/osmapDP.dat
  python3 - "${XDIR}/orrs/src/osmap.f" "${OSMAP_DEST}" <<'PYEOF'
import sys, pathlib, re
osmap_path = pathlib.Path(sys.argv[1])
dest = sys.argv[2]
txt = osmap_path.read_text()
txt_new = re.sub(
    r"OSFILE\s*=\s*'[^']*'",
    f"OSFILE = '{dest}'",
    txt,
)
# Ensure OSFILE character declaration is long enough
txt_new = re.sub(r"CHARACTER\*80\s+OSFILE", "CHARACTER*256 OSFILE", txt_new)
osmap_path.write_text(txt_new)
PYEOF

  # Patch Makefiles for modern gfortran + zero SIGFPE trapping
  for mk in "${XDIR}/orrs/bin/Makefile_DP" "${XDIR}/orrs/bin/Makefile" "${XDIR}/plotlib/config.make" "${XDIR}/bin/Makefile"; do
    if [ -f "$mk" ]; then
      python3 - "$mk" <<'PYEOF'
import sys, pathlib, re
p = pathlib.Path(sys.argv[1])
txt = p.read_text()
txt = re.sub(r"^FC\s*=.*$", "FC = gfortran", txt, flags=re.M)
txt = re.sub(r"^FFLAGS\s*=.*$", "FFLAGS = -O2 -fallow-argument-mismatch -fno-range-check -ffpe-trap=none", txt, flags=re.M)
txt = re.sub(r"^FFLOPT\s*=.*$", "FFLOPT = -O2 -fallow-argument-mismatch -fno-range-check -ffpe-trap=none", txt, flags=re.M)
txt = re.sub(r"^DP\s*=.*$", "DP = -fdefault-real-8", txt, flags=re.M)
p.write_text(txt)
PYEOF
    fi
  done

  # Build Orr-Sommerfeld database
  make -C "${XDIR}/orrs/bin" -f Makefile_DP osgen
  (cd "${XDIR}/orrs" && ./bin/osgen osmaps_ns.lst)
  cp "${XDIR}/orrs/osmapDP.dat" "${OSMAP_DEST}"

  # Build plotlib
  make -C "${XDIR}/plotlib" libPltDP.a || make -C "${XDIR}/plotlib" libPlt.a

  # Build xfoil
  make -C "${XDIR}/bin" xfoil
  cp "${XDIR}/bin/xfoil" "${TARGET_DIR}/bin/xfoil_patched"
  chmod +x "${TARGET_DIR}/bin/xfoil_patched"
fi

echo "Running XFoil 6.99 NACA 0012 viscous smoke test (zero SIGFPE check)..."
XFOIL_TEST_DIR=$(mktemp -d /tmp/xfoil_test.XXXXXX)
(
  cd "${XFOIL_TEST_DIR}"
  printf "NACA 0012\nOPER\nVISC 500000\nITER 100\nPACC\nnaca0012.pol\n\nALFA 0\nALFA 4\nALFA 8\nPACC\n\nQUIT\n" \
    | "${TARGET_DIR}/bin/xfoil_patched" >/dev/null
  cat naca0012.pol
)
rm -rf "${XFOIL_TEST_DIR}"

echo "=== [6/6] Installing Official SU2 v8.6.0 Linux OpenMP -> bin/SU2_CFD ==="
if [ ! -x "${TARGET_DIR}/bin/SU2_CFD" ]; then
  WORK_SU2=$(mktemp -d /tmp/su2_dl.XXXXXX)
  curl -fL "https://github.com/su2code/SU2/releases/download/v8.6.0/SU2_v8.6.0_linux64-omp.zip" -o "${WORK_SU2}/su2.zip"
  unzip -q "${WORK_SU2}/su2.zip" -d "${WORK_SU2}/extracted"
  SU2_BIN=$(find "${WORK_SU2}/extracted" -type f -name "SU2_CFD" | head -n 1)
  if [ -z "${SU2_BIN}" ]; then
    echo "ERROR: SU2_CFD binary not found inside release archive."
    exit 1
  fi
  cp "${SU2_BIN}" "${TARGET_DIR}/bin/SU2_CFD"
  chmod +x "${TARGET_DIR}/bin/SU2_CFD"
  rm -rf "${WORK_SU2}"
fi

"${TARGET_DIR}/bin/SU2_CFD" --version || true

echo "=== Running Verification Suite ==="
OPENBLAS_NUM_THREADS=1 uv run pytest -v
echo "Bootstrap complete on C2-WSL2!"
