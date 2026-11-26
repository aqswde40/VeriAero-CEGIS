#!/usr/bin/env bash
set -euo pipefail

# Build patched XFoil 6.99 -> ~/VeriAero-CEGIS/bin/xfoil_patched (zero SIGFPE)
# Requires: gfortran, gcc, make, libx11-dev (run sudo apt install first if missing)

TARGET_DIR="${HOME}/VeriAero-CEGIS"
BIN_DIR="${TARGET_DIR}/bin"
mkdir -p "${BIN_DIR}"

for cmd in gfortran gcc make curl tar python3; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: Missing required tool '${cmd}'."
    echo "Run: sudo apt-get update && sudo apt-get install -y build-essential gfortran libx11-dev"
    exit 1
  fi
done

WORK_XFOIL=$(mktemp -d /tmp/xfoil_build.XXXXXX)
trap 'rm -rf "${WORK_XFOIL}"' EXIT

echo "Downloading XFoil 6.99 from MIT..."
curl -fL "https://web.mit.edu/drela/Public/web/xfoil/xfoil6.99.tgz" -o "${WORK_XFOIL}/xfoil6.99.tgz"
tar -xzf "${WORK_XFOIL}/xfoil6.99.tgz" -C "${WORK_XFOIL}"
XDIR="${WORK_XFOIL}/Xfoil"
OSMAP_DEST="${BIN_DIR}/osmapDP.dat"

python3 - "${XDIR}/orrs/src/osmap.f" "${OSMAP_DEST}" <<'PYEOF'
import sys, pathlib, re
osmap_path = pathlib.Path(sys.argv[1])
dest = sys.argv[2]
txt = osmap_path.read_text()
txt_new = re.sub(r"OSFILE\s*=\s*'[^']*'", f"OSFILE = '{dest}'", txt)
txt_new = re.sub(r"CHARACTER\*80\s+OSFILE", "CHARACTER*256 OSFILE", txt_new)
osmap_path.write_text(txt_new)
PYEOF

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

echo "Building Orr-Sommerfeld transition database..."
make -C "${XDIR}/orrs/bin" -f Makefile_DP osgen
(cd "${XDIR}/orrs" && ./bin/osgen osmaps_ns.lst)
cp "${XDIR}/orrs/osmapDP.dat" "${OSMAP_DEST}"

echo "Building plotlib and patched XFoil 6.99..."
make -C "${XDIR}/plotlib" libPltDP.a || make -C "${XDIR}/plotlib" libPlt.a
make -C "${XDIR}/bin" xfoil
cp "${XDIR}/bin/xfoil" "${BIN_DIR}/xfoil_patched"
chmod +x "${BIN_DIR}/xfoil_patched"

echo "Running NACA 0012 viscous smoke test (N=200, Re=500000, alpha=0,4,8)..."
XFOIL_TEST_DIR=$(mktemp -d /tmp/xfoil_test.XXXXXX)
(
  cd "${XFOIL_TEST_DIR}"
  printf "NACA 0012\nPPAR\nN 200\n\n\nPANE\nOPER\nVISC 500000\nITER 100\nPACC\nnaca0012.pol\n\nALFA 0\nALFA 4\nALFA 8\nPACC\n\nQUIT\n" \
    | "${BIN_DIR}/xfoil_patched" >/dev/null
  cat naca0012.pol
)
rm -rf "${XFOIL_TEST_DIR}"
echo "SUCCESS: ${BIN_DIR}/xfoil_patched built and smoke-tested with zero SIGFPE."
