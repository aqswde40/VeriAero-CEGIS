#!/usr/bin/env bash
set -euo pipefail

# Build patched XFoil 6.99 -> ~/VeriAero-CEGIS/bin/xfoil_patched (zero SIGFPE)
# Requires: gfortran, gcc, make, curl, tar, python3, libx11-dev

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
OSMAP_DEST="${BIN_DIR}/osmap.dat"

# Patch osrc/osmap.f (used by bin/Makefile) and orrs/src/osmap.f so OSFILE points to BIN_DIR/osmap.dat
# even when the OSMAP environment variable is unset.
python3 - "${XDIR}/osrc/osmap.f" "${XDIR}/orrs/src/osmap.f" "${OSMAP_DEST}" <<'PYEOF'
import pathlib, re, sys

osrc_osmap = pathlib.Path(sys.argv[1])
orrs_osmap = pathlib.Path(sys.argv[2])
dest = sys.argv[3]

if osrc_osmap.is_file():
    txt = osrc_osmap.read_text(encoding="latin-1")
    txt = re.sub(r"CHARACTER\*(?:80|128)\s+OSFILE", "CHARACTER*256 OSFILE", txt)
    txt = re.sub(r"DATA\s+OSFILE\s*/\s*'[^']*'\s*/", f"DATA OSFILE / '{dest}' /", txt)
    txt = re.sub(r"IF\(LOSF\.EQ\.0\)\s+GO\s+TO\s+800", "IF(LOSF.EQ.0) LOSF = LEN_TRIM(OSFILE)", txt)
    osrc_osmap.write_text(txt, encoding="latin-1")

if orrs_osmap.is_file():
    txt = orrs_osmap.read_text(encoding="latin-1")
    txt = re.sub(r"CHARACTER\*(?:80|128)\s+OSFILE", "CHARACTER*256 OSFILE", txt)
    txt = re.sub(r"DATA\s+OSFILE\s*/\s*'[^']*'\s*/", f"DATA OSFILE / '{dest}' /", txt)
    txt = re.sub(r"OSFILE\s*=\s*'[^']*'", f"OSFILE = '{dest}'", txt)
    orrs_osmap.write_text(txt, encoding="latin-1")
PYEOF

# Patch xpol.f BOTTOM/BOTTOMX for gfortran (BACKSPACE required after reading to EOF before WRITE)
python3 - "${XDIR}/src/xpol.f" <<'PYEOF'
import pathlib, re, sys

xpol_path = pathlib.Path(sys.argv[1])
txt = xpol_path.read_text(encoding="latin-1")
txt_new, n_sub = re.subn(
    r"(READ\(LU(?:,1000)?,END=90,ERR=90\)\s+DUMMY\s*\n(?:\s*1000\s+FORMAT\(A\)\s*\n)?\s*GO TO 10\s*\nC\s*\n\s*90\s+)RETURN",
    r"\1BACKSPACE(LU)\n      RETURN",
    txt,
)
if n_sub != 2:
    raise RuntimeError(f"Expected 2 BOTTOM/BOTTOMX patches in xpol.f, got {n_sub}")
xpol_path.write_text(txt_new, encoding="latin-1")
PYEOF

# Patch Makefiles cleanly for modern gfortran/gcc 13+ on Ubuntu 24.04
python3 - "${XDIR}" <<'PYEOF'
import pathlib, re, sys

xdir = pathlib.Path(sys.argv[1])

# 1. orrs/bin/Makefile (single-precision osgen matching osrc/osmap.f REAL*4 arrays)
orrs_mk = xdir / "orrs" / "bin" / "Makefile"
txt = orrs_mk.read_text(encoding="latin-1")
txt = re.sub(r"^FC\s*=.*$", "FC = gfortran", txt, flags=re.M)
txt = re.sub(
    r"^FLG\s*=.*$",
    "FLG = -O2 -fallow-argument-mismatch -fno-range-check",
    txt,
    flags=re.M,
)
txt = re.sub(r"^PLTLIB\s*=.*$", "PLTLIB = -lX11", txt, flags=re.M)
txt = txt.replace("-r8", "-fdefault-real-8")
orrs_mk.write_text(txt, encoding="latin-1")

# 2. plotlib/config.make (double-precision libPlt_gDP.a required by bin/Makefile)
plt_cfg = xdir / "plotlib" / "config.make"
txt = plt_cfg.read_text(encoding="latin-1")
txt = re.sub(r"^PLTLIB\s*=.*$", "PLTLIB = libPlt_gDP.a", txt, flags=re.M)
txt = re.sub(r"^FC\s*=.*$", "FC = gfortran", txt, flags=re.M)
txt = re.sub(r"^CC\s*=.*$", "CC = gcc", txt, flags=re.M)
txt = re.sub(r"^#?DP\s*=.*$", "DP = -fdefault-real-8", txt, flags=re.M)
txt = re.sub(
    r"^FFLAGS\s*=.*$",
    "FFLAGS = -m64 -O2 $(DP) -fallow-argument-mismatch -fno-range-check",
    txt,
    flags=re.M,
)
txt = re.sub(
    r"^CFLAGS\s*=.*$",
    "CFLAGS = -m64 -O2 $(DEFINE) -Wno-implicit-function-declaration -Wno-implicit-int",
    txt,
    flags=re.M,
)
txt = re.sub(r"^LINKLIB\s*=.*$", "LINKLIB = -lX11", txt, flags=re.M)
plt_cfg.write_text(txt, encoding="latin-1")

# 3. bin/Makefile (xfoil binary with zero SIGFPE trapping and local install disabled)
bin_mk = xdir / "bin" / "Makefile"
txt = bin_mk.read_text(encoding="latin-1")
txt = re.sub(r"^FC\s*=.*$", "FC = gfortran", txt, flags=re.M)
txt = re.sub(r"^CC\s*=.*$", "CC = gcc", txt, flags=re.M)
txt = re.sub(
    r"^CFLAGS\s*=.*$",
    "CFLAGS = -O2 -DUNDERSCORE -Wno-implicit-function-declaration -Wno-implicit-int",
    txt,
    flags=re.M,
)
txt = re.sub(r"^CHK\s*=.*$", "CHK =", txt, flags=re.M)
txt = re.sub(r"^DBL\s*=.*$", "DBL = -fdefault-real-8", txt, flags=re.M)
txt = re.sub(
    r"^FFLAGS\s*=.*$",
    "FFLAGS = -O2 $(DBL) -fallow-argument-mismatch -fno-range-check",
    txt,
    flags=re.M,
)
txt = re.sub(
    r"^FFLOPT\s*=.*$",
    "FFLOPT = -O2 $(DBL) -fallow-argument-mismatch -fno-range-check",
    txt,
    flags=re.M,
)
txt = re.sub(r"^PLTOBJ\s*=.*$", "PLTOBJ = ../plotlib/libPlt_gDP.a", txt, flags=re.M)
txt = re.sub(r"^PLTLIB\s*=.*$", "PLTLIB = -lX11", txt, flags=re.M)
txt = re.sub(r"^INSTALLCMD\s*=.*$", "INSTALLCMD = :", txt, flags=re.M)
bin_mk.write_text(txt, encoding="latin-1")
PYEOF

echo "Building single-precision Orr-Sommerfeld transition database..."
make -C "${XDIR}/orrs/bin" -f Makefile osgen
(cd "${XDIR}/orrs" && ./bin/osgen osmaps_ns.lst)
cp "${XDIR}/orrs/osmap.dat" "${OSMAP_DEST}"

echo "Building plotlib (libPlt_gDP.a) and patched XFoil 6.99..."
make -C "${XDIR}/plotlib" libPlt_gDP.a
make -C "${XDIR}/bin" xfoil
cp "${XDIR}/bin/xfoil" "${BIN_DIR}/xfoil_patched"
chmod +x "${BIN_DIR}/xfoil_patched"

echo "Running headless NACA 0012 viscous smoke test (N=200, Re=500000, alpha=0,4,8)..."
XFOIL_TEST_DIR=$(mktemp -d /tmp/xfoil_test.XXXXXX)
(
  cd "${XFOIL_TEST_DIR}"
  printf "PLOP\nG F\n\nNACA 0012\nPPAR\nN 200\n\n\nPANE\nOPER\nVISC 500000\nITER 100\nPACC\nnaca0012.pol\n\nALFA 0\nALFA 4\nALFA 8\nPACC\n\nQUIT\n" \
    | DISPLAY= "${BIN_DIR}/xfoil_patched" >/dev/null
  if [ ! -s naca0012.pol ]; then
    echo "ERROR: naca0012.pol was not generated or is empty."
    exit 1
  fi
  cat naca0012.pol
)
rm -rf "${XFOIL_TEST_DIR}"
echo "SUCCESS: ${BIN_DIR}/xfoil_patched built and smoke-tested with zero SIGFPE."
