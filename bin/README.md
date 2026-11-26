# Native Solver Binaries (`bin/`)

On **C2-WSL2 (Ubuntu 24.04)**, run `bash scripts/bootstrap_wsl2.sh` to provision:
- `bin/xfoil_patched`: Patched XFoil 6.99 compiled with `gfortran -fallow-argument-mismatch -fno-range-check -ffpe-trap=none`
- `bin/osmapDP.dat`: Orr-Sommerfeld transition database linked by `xfoil_patched`
- `bin/SU2_CFD`: Official Linux OpenMP SU2 v8.6.0 CFD binary
