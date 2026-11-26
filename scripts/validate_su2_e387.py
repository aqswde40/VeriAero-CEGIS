"""Day-3 SU2 v8.6.0 Eppler 387 & TMR NACA 0012 Validation Parity Check Script.

Generates structured quadrilateral C-grids using Gmsh (`gmsh==4.15.2`), writes
locked Section 6.5 steady-state SST-LM RANS configs (`TIME_DOMAIN= NO`, `L2ROE`,
`BOUNDED_SCALAR`, `WEIGHTED_LEAST_SQUARES`), and runs `bin/SU2_CFD` with the
dual low-Re pass gate (`STRICT_PASS` / `LSB_DAMPED_PASS`) on C2-WSL2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import aerosandbox as asb
import numpy as np

from veriaero.solvers.su2_runner import (
    SU2Runner,
    generate_airfoil_cgrid_su2,
    write_su2_rans_cfg,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Day-3 SU2 E387 & NACA 0012 parity check")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("outputs/su2_validation"),
        help="Root directory for SU2 validation cases",
    )
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="Generate and verify Gmsh C-grids + SU2 .cfg files without invoking SU2_CFD",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Use fast coarse mesh density (~3k cells) instead of production ~40k cells",
    )
    args = parser.parse_args()

    cases = [
        ("e387_re2e5_a4", "e387", 4.0, 2.0e5, 0.06, 5.0),
        ("naca0012_tmr_re6e6_a0", "naca0012", 0.0, 6.0e6, 0.15, 3.0),
    ]

    grid_kwargs = (
        {"n_airfoil_half": 41, "n_wake_pts": 19, "n_normal_pts": 25, "progression_normal": 1.15}
        if args.coarse
        else {"n_airfoil_half": 131, "n_wake_pts": 45, "n_normal_pts": 150, "progression_normal": 1.08}
    )

    runner = SU2Runner(omp_threads=8)

    for case_name, af_name, alpha_deg, re_num, mach, t2l in cases:
        case_dir = args.workdir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = case_dir / f"{af_name}_cgrid.su2"
        cfg_path = case_dir / "rans_steady.cfg"

        coords = np.asarray(asb.Airfoil(af_name).coordinates, dtype=float)
        generate_airfoil_cgrid_su2(coords, mesh_path, chord=1.0, farfield_radius=20.0, **grid_kwargs)
        write_su2_rans_cfg(
            cfg_path=cfg_path,
            mesh_filename=mesh_path.name,
            mach=mach,
            reynolds=re_num,
            alpha_deg=alpha_deg,
            turb2lam_visc_ratio=t2l,
        )
        print(f"[{case_name}] Gmsh C-grid: {mesh_path} ({mesh_path.stat().st_size} bytes) | cfg: {cfg_path}")

        if not args.mesh_only:
            status, report, reason = runner.run_case(workdir=case_dir, cfg_filename=cfg_path.name)
            print(f"[{case_name}] SU2 Status: {status.value}")
            if report is not None:
                print(
                    f"  Gate={report.gate_tier} | log10(rms[Rho])={report.final_log10_rms_rho:.2f} | "
                    f"CL={report.cl_final:.4f} | CD={report.cd_final:.5f}"
                )
            if reason:
                print(f"  Note: {reason}")


if __name__ == "__main__":
    main()
