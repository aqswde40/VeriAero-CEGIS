"""Day-3 SU2 v8.6.0 Eppler 387 & TMR NACA 0012 Validation Parity Check Script.

Generates structured quadrilateral C-grids using Gmsh (`gmsh==4.15.2`), writes
locked Section 6.5 steady-state SST-LM RANS configs (`TIME_DOMAIN= NO`, `L2ROE`,
`BOUNDED_SCALAR`, `WEIGHTED_LEAST_SQUARES`), and runs `bin/SU2_CFD` with the
dual low-Re pass gate (`STRICT_PASS` / `LSB_DAMPED_PASS`) on C2-WSL2.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import aerosandbox as asb
import numpy as np

from veriaero.contracts.types import HostProvenance
from veriaero.solvers.su2_runner import (
    SU2Runner,
    generate_airfoil_cgrid_su2_with_metrics,
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
        help="Use fast coarse mesh density (~7k cells) instead of production ~40k cells",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional output path for JSON parity summary (defaults to data/validation/su2_day3_parity.json on production runs)",
    )
    args = parser.parse_args()

    cases = [
        ("e387_re2e5_a4", "e387", 4.0, 2.0e5, 0.06, 5.0),
        ("naca0012_tmr_re6e6_a0", "naca0012", 0.0, 6.0e6, 0.15, 3.0),
    ]

    grid_kwargs = (
        {"n_airfoil_half": 51, "n_wake_pts": 23, "n_normal_pts": 75}
        if args.coarse
        else {"n_airfoil_half": 131, "n_wake_pts": 45, "n_normal_pts": 150}
    )

    runner = SU2Runner(omp_threads=8)
    results_out: list[dict[str, Any]] = []

    for case_name, af_name, alpha_deg, re_num, mach, t2l in cases:
        case_dir = args.workdir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = case_dir / f"{af_name}_cgrid.su2"
        cfg_path = case_dir / "rans_steady.cfg"

        coords = np.asarray(asb.Airfoil(af_name).coordinates, dtype=float)
        cgrid = generate_airfoil_cgrid_su2_with_metrics(
            coords,
            mesh_path,
            chord=1.0,
            farfield_radius=20.0,
            reynolds=re_num,
            target_yplus=0.5,
            **grid_kwargs,
        )
        write_su2_rans_cfg(
            cfg_path=cfg_path,
            mesh_filename=mesh_path.name,
            mach=mach,
            reynolds=re_num,
            alpha_deg=alpha_deg,
            turb2lam_visc_ratio=t2l,
        )
        print(
            f"[{case_name}] Gmsh C-grid: {mesh_path} ({mesh_path.stat().st_size} bytes, "
            f"{cgrid.n_quads} quads, minSJ={cgrid.min_scaled_jacobian:.3f}) | cfg: {cfg_path}"
        )

        case_entry: dict[str, Any] = {
            "case_name": case_name,
            "airfoil": af_name,
            "alpha_deg": alpha_deg,
            "reynolds": re_num,
            "mach": mach,
            "mesh_n_quads": cgrid.n_quads,
            "mesh_min_scaled_jacobian": cgrid.min_scaled_jacobian,
            "mesh_est_yplus": cgrid.estimated_yplus,
        }

        if not args.mesh_only:
            t0 = time.perf_counter()
            status, report, reason = runner.run_case(workdir=case_dir, cfg_filename=cfg_path.name)
            wall_s = time.perf_counter() - t0
            print(f"[{case_name}] SU2 Status: {status.value} ({wall_s:.1f}s)")
            case_entry["status"] = status.value
            case_entry["wall_time_s"] = round(wall_s, 2)
            case_entry["note"] = reason
            if report is not None:
                yp_str = f"{report.mean_yplus:.3f}" if report.mean_yplus is not None else "n/a"
                print(
                    f"  Gate={report.gate_tier} | log10(rms[Rho])={report.final_log10_rms_rho:.2f} | "
                    f"mean_y+={yp_str} | CL={report.cl_final:.4f} | CD={report.cd_final:.5f}"
                )
                case_entry.update(
                    {
                        "gate_tier": report.gate_tier,
                        "final_log10_rms_rho": report.final_log10_rms_rho,
                        "mean_yplus": report.mean_yplus,
                        "cl_final": report.cl_final,
                        "cd_final": report.cd_final,
                        "cl_delta_window": report.cl_delta_window,
                        "cd_delta_window": report.cd_delta_window,
                        "iterations_completed": report.iterations_completed,
                    }
                )
            if reason:
                print(f"  Note: {reason}")

        results_out.append(case_entry)

    summary_path = args.summary_json
    if summary_path is None and not args.coarse and not args.mesh_only:
        summary_path = Path("data/validation/su2_day3_parity.json")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "coarse": bool(args.coarse),
            "mesh_only": bool(args.mesh_only),
            "host_provenance": HostProvenance.capture().model_dump(),
            "cases": results_out,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved SU2 validation summary: {summary_path}")


if __name__ == "__main__":
    main()
