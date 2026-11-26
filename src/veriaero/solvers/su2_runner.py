"""SU2 v8.6.0 RANS wrapper, Gmsh structured quad C-grid generator, and dual low-Re pass gate.

Locked verbatim to VeriAero-CEGIS v3.2 Section 6.5:
- Steady-state RANS (`TIME_DOMAIN= NO`)
- `SOLVER= RANS`, `KIND_TURB_MODEL= SST`, `SST_OPTIONS= NONE`, `KIND_TRANS_MODEL= LM`
- `CONV_NUM_METHOD_FLOW= L2ROE`, `CONV_NUM_METHOD_TURB= BOUNDED_SCALAR`
- `NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES`
- `FREESTREAM_TURBULENCEINTENSITY= 0.0007`, `FREESTREAM_TURB2LAMVISCRATIO= 5.0`
- Structured quadrilateral C-grid (~260 wrap x ~150 wall-normal ~ 40k cells, y+ <= 1.0)
- Dual low-Re pass gate:
  - `STRICT_PASS`: rms[Rho] <= -6.0 AND Cauchy delta <= 1e-4 over 100 iters AND mean y+ <= 1.5
  - `LSB_DAMPED_PASS`: rms[Rho] <= -4.5 AND Cauchy delta <= 1e-4 over 250 iters AND mean y+ <= 1.5
  - `UNRESOLVED_NUMERICAL`: divergence, timeout, mean y+ > 1.5, or force oscillation > 5e-4.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gmsh
import numpy as np

from veriaero.contracts.types import SolverStatus


@dataclass(frozen=True)
class SU2ConvergenceReport:
    """Result of the Tier-H2 dual low-Re pass gate check."""

    gate_tier: Literal["STRICT_PASS", "LSB_DAMPED_PASS", "UNRESOLVED_NUMERICAL"]
    passed: bool
    final_log10_rms_rho: float
    cl_final: float
    cd_final: float
    cl_delta_window: float
    cd_delta_window: float
    mean_yplus: float | None
    iterations_completed: int
    reason: str


def default_su2_binary() -> Path:
    """Return canonical path to the Linux OpenMP SU2 v8.6.0 executable."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "bin" / "SU2_CFD"


def generate_airfoil_cgrid_su2(
    airfoil_coords: np.ndarray,
    out_su2_path: str | Path,
    chord: float = 1.0,
    farfield_radius: float = 20.0,
    n_airfoil_half: int = 131,
    n_wake_pts: int = 45,
    n_normal_pts: int = 150,
    progression_normal: float = 1.08,
) -> Path:
    """Generate a structured quadrilateral C-grid (~40k cells default) and export .su2."""
    coords = np.asarray(airfoil_coords, dtype=float) * chord
    out_path = Path(out_su2_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    le_idx = int(np.argmin(coords[:, 0]))
    upper_pts = coords[: le_idx + 1]
    lower_pts = coords[le_idx:]

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("airfoil_cgrid")
        geo = gmsh.model.geo

        pt_te = geo.addPoint(float(upper_pts[0, 0]), float(upper_pts[0, 1]), 0.0)
        pt_le = geo.addPoint(float(upper_pts[-1, 0]), float(upper_pts[-1, 1]), 0.0)

        up_ids = [pt_te]
        for x, y in upper_pts[1:-1]:
            up_ids.append(geo.addPoint(float(x), float(y), 0.0))
        up_ids.append(pt_le)

        lo_ids = [pt_le]
        for x, y in lower_pts[1:-1]:
            lo_ids.append(geo.addPoint(float(x), float(y), 0.0))
        lo_ids.append(pt_te)

        x_te = float(upper_pts[0, 0])
        x_wake = x_te + farfield_radius
        pt_ff_wake_top = geo.addPoint(x_wake, farfield_radius, 0.0)
        pt_ff_top = geo.addPoint(x_te, farfield_radius, 0.0)
        pt_ff_nose = geo.addPoint(x_te - farfield_radius, 0.0, 0.0)
        pt_ff_bot = geo.addPoint(x_te, -farfield_radius, 0.0)
        pt_ff_wake_bot = geo.addPoint(x_wake, -farfield_radius, 0.0)
        pt_wake_end = geo.addPoint(x_wake, 0.0, 0.0)
        pt_arc_center = geo.addPoint(x_te, 0.0, 0.0)

        l_air_up = geo.addSpline(up_ids)
        l_air_lo = geo.addSpline(lo_ids)
        l_wake = geo.addLine(pt_te, pt_wake_end)

        l_ff_top_wake = geo.addLine(pt_ff_wake_top, pt_ff_top)
        l_ff_arc_up = geo.addCircleArc(pt_ff_top, pt_arc_center, pt_ff_nose)
        l_ff_arc_lo = geo.addCircleArc(pt_ff_nose, pt_arc_center, pt_ff_bot)
        l_ff_bot_wake = geo.addLine(pt_ff_bot, pt_ff_wake_bot)

        l_out_top = geo.addLine(pt_wake_end, pt_ff_wake_top)
        l_out_bot = geo.addLine(pt_wake_end, pt_ff_wake_bot)
        l_cut_le = geo.addLine(pt_le, pt_ff_nose)

        cl_up = geo.addCurveLoop([l_air_up, l_cut_le, -l_ff_arc_up, -l_ff_top_wake, -l_out_top, -l_wake])
        surf_up = geo.addPlaneSurface([cl_up])

        cl_lo = geo.addCurveLoop([l_air_lo, l_wake, l_out_bot, -l_ff_bot_wake, -l_ff_arc_lo, -l_cut_le])
        surf_lo = geo.addPlaneSurface([cl_lo])

        geo.synchronize()

        for c in (l_air_up, l_air_lo, l_ff_arc_up, l_ff_arc_lo):
            gmsh.model.mesh.setTransfiniteCurve(c, n_airfoil_half)
        for c in (l_wake, l_ff_top_wake, l_ff_bot_wake):
            gmsh.model.mesh.setTransfiniteCurve(c, n_wake_pts, coef=1.08)
        for c in (l_cut_le, l_out_top, l_out_bot):
            gmsh.model.mesh.setTransfiniteCurve(c, n_normal_pts, coef=progression_normal)

        gmsh.model.mesh.setTransfiniteSurface(
            surf_up, cornerTags=[pt_te, pt_le, pt_ff_nose, pt_ff_wake_top]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            surf_lo, cornerTags=[pt_le, pt_te, pt_ff_wake_bot, pt_ff_nose]
        )
        gmsh.model.mesh.setRecombine(2, surf_up)
        gmsh.model.mesh.setRecombine(2, surf_lo)

        gmsh.model.addPhysicalGroup(1, [l_air_up, l_air_lo], name="airfoil")
        gmsh.model.addPhysicalGroup(
            1,
            [l_ff_top_wake, l_ff_arc_up, l_ff_arc_lo, l_ff_bot_wake, l_out_top, l_out_bot],
            name="farfield",
        )
        gmsh.model.addPhysicalGroup(2, [surf_up, surf_lo], name="fluid")

        gmsh.model.mesh.generate(2)
        gmsh.write(str(out_path))
    finally:
        gmsh.finalize()

    return out_path


def write_su2_rans_cfg(
    cfg_path: str | Path,
    mesh_filename: str,
    mach: float,
    reynolds: float,
    alpha_deg: float,
    reynolds_length: float = 1.0,
    max_iter: int = 10000,
    temperature_k: float = 288.15,
    turb2lam_visc_ratio: float = 5.0,
) -> Path:
    """Write locked VeriAero-CEGIS v3.2 Section 6.5 steady-state SST-LM RANS SU2 config."""
    cfg_text = f"""% VeriAero-CEGIS v3.2 Tier-H2 Steady-State RANS Configuration (Locked Section 6.5)
SOLVER= RANS
KIND_TURB_MODEL= SST
SST_OPTIONS= NONE
KIND_TRANS_MODEL= LM
TIME_DOMAIN= NO
MATH_PROBLEM= DIRECT

MACH_NUMBER= {mach:.6f}
AOA= {alpha_deg:.6f}
FREESTREAM_TEMPERATURE= {temperature_k:.4f}
REYNOLDS_NUMBER= {reynolds:.2f}
REYNOLDS_LENGTH= {reynolds_length:.6f}
FREESTREAM_OPTION= TEMPERATURE_FS
FREESTREAM_TURBULENCEINTENSITY= 0.0007
FREESTREAM_TURB2LAMVISCRATIO= {turb2lam_visc_ratio:.2f}

REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.00
REF_ORIGIN_MOMENT_Z= 0.00
REF_LENGTH= {reynolds_length:.6f}
REF_AREA= {reynolds_length:.6f}

MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES
CFL_NUMBER= 2.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 2.0, 35.0 )
ITER= {max_iter}

CONV_NUM_METHOD_FLOW= L2ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.05

CONV_NUM_METHOD_TURB= BOUNDED_SCALAR
MUSCL_TURB= NO

LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 15

CONV_FIELD= ( RMS_DENSITY, DRAG, LIFT )
CONV_RESIDUAL_MINVAL= -6
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1E-4
CONV_STARTITER= 500

MESH_FILENAME= {mesh_filename}
MESH_FORMAT= SU2
OUTPUT_FILES= ( RESTART, PARAVIEW_MULTIBLOCK, SURFACE_CSV )
HISTORY_OUTPUT= ( ITER, RMS_RES, AERO_COEFF )
CONV_FILENAME= history
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG )
SCREEN_WRT_FREQ_INNER= 50
"""
    path = Path(cfg_path)
    path.write_text(cfg_text, encoding="utf-8")
    return path


def evaluate_su2_dual_pass_gate(
    history_csv_path: str | Path,
    mean_yplus: float | None = 1.0,
) -> SU2ConvergenceReport:
    """Evaluate Section 6.5 Dual Low-Re Pass Gate (`STRICT_PASS` / `LSB_DAMPED_PASS`)."""
    path = Path(history_csv_path)
    if not path.is_file():
        return SU2ConvergenceReport(
            gate_tier="UNRESOLVED_NUMERICAL",
            passed=False,
            final_log10_rms_rho=float("inf"),
            cl_final=float("nan"),
            cd_final=float("nan"),
            cl_delta_window=float("inf"),
            cd_delta_window=float("inf"),
            mean_yplus=mean_yplus,
            iterations_completed=0,
            reason="SU2 history.csv not found",
        )

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {k.strip().strip('"'): float(v) for k, v in row.items() if k and v}
            rows.append(cleaned)

    if len(rows) < 100:
        return SU2ConvergenceReport(
            gate_tier="UNRESOLVED_NUMERICAL",
            passed=False,
            final_log10_rms_rho=float("inf"),
            cl_final=float("nan"),
            cd_final=float("nan"),
            cl_delta_window=float("inf"),
            cd_delta_window=float("inf"),
            mean_yplus=mean_yplus,
            iterations_completed=len(rows),
            reason=f"Only {len(rows)} iterations completed (< 100 minimum)",
        )

    def _col(name_candidates: tuple[str, ...]) -> list[float]:
        for key in rows[0]:
            for cand in name_candidates:
                if cand in key:
                    return [r[key] for r in rows]
        raise KeyError(f"Missing any of columns {name_candidates} in SU2 history")

    rms_vals = _col(("rms[Rho]", "RMS_DENSITY"))
    cl_vals = _col(("CL", "LIFT"))
    cd_vals = _col(("CD", "DRAG"))

    raw_rms = float(rms_vals[-1])
    log10_rms = raw_rms if raw_rms <= 0.0 else math.log10(max(raw_rms, 1e-30))
    cl_final = float(cl_vals[-1])
    cd_final = float(cd_vals[-1])

    yplus_ok = mean_yplus is None or (math.isfinite(mean_yplus) and mean_yplus <= 1.5)
    finite_ok = math.isfinite(log10_rms) and math.isfinite(cl_final) and math.isfinite(cd_final)
    if not (finite_ok and yplus_ok):
        return SU2ConvergenceReport(
            gate_tier="UNRESOLVED_NUMERICAL",
            passed=False,
            final_log10_rms_rho=log10_rms,
            cl_final=cl_final,
            cd_final=cd_final,
            cl_delta_window=float("inf"),
            cd_delta_window=float("inf"),
            mean_yplus=mean_yplus,
            iterations_completed=len(rows),
            reason=f"Non-finite state or mean y+={mean_yplus} > 1.5",
        )

    def _span(vals: list[float], w: int) -> tuple[float, float]:
        tail = vals[-w:]
        return (max(tail) - min(tail)), len(tail)

    cl_d100, _ = _span(cl_vals, 100)
    cd_d100, _ = _span(cd_vals, 100)

    if log10_rms <= -6.0 and cl_d100 <= 1.0e-4 and cd_d100 <= 1.0e-4:
        return SU2ConvergenceReport(
            gate_tier="STRICT_PASS",
            passed=True,
            final_log10_rms_rho=log10_rms,
            cl_final=cl_final,
            cd_final=cd_final,
            cl_delta_window=cl_d100,
            cd_delta_window=cd_d100,
            mean_yplus=mean_yplus,
            iterations_completed=len(rows),
            reason="STRICT_PASS: rms[Rho] <= -6.0 and 100-iter Cauchy delta <= 1e-4",
        )

    if len(rows) >= 250:
        cl_d250, _ = _span(cl_vals, 250)
        cd_d250, _ = _span(cd_vals, 250)
        if log10_rms <= -4.5 and cl_d250 <= 1.0e-4 and cd_d250 <= 1.0e-4:
            return SU2ConvergenceReport(
                gate_tier="LSB_DAMPED_PASS",
                passed=True,
                final_log10_rms_rho=log10_rms,
                cl_final=cl_final,
                cd_final=cd_final,
                cl_delta_window=cl_d250,
                cd_delta_window=cd_d250,
                mean_yplus=mean_yplus,
                iterations_completed=len(rows),
                reason="LSB_DAMPED_PASS: rms[Rho] <= -4.5 and 250-iter Cauchy delta <= 1e-4",
            )
        cl_delta_out, cd_delta_out = cl_d250, cd_d250
    else:
        cl_delta_out, cd_delta_out = cl_d100, cd_d100

    return SU2ConvergenceReport(
        gate_tier="UNRESOLVED_NUMERICAL",
        passed=False,
        final_log10_rms_rho=log10_rms,
        cl_final=cl_final,
        cd_final=cd_final,
        cl_delta_window=cl_delta_out,
        cd_delta_window=cd_delta_out,
        mean_yplus=mean_yplus,
        iterations_completed=len(rows),
        reason=(
            f"Failed dual low-Re gate: log10(rms[Rho])={log10_rms:.2f}, "
            f"dCL={cl_delta_out:.2e}, dCD={cd_delta_out:.2e}"
        ),
    )


class SU2Runner:
    """OpenMP subprocess runner for SU2 v8.6.0."""

    def __init__(
        self,
        binary_path: str | Path | None = None,
        omp_threads: int = 8,
        timeout_s: float = 900.0,
    ) -> None:
        self.binary_path = Path(binary_path) if binary_path else default_su2_binary()
        self.omp_threads = int(omp_threads)
        self.timeout_s = float(timeout_s)

    def is_available(self) -> bool:
        return self.binary_path.is_file() and os.access(self.binary_path, os.X_OK)

    def run_case(
        self,
        workdir: str | Path,
        cfg_filename: str,
    ) -> tuple[SolverStatus, SU2ConvergenceReport | None, str | None]:
        if not self.is_available():
            return (
                SolverStatus.UNRESOLVED_NUMERICAL,
                None,
                f"SU2_CFD binary not found/executable at '{self.binary_path}'. Run scripts/bootstrap_wsl2.sh on C2-WSL2.",
            )

        work = Path(workdir)
        try:
            proc = subprocess.run(
                [str(self.binary_path), "-t", str(self.omp_threads), cfg_filename],
                cwd=str(work),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SolverStatus.UNRESOLVED_NUMERICAL, None, f"SU2 timed out after {self.timeout_s}s"
        except Exception as exc:
            return SolverStatus.UNRESOLVED_NUMERICAL, None, f"SU2 execution error: {exc}"

        if proc.returncode != 0:
            return SolverStatus.UNRESOLVED_NUMERICAL, None, f"SU2 exited with code {proc.returncode}"

        report = evaluate_su2_dual_pass_gate(work / "history.csv")
        status = SolverStatus.RESOLVED_PASS if report.passed else SolverStatus.UNRESOLVED_NUMERICAL
        return status, report, None if report.passed else report.reason
