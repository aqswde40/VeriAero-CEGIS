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
from scipy.optimize import brentq

from veriaero.contracts.types import SolverStatus


@dataclass(frozen=True)
class CGridQualityReport:
    """Mesh quality and boundary-layer spacing metrics for a Gmsh C-grid."""

    mesh_path: Path
    n_quads: int
    min_scaled_jacobian: float
    first_layer_dy: float
    progression_normal: float
    estimated_yplus: float | None


def compute_first_layer_height(
    reynolds: float,
    chord: float = 1.0,
    target_yplus: float = 0.5,
) -> float:
    """Compute turbulent wall-normal first-layer height dy1 for target y+ <= 1.0."""
    re = max(float(reynolds), 1.0e4)
    cf = 0.026 / (re ** (1.0 / 7.0))
    u_tau_ratio = math.sqrt(cf / 2.0)
    return float((target_yplus * chord) / (re * u_tau_ratio))


def solve_wall_normal_progression(
    farfield_radius: float,
    n_normal_pts: int,
    first_layer_dy: float,
) -> float:
    """Solve L * (r - 1) / (r^(N-1) - 1) = dy1 for geometric progression ratio r > 1."""
    n_seg = max(int(n_normal_pts) - 1, 2)
    uniform_dy = farfield_radius / n_seg
    if first_layer_dy >= uniform_dy:
        return 1.01

    def residual(r: float) -> float:
        return farfield_radius * (r - 1.0) / (r**n_seg - 1.0) - first_layer_dy

    r_lo = 1.0001
    r_hi = 1.60
    while residual(r_hi) > 0.0 and r_hi < 4.0:
        r_hi *= 1.5
    return float(brentq(residual, r_lo, r_hi, xtol=1e-7))


def parse_su2_surface_yplus(surface_csv_path: str | Path) -> float | None:
    """Parse mean wall y+ from SU2 surface_flow.csv output."""
    path = Path(surface_csv_path)
    if not path.is_file():
        return None
    vals: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return None
        yplus_col: str | None = None
        for col in reader.fieldnames:
            norm = col.strip().strip('"').lower().replace("_", "").replace(" ", "")
            if norm in ("yplus", "y+"):
                yplus_col = col
                break
        if yplus_col is None:
            return None
        for row in reader:
            raw = row.get(yplus_col)
            if raw is None:
                continue
            try:
                v = float(raw)
                if math.isfinite(v) and v >= 0.0:
                    vals.append(v)
            except ValueError:
                continue
    if not vals:
        return None
    return float(np.mean(vals))


def parse_su2_vtu_yplus(vtu_path: str | Path) -> float | None:
    """Parse mean wall y+ from SU2 v8.6.0 ParaView surface mesh (`airfoil.vtu`)."""
    path = Path(vtu_path)
    if not path.is_file():
        return None
    try:
        import pyvista as pv

        mesh = pv.read(str(path))
    except Exception:
        return None

    for container in (mesh.point_data, mesh.cell_data):
        for name in container.keys():
            norm = str(name).strip().lower().replace("_", "").replace(" ", "")
            if norm in ("yplus", "y+"):
                arr = np.asarray(container[name], dtype=float)
                finite = arr[np.isfinite(arr) & (arr >= 0.0)]
                if finite.size > 0:
                    return float(np.mean(finite))
    return None


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


def generate_airfoil_cgrid_su2_with_metrics(
    airfoil_coords: np.ndarray,
    out_su2_path: str | Path,
    chord: float = 1.0,
    farfield_radius: float = 20.0,
    n_airfoil_half: int = 131,
    n_wake_pts: int = 45,
    n_normal_pts: int = 150,
    progression_normal: float | None = None,
    reynolds: float | None = None,
    target_yplus: float = 0.5,
    min_scaled_jacobian: float = 0.20,
) -> CGridQualityReport:
    """Generate a structured quad C-grid with Re-adaptive y+ spacing and minSJ quality gate."""
    coords = np.asarray(airfoil_coords, dtype=float) * chord
    out_path = Path(out_su2_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if progression_normal is None:
        if reynolds is not None:
            dy1_target = compute_first_layer_height(reynolds, chord=chord, target_yplus=target_yplus)
            prog = solve_wall_normal_progression(farfield_radius, n_normal_pts, dy1_target)
        else:
            prog = 1.08
    else:
        prog = float(progression_normal)

    n_seg = max(int(n_normal_pts) - 1, 2)
    dy1_actual = (
        farfield_radius * (prog - 1.0) / (prog**n_seg - 1.0)
        if abs(prog - 1.0) > 1e-9
        else farfield_radius / n_seg
    )
    est_yplus: float | None = None
    if reynolds is not None:
        cf = 0.026 / (max(float(reynolds), 1.0e4) ** (1.0 / 7.0))
        est_yplus = float((dy1_actual / chord) * reynolds * math.sqrt(cf / 2.0))

    le_idx = int(np.argmin(coords[:, 0]))
    upper_pts = coords[: le_idx + 1]
    lower_pts = coords[le_idx:]

    # Close blunt trailing edges symmetrically at TE midpoint to avoid skewed TE cells
    te_x = 0.5 * (float(upper_pts[0, 0]) + float(lower_pts[-1, 0]))
    te_y = 0.5 * (float(upper_pts[0, 1]) + float(lower_pts[-1, 1]))

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("airfoil_cgrid")
        geo = gmsh.model.geo

        pt_te = geo.addPoint(te_x, te_y, 0.0)
        pt_le = geo.addPoint(float(upper_pts[-1, 0]), float(upper_pts[-1, 1]), 0.0)

        up_ids = [pt_te]
        for x, y in upper_pts[1:-1]:
            up_ids.append(geo.addPoint(float(x), float(y), 0.0))
        up_ids.append(pt_le)

        lo_ids = [pt_le]
        for x, y in lower_pts[1:-1]:
            lo_ids.append(geo.addPoint(float(x), float(y), 0.0))
        lo_ids.append(pt_te)

        x_te = te_x
        x_wake = x_te + farfield_radius
        pt_ff_wake_top = geo.addPoint(x_wake, farfield_radius, 0.0)
        pt_ff_top = geo.addPoint(x_te, farfield_radius, 0.0)
        pt_ff_nose = geo.addPoint(x_te - farfield_radius, 0.0, 0.0)
        pt_ff_bot = geo.addPoint(x_te, -farfield_radius, 0.0)
        pt_ff_wake_bot = geo.addPoint(x_wake, -farfield_radius, 0.0)
        pt_wake_end = geo.addPoint(x_wake, te_y, 0.0)
        pt_arc_center = geo.addPoint(x_te, 0.0, 0.0)

        l_air_up = geo.addSpline(up_ids)
        l_air_lo = geo.addSpline(lo_ids)
        l_wake = geo.addLine(pt_te, pt_wake_end)

        # Farfield edges oriented to match airfoil & wake directions for consistent clustering
        l_ff_top_wake = geo.addLine(pt_ff_top, pt_ff_wake_top)
        l_ff_arc_up = geo.addCircleArc(pt_ff_top, pt_arc_center, pt_ff_nose)
        l_ff_arc_lo = geo.addCircleArc(pt_ff_nose, pt_arc_center, pt_ff_bot)
        l_ff_bot_wake = geo.addLine(pt_ff_bot, pt_ff_wake_bot)

        # Wall-normal radial/vertical connectors (all oriented wall -> farfield for coef=prog)
        l_cut_le = geo.addLine(pt_le, pt_ff_nose)
        l_cut_te_up = geo.addLine(pt_te, pt_ff_top)
        l_cut_te_lo = geo.addLine(pt_te, pt_ff_bot)
        l_out_top = geo.addLine(pt_wake_end, pt_ff_wake_top)
        l_out_bot = geo.addLine(pt_wake_end, pt_ff_wake_bot)

        # 4-block structured C-grid: upper airfoil, lower airfoil, upper wake, lower wake
        cl_air_up = geo.addCurveLoop([l_air_up, l_cut_le, -l_ff_arc_up, -l_cut_te_up])
        surf_air_up = geo.addPlaneSurface([cl_air_up])

        cl_air_lo = geo.addCurveLoop([l_air_lo, l_cut_te_lo, -l_ff_arc_lo, -l_cut_le])
        surf_air_lo = geo.addPlaneSurface([cl_air_lo])

        cl_wake_up = geo.addCurveLoop([l_wake, l_out_top, -l_ff_top_wake, -l_cut_te_up])
        surf_wake_up = geo.addPlaneSurface([cl_wake_up])

        cl_wake_lo = geo.addCurveLoop([l_wake, l_out_bot, -l_ff_bot_wake, -l_cut_te_lo])
        surf_wake_lo = geo.addPlaneSurface([cl_wake_lo])

        geo.synchronize()

        for c in (l_air_up, l_air_lo, l_ff_arc_up, l_ff_arc_lo):
            gmsh.model.mesh.setTransfiniteCurve(c, n_airfoil_half, meshType="Bump", coef=0.25)
        for c in (l_wake, l_ff_top_wake, l_ff_bot_wake):
            gmsh.model.mesh.setTransfiniteCurve(c, n_wake_pts, coef=1.08)
        for c in (l_cut_le, l_cut_te_up, l_cut_te_lo, l_out_top, l_out_bot):
            gmsh.model.mesh.setTransfiniteCurve(c, n_normal_pts, coef=prog)

        gmsh.model.mesh.setTransfiniteSurface(
            surf_air_up, cornerTags=[pt_te, pt_le, pt_ff_nose, pt_ff_top]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            surf_air_lo, cornerTags=[pt_le, pt_te, pt_ff_bot, pt_ff_nose]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            surf_wake_up, cornerTags=[pt_te, pt_wake_end, pt_ff_wake_top, pt_ff_top]
        )
        gmsh.model.mesh.setTransfiniteSurface(
            surf_wake_lo, cornerTags=[pt_te, pt_wake_end, pt_ff_wake_bot, pt_ff_bot]
        )
        all_surfs = [surf_air_up, surf_air_lo, surf_wake_up, surf_wake_lo]
        for s_tag in all_surfs:
            gmsh.model.mesh.setRecombine(2, s_tag)

        gmsh.model.addPhysicalGroup(1, [l_air_up, l_air_lo], name="airfoil")
        gmsh.model.addPhysicalGroup(
            1,
            [l_ff_top_wake, l_ff_arc_up, l_ff_arc_lo, l_ff_bot_wake, l_out_top, l_out_bot],
            name="farfield",
        )
        gmsh.model.addPhysicalGroup(2, all_surfs, name="fluid")

        gmsh.model.mesh.generate(2)

        _elem_types, elem_tags_list, _node_tags = gmsh.model.mesh.getElements(dim=2)
        all_quad_tags = np.concatenate(elem_tags_list) if elem_tags_list else np.array([], dtype=np.uint64)
        if all_quad_tags.size == 0:
            raise ValueError("Gmsh generated 0 2D elements for C-grid")
        qualities = np.asarray(
            gmsh.model.mesh.getElementQualities(all_quad_tags.tolist(), "minSJ"), dtype=float
        )
        min_sj = float(np.min(qualities))
        if min_sj < min_scaled_jacobian:
            raise ValueError(
                f"Gmsh C-grid failed scaled Jacobian gate: minSJ={min_sj:.4f} < {min_scaled_jacobian:.2f}"
            )

        gmsh.write(str(out_path))
    finally:
        gmsh.finalize()

    return CGridQualityReport(
        mesh_path=out_path,
        n_quads=int(all_quad_tags.size),
        min_scaled_jacobian=min_sj,
        first_layer_dy=float(dy1_actual),
        progression_normal=float(prog),
        estimated_yplus=est_yplus,
    )


def generate_airfoil_cgrid_su2(
    airfoil_coords: np.ndarray,
    out_su2_path: str | Path,
    chord: float = 1.0,
    farfield_radius: float = 20.0,
    n_airfoil_half: int = 131,
    n_wake_pts: int = 45,
    n_normal_pts: int = 150,
    progression_normal: float | None = None,
    reynolds: float | None = None,
    target_yplus: float = 0.5,
    min_scaled_jacobian: float = 0.20,
) -> Path:
    """Generate a structured quadrilateral C-grid (~40k cells default) and export .su2."""
    rep = generate_airfoil_cgrid_su2_with_metrics(
        airfoil_coords=airfoil_coords,
        out_su2_path=out_su2_path,
        chord=chord,
        farfield_radius=farfield_radius,
        n_airfoil_half=n_airfoil_half,
        n_wake_pts=n_wake_pts,
        n_normal_pts=n_normal_pts,
        progression_normal=progression_normal,
        reynolds=reynolds,
        target_yplus=target_yplus,
        min_scaled_jacobian=min_scaled_jacobian,
    )
    return rep.mesh_path


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
    mean_yplus: float | None = None,
) -> SU2ConvergenceReport:
    """Evaluate Section 6.5 Dual Low-Re Pass Gate (`STRICT_PASS` / `LSB_DAMPED_PASS`).

    Never defaults `mean_yplus` to a passing value: missing or > 1.5 wall y+ always
    returns `UNRESOLVED_NUMERICAL`.
    """
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

    yplus_ok = mean_yplus is not None and math.isfinite(mean_yplus) and mean_yplus <= 1.5
    finite_ok = math.isfinite(log10_rms) and math.isfinite(cl_final) and math.isfinite(cd_final)
    if not (finite_ok and yplus_ok):
        reason_msg = (
            "Missing or non-finite mean y+ (surface_flow.csv not verified)"
            if mean_yplus is None or not math.isfinite(mean_yplus)
            else f"Non-finite state or mean y+={mean_yplus:.3f} > 1.5"
        )
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
            reason=reason_msg,
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

        surface_candidates = [
            work / "surface_flow.csv",
            work / "surface.csv",
            *sorted(work.glob("surface_flow*.csv")),
            *sorted(work.glob("surface*.csv")),
        ]
        parsed_yplus: float | None = None
        for surf_path in surface_candidates:
            if surf_path.is_file():
                parsed_yplus = parse_su2_surface_yplus(surf_path)
                if parsed_yplus is not None:
                    break

        if parsed_yplus is None:
            vtu_candidates = [
                *sorted(work.glob("vol_solution/**/airfoil.vtu")),
                *sorted(work.glob("*airfoil*.vtu")),
            ]
            for vtu_path in vtu_candidates:
                if vtu_path.is_file():
                    parsed_yplus = parse_su2_vtu_yplus(vtu_path)
                    if parsed_yplus is not None:
                        break

        report = evaluate_su2_dual_pass_gate(work / "history.csv", mean_yplus=parsed_yplus)
        status = SolverStatus.RESOLVED_PASS if report.passed else SolverStatus.UNRESOLVED_NUMERICAL
        return status, report, None if report.passed else report.reason
