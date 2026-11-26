"""Sweep design parameters comparing Tier-L analytical vs Tier-H1 OpenAeroStruct.

Tests what drives discrepancies in induced drag (CDi), effective Oswald span
efficiency (e), and Pratt discrete gust / +3.0g structural stress across:
  1. Relative wing twist (tip_twist - root_twist) from -8 deg to +4 deg
  2. Taper ratio (lambda = c_t / c_r) from 0.30 to 0.85
  3. Aspect ratio (AR) from 8.0 to 16.0
  4. Spar cap thickness (t_cap) from 1.5 mm to 10.0 mm
  5. 2D Twist x Taper interaction grid (49 designs) to detect sign crossovers

Generates high-resolution figures and JSON summary in `outputs/diagnostics/`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from veriaero.contracts.schema import ProblemSchema
from veriaero.physics.tier_h1_uav import run_openaerostruct_wing
from veriaero.physics.tier_l_uav import (
    compute_aero_polar,
    compute_geometry,
    compute_mass_and_bom,
    compute_root_bending_stress,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "diagnostics"


def _eval_pair(
    schema: ProblemSchema,
    base_params: dict[str, float],
    selected_skus: dict[str, Any],
    airfoil_cst: dict[str, Any],
    overrides: dict[str, float],
) -> dict[str, float]:
    """Evaluate coupled Tier-L and Tier-H1 OpenAeroStruct metrics for one design."""
    params = dict(base_params)
    params.update(overrides)
    scen = schema.scenario_defaults

    geom = compute_geometry(params, airfoil_cst=airfoil_cst)
    mass = compute_mass_and_bom(params, geom, selected_skus)
    aero = compute_aero_polar(
        params=params,
        geom=geom,
        mass_kg=mass["m_total_kg"],
        airfoil_cst=airfoil_cst,
        selected_skus=selected_skus,
        scenario=scen,
    )
    struct = compute_root_bending_stress(
        params=params,
        geom=geom,
        mass_kg=mass["m_total_kg"],
        m_wing_kg=mass["m_wing_kg"],
        a_wing_per_rad=aero["a_wing_per_rad"],
        scenario=scen,
    )
    oas = run_openaerostruct_wing(
        params=params,
        geom=geom,
        mass_kg=mass["m_total_kg"],
        m_wing_kg=mass["m_wing_kg"],
        airfoil_cst=airfoil_cst,
        scenario=scen,
    )

    ar = geom["aspect_ratio"]
    cl_l = aero["cl_cruise"]
    cdi_l = aero["cd_induced"]
    e_l = aero["e_oswald"]

    cl_h1 = oas["cl_oas"]
    cdi_h1 = oas["cdi_oas"]
    e_h1 = (cl_h1 * cl_h1) / (math.pi * ar * max(cdi_h1, 1.0e-9))

    sig_gust_l = struct["sigma_gust_pratt_fs15_mpa"]
    sig_gust_h1 = oas["vonmises_gust_pratt_fs15_mpa"]
    sig_man_l = struct["sigma_maneuver_3g_fs15_mpa"]
    sig_man_h1 = oas["vonmises_maneuver_3g_fs15_mpa"]

    return {
        "aspect_ratio": ar,
        "taper_ratio": params["taper_ratio"],
        "root_twist_deg": params["root_twist_deg"],
        "tip_twist_deg": params["tip_twist_deg"],
        "delta_twist_deg": params["tip_twist_deg"] - params["root_twist_deg"],
        "t_cap_mm": params["spar_cap_thickness_m"] * 1000.0,
        "m_total_kg": mass["m_total_kg"],
        "cl_l": cl_l,
        "cl_h1": cl_h1,
        "cdi_l": cdi_l,
        "cdi_h1": cdi_h1,
        "delta_cdi": cdi_h1 - cdi_l,
        "e_l": e_l,
        "e_h1": e_h1,
        "sigma_man_l_mpa": sig_man_l,
        "sigma_man_h1_mpa": sig_man_h1,
        "sigma_gust_l_mpa": sig_gust_l,
        "sigma_gust_h1_mpa": sig_gust_h1,
        "delta_sigma_gust_mpa": sig_gust_h1 - sig_gust_l,
        "tip_disp_gust_mm": oas["tip_disp_gust_m"] * 1000.0,
        "n_gust_limit": oas["n_gust_limit"],
    }


def _twist_schedule() -> list[tuple[float, float]]:
    """Bounded (root_twist_deg, tip_twist_deg) pairs spanning delta_twist in [-8, +4] deg."""
    pairs = [
        (2.0, -6.0),   # delta = -8.0
        (1.0, -6.0),   # delta = -7.0
        (0.0, -6.0),   # delta = -6.0
        (0.0, -5.0),   # delta = -5.0
        (0.0, -4.0),   # delta = -4.0
        (0.0, -3.0),   # delta = -3.0
        (0.0, -2.0),   # delta = -2.0 (baseline)
        (0.0, -1.0),   # delta = -1.0
        (0.0, 0.0),    # delta =  0.0
        (0.0, 1.0),    # delta = +1.0
        (0.0, 2.0),    # delta = +2.0
        (-1.0, 2.0),   # delta = +3.0
        (-2.0, 2.0),   # delta = +4.0
    ]
    return sorted(pairs, key=lambda pair: pair[1] - pair[0])


def _twist_pair_for_delta(delta_deg: float) -> tuple[float, float]:
    """Map target delta_twist in [-8, +4] inside schema bounds root in [-2,4], tip in [-6,2]."""
    tip = min(2.0, max(-6.0, delta_deg))
    root = tip - delta_deg
    if root < -2.0:
        root = -2.0
        tip = root + delta_deg
    elif root > 4.0:
        root = 4.0
        tip = root + delta_deg
    return float(root), float(tip)


def _plot_cdi_figure(
    twist_rows: list[dict[str, float]],
    taper_rows: list[dict[str, float]],
    ar_rows: list[dict[str, float]],
    taper_grid: list[float],
    twist_grid: list[float],
    delta_cdi_grid: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=True)

    # Panel A: CDi & Oswald e vs Wing Twist
    ax = axes[0, 0]
    tx = [r["delta_twist_deg"] for r in twist_rows]
    ax.plot(tx, [r["cdi_l"] * 1e4 for r in twist_rows], "o-", color="#1f77b4", label="Tier-L CDi (x1e-4)")
    ax.plot(tx, [r["cdi_h1"] * 1e4 for r in twist_rows], "s-", color="#d62728", label="Tier-H1 OAS CDi (x1e-4)")
    ax.axvline(-2.0, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.set_title("(A) Induced Drag & Span Efficiency vs. Wing Twist", fontweight="bold")
    ax.set_xlabel("Relative Wing Twist (tip - root, deg)  [neg = washout]")
    ax.set_ylabel("Induced Drag Coefficient CDi (x1e-4)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(tx, [r["e_l"] for r in twist_rows], "o--", color="#6baed6", lw=1.3, label="Tier-L e_oswald")
    ax2.plot(tx, [r["e_h1"] for r in twist_rows], "s--", color="#fb6a4a", lw=1.3, label="Tier-H1 eff. e")
    ax2.set_ylabel("Effective Oswald Efficiency e")
    ax2.set_ylim(0.55, 1.10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", bbox_to_anchor=(0.98, 0.36), fontsize=8.0, framealpha=0.92)

    # Panel B: CDi & Oswald e vs Taper Ratio
    ax = axes[0, 1]
    lx = [r["taper_ratio"] for r in taper_rows]
    ax.plot(lx, [r["cdi_l"] * 1e4 for r in taper_rows], "o-", color="#1f77b4", label="Tier-L CDi (x1e-4)")
    ax.plot(lx, [r["cdi_h1"] * 1e4 for r in taper_rows], "s-", color="#d62728", label="Tier-H1 OAS CDi (x1e-4)")
    ax.axvline(0.45, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.set_title("(B) Induced Drag & Span Efficiency vs. Taper Ratio", fontweight="bold")
    ax.set_xlabel("Taper Ratio lambda = c_tip / c_root")
    ax.set_ylabel("Induced Drag Coefficient CDi (x1e-4)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(lx, [r["e_l"] for r in taper_rows], "o--", color="#6baed6", lw=1.3, label="Tier-L e_oswald")
    ax2.plot(lx, [r["e_h1"] for r in taper_rows], "s--", color="#fb6a4a", lw=1.3, label="Tier-H1 eff. e")
    ax2.set_ylabel("Effective Oswald Efficiency e")
    ax2.set_ylim(0.55, 1.10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", bbox_to_anchor=(0.98, 0.36), fontsize=8.0, framealpha=0.92)

    # Panel C: CDi vs Aspect Ratio
    ax = axes[1, 0]
    arx = [r["aspect_ratio"] for r in ar_rows]
    ax.plot(arx, [r["cdi_l"] * 1e4 for r in ar_rows], "o-", color="#1f77b4", label="Tier-L CDi")
    ax.plot(arx, [r["cdi_h1"] * 1e4 for r in ar_rows], "s-", color="#d62728", label="Tier-H1 OAS CDi")
    ax.axvline(12.0, color="gray", ls="--", lw=1.0, alpha=0.7, label="Baseline AR=12")
    ax.set_title("(C) Induced Drag vs. Aspect Ratio (1/AR Physics)", fontweight="bold")
    ax.set_xlabel("Aspect Ratio AR")
    ax.set_ylabel("Induced Drag Coefficient CDi (x1e-4)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel D: 2D Heatmap of Delta CDi = H1 - L
    ax = axes[1, 1]
    vmax = float(np.max(np.abs(delta_cdi_grid * 1e4)))
    im = ax.pcolormesh(
        taper_grid,
        twist_grid,
        delta_cdi_grid * 1e4,
        shading="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.plot(0.45, -2.0, "*", color="black", markersize=11, label="Baseline (0.45, -2 deg)")
    try:
        cs = ax.contour(
            taper_grid,
            twist_grid,
            delta_cdi_grid * 1e4,
            levels=[0.0],
            colors="black",
            linestyles="--",
            linewidths=1.6,
        )
        ax.clabel(cs, fmt="0", fontsize=8)
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta CDi = CDi_H1 - CDi_L (x1e-4)")
    ax.set_title("(D) 2D CDi Residual (Tier-H1 minus Tier-L)", fontweight="bold")
    ax.set_xlabel("Taper Ratio lambda")
    ax.set_ylabel("Relative Wing Twist (deg)")
    ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "VeriAero-CEGIS: What Drives Induced Drag Discrepancy Between Tier-L and Tier-H1?",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_stress_figure(
    twist_rows: list[dict[str, float]],
    taper_rows: list[dict[str, float]],
    tcap_rows: list[dict[str, float]],
    taper_grid: list[float],
    twist_grid: list[float],
    delta_sig_grid: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.6), constrained_layout=True)

    # Panel A: Pratt Gust Stress vs Wing Twist
    ax = axes[0, 0]
    tx = [r["delta_twist_deg"] for r in twist_rows]
    ax.plot(tx, [r["sigma_gust_l_mpa"] for r in twist_rows], "o-", color="#1f77b4", label="Tier-L Pratt Gust (FS=1.5)")
    ax.plot(tx, [r["sigma_gust_h1_mpa"] for r in twist_rows], "s-", color="#d62728", label="Tier-H1 OAS Wingbox (FS=1.5)")
    ax.axvline(-2.0, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.set_title("(A) Pratt Gust Wing Stress vs. Wing Twist", fontweight="bold")
    ax.set_xlabel("Relative Wing Twist (tip - root, deg)")
    ax.set_ylabel("Factored Stress (MPa, FS=1.5)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5)

    # Panel B: Pratt Gust Stress vs Taper Ratio
    ax = axes[0, 1]
    lx = [r["taper_ratio"] for r in taper_rows]
    ax.plot(lx, [r["sigma_gust_l_mpa"] for r in taper_rows], "o-", color="#1f77b4", label="Tier-L Pratt Gust (FS=1.5)")
    ax.plot(lx, [r["sigma_gust_h1_mpa"] for r in taper_rows], "s-", color="#d62728", label="Tier-H1 OAS Wingbox (FS=1.5)")
    ax.axvline(0.45, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.set_title("(B) Pratt Gust Wing Stress vs. Taper Ratio", fontweight="bold")
    ax.set_xlabel("Taper Ratio lambda = c_tip / c_root")
    ax.set_ylabel("Factored Stress (MPa, FS=1.5)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)

    # Panel C: Pratt Gust Stress & Deflection vs Spar Cap Thickness
    ax = axes[1, 0]
    cx = [r["t_cap_mm"] for r in tcap_rows]
    ax.plot(cx, [r["sigma_gust_l_mpa"] for r in tcap_rows], "o-", color="#1f77b4", label="Tier-L Stress (MPa)")
    ax.plot(cx, [r["sigma_gust_h1_mpa"] for r in tcap_rows], "s-", color="#d62728", label="Tier-H1 OAS Stress (MPa)")
    ax.axvline(4.5, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.set_title("(C) Stress & Aeroelastic Tip Deflection vs. Spar Cap", fontweight="bold")
    ax.set_xlabel("Spar Cap Thickness t_cap (mm)")
    ax.set_ylabel("Factored Gust Stress (MPa, FS=1.5)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(cx, [r["tip_disp_gust_mm"] for r in tcap_rows], "^--", color="#2ca02c", label="OAS Gust Tip Disp (mm)")
    ax2.set_ylabel("OAS Gust Tip Deflection (mm)", color="#2ca02c")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.0, framealpha=0.92)

    # Panel D: Signed Stress Residual Heatmap (Delta sigma = H1 - L)
    ax = axes[1, 1]
    vmax = max(5.0, float(np.max(np.abs(delta_sig_grid))))
    im = ax.pcolormesh(
        taper_grid,
        twist_grid,
        delta_sig_grid,
        shading="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.plot(0.45, -2.0, "*", color="black", markersize=11, label="Baseline (0.45, -2 deg)")
    if float(np.min(delta_sig_grid)) < 0.0 < float(np.max(delta_sig_grid)):
        cs = ax.contour(
            taper_grid,
            twist_grid,
            delta_sig_grid,
            levels=[0.0],
            colors="black",
            linestyles="--",
            linewidths=1.8,
        )
        ax.clabel(cs, fmt="0 MPa", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta sigma = H1 - L (MPa)  [red: H1 > L]")
    ax.set_title("(D) Gust Stress Residual & Tier-H1 > Tier-L Crossover", fontweight="bold")
    ax.set_xlabel("Taper Ratio lambda")
    ax.set_ylabel("Relative Wing Twist (deg)")
    ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "VeriAero-CEGIS: Where Does Tier-L Underestimate vs. Overestimate Wing Stress?",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = ProblemSchema.from_json_file(REPO_ROOT / "configs" / "uav25kg_schema.json")
    base_ir = schema.snap_design(schema.baseline_vector(), design_id="baseline")
    base_params = dict(base_ir.params)
    selected_skus = {cat: schema.catalog.get(sid) for cat, sid in base_ir.selected_skus.items()}
    airfoil_cst = dict(base_ir.airfoil_cst or {})

    # 1. Twist sweep
    twist_rows = [
        _eval_pair(
            schema,
            base_params,
            selected_skus,
            airfoil_cst,
            {"root_twist_deg": rt, "tip_twist_deg": tt},
        )
        for rt, tt in _twist_schedule()
    ]

    # 2. Taper ratio sweep
    taper_rows = [
        _eval_pair(schema, base_params, selected_skus, airfoil_cst, {"taper_ratio": float(lam)})
        for lam in np.linspace(0.30, 0.85, 12)
    ]

    # 3. Aspect ratio sweep
    ar_rows = [
        _eval_pair(schema, base_params, selected_skus, airfoil_cst, {"aspect_ratio": float(ar)})
        for ar in np.linspace(8.0, 16.0, 13)
    ]

    # 4. Spar cap thickness sweep
    tcap_rows = [
        _eval_pair(
            schema, base_params, selected_skus, airfoil_cst, {"spar_cap_thickness_m": float(t_mm / 1000.0)}
        )
        for t_mm in np.linspace(1.5, 10.0, 12)
    ]

    # 5. 2D Taper x Twist grid (49 designs)
    taper_grid = [float(x) for x in np.linspace(0.30, 0.85, 7)]
    twist_grid = [float(x) for x in np.linspace(-8.0, 4.0, 7)]
    delta_cdi_grid = np.zeros((len(twist_grid), len(taper_grid)), dtype=float)
    delta_sig_grid = np.zeros((len(twist_grid), len(taper_grid)), dtype=float)
    grid_rows: list[dict[str, float]] = []

    for i, dtw in enumerate(twist_grid):
        rt, tt = _twist_pair_for_delta(dtw)
        for j, lam in enumerate(taper_grid):
            row = _eval_pair(
                schema,
                base_params,
                selected_skus,
                airfoil_cst,
                {"taper_ratio": lam, "root_twist_deg": rt, "tip_twist_deg": tt},
            )
            delta_cdi_grid[i, j] = row["delta_cdi"]
            delta_sig_grid[i, j] = row["delta_sigma_gust_mpa"]
            grid_rows.append(row)

    cdi_png = OUT_DIR / "tier_l_vs_h1_cdi_and_e.png"
    stress_png = OUT_DIR / "tier_l_vs_h1_stress_crossover.png"
    _plot_cdi_figure(twist_rows, taper_rows, ar_rows, taper_grid, twist_grid, delta_cdi_grid, cdi_png)
    _plot_stress_figure(twist_rows, taper_rows, tcap_rows, taper_grid, twist_grid, delta_sig_grid, stress_png)

    baseline_row = _eval_pair(schema, base_params, selected_skus, airfoil_cst, {})
    crossover_pts = [r for r in grid_rows if r["delta_sigma_gust_mpa"] > 0.0]
    cdi_crossover_pts = [r for r in grid_rows if r["delta_cdi"] > 0.0]
    max_h1_over_l_stress = max(grid_rows + twist_rows + taper_rows + tcap_rows, key=lambda r: r["delta_sigma_gust_mpa"])
    min_h1_over_l_stress = min(grid_rows + twist_rows + taper_rows + tcap_rows, key=lambda r: r["delta_sigma_gust_mpa"])
    max_e_h1 = max(twist_rows + taper_rows + grid_rows, key=lambda r: r["e_h1"])
    min_e_h1 = min(twist_rows + taper_rows + grid_rows, key=lambda r: r["e_h1"])

    summary = {
        "baseline": baseline_row,
        "twist_sweep": twist_rows,
        "taper_sweep": taper_rows,
        "aspect_ratio_sweep": ar_rows,
        "spar_cap_sweep": tcap_rows,
        "grid_2d_twist_x_taper": grid_rows,
        "highlights": {
            "n_grid_stress_h1_exceeds_l": len(crossover_pts),
            "n_grid_total": len(grid_rows),
            "n_grid_cdi_h1_exceeds_l": len(cdi_crossover_pts),
            "max_delta_sigma_gust_mpa": max_h1_over_l_stress,
            "min_delta_sigma_gust_mpa": min_h1_over_l_stress,
            "max_effective_e_h1": max_e_h1,
            "min_effective_e_h1": min_e_h1,
        },
        "artifacts": {
            "cdi_plot": str(cdi_png),
            "stress_plot": str(stress_png),
        },
    }
    json_path = OUT_DIR / "tier_l_vs_h1_sweep.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== Tier-L vs Tier-H1 Sweep Summary (98 coupled evaluations) ===")
    print(
        f"Baseline (AR=12, lam=0.45, twist=-2 deg, t_cap=4.5 mm): "
        f"CDi_L={baseline_row['cdi_l']:.5f} (e_L={baseline_row['e_l']:.3f}) | "
        f"CDi_H1={baseline_row['cdi_h1']:.5f} (e_H1={baseline_row['e_h1']:.3f}) | "
        f"Gust Stress L={baseline_row['sigma_gust_l_mpa']:.2f} vs H1={baseline_row['sigma_gust_h1_mpa']:.2f} MPa"
    )
    print(
        f"Oswald e_H1 range across Twist x Taper: {min_e_h1['e_h1']:.3f} "
        f"(lam={min_e_h1['taper_ratio']:.2f}, dtw={min_e_h1['delta_twist_deg']:+.1f} deg) "
        f"to {max_e_h1['e_h1']:.3f} (lam={max_e_h1['taper_ratio']:.2f}, dtw={max_e_h1['delta_twist_deg']:+.1f} deg)"
    )
    print(
        f"2D Grid points where Tier-H1 gust stress EXCEEDS Tier-L: {len(crossover_pts)}/{len(grid_rows)} | "
        f"max(H1-L)={max_h1_over_l_stress['delta_sigma_gust_mpa']:+.2f} MPa "
        f"(lam={max_h1_over_l_stress['taper_ratio']:.2f}, dtw={max_h1_over_l_stress['delta_twist_deg']:+.1f} deg, "
        f"t_cap={max_h1_over_l_stress['t_cap_mm']:.2f} mm: "
        f"L={max_h1_over_l_stress['sigma_gust_l_mpa']:.1f} vs H1={max_h1_over_l_stress['sigma_gust_h1_mpa']:.1f} MPa)"
    )
    print(
        f"2D Grid points where Tier-H1 CDi EXCEEDS Tier-L: {len(cdi_crossover_pts)}/{len(grid_rows)} | "
        f"max(H1-L)={np.max(delta_cdi_grid):+.5f}"
    )
    print(f"Saved plots:\n  {cdi_png}\n  {stress_png}\n  {json_path}")


if __name__ == "__main__":
    main()
