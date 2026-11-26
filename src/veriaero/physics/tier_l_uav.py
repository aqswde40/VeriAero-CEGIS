"""Honest Tier-L Multi-Physics Oracle for the 25 kg Lift+Cruise eVTOL UAV.

Aligned with VeriAero-CEGIS v3.2 Architecture Reference (Sections 6.1 & 6.5):
1. Reads the locked 26-D schema (`wing_area_m2`, `aspect_ratio`, `taper_ratio`,
   `airfoil_pca_0..3`, 4 catalog categories `cell`, `lift_motor`, `esc`, `cruise_motor`,
   and discrete `(N_s, N_p)` topology).
2. Closed 25.00 kg mass balance (no clamping on perturbed designs; Vertiq 81-08 G2
   integrated ESC enforces lift-ESC mass & cost = 0).
3. True trapezoidal Mean Aerodynamic Chord (MAC = 0.3144 m at baseline).
4. `NeuralFoil 0.3.3` on 18-param Kulfan CST (`kulfan_parameters`) + full parasitic
   drag buildup + 3D induced drag (baseline L/D ~ 12.45, range ~ 151.2 km).
5. Separated `+3.0g` maneuver vs. dynamic Pratt discrete gust (`U_de = 10 m/s`)
   Euler-Bernoulli root stress at `FS = 1.5` (`sigma_allow = 350 MPa`).
6. 6-rotor twin-boom static OEI trim convex QP (`cvxpy`) across 6 failures x +-5% MAC CG.
7. DCIR + Arrhenius 0 C resistance scaling (`E_a = 25 kJ/mol`, 20-30 kJ/mol band) +
   exact quadratic voltage sag + two-phase cruise-preheated (`T_cruise_eq`) feeding
   90-s OEI thermal ODE (`c_p = 850 J/(kg*K)`) with live Arrhenius `R_pack(T)`.
8. Evaluates both operational scenario and 0 C cold-soak OEI to output the 3 Pareto
   objectives (`verified_range_km`, `domestic_bom_cost_usd`, `cold_oei_margin`) + `m_safe`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import neuralfoil as nf
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from veriaero.contracts.schema import DesignIR, ProblemSchema, SKUItem
from veriaero.contracts.types import (
    BudgetLedger,
    EvalResult,
    HostProvenance,
    OracleTier,
    SolverStatus,
)
from veriaero.physics.airfoil_cst import (
    kulfan_dict_to_neuralfoil_params,
    reconstruct_kulfan_from_pca,
)

G_ACCEL = 9.80665
R_GAS = 8.314462618  # J/(mol*K)
CELL_CP_J_KG_K = 850.0  # Conservative cylindrical 21700 specific heat (Steinhardt et al. 2022)
R_BUS_OHM = 0.004  # Harness, connector & busbar parasitic resistance

# Fixed baseline airframe dimensions & mission payload
FUSE_LENGTH_M = 1.55
FUSE_DIAMETER_M = 0.24
BOOM_LENGTH_M = 1.70
BOOM_Y_FRAC = 0.36
PAYLOAD_MASS_KG = 4.000
LIFT_ROTOR_BLADE_MASS_KG = 0.065  # per rotor
CRUISE_PROP_MASS_KG = 0.090
CRUISE_PROP_AERODYN_EFF = 0.833333  # combined with esc 0.96 * motor 0.90 -> 0.720 total
AVIONICS_MISC_MASS_KG = 1.531137  # Pixhawk Cube, RTK, telemetry, landing skids, gimbal


@dataclass(frozen=True)
class QuadraticCircuitSolution:
    """Exact solution of P_req = (V_oc - I * R_pack) * I."""

    current_a: float
    terminal_voltage_v: float
    p_max_w: float
    discriminant: float
    voltage_collapse: bool
    residual_w: float


def arrhenius_resistance_scale(
    temp_c: float,
    ea_j_mol: float = 25000.0,
    temp_ref_c: float = 25.0,
) -> float:
    """Arrhenius DCIR scaling factor relative to 25 C."""
    t_k = temp_c + 273.15
    t_ref_k = temp_ref_c + 273.15
    exponent = (ea_j_mol / R_GAS) * ((1.0 / t_k) - (1.0 / t_ref_k))
    return float(math.exp(exponent))


def solve_quadratic_circuit(
    v_oc: float,
    r_pack: float,
    p_req_w: float,
) -> QuadraticCircuitSolution:
    """Solve R_pack * I^2 - V_oc * I + P_req = 0 on the physical lower-current branch."""
    if r_pack <= 0.0 or v_oc <= 0.0:
        return QuadraticCircuitSolution(
            current_a=float("nan"),
            terminal_voltage_v=float("nan"),
            p_max_w=0.0,
            discriminant=-1.0,
            voltage_collapse=True,
            residual_w=float("nan"),
        )

    p_max_w = (v_oc * v_oc) / (4.0 * r_pack)
    disc = (v_oc * v_oc) - 4.0 * p_req_w * r_pack
    if disc < 0.0:
        return QuadraticCircuitSolution(
            current_a=v_oc / (2.0 * r_pack),
            terminal_voltage_v=v_oc / 2.0,
            p_max_w=p_max_w,
            discriminant=disc,
            voltage_collapse=True,
            residual_w=abs(p_req_w - p_max_w),
        )

    current_a = (v_oc - math.sqrt(disc)) / (2.0 * r_pack)
    v_term = v_oc - current_a * r_pack
    residual_w = abs((v_term * current_a) - p_req_w)
    return QuadraticCircuitSolution(
        current_a=current_a,
        terminal_voltage_v=v_term,
        p_max_w=p_max_w,
        discriminant=disc,
        voltage_collapse=False,
        residual_w=residual_w,
    )


def compute_geometry(
    params: dict[str, float],
    airfoil_cst: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute exact trapezoidal wing planform, MAC, tail areas, and wetted areas."""
    s_wing = params["wing_area_m2"]
    ar = params["aspect_ratio"]
    lam = params["taper_ratio"]
    b = math.sqrt(s_wing * ar)
    c_avg = s_wing / b
    c_r = (2.0 * c_avg) / (1.0 + lam)
    c_t = c_r * lam
    mac = (2.0 / 3.0) * c_r * (1.0 + lam + lam * lam) / (1.0 + lam)

    l_tail = 0.85 * BOOM_LENGTH_M
    s_h = 0.55 * s_wing * mac / l_tail
    s_v = 0.045 * s_wing * b / l_tail

    s_wet_fuse = 0.78 * math.pi * FUSE_DIAMETER_M * FUSE_LENGTH_M
    s_wet_booms = 2.0 * math.pi * 0.036 * BOOM_LENGTH_M
    tc = float(airfoil_cst["max_thickness_over_c"]) if airfoil_cst else 0.1108

    return {
        "wing_span_m": b,
        "c_root_m": c_r,
        "c_tip_m": c_t,
        "c_avg_m": c_avg,
        "wing_area_m2": s_wing,
        "aspect_ratio": ar,
        "mac_m": mac,
        "airfoil_tc": tc,
        "l_tail_m": l_tail,
        "s_h_m2": s_h,
        "s_v_m2": s_v,
        "s_wet_fuse_m2": s_wet_fuse,
        "s_wet_booms_m2": s_wet_booms,
    }


def compute_mass_and_bom(
    params: dict[str, float],
    geom: dict[str, float],
    selected_skus: dict[str, SKUItem],
) -> dict[str, float]:
    """Compute un-clamped closed mass breakdown (sums to 25.0000 kg at baseline) & BOM cost."""
    n_s = params["batt_n_series"]
    n_p = params["batt_n_parallel"]
    n_cells = n_s * n_p

    cell_sku = selected_skus["cell"]
    lift_sku = selected_skus["lift_motor"]
    esc_sku = selected_skus["esc"]
    cruise_sku = selected_skus["cruise_motor"]

    m_cells = n_cells * cell_sku.specs["cell_mass_kg"]
    m_batt = 1.15 * m_cells  # +15% pack casing, BMS, nickel/copper busbars

    m_lift_motors = 6.0 * lift_sku.specs["lift_motor_mass_kg"]
    m_lift_rotors = 6.0 * LIFT_ROTOR_BLADE_MASS_KG
    # Integrated ESC (e.g. Vertiq 81-08 G2) enforces lift ESC mass & cost = 0 (only 1 cruise ESC)
    n_escs = 1.0 if lift_sku.has_integrated_esc else 7.0
    m_escs = n_escs * esc_sku.specs["esc_mass_kg"]

    m_cruise_motor = cruise_sku.specs["cruise_motor_mass_kg"]
    m_cruise_prop = CRUISE_PROP_MASS_KG

    s_wing = geom["wing_area_m2"]
    b = geom["wing_span_m"]
    c_avg = geom["c_avg_m"]
    tc = geom["airfoil_tc"]
    t_cap = params["spar_cap_thickness_m"]
    t_web = params["web_thickness_m"]
    t_skin = params["skin_thickness_m"]

    m_wing_skin = 2.0 * s_wing * t_skin * 1600.0 * 0.65
    m_wing_spar = 2.0 * (0.10 * c_avg) * t_cap * b * 1600.0
    m_wing_web = 0.85 * tc * c_avg * t_web * b * 1600.0
    m_wing = m_wing_skin + m_wing_spar + m_wing_web

    m_fuse = 2.60 * geom["s_wet_fuse_m2"]
    m_booms = 2.0 * 0.55 * BOOM_LENGTH_M
    m_tail = 1.80 * (geom["s_h_m2"] + geom["s_v_m2"])
    m_payload = PAYLOAD_MASS_KG
    m_avionics_misc = AVIONICS_MISC_MASS_KG

    m_total = (
        m_batt
        + m_lift_motors
        + m_lift_rotors
        + m_escs
        + m_cruise_motor
        + m_cruise_prop
        + m_wing
        + m_fuse
        + m_booms
        + m_tail
        + m_payload
        + m_avionics_misc
    )

    bom_usd = (
        n_cells * cell_sku.unit_cost_usd
        + 6.0 * lift_sku.unit_cost_usd
        + n_escs * esc_sku.unit_cost_usd
        + cruise_sku.unit_cost_usd
    )

    return {
        "n_cells": float(n_cells),
        "m_cells_kg": m_cells,
        "m_batt_kg": m_batt,
        "m_lift_motors_kg": m_lift_motors,
        "m_lift_rotors_kg": m_lift_rotors,
        "m_escs_kg": m_escs,
        "m_cruise_motor_kg": m_cruise_motor,
        "m_cruise_prop_kg": m_cruise_prop,
        "m_wing_kg": m_wing,
        "m_fuse_kg": m_fuse,
        "m_booms_kg": m_booms,
        "m_tail_kg": m_tail,
        "m_payload_kg": m_payload,
        "m_avionics_misc_kg": m_avionics_misc,
        "m_total_kg": m_total,
        "domestic_bom_cost_usd": bom_usd,
    }


def compute_aero_polar(
    params: dict[str, float],
    geom: dict[str, float],
    mass_kg: float,
    airfoil_cst: dict[str, Any],
    selected_skus: dict[str, SKUItem],
    scenario: dict[str, float],
) -> dict[str, float]:
    """Compute NeuralFoil 18-param Kulfan CST polar + full parasitic buildup + 3D induced drag."""
    rho = scenario["rho_air"]
    mu = scenario["mu_air"]
    v = scenario.get("cruise_speed_mps", 24.0)
    s_wing = geom["wing_area_m2"]
    mac = geom["mac_m"]
    ar = geom["aspect_ratio"]
    lam = params["taper_ratio"]

    weight_n = mass_kg * G_ACCEL
    q_dyn = 0.5 * rho * v * v
    cl_req = weight_n / (q_dyn * s_wing)
    re_mac = rho * v * mac / mu

    kf_params = kulfan_dict_to_neuralfoil_params(airfoil_cst)
    alpha_grid = np.linspace(-4.0, 15.0, 39)
    nf_out = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=kf_params,
        alpha=alpha_grid,
        Re=re_mac,
        model_size="large",
    )
    cl_grid = np.asarray(nf_out["CL"], dtype=float)
    cd_grid = np.asarray(nf_out["CD"], dtype=float)
    conf_grid = np.asarray(nf_out["analysis_confidence"], dtype=float)

    if not (np.all(np.isfinite(cl_grid)) and np.all(np.isfinite(cd_grid))):
        raise RuntimeError("NeuralFoil returned non-finite polar values")

    stall_idx = int(np.argmax(cl_grid))
    cl_max = float(cl_grid[stall_idx])
    alpha_pre = alpha_grid[: stall_idx + 1]
    cl_pre = cl_grid[: stall_idx + 1]

    if cl_req <= cl_pre[0]:
        alpha_cruise = float(alpha_pre[0])
    elif cl_req >= cl_pre[-1]:
        alpha_cruise = float(alpha_pre[-1])
    else:
        alpha_cruise = float(np.interp(cl_req, cl_pre, alpha_pre))

    nf_cruise = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=kf_params,
        alpha=np.array([alpha_cruise]),
        Re=re_mac,
        model_size="large",
    )
    cd_wing_prof = float(np.asarray(nf_cruise["CD"], dtype=float)[0])
    conf_cruise = float(np.asarray(nf_cruise["analysis_confidence"], dtype=float)[0])
    if not (math.isfinite(cd_wing_prof) and cd_wing_prof > 0.0):
        raise RuntimeError("NeuralFoil returned invalid cruise CD")

    def cf_turb(re_val: float) -> float:
        return 0.455 / (math.log10(max(re_val, 1.0e4)) ** 2.58)

    re_fuse = rho * v * FUSE_LENGTH_M / mu
    re_boom = rho * v * BOOM_LENGTH_M / mu
    re_tail = rho * v * (0.65 * mac) / mu

    fr_fuse = FUSE_LENGTH_M / FUSE_DIAMETER_M
    ff_fuse = 1.0 + (60.0 / (fr_fuse**3)) + (fr_fuse / 400.0)

    cd_fuse = cf_turb(re_fuse) * ff_fuse * 1.05 * geom["s_wet_fuse_m2"] / s_wing
    cd_booms = cf_turb(re_boom) * 1.15 * 1.05 * geom["s_wet_booms_m2"] / s_wing
    cd_tail = cf_turb(re_tail) * 1.22 * 1.08 * (2.04 * (geom["s_h_m2"] + geom["s_v_m2"])) / s_wing

    d_lift = params["lift_rotor_diameter_m"]
    fe_rotors_pods = 6.0 * (0.0042 * d_lift + 0.00105)
    fe_skids_gimbal = 0.00472
    cd_rotors_pods = fe_rotors_pods / s_wing
    cd_skids_gimbal = fe_skids_gimbal / s_wing
    cd_parasitic = cd_fuse + cd_booms + cd_tail + cd_rotors_pods + cd_skids_gimbal

    e_oswald = 0.84 / (1.0 + 0.20 * ((lam - 0.45) ** 2))
    twist_diff = abs((params["tip_twist_deg"] - params["root_twist_deg"]) - (-2.0))
    e_oswald *= max(0.90, 1.0 - 0.01 * twist_diff)

    cd_induced = (cl_req * cl_req) / (math.pi * e_oswald * ar)
    cd_total = cd_wing_prof + cd_parasitic + cd_induced
    l_over_d = cl_req / cd_total

    eta_esc = selected_skus["esc"].specs["drive_elec_eff"]
    eta_cruise_m = selected_skus["cruise_motor"].specs["cruise_motor_eff"]
    eta_cruise_total = eta_esc * eta_cruise_m * CRUISE_PROP_AERODYN_EFF

    drag_n = weight_n / l_over_d
    p_shaft_cruise_w = drag_n * v
    p_elec_cruise_w = p_shaft_cruise_w / eta_cruise_total

    # Lift-curve slope for Pratt discrete gust
    a_wing_rad = (2.0 * math.pi * ar) / (2.0 + math.sqrt(4.0 + ar * ar))

    return {
        "re_mac": re_mac,
        "mach": v / 340.3,
        "cl_cruise": cl_req,
        "cl_max": cl_max,
        "alpha_cruise_deg": alpha_cruise,
        "a_wing_per_rad": a_wing_rad,
        "nf_confidence": min(conf_cruise, float(np.min(conf_grid))),
        "cd_wing_prof": cd_wing_prof,
        "cd_fuse": cd_fuse,
        "cd_booms": cd_booms,
        "cd_tail": cd_tail,
        "cd_rotors_pods": cd_rotors_pods,
        "cd_skids_gimbal": cd_skids_gimbal,
        "cd_parasitic": cd_parasitic,
        "cd_induced": cd_induced,
        "cd_total": cd_total,
        "e_oswald": e_oswald,
        "l_over_d": l_over_d,
        "drag_n": drag_n,
        "p_shaft_cruise_w": p_shaft_cruise_w,
        "p_elec_cruise_w": p_elec_cruise_w,
    }


def compute_pratt_gust_load_factor(
    mass_kg: float,
    geom: dict[str, float],
    a_wing_per_rad: float,
    scenario: dict[str, float],
) -> float:
    """Compute FAR/Pratt discrete gust load factor n_gust = 1 + K_g * rho * V * U_de * a / (2 W/S)."""
    rho = scenario["rho_air"]
    v = scenario.get("cruise_speed_mps", 24.0)
    u_de = scenario.get("gust_u_de_mps", 10.0)
    wing_loading_pa = (mass_kg * G_ACCEL) / geom["wing_area_m2"]
    mu_g = (2.0 * wing_loading_pa) / (rho * geom["mac_m"] * a_wing_per_rad * G_ACCEL)
    k_g = (0.88 * mu_g) / (5.3 + mu_g)
    delta_n = (k_g * rho * v * u_de * a_wing_per_rad) / (2.0 * wing_loading_pa)
    return float(1.0 + delta_n)


def compute_root_bending_stress(
    params: dict[str, float],
    geom: dict[str, float],
    mass_kg: float,
    m_wing_kg: float,
    a_wing_per_rad: float,
    scenario: dict[str, float],
) -> dict[str, float]:
    """Euler-Bernoulli wing-root box-spar stress under separated +3.0g maneuver & Pratt gust (FS=1.5)."""
    fs = 1.5
    n_man = 3.0
    n_gust = compute_pratt_gust_load_factor(mass_kg, geom, a_wing_per_rad, scenario)

    b = geom["wing_span_m"]
    c_r = geom["c_root_m"]
    lam = params["taper_ratio"]
    tc = geom["airfoil_tc"]
    t_cap = params["spar_cap_thickness_m"]

    y_cp = (b / 6.0) * (1.0 + 2.0 * lam) / (1.0 + lam)
    net_half_lift_n = 0.5 * (mass_kg - m_wing_kg) * G_ACCEL

    h_spar = 0.85 * tc * c_r
    w_cap = 0.10 * c_r
    d_offset = max(0.5 * (h_spar - t_cap), 1.0e-4)
    i_root = 2.0 * ((w_cap * (t_cap**3)) / 12.0 + w_cap * t_cap * (d_offset**2))

    def stress_at_n(n_limit: float) -> tuple[float, float]:
        n_ult = n_limit * fs
        m_root_nm = n_ult * net_half_lift_n * y_cp
        sigma_mpa = (m_root_nm * (0.5 * h_spar) / i_root) / 1.0e6
        return m_root_nm, sigma_mpa

    m_man_nm, sigma_man_mpa = stress_at_n(n_man)
    m_gust_nm, sigma_gust_mpa = stress_at_n(n_gust)

    return {
        "y_cp_m": y_cp,
        "h_spar_m": h_spar,
        "w_cap_m": w_cap,
        "i_root_m4": i_root,
        "n_gust_limit": n_gust,
        "m_root_man_nm": m_man_nm,
        "m_root_gust_nm": m_gust_nm,
        "sigma_maneuver_3g_fs15_mpa": sigma_man_mpa,
        "sigma_gust_pratt_fs15_mpa": sigma_gust_mpa,
        "sigma_root_mpa": max(sigma_man_mpa, sigma_gust_mpa),
    }


def _solve_5rotor_min_norm_nonneg(
    a_5: np.ndarray,
    b_eq: np.ndarray,
    tol: float = 1e-10,
) -> tuple[bool, np.ndarray | None]:
    """Exact closed-form solver for min ||T||_2^2 s.t. A_5 T = b_eq, T >= 0 (A_5 is 4x5).

    Because A_5 has 4 rows and 5 columns, when rank(A_5) == 4 the feasible affine
    subspace is a 1-D line T(s) = T_min + s * n, where T_min = A_5^+ b is orthogonal
    to the unit nullspace vector n. Hence ||T(s)||_2^2 = ||T_min||_2^2 + s^2 is
    uniquely minimized by s* = clip(0, s_lo, s_hi) over the nonnegativity interval.
    """
    u_mat, s_vals, vt_mat = np.linalg.svd(a_5, full_matrices=True)
    if len(s_vals) < 4 or s_vals[3] < 1e-10 * max(s_vals[0], 1.0):
        return False, None

    t_min = vt_mat[:4, :].T @ ((u_mat.T @ b_eq) / s_vals)
    null_vec = vt_mat[4, :]

    s_lo = -float("inf")
    s_hi = float("inf")
    for t_j, n_j in zip(t_min, null_vec):
        if n_j > 1e-12:
            s_lo = max(s_lo, -t_j / n_j)
        elif n_j < -1e-12:
            s_hi = min(s_hi, -t_j / n_j)
        elif t_j < -tol:
            return True, None  # Proven infeasible

    if s_lo > s_hi + tol:
        return True, None  # Proven infeasible

    s_opt = float(np.clip(0.0, s_lo, s_hi))
    sol5 = np.clip(t_min + s_opt * null_vec, 0.0, None)
    return True, sol5


def solve_oei_trim_qp(
    params: dict[str, float],
    geom: dict[str, float],
    mass_kg: float,
    selected_skus: dict[str, SKUItem],
    scenario: dict[str, float],
    backend: str = "nullspace",
) -> dict[str, Any]:
    """Solve 6-rotor twin-boom static OEI trim convex QP across all 6 failures x 3 CG shifts.

    Uses exact closed-form 1-D nullspace projection by default (~0.2 ms for all 18 cases)
    with automatic fallback to CVXPY on rank-deficient cases (or explicit `backend="cvxpy"`
    for independent cross-verification).
    """
    weight_n = mass_kg * G_ACCEL
    b = geom["wing_span_m"]
    y_b = BOOM_Y_FRAC * (0.5 * b)
    x_f = 0.5 * BOOM_LENGTH_M
    x_r = 0.5 * BOOM_LENGTH_M

    coords = [
        (x_f, -y_b),
        (x_f, y_b),
        (0.0, -y_b),
        (0.0, y_b),
        (-x_r, -y_b),
        (-x_r, y_b),
    ]
    spins = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    cants_rad = [math.radians(d) for d in (2.0, -2.0, -4.0, 4.0, 2.0, -2.0)]

    d_lift = params["lift_rotor_diameter_m"]
    k_tau = 0.030 * d_lift

    mac = geom["mac_m"]
    cg_frac = scenario["cg_shift_mac_frac"]
    dcg_shifts = [-cg_frac * mac, 0.0, cg_frac * mac]

    worst_peak_n = 0.0
    worst_case_label = ""
    max_eq_residual = 0.0
    worst_thrusts = [0.0] * 6
    worst_power_elec_w = 0.0
    worst_peak_motor_elec_w = 0.0
    all_feasible = True

    rho = scenario["rho_air"]
    a_disk = math.pi * (d_lift**2) / 4.0
    fm_hover = 0.72
    lift_sku = selected_skus["lift_motor"]
    esc_sku = selected_skus["esc"]
    eta_lift_motor = lift_sku.specs["lift_motor_eff"]
    eta_lift_esc = 1.0 if lift_sku.has_integrated_esc else esc_sku.specs["drive_elec_eff"]
    eta_lift_em = eta_lift_motor * eta_lift_esc
    power_coeff = 1.0 / (fm_hover * math.sqrt(2.0 * rho * a_disk) * eta_lift_em)

    row_fz = np.array([math.cos(cants_rad[i]) for i in range(6)], dtype=float)
    row_mx = np.array([coords[i][1] * math.cos(cants_rad[i]) for i in range(6)], dtype=float)
    row_mz = np.array(
        [
            spins[i] * k_tau * math.cos(cants_rad[i]) - coords[i][1] * math.sin(cants_rad[i])
            for i in range(6)
        ],
        dtype=float,
    )
    b_eq = np.array([weight_n, 0.0, 0.0, 0.0], dtype=float)
    t_vec_cvx = cp.Variable(6, nonneg=True) if backend == "cvxpy" else None

    for fail_idx in range(6):
        active_idx = [i for i in range(6) if i != fail_idx]
        for dcg in dcg_shifts:
            row_my = np.array(
                [(coords[i][0] - dcg) * math.cos(cants_rad[i]) for i in range(6)],
                dtype=float,
            )
            a_eq = np.vstack([row_fz, row_mx, row_my, row_mz])

            sol: np.ndarray | None = None
            resolved_exact = False
            if backend == "nullspace":
                resolved_exact, sol5 = _solve_5rotor_min_norm_nonneg(a_eq[:, active_idx], b_eq)
                if resolved_exact:
                    if sol5 is None:
                        all_feasible = False
                        continue
                    sol = np.zeros(6, dtype=float)
                    sol[active_idx] = sol5

            if not resolved_exact:
                t_var = t_vec_cvx if t_vec_cvx is not None else cp.Variable(6, nonneg=True)
                prob = cp.Problem(
                    cp.Minimize(cp.sum_squares(t_var)),
                    [a_eq @ t_var == b_eq, t_var[fail_idx] == 0.0],
                )
                solved = False
                for solver_name, kwargs in [
                    (cp.CLARABEL, {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}),
                    (cp.OSQP, {"eps_abs": 1e-9, "eps_rel": 1e-9}),
                    (cp.SCS, {"eps": 1e-9}),
                ]:
                    try:
                        prob.solve(solver=solver_name, **kwargs)
                        if prob.status in (
                            cp.OPTIMAL,
                            cp.OPTIMAL_INACCURATE,
                            cp.INFEASIBLE,
                            cp.INFEASIBLE_INACCURATE,
                        ):
                            solved = True
                            break
                    except Exception:
                        continue

                if not solved:
                    raise RuntimeError(f"CVXPY solver failed on OEI fail={fail_idx}, dcg={dcg:.4f}")

                if prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE) or t_var.value is None:
                    all_feasible = False
                    continue

                sol = np.clip(np.asarray(t_var.value, dtype=float), 0.0, None)
                sol[fail_idx] = 0.0

            assert sol is not None
            res = float(np.max(np.abs(a_eq @ sol - b_eq)))
            max_eq_residual = max(max_eq_residual, res)

            peak_t = float(np.max(sol))
            motor_powers_w = np.power(sol, 1.5) * power_coeff
            p_elec_w = float(np.sum(motor_powers_w))
            peak_motor_elec_w = float(np.max(motor_powers_w))

            if peak_t > worst_peak_n:
                worst_peak_n = peak_t
                worst_thrusts = [float(v) for v in sol]
                worst_case_label = f"rotor_{fail_idx}_dcg_{dcg:+.4f}m"
            worst_power_elec_w = max(worst_power_elec_w, p_elec_w)
            worst_peak_motor_elec_w = max(worst_peak_motor_elec_w, peak_motor_elec_w)

    if not all_feasible:
        worst_peak_n = max(worst_peak_n, 1e6)
        worst_power_elec_w = max(worst_power_elec_w, 1e6)
        worst_peak_motor_elec_w = max(worst_peak_motor_elec_w, 1e6)

    return {
        "oei_all_18_cases_feasible": all_feasible,
        "worst_peak_thrust_n": worst_peak_n,
        "worst_case_label": worst_case_label,
        "worst_thrusts_n": worst_thrusts,
        "max_equilibrium_residual": max_eq_residual,
        "p_oei_elec_w": worst_power_elec_w,
        "worst_peak_motor_elec_w": worst_peak_motor_elec_w,
        "lift_motor_max_thrust_n": lift_sku.specs["lift_motor_max_thrust_n"],
    }


def compute_battery_electrothermal(
    params: dict[str, float],
    mass_info: dict[str, float],
    p_cruise_elec_w: float,
    p_oei_elec_w: float,
    selected_skus: dict[str, SKUItem],
    scenario: dict[str, float],
) -> dict[str, Any]:
    """Compute Arrhenius DCIR, quadratic voltage sag, cruise pre-heat, and 90-s OEI ODE."""
    n_s = params["batt_n_series"]
    n_p = params["batt_n_parallel"]
    cell_sku = selected_skus["cell"]
    r_cell_25c = cell_sku.specs["cell_dcir_25c_mohm"] / 1000.0
    cap_ah = cell_sku.specs["cell_capacity_ah"]
    max_c_rate = cell_sku.specs["cell_max_c_rate"]
    v_cell_nom = cell_sku.specs["cell_v_nom"]
    v_cell_min = cell_sku.specs["cell_v_min"]
    m_cells = mass_info["m_cells_kg"]

    t_amb_c = scenario["t_amb_c"]
    h_conv = scenario["h_conv_w_m2k"]
    ea = scenario["ea_j_mol"]
    oei_dur_s = scenario["oei_duration_s"]
    usable_frac = scenario["usable_batt_frac"]
    v_cruise = scenario.get("cruise_speed_mps", 24.0)

    v_oc_pack = n_s * v_cell_nom
    v_cutoff_pack = n_s * v_cell_min
    i_pack_rated_max = n_p * cap_ah * max_c_rate

    n_cells = n_s * n_p
    v_pack_m3 = n_cells * (math.pi * (0.0105**2) * 0.070) * 1.28
    a_pack_m2 = 6.325 * (v_pack_m3 ** (2.0 / 3.0))

    def r_cells_pack(temp_c: float) -> float:
        return (n_s / n_p) * r_cell_25c * arrhenius_resistance_scale(temp_c, ea_j_mol=ea)

    def r_total_pack(temp_c: float) -> float:
        return r_cells_pack(temp_c) + R_BUS_OHM

    arrh_factor_amb = arrhenius_resistance_scale(t_amb_c, ea_j_mol=ea)
    arrh_factor_20k = arrhenius_resistance_scale(t_amb_c, ea_j_mol=20000.0)
    arrh_factor_30k = arrhenius_resistance_scale(t_amb_c, ea_j_mol=30000.0)

    def cruise_heat_balance(temp_c: float) -> float:
        sol = solve_quadratic_circuit(v_oc_pack, r_total_pack(temp_c), p_cruise_elec_w)
        q_joule = (sol.current_a**2) * r_cells_pack(temp_c)
        q_conv = h_conv * a_pack_m2 * (temp_c - t_amb_c)
        return q_joule - q_conv

    cruise_circuit_amb = solve_quadratic_circuit(v_oc_pack, r_total_pack(t_amb_c), p_cruise_elec_w)
    if scenario.get("cold_soak_takeoff", 0.0) >= 0.5 or cruise_circuit_amb.voltage_collapse:
        t_cruise_eq_c = t_amb_c
        cruise_circuit_eq = cruise_circuit_amb
    else:
        f_low = cruise_heat_balance(t_amb_c)
        t_hi = t_amb_c + 120.0
        f_hi = cruise_heat_balance(t_hi)
        if f_low >= 0.0 and f_hi <= 0.0:
            t_cruise_eq_c = float(brentq(cruise_heat_balance, t_amb_c, t_hi, xtol=1e-6))
        else:
            t_cruise_eq_c = t_amb_c
        cruise_circuit_eq = solve_quadratic_circuit(
            v_oc_pack, r_total_pack(t_cruise_eq_c), p_cruise_elec_w
        )

    e_pack_j = n_s * n_p * cap_ah * v_cell_nom * 3600.0
    e_usable_j = usable_frac * e_pack_j
    if cruise_circuit_eq.voltage_collapse or cruise_circuit_eq.current_a <= 0.0:
        range_km = 0.0
    else:
        p_chem_cruise_w = v_oc_pack * cruise_circuit_eq.current_a
        t_cruise_s = e_usable_j / p_chem_cruise_w
        range_km = v_cruise * t_cruise_s / 1000.0

    oei_circuit_t0 = solve_quadratic_circuit(v_oc_pack, r_total_pack(t_cruise_eq_c), p_oei_elec_w)

    if oei_circuit_t0.voltage_collapse:
        return {
            "v_oc_pack_v": v_oc_pack,
            "v_cutoff_pack_v": v_cutoff_pack,
            "i_pack_rated_max_a": i_pack_rated_max,
            "a_pack_m2": a_pack_m2,
            "arrhenius_factor_at_tamb": arrh_factor_amb,
            "arrhenius_band_20_30_kj": [arrh_factor_20k, arrh_factor_30k],
            "r_pack_tamb_ohm": r_total_pack(t_amb_c),
            "r_pack_tcruise_ohm": r_total_pack(t_cruise_eq_c),
            "t_cruise_eq_c": t_cruise_eq_c,
            "cruise_current_a": cruise_circuit_eq.current_a,
            "cruise_v_term_v": cruise_circuit_eq.terminal_voltage_v,
            "cruise_voltage_collapse": cruise_circuit_eq.voltage_collapse,
            "range_km": range_km,
            "oei_voltage_collapse": True,
            "oei_p_max_t0_w": oei_circuit_t0.p_max_w,
            "oei_discriminant_t0": oei_circuit_t0.discriminant,
            "oei_current_max_a": oei_circuit_t0.current_a,
            "oei_v_term_min_v": oei_circuit_t0.terminal_voltage_v,
            "oei_circuit_residual_w": oei_circuit_t0.residual_w,
            "t_oei_peak_c": t_cruise_eq_c,
            "delta_t_oei_c": 0.0,
            "q_gen_j": 0.0,
            "q_stored_j": 0.0,
            "q_conv_j": 0.0,
            "trapped_heat_frac": 1.0,
        }

    thermal_cap = m_cells * CELL_CP_J_KG_K
    collapse_mid_burst = False

    def oei_ode(_t: float, y: np.ndarray) -> np.ndarray:
        nonlocal collapse_mid_burst
        temp_c = float(y[0])
        r_tot = r_total_pack(temp_c)
        r_cells = r_cells_pack(temp_c)
        circ = solve_quadratic_circuit(v_oc_pack, r_tot, p_oei_elec_w)
        if circ.voltage_collapse:
            collapse_mid_burst = True
        q_gen_w = (circ.current_a**2) * r_cells
        q_conv_w = h_conv * a_pack_m2 * (temp_c - t_amb_c)
        dtemp_dt = (q_gen_w - q_conv_w) / thermal_cap
        return np.array([dtemp_dt, q_gen_w, q_conv_w], dtype=float)

    sol_ivp = solve_ivp(
        oei_ode,
        t_span=(0.0, oei_dur_s),
        y0=np.array([t_cruise_eq_c, 0.0, 0.0], dtype=float),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol_ivp.success:
        raise RuntimeError(f"Thermal ODE solve_ivp failed: {sol_ivp.message}")

    temps = sol_ivp.y[0, :]
    t_oei_peak_c = float(np.max(temps))
    t_oei_end_c = float(temps[-1])
    q_gen_j = float(sol_ivp.y[1, -1])
    q_conv_j = float(sol_ivp.y[2, -1])
    q_stored_j = thermal_cap * (t_oei_end_c - t_cruise_eq_c)
    trapped_frac = q_stored_j / q_gen_j if q_gen_j > 0.0 else 1.0

    oei_currents, oei_vterms, oei_residuals = [], [], []
    for temp_c in temps:
        c_node = solve_quadratic_circuit(v_oc_pack, r_total_pack(float(temp_c)), p_oei_elec_w)
        oei_currents.append(c_node.current_a)
        oei_vterms.append(c_node.terminal_voltage_v)
        oei_residuals.append(c_node.residual_w)

    return {
        "v_oc_pack_v": v_oc_pack,
        "v_cutoff_pack_v": v_cutoff_pack,
        "i_pack_rated_max_a": i_pack_rated_max,
        "a_pack_m2": a_pack_m2,
        "arrhenius_factor_at_tamb": arrh_factor_amb,
        "arrhenius_band_20_30_kj": [arrh_factor_20k, arrh_factor_30k],
        "r_pack_tamb_ohm": r_total_pack(t_amb_c),
        "r_pack_tcruise_ohm": r_total_pack(t_cruise_eq_c),
        "t_cruise_eq_c": t_cruise_eq_c,
        "cruise_current_a": cruise_circuit_eq.current_a,
        "cruise_v_term_v": cruise_circuit_eq.terminal_voltage_v,
        "cruise_voltage_collapse": cruise_circuit_eq.voltage_collapse,
        "range_km": range_km,
        "oei_voltage_collapse": oei_circuit_t0.voltage_collapse or collapse_mid_burst,
        "oei_p_max_t0_w": oei_circuit_t0.p_max_w,
        "oei_discriminant_t0": oei_circuit_t0.discriminant,
        "oei_current_max_a": float(max(oei_currents)),
        "oei_v_term_min_v": float(min(oei_vterms)),
        "oei_circuit_residual_w": float(max(oei_residuals)),
        "t_oei_peak_c": t_oei_peak_c,
        "delta_t_oei_c": t_oei_end_c - t_cruise_eq_c,
        "q_gen_j": q_gen_j,
        "q_stored_j": q_stored_j,
        "q_conv_j": q_conv_j,
        "trapped_heat_frac": trapped_frac,
    }


def compute_cold_oei_margin(
    params: dict[str, float],
    mass_info: dict[str, float],
    p_cruise_elec_w: float,
    oei_trim: dict[str, Any],
    selected_skus: dict[str, SKUItem],
    scenario: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    """Compute normalized 0 C cold-soak OEI margin objective across thrust, power, V, I, T."""
    cold_scen = dict(scenario)
    cold_scen["t_amb_c"] = 0.0
    cold_scen["cold_soak_takeoff"] = 1.0
    cold_el = compute_battery_electrothermal(
        params=params,
        mass_info=mass_info,
        p_cruise_elec_w=p_cruise_elec_w,
        p_oei_elec_w=oei_trim["p_oei_elec_w"],
        selected_skus=selected_skus,
        scenario=cold_scen,
    )
    t_max_n = oei_trim["lift_motor_max_thrust_n"]
    m_thrust = (t_max_n - oei_trim["worst_peak_thrust_n"]) / max(t_max_n, 1.0)
    m_power = (cold_el["oei_p_max_t0_w"] - oei_trim["p_oei_elec_w"]) / max(
        oei_trim["p_oei_elec_w"], 1.0
    )
    m_vcut = (cold_el["oei_v_term_min_v"] - cold_el["v_cutoff_pack_v"]) / max(
        cold_el["v_cutoff_pack_v"], 1.0
    )
    m_crate = (cold_el["i_pack_rated_max_a"] - cold_el["oei_current_max_a"]) / max(
        cold_el["i_pack_rated_max_a"], 1.0
    )
    m_temp = (cold_scen["t_max_batt_c"] - cold_el["t_oei_peak_c"]) / 60.0
    return float(min(m_thrust, m_power, m_vcut, m_crate, m_temp)), cold_el


class TierLUAVOracle:
    """Honest Tier-L DisciplineOracle for the 25 kg Lift+Cruise eVTOL UAV."""

    def __init__(
        self,
        schema: ProblemSchema,
        name: str = "tier_l_uav",
        capture_provenance: bool = False,
    ) -> None:
        self._schema = schema
        self._name = name
        self._capture_provenance = capture_provenance
        self._sub_oracles: tuple[Any, ...] = ()

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> OracleTier:
        return OracleTier.L

    @property
    def schema_id(self) -> str:
        return self._schema.schema_id

    @property
    def approx_cost_s(self) -> float:
        return 0.015

    @property
    def provides_outputs(self) -> tuple[str, ...]:
        return ("verified_range_km", "domestic_bom_cost_usd", "cold_oei_margin")

    @property
    def sub_oracles(self) -> tuple[Any, ...]:
        return self._sub_oracles

    def evaluate(
        self,
        design: DesignIR,
        scenario: dict[str, float] | None = None,
        ledger: BudgetLedger | None = None,
    ) -> EvalResult:
        """Evaluate a snapped DesignIR against all Tier-L coupled disciplines."""
        t_start = time.perf_counter()
        scen = dict(self._schema.scenario_defaults)
        if scenario:
            scen.update(scenario)

        provenance = HostProvenance.capture() if self._capture_provenance else None

        try:
            params = design.params
            selected_sku_objs = {
                cat: self._schema.catalog.get(sku_id)
                for cat, sku_id in design.selected_skus.items()
            }
            compat_ok, compat_errors = self._schema.compatibility.check(
                selected_skus=selected_sku_objs,
                params=params,
            )

            airfoil_cst = design.airfoil_cst or reconstruct_kulfan_from_pca(
                np.array([params[f"airfoil_pca_{k}"] for k in range(4)], dtype=float)
            )
            geom = compute_geometry(params, airfoil_cst=airfoil_cst)
            mass_info = compute_mass_and_bom(params, geom, selected_sku_objs)
            mass_kg = mass_info["m_total_kg"]

            aero = compute_aero_polar(
                params=params,
                geom=geom,
                mass_kg=mass_kg,
                airfoil_cst=airfoil_cst,
                selected_skus=selected_sku_objs,
                scenario=scen,
            )
            struct = compute_root_bending_stress(
                params=params,
                geom=geom,
                mass_kg=mass_kg,
                m_wing_kg=mass_info["m_wing_kg"],
                a_wing_per_rad=aero["a_wing_per_rad"],
                scenario=scen,
            )
            oei_trim = solve_oei_trim_qp(params, geom, mass_kg, selected_sku_objs, scen)
            electro = compute_battery_electrothermal(
                params=params,
                mass_info=mass_info,
                p_cruise_elec_w=aero["p_elec_cruise_w"],
                p_oei_elec_w=oei_trim["p_oei_elec_w"],
                selected_skus=selected_sku_objs,
                scenario=scen,
            )
            cold_oei_margin, cold_el = compute_cold_oei_margin(
                params=params,
                mass_info=mass_info,
                p_cruise_elec_w=aero["p_elec_cruise_w"],
                oei_trim=oei_trim,
                selected_skus=selected_sku_objs,
                scenario=scen,
            )

            objectives = {
                "verified_range_km": float(electro["range_km"]),
                "domestic_bom_cost_usd": float(mass_info["domestic_bom_cost_usd"]),
                "cold_oei_margin": float(cold_oei_margin),
            }

            esc_sku = selected_sku_objs["esc"]
            lift_sku = selected_sku_objs["lift_motor"]
            cruise_sku = selected_sku_objs["cruise_motor"]
            v_cruise_term = max(float(electro["cruise_v_term_v"]), 1.0)
            v_oei_term = max(float(electro["oei_v_term_min_v"]), 1.0)
            i_cruise_esc_a = float(aero["p_elec_cruise_w"]) / v_cruise_term
            i_lift_esc_peak_a = (
                0.0
                if lift_sku.has_integrated_esc
                else float(oei_trim["worst_peak_motor_elec_w"]) / v_oei_term
            )
            peak_single_esc_current_a = max(i_cruise_esc_a, i_lift_esc_peak_a)
            esc_i_max_a = float(esc_sku.specs["i_max_a"])
            cruise_motor_max_power_w = float(cruise_sku.specs["max_power_w"])

            constraints = {
                "mass_excess_kg": float(mass_kg - scen["mtow_limit_kg"]),
                "stall_cl_margin": float(aero["cl_cruise"] - 0.85 * aero["cl_max"]),
                "maneuver_stress_margin_mpa": float(
                    struct["sigma_maneuver_3g_fs15_mpa"] - scen["sigma_allow_mpa"]
                ),
                "gust_stress_margin_mpa": float(
                    struct["sigma_gust_pratt_fs15_mpa"] - scen["sigma_allow_mpa"]
                ),
                "oei_thrust_margin_n": float(
                    oei_trim["worst_peak_thrust_n"] - oei_trim["lift_motor_max_thrust_n"]
                ),
                "oei_voltage_collapse_margin_w": float(
                    oei_trim["p_oei_elec_w"] - electro["oei_p_max_t0_w"]
                ),
                "oei_v_cutoff_margin_v": float(
                    electro["v_cutoff_pack_v"] - electro["oei_v_term_min_v"]
                ),
                "oei_c_rate_margin_a": float(
                    electro["oei_current_max_a"] - electro["i_pack_rated_max_a"]
                ),
                "oei_temp_margin_c": float(electro["t_oei_peak_c"] - scen["t_max_batt_c"]),
                "esc_current_margin_a": float(peak_single_esc_current_a - esc_i_max_a),
                "cruise_motor_power_margin_w": float(
                    aero["p_elec_cruise_w"] - cruise_motor_max_power_w
                ),
                "compat_violation_count": float(len(compat_errors)),
            }

            all_nums = [*objectives.values(), *constraints.values()]
            if not all(math.isfinite(val) for val in all_nums):
                raise ValueError("Non-finite objective or constraint produced during evaluation")

            m_safe = self._schema.compute_m_safe(constraints)
            tol_map = {c.name: c.tolerance for c in self._schema.constraints}
            violated = any(val > tol_map.get(name, 1e-6) for name, val in constraints.items())
            status = SolverStatus.RESOLVED_VIOLATION if violated else SolverStatus.RESOLVED_PASS

            result = EvalResult(
                design_id=design.design_id,
                oracle_name=self._name,
                tier=OracleTier.L,
                status=status,
                objectives=objectives,
                constraints=constraints,
                m_safe=m_safe,
                diagnostics={
                    "scenario": scen,
                    "selected_skus": design.selected_skus,
                    "compat_ok": compat_ok,
                    "compat_errors": compat_errors,
                    "airfoil_cst": airfoil_cst,
                    "geometry": geom,
                    "mass_breakdown": mass_info,
                    "aero": aero,
                    "structures": struct,
                    "oei_trim": oei_trim,
                    "electrothermal": electro,
                    "cold_oei_0c": cold_el,
                },
                wall_time_s=time.perf_counter() - t_start,
                provenance=provenance,
            )
        except Exception as exc:
            result = EvalResult(
                design_id=design.design_id,
                oracle_name=self._name,
                tier=OracleTier.L,
                status=SolverStatus.UNRESOLVED_NUMERICAL,
                wall_time_s=time.perf_counter() - t_start,
                failure_reason=f"{type(exc).__name__}: {exc}",
                provenance=provenance,
            )

        if ledger is not None:
            ledger.debit(result)
        return result
