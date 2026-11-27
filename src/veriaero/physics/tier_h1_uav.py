"""Honest Tier-H1 Coupled Multi-Physics Oracle for VeriAero-CEGIS v3.2.

Locked verbatim to Architecture Reference Section 6.5:
1. Coupled 3D `OpenAeroStruct 2.12.1` VLM + CFRP `wingbox` FEM (`with_viscous=False`,
   `with_wave=False`, cached `Problem.setup()`, separated `+3.0g` maneuver vs.
   dynamic Pratt discrete gust, `FS = 1.5`, `sigma_allow = 350 MPa`).
2. Spanwise strip integration of patched `XFoil 6.99` across 8 Gauss-Legendre stations
   (`N = 200`, `N_crit = 9.0`, free + tripped `x_tr = 0.05` roughness check).
3. Tri-state guarantee: if `bin/xfoil_patched` is absent, crashes (`SIGFPE`), times out,
   or fails to converge on any Gauss-Legendre station, returns `SolverStatus.UNRESOLVED_NUMERICAL`
   and NEVER silently falls back to NeuralFoil.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import time
import warnings
from typing import Any

import numpy as np
import openmdao.api as om
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.meshing.mesh_generator import generate_mesh

from veriaero.contracts.schema import DesignIR, ProblemSchema
from veriaero.contracts.types import (
    BudgetLedger,
    EvalResult,
    HostProvenance,
    OracleTier,
    SolverStatus,
)
from veriaero.physics.airfoil_cst import (
    kulfan_dict_to_airfoil,
    kulfan_dict_to_coordinates,
    reconstruct_kulfan_from_pca,
)
from veriaero.physics.tier_l_uav import (
    BOOM_LENGTH_M,
    CRUISE_PROP_AERODYN_EFF,
    FUSE_DIAMETER_M,
    FUSE_LENGTH_M,
    G_ACCEL,
    compute_battery_electrothermal,
    compute_cold_oei_margin,
    compute_geometry,
    compute_mass_and_bom,
    compute_pratt_gust_load_factor,
    solve_oei_trim_qp,
)
from veriaero.solvers.xfoil_runner import XFoilRunner

# Bounded cache of setup() OpenMDAO Problems keyed ONLY by mesh topology, allowable stress,
# and quantized normalized wingbox cross-section shape (all planform/twist/thickness/t_c
# updates happen in-place via prob.set_val() with ZERO Problem.setup() rebuilds).
_OAS_PROBLEM_CACHE: dict[tuple[Any, ...], om.Problem] = {}
OAS_SAFETY_FACTOR = 1.5


def _build_or_get_oas_wingbox_problem(
    params: dict[str, float],
    geom: dict[str, float],
    airfoil_cst: dict[str, Any],
    sigma_allow_mpa: float,
    num_y: int = 29,
    num_x: int = 3,
) -> tuple[om.Problem, np.ndarray]:
    """Build or retrieve a truly reusable cached setup() OpenMDAO Problem for the CFRP wingbox."""
    b = geom["wing_span_m"]
    c_r = geom["c_root_m"]
    lam = params["taper_ratio"]
    tc = float(airfoil_cst["max_thickness_over_c"])

    k_af = kulfan_dict_to_airfoil(airfoil_cst)
    x_box = np.linspace(0.15, 0.55, 21)
    y_upper = np.array(
        [float(k_af.upper_coordinates(np.array([xv]))[0, 1]) for xv in x_box],
        dtype="complex128",
    )
    y_lower = np.array(
        [float(k_af.lower_coordinates(np.array([xv]))[0, 1]) for xv in x_box],
        dtype="complex128",
    )

    # Tip-to-root ordering matching OAS symmetric half-wing mesh y in [-b/2, 0].
    # Preserve exact spar-cap flange area (0.10*c*t_cap distributed across 0.40*c OAS box)
    # plus 25% axial stiffness contribution from +-45 deg CFRP skins.
    spar_thk_cp = np.array([params["web_thickness_m"], params["web_thickness_m"]], dtype=float)
    skin_eq = 0.25 * params["spar_cap_thickness_m"] + 0.25 * params["skin_thickness_m"]
    skin_thk_cp = np.array([0.85 * skin_eq, skin_eq], dtype=float)
    twist_cp = np.array([params["tip_twist_deg"], params["root_twist_deg"]], dtype=float)

    # Key cache on normalized box contour shape (independent of planform, thickness, twist, or t/c)
    tc_safe = max(tc, 1e-4)
    norm_shape_sig = (
        tuple(round(float(v) / tc_safe, 2) for v in y_upper[::7].real),
        tuple(round(float(v) / tc_safe, 2) for v in y_lower[::7].real),
    )
    cache_key = (num_y, num_x, round(sigma_allow_mpa, 2), norm_shape_sig)

    if cache_key in _OAS_PROBLEM_CACHE:
        prob = _OAS_PROBLEM_CACHE[cache_key]
    else:
        # Unit rectangular base mesh; span, taper, and root chord are updated dynamically via set_val()
        mesh = generate_mesh(
            {
                "num_y": num_y,
                "num_x": num_x,
                "wing_type": "rect",
                "symmetry": True,
                "span": 1.0,
                "root_chord": 1.0,
                "span_cos_spacing": 0.5,
                "chord_cos_spacing": 0.0,
            }
        )

        surface: dict[str, Any] = {
            "name": "wing",
            "symmetry": True,
            "S_ref_type": "projected",
            "fem_model_type": "wingbox",
            "mesh": mesh,
            "span": b,
            "taper": lam,
            "chord_cp": np.array([c_r], dtype=float),
            "sweep": 2.0,
            "twist_cp": twist_cp,
            "data_x_upper": x_box.astype("complex128"),
            "data_x_lower": x_box.astype("complex128"),
            "data_y_upper": y_upper,
            "data_y_lower": y_lower,
            "original_wingbox_airfoil_t_over_c": tc,
            "t_over_c_cp": np.array([tc], dtype=float),
            "spar_thickness_cp": spar_thk_cp,
            "skin_thickness_cp": skin_thk_cp,
            "CL0": 0.0,
            "CD0": 0.0,
            "with_viscous": False,
            "with_wave": False,
            "k_lam": 0.05,
            "c_max_t": 0.30,
            "E": 110.0e9,
            "G": 40.0e9,
            "yield": sigma_allow_mpa * 1.0e6,
            "safety_factor": OAS_SAFETY_FACTOR,
            "strength_factor_for_upper_skin": 1.0,
            "mrho": 1600.0,
            "wing_weight_ratio": 1.2,
            "exact_failure_constraint": False,
            "struct_weight_relief": True,
            "distributed_fuel_weight": False,
        }

        prob = om.Problem(reports=False)
        indep = om.IndepVarComp()
        indep.add_output("v", val=24.0, units="m/s")
        indep.add_output("alpha", val=3.0, units="deg")
        indep.add_output("Mach_number", val=24.0 / 340.3)
        indep.add_output("re", val=5.165e5, units="1/m")
        indep.add_output("rho", val=1.225, units="kg/m**3")
        indep.add_output("CT", val=1.0e-5, units="1/s")
        indep.add_output("R", val=150.0e3, units="m")
        indep.add_output("W0", val=21.0, units="kg")
        indep.add_output("speed_of_sound", val=340.3, units="m/s")
        indep.add_output("load_factor", val=1.0)
        indep.add_output("empty_cg", val=np.zeros(3), units="m")
        indep.add_output("beta", val=0.0, units="deg")
        prob.model.add_subsystem("prob_vars", indep, promotes=["*"])

        prob.model.add_subsystem("wing", AerostructGeometry(surface=surface))
        prob.model.add_subsystem(
            "AS_point_0",
            AerostructPoint(surfaces=[surface]),
            promotes_inputs=[
                "v",
                "alpha",
                "Mach_number",
                "re",
                "rho",
                "CT",
                "R",
                "W0",
                "speed_of_sound",
                "load_factor",
                "empty_cg",
                "beta",
            ],
        )
        for src, dst in [
            ("wing.local_stiff_transformed", "AS_point_0.coupled.wing.local_stiff_transformed"),
            ("wing.nodes", "AS_point_0.coupled.wing.nodes"),
            ("wing.nodes", "AS_point_0.wing_perf.nodes"),
            ("wing.mesh", "AS_point_0.coupled.wing.mesh"),
            ("wing.element_mass", "AS_point_0.coupled.wing.element_mass"),
            ("wing.Qz", "AS_point_0.wing_perf.Qz"),
            ("wing.J", "AS_point_0.wing_perf.J"),
            ("wing.A_enc", "AS_point_0.wing_perf.A_enc"),
            ("wing.htop", "AS_point_0.wing_perf.htop"),
            ("wing.hbottom", "AS_point_0.wing_perf.hbottom"),
            ("wing.hfront", "AS_point_0.wing_perf.hfront"),
            ("wing.hrear", "AS_point_0.wing_perf.hrear"),
            ("wing.spar_thickness", "AS_point_0.wing_perf.spar_thickness"),
            ("wing.t_over_c", "AS_point_0.wing_perf.t_over_c"),
            ("wing.structural_mass", "AS_point_0.total_perf.wing_structural_mass"),
            ("wing.cg_location", "AS_point_0.total_perf.wing_cg_location"),
        ]:
            prob.model.connect(src, dst)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.setup(check=False)
            prob.model.AS_point_0.coupled.nonlinear_solver.options["iprint"] = -1

        if len(_OAS_PROBLEM_CACHE) >= 32:
            _OAS_PROBLEM_CACHE.clear()
        _OAS_PROBLEM_CACHE[cache_key] = prob

    # Apply current design's planform, twist, thickness, and t/c in-place (true cache reuse)
    prob.set_val("wing.geometry.mesh.stretch.span", float(b))
    prob.set_val("wing.geometry.mesh.taper.taper", float(lam))
    prob.set_val("wing.geometry.chord_cp", np.array([c_r], dtype=float))
    prob.set_val("wing.twist_cp", twist_cp)
    prob.set_val("wing.spar_thickness_cp", spar_thk_cp)
    prob.set_val("wing.skin_thickness_cp", skin_thk_cp)
    prob.set_val("wing.geometry.t_over_c_bsp.t_over_c_cp", np.array([tc], dtype=float))
    return prob, x_box


def _eval_cached_oas_point(
    prob: om.Problem,
    v_mps: float,
    alpha_deg: float,
    rho: float,
    re_unit: float,
    w0_kg: float,
    load_factor: float,
) -> dict[str, Any]:
    prob.set_val("v", v_mps, units="m/s")
    prob.set_val("alpha", alpha_deg, units="deg")
    prob.set_val("Mach_number", v_mps / 340.3)
    prob.set_val("re", re_unit, units="1/m")
    prob.set_val("rho", rho, units="kg/m**3")
    prob.set_val("W0", w0_kg, units="kg")
    prob.set_val("load_factor", load_factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob.run_model()

    sec_forces = np.array(
        prob["AS_point_0.coupled.aero_states.wing_sec_forces"], dtype=float, copy=True
    )
    chords = np.array(prob["AS_point_0.coupled.wing.chords"], dtype=float, copy=True)
    widths = np.array(prob["AS_point_0.coupled.wing.widths"], dtype=float, copy=True)
    nodes = np.array(prob["wing.nodes"], dtype=float, copy=True)
    disp = np.array(prob["AS_point_0.coupled.wing.disp"], dtype=float, copy=True)
    vonmises = np.array(prob["AS_point_0.wing_perf.vonmises"], dtype=float, copy=True)

    return {
        "cl": float(np.asarray(prob["AS_point_0.wing_perf.CL"]).item()),
        "cdi": float(np.asarray(prob["AS_point_0.wing_perf.CD"]).item()),
        "s_ref_m2": float(np.asarray(prob["AS_point_0.coupled.wing.S_ref"]).item()),
        "structural_mass_kg": float(np.asarray(prob["wing.structural_mass"]).item()),
        "tip_disp_z_m": float(disp[0, 2]),
        "vonmises_max_mpa": float(np.max(vonmises) / 1.0e6),
        "failure_ks": float(np.asarray(prob["AS_point_0.wing_perf.failure"]).item()),
        "sec_forces": sec_forces,
        "chords": chords,
        "widths": widths,
        "nodes_y": nodes[:, 1],
    }


def run_openaerostruct_wing(
    params: dict[str, float],
    geom: dict[str, float],
    mass_kg: float,
    m_wing_kg: float,
    airfoil_cst: dict[str, Any],
    scenario: dict[str, float],
    num_y: int = 29,
    num_x: int = 3,
) -> dict[str, Any]:
    """Run cached OpenAeroStruct CFRP wingbox (`with_viscous=False`) at cruise, +3.0g, & Pratt gust."""
    rho = scenario["rho_air"]
    mu = scenario["mu_air"]
    v_mps = scenario.get("cruise_speed_mps", 24.0)
    s_wing = geom["wing_area_m2"]
    q_dyn = 0.5 * rho * v_mps * v_mps
    cl_req = (mass_kg * G_ACCEL) / (q_dyn * s_wing)
    re_unit = rho * v_mps / mu
    w0_kg = max(mass_kg - m_wing_kg, 5.0)

    prob, _ = _build_or_get_oas_wingbox_problem(
        params=params,
        geom=geom,
        airfoil_cst=airfoil_cst,
        sigma_allow_mpa=scenario["sigma_allow_mpa"],
        num_y=num_y,
        num_x=num_x,
    )

    pt0 = _eval_cached_oas_point(prob, v_mps, 0.0, rho, re_unit, w0_kg, load_factor=1.0)
    pt4 = _eval_cached_oas_point(prob, v_mps, 4.0, rho, re_unit, w0_kg, load_factor=1.0)
    dcl_dalpha = (pt4["cl"] - pt0["cl"]) / 4.0
    if abs(dcl_dalpha) < 1e-6:
        raise RuntimeError("OpenAeroStruct returned degenerate lift-curve slope")
    alpha_aero_deg = float(np.clip((cl_req - pt0["cl"]) / dcl_dalpha, -8.0, 18.0))
    pt_cruise = _eval_cached_oas_point(
        prob, v_mps, alpha_aero_deg, rho, re_unit, w0_kg, load_factor=1.0
    )

    # Separated +3.0g maneuver vs dynamic Pratt discrete gust (FS = 1.5 inside OAS wingbox)
    n_man = 3.0
    pt_man = _eval_cached_oas_point(
        prob,
        v_mps * math.sqrt(n_man),
        alpha_aero_deg,
        rho,
        re_unit * math.sqrt(n_man),
        w0_kg,
        load_factor=n_man,
    )

    ar = geom["aspect_ratio"]
    a_wing_rad = (2.0 * math.pi * ar) / (2.0 + math.sqrt(4.0 + ar * ar))
    n_gust = compute_pratt_gust_load_factor(mass_kg, geom, a_wing_rad, scenario)
    pt_gust = _eval_cached_oas_point(
        prob,
        v_mps * math.sqrt(n_gust),
        alpha_aero_deg,
        rho,
        re_unit * math.sqrt(n_gust),
        w0_kg,
        load_factor=n_gust,
    )

    panel_chords = 0.5 * (pt_cruise["chords"][:-1] + pt_cruise["chords"][1:])
    sec_z = np.sum(pt_cruise["sec_forces"][:, :, 2], axis=0)
    panel_areas = panel_chords * pt_cruise["widths"]
    cl_local = sec_z / (q_dyn * np.maximum(panel_areas, 1e-9))
    panel_y_abs = np.abs(0.5 * (pt_cruise["nodes_y"][:-1] + pt_cruise["nodes_y"][1:]))

    alpha_zero_lift_deg = -math.degrees(1.8 * float(airfoil_cst.get("max_camber_over_c", 0.029)))
    return {
        "alpha_cruise_deg": alpha_aero_deg + alpha_zero_lift_deg,
        "alpha_aerodynamic_deg": alpha_aero_deg,
        "cl_oas": pt_cruise["cl"],
        "cdi_oas": pt_cruise["cdi"],
        "s_ref_oas_m2": pt_cruise["s_ref_m2"],
        "structural_mass_wingbox_kg": pt_cruise["structural_mass_kg"],
        "tip_disp_cruise_m": pt_cruise["tip_disp_z_m"],
        "tip_disp_maneuver_3g_m": pt_man["tip_disp_z_m"],
        "tip_disp_gust_m": pt_gust["tip_disp_z_m"],
        "vonmises_maneuver_3g_unfactored_mpa": pt_man["vonmises_max_mpa"],
        "vonmises_gust_pratt_unfactored_mpa": pt_gust["vonmises_max_mpa"],
        "vonmises_maneuver_3g_fs15_mpa": OAS_SAFETY_FACTOR * pt_man["vonmises_max_mpa"],
        "vonmises_gust_pratt_fs15_mpa": OAS_SAFETY_FACTOR * pt_gust["vonmises_max_mpa"],
        "n_gust_limit": n_gust,
        "failure_maneuver_ks": pt_man["failure_ks"],
        "failure_gust_ks": pt_gust["failure_ks"],
        "panel_y_abs_m": [float(y) for y in panel_y_abs],
        "panel_chords_m": [float(c) for c in panel_chords],
        "panel_cl": [float(c) for c in cl_local],
    }


class TierH1UAVOracle:
    """Tier-H1 coupled OpenAeroStruct CFRP wingbox (`with_viscous=False`) + 8-GL XFoil oracle."""

    def __init__(
        self,
        schema: ProblemSchema,
        xfoil_runner: XFoilRunner | None = None,
        name: str = "tier_h1_uav",
        capture_provenance: bool = False,
        xfoil_workers: int = 8,
    ) -> None:
        self._schema = schema
        self._xfoil = xfoil_runner or XFoilRunner()
        self._name = name
        self._capture_provenance = capture_provenance
        self._xfoil_workers = max(1, int(xfoil_workers))
        self._sub_oracles: tuple[Any, ...] = ()

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> OracleTier:
        return OracleTier.H1

    @property
    def schema_id(self) -> str:
        return self._schema.schema_id

    @property
    def approx_cost_s(self) -> float:
        return 1.5

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
        """Evaluate design using coupled OpenAeroStruct wingbox + 8-GL XFoil 6.99."""
        t_start = time.perf_counter()
        scen = dict(self._schema.scenario_defaults)
        if scenario:
            scen.update(scenario)
        provenance = HostProvenance.capture() if self._capture_provenance else None
        oas_info: dict[str, Any] = {}

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

            oas_info = run_openaerostruct_wing(
                params=params,
                geom=geom,
                mass_kg=mass_kg,
                m_wing_kg=mass_info["m_wing_kg"],
                airfoil_cst=airfoil_cst,
                scenario=scen,
            )

            # 8 Gauss-Legendre spanwise stations over [0, b/2] with free + tripped (x_tr=0.05) XFoil
            coords = kulfan_dict_to_coordinates(airfoil_cst, n_points_per_side=100)
            rho = scen["rho_air"]
            mu = scen["mu_air"]
            v_mps = scen.get("cruise_speed_mps", 24.0)
            mach = v_mps / 340.3
            half_b = 0.5 * geom["wing_span_m"]
            c_r = geom["c_root_m"]
            c_t = geom["c_tip_m"]

            xi_gl, wi_gl = np.polynomial.legendre.leggauss(8)
            y_gl = 0.5 * half_b * (xi_gl + 1.0)
            dy_gl = 0.5 * half_b * wi_gl

            # Sort OAS panel y_abs ascending for interpolation
            y_pan = np.asarray(oas_info["panel_y_abs_m"], dtype=float)
            cl_pan = np.asarray(oas_info["panel_cl"], dtype=float)
            order = np.argsort(y_pan)
            y_pan_s, cl_pan_s = y_pan[order], cl_pan[order]
            camber = float(airfoil_cst.get("max_camber_over_c", 0.029))

            def _eval_gl_station(y_i: float, dy_i: float) -> dict[str, float]:
                c_i = c_r + (c_t - c_r) * (float(y_i) / half_b)
                cl_i = float(np.interp(float(y_i), y_pan_s, cl_pan_s))
                re_i = rho * v_mps * c_i / mu
                alpha_est = (cl_i - 2.0 * math.pi * 1.8 * camber) / (2.0 * math.pi * math.pi / 180.0)
                alpha_est = float(np.clip(alpha_est, -4.0, 12.0))

                xf_free = self._xfoil.run_polar(
                    airfoil_coords=coords,
                    reynolds=re_i,
                    mach=mach,
                    cl_targets=[cl_i],
                    aseq=(6.0, 15.0, 1.5),
                    x_tr_trip=None,
                )
                if xf_free.status != SolverStatus.RESOLVED_PASS or not xf_free.points:
                    raise RuntimeError(
                        xf_free.failure_reason or "XFoil free-transition station failed to converge"
                    )

                xf_trip = self._xfoil.run_polar(
                    airfoil_coords=coords,
                    reynolds=re_i,
                    mach=mach,
                    cl_targets=[cl_i],
                    x_tr_trip=0.05,
                )
                if xf_trip.status != SolverStatus.RESOLVED_PASS or not xf_trip.points:
                    raise RuntimeError(
                        xf_trip.failure_reason or "XFoil tripped x_tr=0.05 station failed to converge"
                    )

                pt_free = min(xf_free.points, key=lambda pt: abs(pt.cl - cl_i))
                pt_trip = min(xf_trip.points, key=lambda pt: abs(pt.cl - cl_i))
                # Blend 80% natural free transition + 20% insect/roughness trip (x_tr=0.05)
                cd_station = 0.80 * pt_free.cd + 0.20 * pt_trip.cd
                cl_max_station = max(pt.cl for pt in xf_free.points)
                return {
                    "y_m": float(y_i),
                    "dy_m": float(dy_i),
                    "chord_m": float(c_i),
                    "re": float(re_i),
                    "cl_target": float(cl_i),
                    "alpha_deg": float(pt_free.alpha_deg),
                    "alpha_est_deg": float(alpha_est),
                    "cd_free": float(pt_free.cd),
                    "cd_tripped_05": float(pt_trip.cd),
                    "cd_blended": float(cd_station),
                    "cl_max": float(cl_max_station),
                }

            station_inputs = list(zip([float(y) for y in y_gl], [float(dy) for dy in dy_gl]))
            with ThreadPoolExecutor(max_workers=min(self._xfoil_workers, len(station_inputs))) as pool:
                xfoil_stations = list(pool.map(lambda pair: _eval_gl_station(pair[0], pair[1]), station_inputs))

            cd_prof_num = sum(st["cd_blended"] * st["chord_m"] * st["dy_m"] for st in xfoil_stations)
            area_gl_den = sum(st["chord_m"] * st["dy_m"] for st in xfoil_stations)
            cl_max_min = min(st["cl_max"] for st in xfoil_stations)

            cd_wing_prof_h1 = cd_prof_num / max(area_gl_den, 1e-9)

            def cf_turb(re_val: float) -> float:
                return 0.455 / (math.log10(max(re_val, 1.0e4)) ** 2.58)

            mac = geom["mac_m"]
            s_wing = geom["wing_area_m2"]
            fr_fuse = FUSE_LENGTH_M / FUSE_DIAMETER_M
            ff_fuse = 1.0 + (60.0 / (fr_fuse**3)) + (fr_fuse / 400.0)
            cd_fuse = (
                cf_turb(rho * v_mps * FUSE_LENGTH_M / mu)
                * ff_fuse
                * 1.05
                * geom["s_wet_fuse_m2"]
                / s_wing
            )
            cd_booms = (
                cf_turb(rho * v_mps * BOOM_LENGTH_M / mu)
                * 1.15
                * 1.05
                * geom["s_wet_booms_m2"]
                / s_wing
            )
            cd_tail = (
                cf_turb(rho * v_mps * (0.65 * mac) / mu)
                * 1.22
                * 1.08
                * (2.04 * (geom["s_h_m2"] + geom["s_v_m2"]))
                / s_wing
            )
            cd_excresc = (
                6.0 * (0.0042 * params["lift_rotor_diameter_m"] + 0.00105) + 0.00472
            ) / s_wing
            cd_total_h1 = (
                cd_wing_prof_h1 + oas_info["cdi_oas"] + cd_fuse + cd_booms + cd_tail + cd_excresc
            )

            cl_cruise = oas_info["cl_oas"]
            l_over_d_h1 = cl_cruise / cd_total_h1

            eta_esc = selected_sku_objs["esc"].specs["drive_elec_eff"]
            eta_cm = selected_sku_objs["cruise_motor"].specs["cruise_motor_eff"]
            eta_cruise_total = eta_esc * eta_cm * CRUISE_PROP_AERODYN_EFF
            p_elec_cruise_w = ((mass_kg * G_ACCEL) / l_over_d_h1) * v_mps / eta_cruise_total

            oei_trim = solve_oei_trim_qp(params, geom, mass_kg, selected_sku_objs, scen)
            electro = compute_battery_electrothermal(
                params=params,
                mass_info=mass_info,
                p_cruise_elec_w=p_elec_cruise_w,
                p_oei_elec_w=oei_trim["p_oei_elec_w"],
                selected_skus=selected_sku_objs,
                scenario=scen,
            )
            cold_oei_margin, cold_el = compute_cold_oei_margin(
                params=params,
                mass_info=mass_info,
                p_cruise_elec_w=p_elec_cruise_w,
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
            i_cruise_esc_a = float(p_elec_cruise_w) / v_cruise_term
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
                "stall_cl_margin": float(cl_cruise - 0.85 * cl_max_min),
                "maneuver_stress_margin_mpa": float(
                    oas_info["vonmises_maneuver_3g_fs15_mpa"] - scen["sigma_allow_mpa"]
                ),
                "gust_stress_margin_mpa": float(
                    oas_info["vonmises_gust_pratt_fs15_mpa"] - scen["sigma_allow_mpa"]
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
                "cruise_motor_power_margin_w": float(p_elec_cruise_w - cruise_motor_max_power_w),
                "compat_violation_count": float(len(compat_errors)),
            }

            if not all(math.isfinite(val) for val in [*objectives.values(), *constraints.values()]):
                raise ValueError("Non-finite output in Tier-H1 evaluation")

            m_safe = self._schema.compute_m_safe(constraints)
            tol_map = {c.name: c.tolerance for c in self._schema.constraints}
            violated = any(val > tol_map.get(name, 1e-6) for name, val in constraints.items())
            status = SolverStatus.RESOLVED_VIOLATION if violated else SolverStatus.RESOLVED_PASS

            result = EvalResult(
                design_id=design.design_id,
                oracle_name=self._name,
                tier=OracleTier.H1,
                status=status,
                objectives=objectives,
                constraints=constraints,
                m_safe=m_safe,
                diagnostics={
                    "scenario": scen,
                    "selected_skus": design.selected_skus,
                    "compat_ok": compat_ok,
                    "compat_errors": compat_errors,
                    "openaerostruct": oas_info,
                    "xfoil_gl8_stations": xfoil_stations,
                    "cd_wing_prof_h1": cd_wing_prof_h1,
                    "l_over_d_h1": l_over_d_h1,
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
                tier=OracleTier.H1,
                status=SolverStatus.UNRESOLVED_NUMERICAL,
                diagnostics={"openaerostruct": oas_info} if oas_info else {},
                wall_time_s=time.perf_counter() - t_start,
                failure_reason=f"{type(exc).__name__}: {exc}",
                provenance=provenance,
            )

        if ledger is not None:
            ledger.debit(result)
        return result
