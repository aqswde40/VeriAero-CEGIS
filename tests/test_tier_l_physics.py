"""Analytical & first-principles verification suite for reconciled Tier-L UAV Physics."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from veriaero.contracts.schema import ProblemSchema
from veriaero.contracts.types import BudgetLedger, SolverStatus
from veriaero.physics.tier_l_uav import (
    TierLUAVOracle,
    arrhenius_resistance_scale,
    solve_quadratic_circuit,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "uav25kg_schema.json"


@pytest.fixture(scope="module")
def schema() -> ProblemSchema:
    return ProblemSchema.from_json_file(SCHEMA_PATH)


def test_baseline_passes_mass_mac_ld_range_and_pareto_objectives(schema: ProblemSchema) -> None:
    oracle = TierLUAVOracle(schema=schema, capture_provenance=True)
    ledger = BudgetLedger()
    base_ir = schema.snap_design(schema.baseline_vector(), design_id="baseline")

    res = oracle.evaluate(base_ir, ledger=ledger)
    assert res.status == SolverStatus.RESOLVED_PASS
    assert res.max_constraint_violation <= 1e-6
    assert res.m_safe is not None and res.m_safe > 0.0
    assert ledger.total_calls() == 1

    # Exact 25.00 kg mass closure
    assert abs(res.diagnostics["mass_breakdown"]["m_total_kg"] - 25.00) < 1e-4

    # True trapezoidal MAC = 0.3144 m
    mac = res.diagnostics["geometry"]["mac_m"]
    assert abs(mac - 0.31439) < 1e-4
    assert round(mac, 4) == 0.3144

    # Honest L/D ~ 12.45 and verified_range_km ~ 151.2 km
    ld = res.diagnostics["aero"]["l_over_d"]
    assert 12.2 <= ld <= 12.7
    assert 148.0 <= res.objectives["verified_range_km"] <= 154.0
    assert res.objectives["domestic_bom_cost_usd"] > 3000.0
    assert res.objectives["cold_oei_margin"] > 0.0


def test_mass_not_clamped_and_vertiq_integrated_esc_zeroes_lift_esc_mass(schema: ProblemSchema) -> None:
    oracle = TierLUAVOracle(schema=schema)
    x = schema.baseline_vector()
    idx = {v.name: v.index for v in schema.variables}
    x[idx["batt_n_series"]] = 14.0
    x[idx["batt_n_parallel"]] = 9.0
    ir_heavy = schema.snap_design(x, design_id="heavy_pack")
    res_heavy = oracle.evaluate(ir_heavy)
    assert res_heavy.status == SolverStatus.RESOLVED_VIOLATION
    assert res_heavy.diagnostics["mass_breakdown"]["m_total_kg"] > 27.0

    # Vertiq 81-08 G2 integrated ESC: lift ESC mass & cost = 0 (only 1 cruise ESC)
    ir_vertiq = schema.snap_design(
        schema.baseline_vector(),
        design_id="vertiq_g2",
        override_skus={"lift_motor": "MOTOR_LIFT_ALLIED_VERTIQ_8108_G2"},
    )
    res_vertiq = oracle.evaluate(ir_vertiq)
    assert res_vertiq.status == SolverStatus.RESOLVED_PASS
    esc_mass_single = schema.catalog.get(ir_vertiq.selected_skus["esc"]).specs["esc_mass_kg"]
    assert abs(res_vertiq.diagnostics["mass_breakdown"]["m_escs_kg"] - esc_mass_single) < 1e-9


def test_separated_3g_maneuver_and_pratt_gust_stress(schema: ProblemSchema) -> None:
    oracle = TierLUAVOracle(schema=schema)
    base_ir = schema.snap_design(schema.baseline_vector(), design_id="base_struct")
    res_base = oracle.evaluate(base_ir)
    st = res_base.diagnostics["structures"]

    assert st["n_gust_limit"] > 3.0
    assert st["sigma_gust_pratt_fs15_mpa"] > st["sigma_maneuver_3g_fs15_mpa"]
    assert res_base.constraints["maneuver_stress_margin_mpa"] < 0.0
    assert res_base.constraints["gust_stress_margin_mpa"] < 0.0

    # Thin spar cap + high aspect ratio + thin PCA airfoil (pca0=-2.5 -> t/c ~ 8.3%):
    # passes +3.0g maneuver (340 MPa < 350) yet FAILS Pratt discrete gust (420 MPa > 350)!
    x = schema.baseline_vector()
    idx = {v.name: v.index for v in schema.variables}
    x[idx["wing_area_m2"]] = 1.08
    x[idx["aspect_ratio"]] = 16.0
    x[idx["airfoil_pca_0"]] = -2.5
    x[idx["spar_cap_thickness_m"]] = 0.0015
    thin_ir = schema.snap_design(x, design_id="thin_spar")
    res_thin = oracle.evaluate(thin_ir)
    assert res_thin.status == SolverStatus.RESOLVED_VIOLATION
    assert res_thin.constraints["maneuver_stress_margin_mpa"] < 0.0
    assert res_thin.constraints["gust_stress_margin_mpa"] > 0.0


def test_arrhenius_0c_scaling_and_sensitivity_band() -> None:
    r_gas = 8.314462618
    dt_inv = (1.0 / 273.15) - (1.0 / 298.15)
    scale_25k = arrhenius_resistance_scale(0.0, ea_j_mol=25000.0)
    scale_20k = arrhenius_resistance_scale(0.0, ea_j_mol=20000.0)
    scale_30k = arrhenius_resistance_scale(0.0, ea_j_mol=30000.0)

    assert abs(scale_25k - math.exp((25000.0 / r_gas) * dt_inv)) < 1e-12
    assert abs(scale_20k - math.exp((20000.0 / r_gas) * dt_inv)) < 1e-12
    assert abs(scale_30k - math.exp((30000.0 / r_gas) * dt_inv)) < 1e-12
    assert 2.08 <= scale_20k < 2.50 <= scale_25k <= 2.53 < scale_30k <= 3.05


def test_quadratic_voltage_sag_and_0c_collapse(schema: ProblemSchema) -> None:
    circ = solve_quadratic_circuit(v_oc=43.2, r_pack=0.030, p_req_w=4200.0)
    assert not circ.voltage_collapse
    assert circ.residual_w < 1e-9

    oracle = TierLUAVOracle(schema=schema)
    prc_ir = schema.snap_design(
        schema.baseline_vector(),
        design_id="prc_eve_0c_cold",
        override_skus={"cell": "CELL_PRC_EVE_50E"},
    )
    res_cold = oracle.evaluate(prc_ir, scenario={"t_amb_c": 0.0, "cold_soak_takeoff": 1.0})
    assert res_cold.status == SolverStatus.RESOLVED_VIOLATION
    assert res_cold.objectives["cold_oei_margin"] < 0.0
    el = res_cold.diagnostics["electrothermal"]
    assert el["oei_voltage_collapse"] is True
    assert el["oei_discriminant_t0"] < 0.0
    assert res_cold.constraints["oei_voltage_collapse_margin_w"] > 0.0


def test_two_phase_thermal_ode_and_adiabatic_trapping(schema: ProblemSchema) -> None:
    oracle = TierLUAVOracle(schema=schema)
    base_ir = schema.snap_design(schema.baseline_vector(), design_id="thermal_check")

    res_low_h = oracle.evaluate(base_ir, scenario={"h_conv_w_m2k": 5.0})
    res_high_h = oracle.evaluate(base_ir, scenario={"h_conv_w_m2k": 10.0})

    for res in (res_low_h, res_high_h):
        el = res.diagnostics["electrothermal"]
        assert el["t_cruise_eq_c"] > res.diagnostics["scenario"]["t_amb_c"]
        assert el["t_oei_peak_c"] > el["t_cruise_eq_c"]
        assert el["trapped_heat_frac"] > 0.96

    assert (
        res_low_h.diagnostics["electrothermal"]["t_cruise_eq_c"]
        > res_high_h.diagnostics["electrothermal"]["t_cruise_eq_c"]
    )


def test_oei_trim_qp_residuals_and_undersized_motor_violation(schema: ProblemSchema) -> None:
    oracle = TierLUAVOracle(schema=schema)
    base_ir = schema.snap_design(schema.baseline_vector(), design_id="oei_base")
    res_base = oracle.evaluate(base_ir)
    oei = res_base.diagnostics["oei_trim"]
    assert oei["oei_all_18_cases_feasible"] is True
    assert oei["max_equilibrium_residual"] < 1e-6

    prc_motor_ir = schema.snap_design(
        schema.baseline_vector(),
        design_id="oei_prc_motor",
        override_skus={"lift_motor": "MOTOR_LIFT_PRC_TMOTOR_U7V2_280KV"},
    )
    res_motor = oracle.evaluate(prc_motor_ir)
    assert res_motor.status == SolverStatus.RESOLVED_VIOLATION
    assert res_motor.constraints["oei_thrust_margin_n"] > 0.0


def test_unresolved_numerical_status_still_debits_ledger(schema: ProblemSchema) -> None:
    oracle = TierLUAVOracle(schema=schema)
    ledger = BudgetLedger()
    bad_ir = schema.snap_design(schema.baseline_vector(), design_id="bad_nan")
    bad_ir.params["wing_area_m2"] = float("nan")

    res = oracle.evaluate(bad_ir, ledger=ledger)
    assert res.status == SolverStatus.UNRESOLVED_NUMERICAL
    assert ledger.total_calls() == 1
    assert ledger.unresolved_count() == 1
    assert ledger.status_count(SolverStatus.RESOLVED_VIOLATION) == 0
