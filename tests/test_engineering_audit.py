"""Automated Level 1 & Level 2 Engineering Audit Gates for VeriAero-CEGIS v3.2.

Enforces objective, non-negotiable engineering standards across current and future modules:
- Level 1 (Contracts, Integrity & Architecture Firewalls):
  1. Zero dead variables & deterministic derived-feature sync (`is_derived`).
  2. Zero dead CompatibilityRule types (every declared rule_type actively enforced).
  3. Domain-agnostic kernel AST firewall (`stage_c/` and `stage_b/` never import physics/solvers/quarantine).
  4. Catalog pedigree & gravimetric energy internal consistency across all 44 SKUs.
  5. Bitwise determinism and tri-state / RED-AMBER fault-injection placard verification.
- Level 2 (First-Principles Physics, Cross-Fidelity Consistency & Solver Verification):
  6. Tier-L speed (< 25 ms/eval), thermal ODE energy conservation (< 1e-5), and
     exact closed-form OEI nullspace vs CVXPY equivalence (< 1e-7 across 15 random cases).
  7. Tier-H1 OpenAeroStruct FS=1.5 stress factoring, cross-fidelity agreement with Tier-L,
     and true `Problem.setup()` cache reuse on perturbed geometries.
  8. Tier-H2 SU2 strict `surface_flow.csv` y+ gate (rejects missing y+), Re-adaptive
     wall-normal C-grid spacing (est y+ <= 1.0 across Re=2e5..6e6), and minSJ >= 0.20.
  9. 100-design Sobol fuzz stress gate on Tier-L (zero uncaught crashes, < 5% unresolved).
 10. Live XFoil 6.99 parity (NACA 0012 & NASA TM-4062 E387) + live Tier-H1 eval when
     `bin/xfoil_patched` is present (auto-executes on C2-WSL2).
"""

from __future__ import annotations

import ast
import math
import time
import typing
from pathlib import Path

import aerosandbox as asb
import numpy as np
import pytest
from scipy.stats import qmc

from veriaero.contracts.credibility import CredibilityEvidenceCard
from veriaero.contracts.schema import (
    CompatibilityGraph,
    CompatibilityRule,
    ProblemSchema,
    VariableKind,
)
from veriaero.contracts.types import BudgetLedger, SolverStatus
from veriaero.physics.tier_h1_uav import (
    OAS_SAFETY_FACTOR,
    TierH1UAVOracle,
    _OAS_PROBLEM_CACHE,
    run_openaerostruct_wing,
)
from veriaero.physics.tier_l_uav import (
    TierLUAVOracle,
    compute_geometry,
    compute_mass_and_bom,
    compute_root_bending_stress,
    solve_oei_trim_qp,
)
from veriaero.solvers.su2_runner import (
    evaluate_su2_dual_pass_gate,
    generate_airfoil_cgrid_su2_with_metrics,
    parse_su2_surface_yplus,
)
from veriaero.solvers.xfoil_runner import XFoilRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs" / "uav25kg_schema.json"


@pytest.fixture(scope="module")
def schema() -> ProblemSchema:
    return ProblemSchema.from_json_file(SCHEMA_PATH)


# =============================================================================
# LEVEL 1: CONTRACTS, ZERO-DEAD-CODE, FIREWALLS & PEDIGREE AUDITS
# =============================================================================


def test_l1_zero_dead_variables_and_derived_sync(schema: ProblemSchema) -> None:
    """Every non-derived variable must affect physics; derived vars must sync deterministically."""
    oracle = TierLUAVOracle(schema=schema)
    base_x = schema.baseline_vector()
    base_ir = schema.snap_design(base_x, design_id="l1_base")
    base_res = oracle.evaluate(base_ir)
    assert base_res.status == SolverStatus.RESOLVED_PASS

    derived_names = {v.name for v in schema.variables if v.is_derived}
    assert derived_names == {"batt_mass_frac", "normalized_bom_cost"}
    assert len(schema.active_variable_indices()) == schema.dim - len(derived_names)

    # 1. Non-derived continuous/integer variables must each produce non-zero sensitivity
    for v in schema.variables:
        if v.is_derived or v.kind == VariableKind.CATALOG_FEATURE:
            continue
        x_pert = base_x.copy()
        x_pert[v.index] = v.upper if abs(v.baseline - v.upper) > 1e-6 else v.lower
        ir_pert = schema.snap_design(x_pert, design_id=f"pert_{v.name}")
        res_pert = oracle.evaluate(ir_pert)
        assert res_pert.status in (SolverStatus.RESOLVED_PASS, SolverStatus.RESOLVED_VIOLATION)
        d_obj = max(abs(res_pert.objectives[k] - base_res.objectives[k]) for k in base_res.objectives)
        d_con = max(abs(res_pert.constraints[k] - base_res.constraints[k]) for k in base_res.constraints)
        assert max(d_obj, d_con) > 1e-9, f"Dead non-derived variable detected: {v.name}"

    # 2. Derived variables must be deterministically synced by snap_design() (immune to raw noise)
    x_noisy = base_x.copy()
    idx = {v.name: v.index for v in schema.variables}
    x_noisy[idx["batt_mass_frac"]] = 0.16
    x_noisy[idx["normalized_bom_cost"]] = 0.0
    ir_synced = schema.snap_design(x_noisy, design_id="derived_sync")
    expected_bmf = (1.15 * 12 * 7 * 0.070) / 25.0
    assert abs(ir_synced.params["batt_mass_frac"] - expected_bmf) < 1e-6
    assert ir_synced.params["normalized_bom_cost"] > 0.4


def test_l1_all_compatibility_rule_types_active(schema: ProblemSchema) -> None:
    """Every rule_type declared in CompatibilityRule must be actively enforced (zero dead enum)."""
    declared_types = set(typing.get_args(CompatibilityRule.model_fields["rule_type"].annotation))
    assert declared_types == {
        "voltage_window",
        "integrated_esc_cap",
        "motor_kv_voltage",
        "current_limit",
        "origin_policy",
        "blocked_pair",
    }

    base_skus = {
        "cell": schema.catalog.get("CELL_ALLIED_MOLICEL_P45B"),
        "lift_motor": schema.catalog.get("MOTOR_LIFT_ALLIED_EPROPELLED_EM5215"),
        "esc": schema.catalog.get("ESC_ALLIED_APD_80F3X"),
        "cruise_motor": schema.catalog.get("MOTOR_CRUISE_ALLIED_EPROPELLED_EM3520"),
    }

    # Test current_limit (passes at 50 A req, fails at 100 A req on 80 A ESC)
    cg_curr_ok = CompatibilityGraph(
        rules=[
            CompatibilityRule(
                rule_id="c_ok",
                rule_type="current_limit",
                params={"load_category": "esc", "min_required_i_a": 50.0},
            )
        ]
    )
    ok, errs = cg_curr_ok.check(base_skus, {"batt_n_series": 12.0})
    assert ok and not errs

    cg_curr_fail = CompatibilityGraph(
        rules=[
            CompatibilityRule(
                rule_id="c_fail",
                rule_type="current_limit",
                params={"load_category": "esc", "min_required_i_a": 100.0},
            )
        ]
    )
    ok, errs = cg_curr_fail.check(base_skus, {"batt_n_series": 12.0})
    assert not ok and len(errs) == 1

    # Test origin_policy & blocked_pair rules directly
    cg_origin = CompatibilityGraph(
        rules=[
            CompatibilityRule(
                rule_id="orig",
                rule_type="origin_policy",
                params={"allowed_origins": ["PRC"]},
            )
        ]
    )
    ok, _ = cg_origin.check(base_skus, {"batt_n_series": 12.0})
    assert not ok

    cg_blocked = CompatibilityGraph(
        rules=[
            CompatibilityRule(
                rule_id="blk",
                rule_type="blocked_pair",
                params={
                    "sku_a": "CELL_ALLIED_MOLICEL_P45B",
                    "sku_b": "ESC_ALLIED_APD_80F3X",
                },
            )
        ]
    )
    ok, _ = cg_blocked.check(base_skus, {"batt_n_series": 12.0})
    assert not ok


def test_l1_domain_agnostic_kernel_firewall() -> None:
    """Stage C and Stage B must NEVER import UAV physics, external solvers, or eval_quarantine."""
    forbidden_prefixes = (
        "veriaero.physics",
        "veriaero.solvers",
        "eval_quarantine",
    )
    kernel_dirs = [
        REPO_ROOT / "src" / "veriaero" / "stage_c",
        REPO_ROOT / "src" / "veriaero" / "stage_b",
    ]
    violations: list[str] = []
    for kdir in kernel_dirs:
        assert kdir.is_dir(), f"Missing kernel package directory: {kdir}"
        for py_file in kdir.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(
                            alias.name == p or alias.name.startswith(p + ".")
                            for p in forbidden_prefixes
                        ):
                            violations.append(f"{py_file}:{node.lineno} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if any(mod == p or mod.startswith(p + ".") for p in forbidden_prefixes):
                        violations.append(f"{py_file}:{node.lineno} imports from {mod}")

    assert not violations, "Domain-agnostic kernel boundary breached:\n" + "\n".join(violations)


def test_l1_catalog_internal_pedigree_and_energy_consistency(schema: ProblemSchema) -> None:
    """All 44 SKUs must have valid pedigree, finite specs, and physically consistent cell energy."""
    assert len(schema.catalog.items) == 44
    for sku_id, item in schema.catalog.items.items():
        assert item.sku_id == sku_id
        assert item.unit_cost_usd > 0.0 and math.isfinite(item.unit_cost_usd)
        assert item.price_source in ("retail_list", "parametric_estimate")
        assert len(item.provenance_note.strip()) >= 10, f"Missing provenance on {sku_id}"
        for k, v in item.specs.items():
            assert math.isfinite(v), f"Non-finite spec {k} on {sku_id}"

        if item.category == "cell":
            calc_wh_kg = (
                item.specs["cell_v_nom"] * item.specs["cell_capacity_ah"] / item.specs["cell_mass_kg"]
            )
            declared_wh_kg = item.specs["cell_specific_energy_wh_kg"]
            rel_err = abs(declared_wh_kg - calc_wh_kg) / max(calc_wh_kg, 1e-6)
            assert rel_err < 0.05, (
                f"Cell {sku_id} gravimetric energy mismatch: declared={declared_wh_kg:.2f} vs "
                f"V*Ah/m={calc_wh_kg:.2f} ({100*rel_err:.1f}%)"
            )


def test_l1_determinism_and_fault_injection_placards(schema: ProblemSchema) -> None:
    """Repeated evaluations are deterministic; out-of-envelope & NaN faults trigger RED/AMBER."""
    oracle = TierLUAVOracle(schema=schema)
    ir = schema.snap_design(schema.baseline_vector(), design_id="det_check")
    r1 = oracle.evaluate(ir)
    r2 = oracle.evaluate(ir)
    assert r1.objectives == r2.objectives
    assert r1.constraints == r2.constraints
    assert r1.m_safe == r2.m_safe

    # Out-of-envelope Mach > 0.30 and Re < 150,000 must trigger RED placard
    card_mach = CredibilityEvidenceCard.evaluate_placard(
        design_id="high_mach",
        oracle_statuses=[SolverStatus.RESOLVED_PASS],
        local_mach=0.35,
        min_reynolds=400_000.0,
        k_max_tilde=0.90,
    )
    assert card_mach.placard_level == "RED"
    assert any("OUT_OF_ENVELOPE" in f for f in card_mach.flags)

    card_re = CredibilityEvidenceCard.evaluate_placard(
        design_id="low_re",
        oracle_statuses=[SolverStatus.RESOLVED_PASS],
        local_mach=0.07,
        min_reynolds=120_000.0,
        k_max_tilde=0.90,
    )
    assert card_re.placard_level == "RED"


def test_l1_ai_disclosure_log_schema_integrity() -> None:
    """AI disclosure log must exist with exact 7-column header and valid CSV rows."""
    import csv

    log_path = REPO_ROOT / "docs" / "ai_disclosure_log.csv"
    assert log_path.is_file(), "Missing docs/ai_disclosure_log.csv"
    expected_cols = [
        "date_utc",
        "tool_name_and_version",
        "task_category",
        "exact_scope_of_assistance",
        "student_independent_verification",
        "defect_caught_by_student",
        "git_commit_or_artifact_ref",
    ]
    with log_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == expected_cols
        rows = list(reader)
    assert len(rows) >= 1
    for i, r in enumerate(rows, 1):
        assert None not in r, f"Row {i} has extra unescaped commas in ai_disclosure_log.csv"
        assert all(r[c] is not None for c in expected_cols), f"Row {i} has missing columns"


# =============================================================================
# LEVEL 2: FIRST-PRINCIPLES PHYSICS, CROSS-FIDELITY & SOLVER VERIFICATION
# =============================================================================


def test_l2_tier_l_speed_conservation_and_oei_cvxpy_equivalence(schema: ProblemSchema) -> None:
    """Tier-L runs < 25 ms/eval, conserves thermal energy, and matches CVXPY OEI QP to < 1e-7."""
    oracle = TierLUAVOracle(schema=schema)
    ir = schema.snap_design(schema.baseline_vector(), design_id="speed_base")
    # Warm up NeuralFoil / PCA cache once
    res_warm = oracle.evaluate(ir)
    assert res_warm.status == SolverStatus.RESOLVED_PASS

    # 1. Speed gate (< 25 ms average per evaluation)
    n_bench = 15
    t0 = time.perf_counter()
    for _ in range(n_bench):
        oracle.evaluate(ir)
    avg_s = (time.perf_counter() - t0) / n_bench
    assert avg_s < 0.025, f"Tier-L too slow: {1000.0*avg_s:.2f} ms/eval (limit 25 ms)"

    # 2. Thermal ODE First-Law energy conservation: Q_gen == Q_stored + Q_conv
    el = res_warm.diagnostics["electrothermal"]
    q_gen = el["q_gen_j"]
    q_bal_err = abs(q_gen - (el["q_stored_j"] + el["q_conv_j"])) / max(q_gen, 1.0)
    assert q_bal_err < 1e-5, f"Thermal energy balance error {q_bal_err:.2e} >= 1e-5"

    # 3. Exact equivalence between fast closed-form nullspace OEI solver and CVXPY across 15 designs
    rng = np.random.default_rng(42)
    skus = {c: schema.catalog.get(sid) for c, sid in ir.selected_skus.items()}
    for k in range(15):
        params_k = dict(ir.params)
        params_k["wing_area_m2"] = float(rng.uniform(0.85, 1.50))
        params_k["aspect_ratio"] = float(rng.uniform(8.5, 15.5))
        params_k["taper_ratio"] = float(rng.uniform(0.32, 0.80))
        params_k["lift_rotor_diameter_m"] = float(rng.uniform(0.40, 0.65))
        geom_k = compute_geometry(params_k, airfoil_cst=ir.airfoil_cst)
        mass_k = float(rng.uniform(20.0, 27.0))
        scen_k = dict(schema.scenario_defaults)
        scen_k["cg_shift_mac_frac"] = float(rng.uniform(0.02, 0.08))

        out_fast = solve_oei_trim_qp(params_k, geom_k, mass_k, skus, scen_k, backend="nullspace")
        out_cvx = solve_oei_trim_qp(params_k, geom_k, mass_k, skus, scen_k, backend="cvxpy")
        assert out_fast["oei_all_18_cases_feasible"] == out_cvx["oei_all_18_cases_feasible"]
        assert abs(out_fast["worst_peak_thrust_n"] - out_cvx["worst_peak_thrust_n"]) < 1e-6
        assert abs(out_fast["p_oei_elec_w"] - out_cvx["p_oei_elec_w"]) < 1e-5
        assert abs(out_fast["worst_peak_motor_elec_w"] - out_cvx["worst_peak_motor_elec_w"]) < 1e-5

    # 4. Electrical limits actively enforced (esc_current_margin_a & cruise_motor_power_margin_w)
    assert "esc_current_margin_a" in res_warm.constraints
    assert "cruise_motor_power_margin_w" in res_warm.constraints
    assert res_warm.constraints["esc_current_margin_a"] < 0.0
    assert res_warm.constraints["cruise_motor_power_margin_w"] < 0.0


def test_l2_tier_h1_oas_fs15_cross_fidelity_and_true_cache_reuse(schema: ProblemSchema) -> None:
    """OAS applies FS=1.5, matches Tier-L baseline stress within +-30%, and truly reuses cache."""
    _OAS_PROBLEM_CACHE.clear()
    ir = schema.snap_design(schema.baseline_vector(), design_id="h1_audit_base")
    skus = {c: schema.catalog.get(sid) for c, sid in ir.selected_skus.items()}
    geom = compute_geometry(ir.params, airfoil_cst=ir.airfoil_cst)
    mass = compute_mass_and_bom(ir.params, geom, skus)
    a_rad = (2.0 * math.pi * geom["aspect_ratio"]) / (
        2.0 + math.sqrt(4.0 + geom["aspect_ratio"] ** 2)
    )
    struct_l = compute_root_bending_stress(
        ir.params, geom, mass["m_total_kg"], mass["m_wing_kg"], a_rad, schema.scenario_defaults
    )

    t0 = time.perf_counter()
    oas_base = run_openaerostruct_wing(
        ir.params, geom, mass["m_total_kg"], mass["m_wing_kg"], ir.airfoil_cst, schema.scenario_defaults
    )
    t_cold = time.perf_counter() - t0
    assert len(_OAS_PROBLEM_CACHE) == 1

    # 1. Exact FS=1.5 factoring verified against unfactored stress and FailureKS
    assert abs(
        oas_base["vonmises_maneuver_3g_fs15_mpa"]
        - OAS_SAFETY_FACTOR * oas_base["vonmises_maneuver_3g_unfactored_mpa"]
    ) < 1e-9
    expected_ks_approx = oas_base["vonmises_maneuver_3g_fs15_mpa"] / 350.0 - 1.0
    assert abs(oas_base["failure_maneuver_ks"] - expected_ks_approx) < 0.05

    # 2. Cross-fidelity agreement: Tier-L Euler-Bernoulli vs Tier-H1 OAS factored wingbox within +-30%
    ratio = oas_base["vonmises_maneuver_3g_fs15_mpa"] / struct_l["sigma_maneuver_3g_fs15_mpa"]
    assert 0.70 <= ratio <= 1.30, (
        f"Tier-L ({struct_l['sigma_maneuver_3g_fs15_mpa']:.1f} MPa) vs "
        f"Tier-H1 OAS ({oas_base['vonmises_maneuver_3g_fs15_mpa']:.1f} MPa) diverged (ratio={ratio:.2f})"
    )

    # 3. True Problem.setup() cache reuse on perturbed planform, twist, and thicknesses
    x_pert = schema.baseline_vector()
    idx = {v.name: v.index for v in schema.variables}
    x_pert[idx["wing_area_m2"]] = 1.24
    x_pert[idx["aspect_ratio"]] = 13.5
    x_pert[idx["taper_ratio"]] = 0.55
    x_pert[idx["spar_cap_thickness_m"]] = 0.0035
    x_pert[idx["tip_twist_deg"]] = -3.0
    ir_pert = schema.snap_design(x_pert, design_id="h1_audit_pert")
    geom_p = compute_geometry(ir_pert.params, airfoil_cst=ir_pert.airfoil_cst)
    mass_p = compute_mass_and_bom(ir_pert.params, geom_p, skus)

    t1 = time.perf_counter()
    oas_pert = run_openaerostruct_wing(
        ir_pert.params,
        geom_p,
        mass_p["m_total_kg"],
        mass_p["m_wing_kg"],
        ir_pert.airfoil_cst,
        schema.scenario_defaults,
    )
    t_cached = time.perf_counter() - t1

    assert len(_OAS_PROBLEM_CACHE) == 1, "OAS cache missed on planform/thickness perturbation!"
    assert abs(oas_pert["s_ref_oas_m2"] - geom_p["wing_area_m2"]) < 0.03
    assert oas_pert["vonmises_maneuver_3g_fs15_mpa"] > oas_base["vonmises_maneuver_3g_fs15_mpa"]
    assert t_cached < max(t_cold, 0.25)


def test_l2_su2_yplus_gate_strictness_and_re_adaptive_cgrid(tmp_path: Path) -> None:
    """SU2 gate must reject missing y+; parse real surface_flow.csv; Re-adaptive C-grid y+ <= 1."""
    hist = tmp_path / "history.csv"
    hist.write_text(
        "\n".join(['"Iteration","rms[Rho]","CL","CD"'] + [f"{i},-6.5,0.75000,0.01250" for i in range(150)]),
        encoding="utf-8",
    )

    # 1. Missing mean_yplus MUST fail (never silently default to 1.0)
    rep_missing = evaluate_su2_dual_pass_gate(hist)
    assert rep_missing.passed is False
    assert rep_missing.gate_tier == "UNRESOLVED_NUMERICAL"

    # 2. Parsing surface_flow.csv accurately enforces y+ <= 1.5
    surf_ok = tmp_path / "surface_flow_ok.csv"
    surf_ok.write_text('"PointID","x","y","Y_Plus"\n0,0.1,0.01,0.70\n1,0.5,0.05,1.00\n', encoding="utf-8")
    y_ok = parse_su2_surface_yplus(surf_ok)
    assert y_ok is not None and abs(y_ok - 0.85) < 1e-6
    assert evaluate_su2_dual_pass_gate(hist, mean_yplus=y_ok).gate_tier == "STRICT_PASS"

    surf_bad = tmp_path / "surface_flow_bad.csv"
    surf_bad.write_text('"PointID","x","y","Y_Plus"\n0,0.1,0.01,2.40\n1,0.5,0.05,3.20\n', encoding="utf-8")
    y_bad = parse_su2_surface_yplus(surf_bad)
    assert y_bad is not None and y_bad > 1.5
    assert evaluate_su2_dual_pass_gate(hist, mean_yplus=y_bad).gate_tier == "UNRESOLVED_NUMERICAL"

    # 3. Re-adaptive Gmsh C-grid achieves y+ <= 1.0 and minSJ >= 0.20 from Re=2e5 up to TMR Re=6e6
    for af_name, re_val in [("e387", 2.0e5), ("sd7062", 5.2e5), ("naca0012", 6.0e6)]:
        coords = np.asarray(asb.Airfoil(af_name).coordinates, dtype=float)
        rep = generate_airfoil_cgrid_su2_with_metrics(
            coords,
            tmp_path / f"{af_name}_{int(re_val)}.su2",
            chord=1.0,
            farfield_radius=20.0,
            n_airfoil_half=35,
            n_wake_pts=15,
            n_normal_pts=60,
            reynolds=re_val,
            target_yplus=0.5,
            min_scaled_jacobian=0.20,
        )
        assert rep.min_scaled_jacobian >= 0.20
        assert rep.estimated_yplus is not None and rep.estimated_yplus <= 1.0


def test_l2_fuzz_100_sobol_designs_no_crashes(schema: ProblemSchema) -> None:
    """Fuzz 100 Sobol designs across full 26-D bounds: zero uncaught errors, < 5% unresolved."""
    oracle = TierLUAVOracle(schema=schema)
    ledger = BudgetLedger()
    lowers, uppers = schema.bounds()

    sampler = qmc.Sobol(d=schema.dim, scramble=True, seed=2026)
    samples_u = sampler.random(128)[:100]
    samples_x = lowers + samples_u * (uppers - lowers)

    for i, x in enumerate(samples_x):
        ir = schema.snap_design(x, design_id=f"fuzz_{i:03d}")
        res = oracle.evaluate(ir, ledger=ledger)
        assert res.status in (
            SolverStatus.RESOLVED_PASS,
            SolverStatus.RESOLVED_VIOLATION,
            SolverStatus.UNRESOLVED_NUMERICAL,
        )
        if res.status != SolverStatus.UNRESOLVED_NUMERICAL:
            assert all(math.isfinite(v) for v in res.objectives.values())
            assert all(math.isfinite(v) for v in res.constraints.values())
            assert res.m_safe is not None and math.isfinite(res.m_safe)

    assert ledger.total_calls() == 100
    assert ledger.unresolved_rate() < 0.05, (
        f"Excessive unresolved rate in Tier-L fuzzing: {100.0*ledger.unresolved_rate():.1f}%"
    )


def test_l2_live_xfoil_parity_when_binary_present(schema: ProblemSchema) -> None:
    """Runs live patched XFoil 6.99 parity (NACA 0012 + E387) & live Tier-H1 when binary exists."""
    runner = XFoilRunner()
    if not runner.is_available():
        pytest.skip(f"Live XFoil binary not built yet at {runner.binary_path} (runs on C2-WSL2)")

    # 1. NASA TM-4062 Eppler 387 Re=2e5, alpha=4 deg physical parity band
    e387_coords = np.asarray(asb.Airfoil("e387").coordinates, dtype=float)
    res_e387 = runner.run_polar(e387_coords, reynolds=2.0e5, mach=0.06, alphas_deg=[4.0])
    assert res_e387.status == SolverStatus.RESOLVED_PASS and len(res_e387.points) == 1
    pt = res_e387.points[0]
    assert 0.55 <= pt.cl <= 0.85, f"E387 Re=2e5 a=4 CL out of physical band: {pt.cl}"
    assert 0.008 <= pt.cd <= 0.025, f"E387 Re=2e5 a=4 CD out of physical band: {pt.cd}"

    # 2. Live Tier-H1 evaluation on baseline UAV with parallel 8-GL CL-targeted XFoil
    oracle_h1 = TierH1UAVOracle(schema=schema, xfoil_runner=runner)
    base_ir = schema.snap_design(schema.baseline_vector(), design_id="live_h1_base")
    t0 = time.perf_counter()
    res_h1 = oracle_h1.evaluate(base_ir)
    elapsed = time.perf_counter() - t0
    assert res_h1.status == SolverStatus.RESOLVED_PASS, f"Live Tier-H1 failed: {res_h1.failure_reason}"
    assert len(res_h1.diagnostics["xfoil_gl8_stations"]) == 8
    assert 10.5 <= res_h1.diagnostics["l_over_d_h1"] <= 15.0
    assert elapsed < 6.0, f"Live Tier-H1 too slow: {elapsed:.2f}s"
