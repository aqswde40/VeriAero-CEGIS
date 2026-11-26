"""Tests for reconciled 44-SKU 4-category catalog, OAS wingbox Tier-H1, 8-GL XFoil, and SU2."""

from __future__ import annotations

from pathlib import Path

import aerosandbox as asb
import numpy as np

from veriaero.contracts.schema import DisciplineOracle, ProblemSchema
from veriaero.contracts.types import BudgetLedger, SolverStatus
from veriaero.physics.tier_h1_uav import TierH1UAVOracle, run_openaerostruct_wing
from veriaero.physics.tier_l_uav import compute_geometry, compute_mass_and_bom
from veriaero.solvers.su2_runner import (
    evaluate_su2_dual_pass_gate,
    generate_airfoil_cgrid_su2,
    write_su2_rans_cfg,
)
from veriaero.solvers.xfoil_runner import (
    XFoilPolarPoint,
    XFoilRunResult,
    XFoilRunner,
    parse_xfoil_polar,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "uav25kg_schema.json"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog" / "uav25kg_catalog.json"


def test_reconciled_44_sku_4_category_origin_audit() -> None:
    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    assert CATALOG_PATH.is_file()
    items = schema.catalog.items
    assert len(items) == 44

    cells = schema.catalog.by_category("cell")
    lift_motors = schema.catalog.by_category("lift_motor")
    escs = schema.catalog.by_category("esc")
    cruise_motors = schema.catalog.by_category("cruise_motor")
    assert (len(cells), len(lift_motors), len(escs), len(cruise_motors)) == (13, 13, 9, 9)

    # Raw combinations: 13 * 13 * 9 * 9 * 7 (N_s) * 9 (N_p) = 862,407 > 300k
    raw_combos = len(cells) * len(lift_motors) * len(escs) * len(cruise_motors) * 7 * 9
    assert raw_combos > 300_000

    # Origin audit rules from Section 3.2:
    assert all("COBRA" not in sku_id for sku_id in items)
    assert items["MOTOR_LIFT_US_PRC_KDE_7208XF_135"].origin_tier == "US_DESIGNED_PRC_MFG"
    assert items["ESC_PRC_KDE_UAS95HVC"].origin_tier == "PRC"
    assert items["MOTOR_LIFT_ALLIED_VERTIQ_8108_G2"].has_integrated_esc is True
    assert items["MOTOR_LIFT_ALLIED_VERTIQ_8108_G2"].max_series_s == 12
    assert all(s.price_source in ("retail_list", "parametric_estimate") for s in items.values())

    # At least 4 Allied ESCs cover 14S-16S (v_max_v >= 58.8 V)
    allied_hv_escs = [
        e for e in schema.catalog.by_category("esc", allowed_origins={"ALLIED"})
        if e.specs["v_max_v"] >= 58.8 - 1e-6
    ]
    assert len(allied_hv_escs) >= 4


def test_openaerostruct_cfrp_wingbox_with_viscous_false() -> None:
    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    ir = schema.snap_design(schema.baseline_vector(), design_id="oas_wingbox_base")
    skus = {cat: schema.catalog.get(sid) for cat, sid in ir.selected_skus.items()}
    geom = compute_geometry(ir.params, airfoil_cst=ir.airfoil_cst)
    mass = compute_mass_and_bom(ir.params, geom, skus)

    oas = run_openaerostruct_wing(
        params=ir.params,
        geom=geom,
        mass_kg=mass["m_total_kg"],
        m_wing_kg=mass["m_wing_kg"],
        airfoil_cst=ir.airfoil_cst,
        scenario=schema.scenario_defaults,
    )
    assert abs(oas["cl_oas"] - 0.643) < 0.05
    assert 0.005 < oas["cdi_oas"] < 0.020
    panel_cl = np.asarray(oas["panel_cl"], dtype=float)
    assert 0.35 <= float(np.mean(panel_cl)) <= 0.90
    assert float(np.max(panel_cl)) < 1.15
    assert oas["tip_disp_cruise_m"] > 0.0
    assert oas["tip_disp_gust_m"] > oas["tip_disp_maneuver_3g_m"] > oas["tip_disp_cruise_m"]
    assert 5.0 < oas["vonmises_maneuver_3g_fs15_mpa"] < oas["vonmises_gust_pratt_fs15_mpa"] < 350.0


def test_xfoil_deck_n200_tripped_and_tri_state_missing_binary() -> None:
    runner = XFoilRunner(binary_path=Path("bin/nonexistent_xfoil_binary"))
    assert not runner.is_available()

    deck_free = runner.build_command_deck("airfoil.dat", "polar.pol", 500000.0, 0.07, [2.0, 12.0])
    assert "N 200" in deck_free
    assert "N 9.00" in deck_free
    assert "XTR" not in deck_free

    deck_trip = runner.build_command_deck(
        "airfoil.dat", "polar.pol", 500000.0, 0.07, [2.0], x_tr_trip=0.05
    )
    assert "XTR 0.050 0.050" in deck_trip

    sample_pol = (
        "   alpha     CL        CD       CDp       CM     Top_Xtr  Bot_Xtr\n"
        "  ------ -------- --------- --------- -------- -------- --------\n"
        "   2.000   0.6120   0.00745   0.00210  -0.0810   0.4820   0.9850\n"
    )
    pts = parse_xfoil_polar(sample_pol)
    assert len(pts) == 1 and abs(pts[0].cd - 0.00745) < 1e-6

    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    oracle = TierH1UAVOracle(schema=schema, xfoil_runner=runner)
    assert isinstance(oracle, DisciplineOracle)
    ledger = BudgetLedger()
    base_ir = schema.snap_design(schema.baseline_vector(), design_id="h1_missing_xfoil")
    res = oracle.evaluate(base_ir, ledger=ledger)
    assert res.status == SolverStatus.UNRESOLVED_NUMERICAL
    assert ledger.total_calls() == 1
    assert ledger.unresolved_count() == 1
    assert "openaerostruct" in res.diagnostics


def test_tier_h1_oracle_8gl_free_and_tripped_xfoil_mocked() -> None:
    calls_xtr: list[float | None] = []

    class MockXFoilRunner(XFoilRunner):
        def is_available(self) -> bool:
            return True

        def run_polar(
            self,
            airfoil_coords: np.ndarray,
            reynolds: float,
            mach: float,
            alphas_deg: list[float] | None = None,
            x_tr_trip: float | None = None,
            cl_targets: list[float] | None = None,
            aseq: tuple[float, float, float] | None = None,
        ) -> XFoilRunResult:
            calls_xtr.append(x_tr_trip)
            cd_val = 0.0095 if x_tr_trip is not None else 0.0074
            cl_val = cl_targets[0] if cl_targets else 0.64
            alpha_val = alphas_deg[0] if alphas_deg else 2.5
            pts = [XFoilPolarPoint(alpha_val, cl_val, cd_val, 0.0020, -0.08, 0.45, 0.98)]
            if aseq is not None or x_tr_trip is None:
                pts.append(XFoilPolarPoint(12.0, 1.45, 0.0260, 0.0180, -0.06, 0.05, 1.00))
            return XFoilRunResult(
                status=SolverStatus.RESOLVED_PASS,
                points=tuple(pts),
            )

    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    oracle = TierH1UAVOracle(schema=schema, xfoil_runner=MockXFoilRunner())
    ledger = BudgetLedger()
    base_ir = schema.snap_design(schema.baseline_vector(), design_id="h1_mocked_gl8")
    res = oracle.evaluate(base_ir, ledger=ledger)
    assert res.status == SolverStatus.RESOLVED_PASS
    assert len(res.diagnostics["xfoil_gl8_stations"]) == 8
    # 8 stations x (1 free + 1 tripped x_tr=0.05) = 16 XFoil polar calls
    assert len(calls_xtr) == 16
    assert calls_xtr.count(None) == 8 and calls_xtr.count(0.05) == 8
    assert 11.0 < res.diagnostics["l_over_d_h1"] < 15.0


def test_gmsh_e387_cgrid_locked_su2_sst_lm_and_dual_gate(tmp_path: Path) -> None:
    coords = np.asarray(asb.Airfoil("e387").coordinates, dtype=float)
    mesh_path = tmp_path / "e387_test.su2"
    cfg_path = tmp_path / "rans.cfg"

    generate_airfoil_cgrid_su2(
        coords,
        mesh_path,
        chord=1.0,
        farfield_radius=20.0,
        n_airfoil_half=35,
        n_wake_pts=15,
        n_normal_pts=21,
        progression_normal=1.15,
    )
    mesh_txt = mesh_path.read_text(encoding="utf-8", errors="ignore")
    assert "NDIME= 2" in mesh_txt
    assert "MARKER_TAG= airfoil" in mesh_txt
    assert "MARKER_TAG= farfield" in mesh_txt

    write_su2_rans_cfg(cfg_path, mesh_path.name, mach=0.06, reynolds=2.0e5, alpha_deg=4.0)
    cfg_txt = cfg_path.read_text(encoding="utf-8")
    for required in (
        "TIME_DOMAIN= NO",
        "KIND_TURB_MODEL= SST",
        "KIND_TRANS_MODEL= LM",
        "CONV_NUM_METHOD_FLOW= L2ROE",
        "CONV_NUM_METHOD_TURB= BOUNDED_SCALAR",
        "NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES",
    ):
        assert required in cfg_txt

    # STRICT_PASS: rms <= -6.0 and 100-iter Cauchy delta <= 1e-4
    hist_strict = tmp_path / "hist_strict.csv"
    hist_strict.write_text(
        "\n".join(['"Iteration","rms[Rho]","CL","CD"'] + [f"{i},-6.3,0.75000,0.01250" for i in range(150)]),
        encoding="utf-8",
    )
    rep_strict = evaluate_su2_dual_pass_gate(hist_strict, mean_yplus=0.9)
    assert rep_strict.passed is True and rep_strict.gate_tier == "STRICT_PASS"

    # LSB_DAMPED_PASS: rms <= -4.5 and 250-iter Cauchy delta <= 1e-4
    hist_lsb = tmp_path / "hist_lsb.csv"
    hist_lsb.write_text(
        "\n".join(['"Iteration","rms[Rho]","CL","CD"'] + [f"{i},-4.9,0.75000,0.01250" for i in range(260)]),
        encoding="utf-8",
    )
    rep_lsb = evaluate_su2_dual_pass_gate(hist_lsb, mean_yplus=1.1)
    assert rep_lsb.passed is True and rep_lsb.gate_tier == "LSB_DAMPED_PASS"

    # UNRESOLVED_NUMERICAL on oscillating LSB > 5e-4 or y+ > 1.5
    hist_osc = tmp_path / "hist_osc.csv"
    hist_osc.write_text(
        "\n".join(
            ['"Iteration","rms[Rho]","CL","CD"']
            + [f"{i},-5.2,{0.75 + (0.01 if i % 2 == 0 else -0.01):.5f},0.01250" for i in range(260)]
        ),
        encoding="utf-8",
    )
    rep_osc = evaluate_su2_dual_pass_gate(hist_osc, mean_yplus=1.0)
    assert rep_osc.passed is False and rep_osc.gate_tier == "UNRESOLVED_NUMERICAL"
