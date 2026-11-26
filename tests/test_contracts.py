"""Unit tests for VeriAero-CEGIS v3.2 contracts, CST PCA, and NASA credibility cards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from veriaero.contracts.credibility import (
    CredibilityEvidenceCard,
    boltzmann_softmax_proximity,
)
from veriaero.contracts.schema import (
    DesignIR,
    DisciplineOracle,
    ProblemSchema,
)
from veriaero.contracts.types import (
    BudgetLedger,
    EvalResult,
    HostProvenance,
    OracleTier,
    SolverStatus,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "uav25kg_schema.json"


def test_schema_loads_and_roundtrips(tmp_path: Path) -> None:
    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    assert schema.schema_id == "uav25kg_v3.2"
    assert schema.dim == 26
    assert schema.catalog_categories() == ["cell", "lift_motor", "esc", "cruise_motor"]
    assert [o.name for o in schema.objectives] == [
        "verified_range_km",
        "domestic_bom_cost_usd",
        "cold_oei_margin",
    ]
    lowers, uppers = schema.bounds()
    base = schema.baseline_vector()
    assert lowers.shape == (26,)
    assert uppers.shape == (26,)
    assert np.all(lowers <= base) and np.all(base <= uppers)

    roundtrip_file = tmp_path / "schema_copy.json"
    schema.to_json_file(roundtrip_file)
    schema2 = ProblemSchema.from_json_file(roundtrip_file)
    assert schema2.dim == schema.dim
    assert len(schema2.catalog.items) == 44


def test_design_ir_snapping_origin_filtering_and_hole_dropout(tmp_path: Path) -> None:
    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    x = schema.baseline_vector()
    idx = {v.name: v.index for v in schema.variables}
    # Perturb cell features toward PRC EVE 50E (260.87 Wh/kg, 28 mOhm, 3.0 C)
    x[idx["cell_specific_energy_wh_kg"]] = 260.9
    x[idx["cell_dcir_25c_mohm"]] = 28.0
    x[idx["cell_max_c_rate"]] = 3.0
    x[idx["batt_n_series"]] = 12.4

    ir_any = schema.snap_design(x, design_id="snap_any")
    assert isinstance(ir_any, DesignIR)
    assert ir_any.topology["batt_n_series"] == 12
    assert ir_any.selected_skus["cell"] == "CELL_PRC_EVE_50E"

    # Origin filtering to ALLIED only
    ir_allied = schema.snap_design(x, design_id="snap_allied", allowed_origins={"ALLIED"})
    assert schema.catalog.get(ir_allied.selected_skus["cell"]).origin_tier == "ALLIED"
    assert ir_allied.selected_skus["cell"] == "CELL_ALLIED_LG_M50T"
    assert ir_allied.metadata["compat_ok"] is True

    # Catalog-hole dropout excluding LG_M50T must route cleanly to next nearest ALLIED cell
    ir_hole = schema.snap_design(
        x,
        design_id="snap_hole",
        allowed_origins={"ALLIED"},
        excluded_skus={"CELL_ALLIED_LG_M50T"},
    )
    assert ir_hole.selected_skus["cell"] != "CELL_ALLIED_LG_M50T"
    assert schema.catalog.get(ir_hole.selected_skus["cell"]).origin_tier == "ALLIED"

    # 18-param Kulfan CST LE radius / blunt TE & .dat / post-Dec export guards
    assert ir_allied.airfoil_cst["r_le_over_c"] >= 0.008 - 1e-9
    assert ir_allied.airfoil_cst["TE_thickness"] == 0.002
    assert len(ir_allied.airfoil_cst["upper_weights"]) == 8
    assert len(ir_allied.airfoil_cst["lower_weights"]) == 8

    dat_file = tmp_path / "airfoil.dat"
    dat_txt = ir_allied.export("dat", out_path=dat_file)
    assert dat_file.is_file() and "snap_allied" in dat_txt
    with pytest.raises(NotImplementedError):
        ir_allied.export("vsp3")


def test_compatibility_graph_and_vertiq_12s_cap() -> None:
    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    idx = {v.name: v.index for v in schema.variables}
    x = schema.baseline_vector()
    x[idx["batt_n_series"]] = 14.0  # 14S * 4.2V = 58.8 V

    # 14S + APD 80F3x (58.8 V max) -> compatible
    ir_14s_ok = schema.snap_design(
        x,
        design_id="14s_apd80",
        override_skus={
            "cell": "CELL_ALLIED_MOLICEL_P45B",
            "lift_motor": "MOTOR_LIFT_ALLIED_EPROPELLED_EM5215",
            "esc": "ESC_ALLIED_APD_80F3X",
            "cruise_motor": "MOTOR_CRUISE_ALLIED_EPROPELLED_EM3520",
        },
    )
    assert ir_14s_ok.metadata["compat_ok"] is True

    # 14S + Castle Edge HV 80 (12S / 50.4 V max) -> incompatible
    ir_14s_bad_esc = schema.snap_design(
        x,
        design_id="14s_castle",
        override_skus={
            "cell": "CELL_ALLIED_MOLICEL_P45B",
            "lift_motor": "MOTOR_LIFT_ALLIED_EPROPELLED_EM5215",
            "esc": "ESC_ALLIED_CASTLE_EDGE_HV80",
            "cruise_motor": "MOTOR_CRUISE_ALLIED_EPROPELLED_EM3520",
        },
    )
    assert ir_14s_bad_esc.metadata["compat_ok"] is False

    # Vertiq 81-08 G2 enforces 12S cap: passes at 12S, fails at 14S
    x_12 = schema.baseline_vector()
    ir_vertiq_12 = schema.snap_design(
        x_12,
        design_id="vertiq_12s",
        override_skus={"lift_motor": "MOTOR_LIFT_ALLIED_VERTIQ_8108_G2"},
    )
    assert ir_vertiq_12.metadata["compat_ok"] is True

    ir_vertiq_14 = schema.snap_design(
        x,
        design_id="vertiq_14s",
        override_skus={"lift_motor": "MOTOR_LIFT_ALLIED_VERTIQ_8108_G2"},
    )
    assert ir_vertiq_14.metadata["compat_ok"] is False


def test_boltzmann_proximity_and_nasa_credibility_card() -> None:
    arch = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    k_at_witness = boltzmann_softmax_proximity(np.array([0.0, 0.0]), arch)
    k_far = boltzmann_softmax_proximity(np.array([10.0, 10.0]), arch)
    assert 0.99 < k_at_witness <= 1.0
    assert 0.0 <= k_far < 0.01

    # Archive-size invariance: adding 50 distant duplicates keeps K_tilde > 0.99 (no 1/N collapse)
    arch_dup = np.vstack([arch, np.tile(np.array([5.0, 5.0]), (50, 1))])
    k_with_dups = boltzmann_softmax_proximity(np.array([0.0, 0.0]), arch_dup)
    assert k_with_dups > 0.99
    assert abs(k_with_dups - k_at_witness) < 0.01

    card_green = CredibilityEvidenceCard.evaluate_placard(
        design_id="d_ok",
        oracle_statuses=[SolverStatus.RESOLVED_PASS],
        local_mach=0.07,
        min_reynolds=400_000.0,
        k_max_tilde=0.85,
        verified_tiers=["L", "H1"],
    )
    assert card_green.placard_level == "GREEN"
    assert "Does NOT constitute NASA certification" in card_green.disclaimer
    assert len(card_green.coverage_ledger.mandatory_physical_gates_remaining) == 5

    card_amber = CredibilityEvidenceCard.evaluate_placard(
        design_id="d_extrap",
        oracle_statuses=[SolverStatus.RESOLVED_PASS],
        local_mach=0.07,
        min_reynolds=400_000.0,
        k_max_tilde=0.10,
    )
    assert card_amber.placard_level == "AMBER"

    card_red = CredibilityEvidenceCard.evaluate_placard(
        design_id="d_bad",
        oracle_statuses=[SolverStatus.UNRESOLVED_NUMERICAL],
        local_mach=0.07,
        min_reynolds=400_000.0,
        k_max_tilde=0.90,
    )
    assert card_red.placard_level == "RED"


def test_budget_ledger_provenance_and_protocol() -> None:
    from veriaero.physics.tier_l_uav import TierLUAVOracle

    ledger = BudgetLedger()
    ledger.debit(
        EvalResult(
            design_id="d1",
            oracle_name="o",
            tier=OracleTier.L,
            status=SolverStatus.RESOLVED_PASS,
            wall_time_s=0.1,
        )
    )
    ledger.debit(
        EvalResult(
            design_id="d2",
            oracle_name="o",
            tier=OracleTier.H1,
            status=SolverStatus.UNRESOLVED_NUMERICAL,
            failure_reason="Simulated timeout",
            wall_time_s=0.5,
        )
    )
    assert ledger.total_calls() == 2
    assert ledger.unresolved_count() == 1
    assert ledger.status_count(SolverStatus.RESOLVED_VIOLATION, OracleTier.H1) == 0

    prov = HostProvenance.capture(worker_count=8)
    assert len(prov.host_id) > 0
    assert prov.ram_total_gb > 0.0

    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    oracle = TierLUAVOracle(schema=schema)
    assert isinstance(oracle, DisciplineOracle)
    assert oracle.approx_cost_s > 0.0
    assert len(oracle.provides_outputs) == 3
