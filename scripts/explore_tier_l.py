"""Interactive & CLI Tier-L Physics Explorer for VeriAero-CEGIS v3.2.

Supports swapping SKUs across all 4 categories (`cell`, `lift_motor`, `esc`,
`cruise_motor`), temperature / cold-soak conditions, and pack topology `(N_s, N_p)`:
    python scripts/explore_tier_l.py --cell CELL_PRC_EVE_50E --temp-c 0 --cold-soak
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from veriaero.contracts.credibility import CredibilityEvidenceCard
from veriaero.contracts.schema import ProblemSchema
from veriaero.contracts.types import BudgetLedger, EvalResult
from veriaero.physics.tier_l_uav import TierLUAVOracle

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "uav25kg_schema.json"


def _explain_outcome(res: EvalResult) -> str:
    if res.status.value == "RESOLVED_PASS":
        return "ALL physical & compatibility constraints satisfied (safe to fly!)."
    if res.status.value == "UNRESOLVED_NUMERICAL":
        return f"Numerical solver failure: {res.failure_reason}"

    reasons: list[str] = []
    c = res.constraints
    d = res.diagnostics
    if c.get("compat_violation_count", 0.0) > 0:
        reasons.append("Compatibility: " + "; ".join(d.get("compat_errors", [])))
    if c.get("mass_excess_kg", 0.0) > 1e-6:
        reasons.append(f"Overweight by +{c['mass_excess_kg']:.3f} kg above 25.00 kg MTOW limit.")
    if c.get("stall_cl_margin", 0.0) > 1e-6:
        reasons.append("Wing stalls or lacks 15% CL margin in cruise.")
    if c.get("maneuver_stress_margin_mpa", 0.0) > 1e-6:
        reasons.append(
            f"+3.0g maneuver stress {d['structures']['sigma_maneuver_3g_fs15_mpa']:.1f} MPa > 350 MPa."
        )
    if c.get("gust_stress_margin_mpa", 0.0) > 1e-6:
        reasons.append(
            f"Pratt gust stress {d['structures']['sigma_gust_pratt_fs15_mpa']:.1f} MPa > 350 MPa."
        )
    if c.get("oei_thrust_margin_n", 0.0) > 1e-6:
        reasons.append(
            f"6-Rotor OEI trim needs {d['oei_trim']['worst_peak_thrust_n']:.1f} N > motor limit!"
        )
    if c.get("oei_voltage_collapse_margin_w", 0.0) > 1e-6:
        el = d["electrothermal"]
        reasons.append(
            f"VOLTAGE COLLAPSE! OEI needs {d['oei_trim']['p_oei_elec_w']:.0f} W > "
            f"max battery deliverable V_oc^2/(4R) = {el['oei_p_max_t0_w']:.0f} W "
            f"(Arrhenius R_pack = {el['r_pack_tcruise_ohm']:.4f} ohm)."
        )
    if c.get("oei_v_cutoff_margin_v", 0.0) > 1e-6:
        el = d["electrothermal"]
        reasons.append(
            f"Voltage sag below cutoff: V_term = {el['oei_v_term_min_v']:.2f} V < "
            f"{el['v_cutoff_pack_v']:.2f} V."
        )
    if c.get("oei_c_rate_margin_a", 0.0) > 1e-6:
        el = d["electrothermal"]
        reasons.append(
            f"OEI current {el['oei_current_max_a']:.1f} A exceeds pack rating "
            f"{el['i_pack_rated_max_a']:.1f} A."
        )
    if c.get("oei_temp_margin_c", 0.0) > 1e-6:
        el = d["electrothermal"]
        reasons.append(
            f"Battery thermal overtemp: {el['t_oei_peak_c']:.1f} C > 60.0 C limit!"
        )
    return " | ".join(reasons)


def print_comparison(
    label_a: str,
    res_a: EvalResult,
    label_b: str,
    res_b: EvalResult,
) -> None:
    da, db = res_a.diagnostics, res_b.diagnostics
    ea, eb = da["electrothermal"], db["electrothermal"]
    oa, ob = da["oei_trim"], db["oei_trim"]
    sa, sb = da["selected_skus"], db["selected_skus"]

    card_b = CredibilityEvidenceCard.evaluate_placard(
        design_id=res_b.design_id,
        oracle_statuses=[res_b.status],
        local_mach=db["aero"]["mach"],
        min_reynolds=db["aero"]["re_mac"],
        k_max_tilde=0.95,
        verified_tiers=["L"],
    )

    rows: list[tuple[str, str, str]] = [
        ("Solver Status", res_a.status.value, res_b.status.value),
        ("Signed Safety Margin m_safe", f"{res_a.m_safe:+.3f}", f"{res_b.m_safe:+.3f}"),
        ("Battery Cell SKU", sa["cell"], sb["cell"]),
        ("Lift Motor SKU", sa["lift_motor"], sb["lift_motor"]),
        ("ESC SKU", sa["esc"], sb["esc"]),
        ("Cruise Motor SKU", sa["cruise_motor"], sb["cruise_motor"]),
        ("Ambient Temp T_amb (C)", f"{da['scenario']['t_amb_c']:.1f}", f"{db['scenario']['t_amb_c']:.1f}"),
        ("Total Mass (kg)", f"{da['mass_breakdown']['m_total_kg']:.4f}", f"{db['mass_breakdown']['m_total_kg']:.4f}"),
        ("Wing MAC (m)", f"{da['geometry']['mac_m']:.4f}", f"{db['geometry']['mac_m']:.4f}"),
        ("Cruise L/D", f"{da['aero']['l_over_d']:.2f}", f"{db['aero']['l_over_d']:.2f}"),
        ("Obj 1: Verified Range (km)", f"{res_a.objectives['verified_range_km']:.2f}", f"{res_b.objectives['verified_range_km']:.2f}"),
        ("Obj 2: Domestic BOM (USD)", f"${res_a.objectives['domestic_bom_cost_usd']:.0f}", f"${res_b.objectives['domestic_bom_cost_usd']:.0f}"),
        ("Obj 3: Cold-OEI Margin (0C)", f"{res_a.objectives['cold_oei_margin']:+.3f}", f"{res_b.objectives['cold_oei_margin']:+.3f}"),
        ("+3.0g Maneuver Stress (MPa)", f"{da['structures']['sigma_maneuver_3g_fs15_mpa']:.1f}", f"{db['structures']['sigma_maneuver_3g_fs15_mpa']:.1f}"),
        ("Pratt Gust Stress (MPa)", f"{da['structures']['sigma_gust_pratt_fs15_mpa']:.1f}", f"{db['structures']['sigma_gust_pratt_fs15_mpa']:.1f}"),
        ("Arrhenius DCIR Scale", f"{ea['arrhenius_factor_at_tamb']:.3f}x", f"{eb['arrhenius_factor_at_tamb']:.3f}x"),
        ("R_pack at Cruise Eq (ohm)", f"{ea['r_pack_tcruise_ohm']:.4f}", f"{eb['r_pack_tcruise_ohm']:.4f}"),
        ("OEI Max Power V_oc^2/4R (W)", f"{ea['oei_p_max_t0_w']:.0f}", f"{eb['oei_p_max_t0_w']:.0f}"),
        ("OEI Required Elec Power (W)", f"{oa['p_oei_elec_w']:.0f}", f"{ob['p_oei_elec_w']:.0f}"),
        ("OEI Voltage Collapse?", str(ea["oei_voltage_collapse"]), str(eb["oei_voltage_collapse"])),
        ("90s OEI Peak Temp (C)", f"{ea['t_oei_peak_c']:.2f}", f"{eb['t_oei_peak_c']:.2f}"),
        ("90s Adiabatic Heat Trapped", f"{100.0*ea['trapped_heat_frac']:.1f}%", f"{100.0*eb['trapped_heat_frac']:.1f}%"),
        ("6-Rotor OEI Peak Thrust (N)", f"{oa['worst_peak_thrust_n']:.2f}", f"{ob['worst_peak_thrust_n']:.2f}"),
        ("OEI Equilibrium Residual", f"{oa['max_equilibrium_residual']:.2e}", f"{ob['max_equilibrium_residual']:.2e}"),
    ]

    w_m, w_a, w_b = 30, 38, 38
    sep = f"+{'-'*(w_m+2)}+{'-'*(w_a+2)}+{'-'*(w_b+2)}+"
    print("\n" + sep)
    print(f"| {'Metric':<{w_m}} | {label_a:<{w_a}} | {label_b:<{w_b}} |")
    print(sep)
    for k, va, vb in rows:
        print(f"| {k:<{w_m}} | {va:<{w_a}} | {vb:<{w_b}} |")
    print(sep)
    print(f"\n[{label_a}] Verdict: {_explain_outcome(res_a)}")
    print(f"[{label_b}] Verdict: {_explain_outcome(res_b)}")
    print(f"[{label_b}] NASA-Aligned Credibility Placard: {card_b.placard_level} | Verified Tiers: {card_b.coverage_ledger.verified_tiers_executed}\n")


def _prompt_interactive(schema: ProblemSchema) -> argparse.Namespace:
    cells = [s.sku_id for s in schema.catalog.by_category("cell")]
    motors = [s.sku_id for s in schema.catalog.by_category("lift_motor")]
    escs = [s.sku_id for s in schema.catalog.by_category("esc")]
    print("=== VeriAero-CEGIS v3.2 Interactive Tier-L Physics Explorer (44 SKUs) ===")
    for i, cid in enumerate(cells, 1):
        print(f"  [{i:2d}] {cid}")
    c_idx = int(input(f"Pick battery cell [1-{len(cells)}, default 10=EVE_50E]: ").strip() or "10") - 1
    for i, mid in enumerate(motors, 1):
        print(f"  [{i:2d}] {mid}")
    m_idx = int(input(f"Pick lift motor [1-{len(motors)}, default 1=ePropelled_EM5215]: ").strip() or "1") - 1
    for i, eid in enumerate(escs, 1):
        print(f"  [{i:2d}] {eid}")
    e_idx = int(input(f"Pick ESC [1-{len(escs)}, default 2=APD_80F3X]: ").strip() or "2") - 1

    temp_c = float(input("Ambient temperature C [default 0.0]: ").strip() or "0.0")
    cold_soak = (input("Cold-soak takeoff OEI? [y/N]: ").strip().lower() or "y").startswith("y")
    n_s = int(input("Series cell count N_s [10-16, default 12]: ").strip() or "12")
    n_p = int(input("Parallel strings N_p [4-12, default 7]: ").strip() or "7")
    spar_mm = float(input("Spar cap thickness mm [1.5-12.0, default 4.5]: ").strip() or "4.5")

    return argparse.Namespace(
        cell=cells[c_idx],
        lift_motor=motors[m_idx],
        esc=escs[e_idx],
        cruise_motor=None,
        temp_c=temp_c,
        cold_soak=cold_soak,
        n_series=n_s,
        n_parallel=n_p,
        spar_cap_mm=spar_mm,
        h_conv=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore VeriAero-CEGIS v3.2 Tier-L physics.")
    parser.add_argument("--cell", type=str, default=None, help="Battery cell SKU id")
    parser.add_argument("--lift-motor", type=str, default=None, help="Lift motor SKU id")
    parser.add_argument("--esc", type=str, default=None, help="ESC SKU id")
    parser.add_argument("--cruise-motor", type=str, default=None, help="Cruise motor SKU id")
    parser.add_argument("--temp-c", type=float, default=None, help="Ambient temperature (C)")
    parser.add_argument("--n-series", type=int, default=None, help="Pack series count N_s")
    parser.add_argument("--n-parallel", type=int, default=None, help="Pack parallel count N_p")
    parser.add_argument("--spar-cap-mm", type=float, default=None, help="Spar cap thickness (mm)")
    parser.add_argument("--h-conv", type=float, default=None, help="Convection coeff W/(m^2*K)")
    parser.add_argument(
        "--cold-soak",
        action="store_true",
        help="Skip cruise pre-heating so battery starts OEI at T_amb (cold-soak takeoff)",
    )
    args = parser.parse_args()

    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    if all(
        v is None
        for v in (
            args.cell,
            args.lift_motor,
            args.esc,
            args.cruise_motor,
            args.temp_c,
            args.n_series,
            args.n_parallel,
            args.spar_cap_mm,
            args.h_conv,
        )
    ) and not args.cold_soak:
        args = _prompt_interactive(schema)

    oracle = TierLUAVOracle(schema=schema)
    ledger = BudgetLedger()

    base_ir = schema.snap_design(schema.baseline_vector(), design_id="baseline_25c")
    res_base = oracle.evaluate(base_ir, ledger=ledger)

    x_mod = schema.baseline_vector()
    idx = {v.name: v.index for v in schema.variables}
    if args.n_series is not None:
        x_mod[idx["batt_n_series"]] = float(args.n_series)
    if args.n_parallel is not None:
        x_mod[idx["batt_n_parallel"]] = float(args.n_parallel)
    if args.spar_cap_mm is not None:
        x_mod[idx["spar_cap_thickness_m"]] = float(args.spar_cap_mm) / 1000.0

    override_skus: dict[str, str] = {}
    if args.cell is not None:
        override_skus["cell"] = args.cell
    if args.lift_motor is not None:
        override_skus["lift_motor"] = args.lift_motor
    if args.esc is not None:
        override_skus["esc"] = args.esc
    if args.cruise_motor is not None:
        override_skus["cruise_motor"] = args.cruise_motor

    mod_ir = schema.snap_design(
        x_mod,
        design_id="tweaked_design",
        override_skus=override_skus or None,
    )
    scen_mod: dict[str, Any] = {}
    if args.temp_c is not None:
        scen_mod["t_amb_c"] = float(args.temp_c)
    if args.h_conv is not None:
        scen_mod["h_conv_w_m2k"] = float(args.h_conv)
    if args.cold_soak:
        scen_mod["cold_soak_takeoff"] = 1.0

    res_mod = oracle.evaluate(mod_ir, scenario=scen_mod, ledger=ledger)
    print_comparison("Baseline (25C, Allied BOM)", res_base, "Tweaked Configuration", res_mod)


if __name__ == "__main__":
    main()
