"""Compare Tier-L analytical physics vs Tier-H1 OpenAeroStruct CFRP wingbox VLM+FEM."""

from __future__ import annotations

import argparse
from pathlib import Path

from veriaero.contracts.schema import ProblemSchema
from veriaero.physics.tier_h1_uav import run_openaerostruct_wing
from veriaero.physics.tier_l_uav import (
    compute_aero_polar,
    compute_geometry,
    compute_mass_and_bom,
    compute_root_bending_stress,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "uav25kg_schema.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Tier-L vs OpenAeroStruct CFRP wingbox.")
    parser.add_argument("--wing-area", type=float, default=1.08, help="Wing area S in m^2 (default 1.08)")
    parser.add_argument("--aspect-ratio", type=float, default=12.0, help="Aspect ratio AR (default 12.0)")
    parser.add_argument("--spar-cap-mm", type=float, default=4.5, help="Spar cap thickness mm (default 4.5)")
    parser.add_argument("--web-mm", type=float, default=2.0, help="Wingbox web thickness mm (default 2.0)")
    parser.add_argument("--skin-mm", type=float, default=1.2, help="Wingbox skin thickness mm (default 1.2)")
    args = parser.parse_args()

    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    x = schema.baseline_vector()
    idx = {v.name: v.index for v in schema.variables}
    x[idx["wing_area_m2"]] = args.wing_area
    x[idx["aspect_ratio"]] = args.aspect_ratio
    x[idx["spar_cap_thickness_m"]] = args.spar_cap_mm / 1000.0
    x[idx["web_thickness_m"]] = args.web_mm / 1000.0
    x[idx["skin_thickness_m"]] = args.skin_mm / 1000.0

    ir = schema.snap_design(x, design_id="compare_l_h1")
    skus = {cat: schema.catalog.get(sid) for cat, sid in ir.selected_skus.items()}
    geom = compute_geometry(ir.params, airfoil_cst=ir.airfoil_cst)
    mass = compute_mass_and_bom(ir.params, geom, skus)
    aero_l = compute_aero_polar(
        ir.params, geom, mass["m_total_kg"], ir.airfoil_cst, skus, schema.scenario_defaults
    )
    struct_l = compute_root_bending_stress(
        ir.params, geom, mass["m_total_kg"], mass["m_wing_kg"], aero_l["a_wing_per_rad"], schema.scenario_defaults
    )
    oas = run_openaerostruct_wing(
        params=ir.params,
        geom=geom,
        mass_kg=mass["m_total_kg"],
        m_wing_kg=mass["m_wing_kg"],
        airfoil_cst=ir.airfoil_cst,
        scenario=schema.scenario_defaults,
    )

    print("\n=== Tier-L (Analytical) vs Tier-H1 (OpenAeroStruct 2.12.1 CFRP Wingbox FEM) ===")
    print(
        f"Wing: S={args.wing_area:.2f} m^2 | AR={args.aspect_ratio:.1f} (b={geom['wing_span_m']:.2f} m) | "
        f"cap={args.spar_cap_mm:.1f} mm | MTOW={mass['m_total_kg']:.2f} kg"
    )
    print(f"  CL cruise:                Tier-L = {aero_l['cl_cruise']:.4f}   | Tier-H1 OAS = {oas['cl_oas']:.4f}")
    print(f"  3D Induced Drag CDi:      Tier-L = {aero_l['cd_induced']:.5f}  | Tier-H1 OAS = {oas['cdi_oas']:.5f}")
    print(
        f"  +3.0g Maneuver (FS=1.5):  Tier-L = {struct_l['sigma_maneuver_3g_fs15_mpa']:.1f} MPa | "
        f"Tier-H1 Wingbox = {oas['vonmises_maneuver_3g_fs15_mpa']:.1f} MPa"
    )
    print(
        f"  Pratt Gust ({oas['n_gust_limit']:.2f}g, FS=1.5): Tier-L = {struct_l['sigma_gust_pratt_fs15_mpa']:.1f} MPa | "
        f"Tier-H1 Wingbox = {oas['vonmises_gust_pratt_fs15_mpa']:.1f} MPa"
    )
    print(f"  1g Cruise Tip Deflect:    Tier-H1 OAS = {1000.0*oas['tip_disp_cruise_m']:.2f} mm")
    print(f"  Pratt Gust Tip Deflect:   Tier-H1 OAS = {1000.0*oas['tip_disp_gust_m']:.2f} mm\n")


if __name__ == "__main__":
    main()
