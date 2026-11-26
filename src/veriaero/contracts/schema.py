"""Domain-agnostic declarative schema, catalog, compatibility, and oracle contracts.

Aligned verbatim with VeriAero-CEGIS v3.2 Architecture Reference (Sections 3.2, 6.1, 6.7):
- `ProblemSchema`: variables (`geometry | hardware | topology`), 4 catalog categories
  (`cell`, `lift_motor`, `esc`, `cruise_motor`), 3 Pareto objectives with Hypervolume
  reference points, constraints with `m_safe_scale`, and composable `CompatibilityGraph`.
- `DesignIR`: canonical vehicle IR holding `schema_version`, `x_norm` in [0,1]^d,
  `s_snap`, `selected_skus`, `topology`, `airfoil_cst`, and `export(fmt)`.
- `SubsystemCatalog`: origin-audited SKUs (`ALLIED`, `PRC`, `US_DESIGNED_PRC_MFG`, `OTHER`)
  with mandatory `price_source` (`retail_list` | `parametric_estimate`) and integrated-ESC
  metadata (`has_integrated_esc`, `max_series_s`).
- `DisciplineOracle` Protocol: `name`, `tier`, `schema_id`, `approx_cost_s`,
  `provides_outputs`, `sub_oracles`, `evaluate(...)`.
"""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field, model_validator

from veriaero.contracts.types import BudgetLedger, EvalResult, OracleTier


class VariableKind(str, Enum):
    """Design variable representation type in the flat optimization vector."""

    CONTINUOUS = "CONTINUOUS"
    INTEGER = "INTEGER"
    CATALOG_FEATURE = "CATALOG_FEATURE"


class VariableSpec(BaseModel):
    """Metadata and bounds for a single dimension of the flat state vector."""

    name: str
    index: int
    kind: VariableKind
    lower: float
    upper: float
    baseline: float
    unit: str = "-"
    group: str = "geometry"
    catalog_category: str | None = None
    catalog_field: str | None = None
    description: str = ""

    @model_validator(mode="after")
    def _validate_bounds(self) -> VariableSpec:
        if self.lower > self.upper:
            raise ValueError(f"Variable {self.name}: lower ({self.lower}) > upper ({self.upper})")
        if not (self.lower <= self.baseline <= self.upper):
            raise ValueError(
                f"Variable {self.name}: baseline ({self.baseline}) outside [{self.lower}, {self.upper}]"
            )
        if self.kind == VariableKind.CATALOG_FEATURE and not self.catalog_category:
            raise ValueError(f"CATALOG_FEATURE variable {self.name} must define catalog_category")
        return self


class ObjectiveSpec(BaseModel):
    """Objective function metadata with Hypervolume reference point."""

    name: str
    sense: Literal["maximize", "minimize"]
    unit: str = "-"
    hv_ref_point: float | None = None
    description: str = ""


class ConstraintSpec(BaseModel):
    """Constraint metadata (strict g(x) <= tolerance; m_safe = min_m(-g_m / m_safe_scale))."""

    name: str
    unit: str = "-"
    tolerance: float = 0.0
    m_safe_scale: float = 1.0
    description: str = ""


class SKUItem(BaseModel):
    """A concrete physical COTS/NDAA hardware component with origin & pricing pedigree."""

    sku_id: str
    category: str
    display_name: str
    manufacturer: str
    origin_country: str
    origin_tier: Literal["ALLIED", "PRC", "US_DESIGNED_PRC_MFG", "OTHER"]
    unit_cost_usd: float
    price_source: Literal["retail_list", "parametric_estimate"] = "retail_list"
    has_integrated_esc: bool = False
    max_series_s: int | None = None
    specs: dict[str, float] = Field(default_factory=dict)
    provenance_note: str = ""


class SubsystemCatalog(BaseModel):
    """Origin-audited catalog of discrete physical hardware SKUs."""

    items: dict[str, SKUItem] = Field(default_factory=dict)

    def get(self, sku_id: str) -> SKUItem:
        if sku_id not in self.items:
            raise KeyError(f"Unknown SKU id: {sku_id}")
        return self.items[sku_id]

    def categories(self) -> list[str]:
        return sorted({item.category for item in self.items.values()})

    def by_category(
        self,
        category: str,
        allowed_origins: set[str] | None = None,
        excluded_skus: set[str] | None = None,
    ) -> list[SKUItem]:
        matches = [
            item
            for item in self.items.values()
            if item.category == category
            and (allowed_origins is None or item.origin_tier in allowed_origins)
            and (excluded_skus is None or item.sku_id not in excluded_skus)
        ]
        return sorted(matches, key=lambda item: item.sku_id)

    def snap_feature_vector(
        self,
        category: str,
        raw_features: dict[str, float],
        allowed_origins: set[str] | None = None,
        excluded_skus: set[str] | None = None,
        feature_scales: dict[str, float] | None = None,
    ) -> tuple[SKUItem, float]:
        """Snap a continuous feature dict to the nearest permitted catalog SKU."""
        candidates = self.by_category(
            category,
            allowed_origins=allowed_origins,
            excluded_skus=excluded_skus,
        )
        if not candidates:
            raise ValueError(
                f"No catalog SKUs found for category='{category}' with origins={allowed_origins}"
            )

        best_item: SKUItem | None = None
        best_dist = float("inf")
        for item in candidates:
            sq_sum = 0.0
            for field, target_val in raw_features.items():
                if field not in item.specs:
                    raise KeyError(
                        f"SKU '{item.sku_id}' missing required catalog field '{field}'"
                    )
                scale = (feature_scales or {}).get(field, 1.0)
                if scale <= 0.0:
                    scale = 1.0
                diff = (item.specs[field] - target_val) / scale
                sq_sum += diff * diff
            dist = math.sqrt(sq_sum)
            if dist < best_dist:
                best_dist = dist
                best_item = item

        assert best_item is not None
        return best_item, best_dist


class CompatibilityRule(BaseModel):
    """Declarative cross-subsystem compatibility predicate."""

    rule_id: str
    rule_type: Literal[
        "voltage_window",
        "integrated_esc_cap",
        "motor_kv_voltage",
        "current_limit",
        "origin_policy",
        "blocked_pair",
    ]
    description: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class CompatibilityGraph(BaseModel):
    """Composable compatibility checker across selected SKUs and (N_s, N_p) topology."""

    rules: list[CompatibilityRule] = Field(default_factory=list)
    blocked_pairs: list[tuple[str, str]] = Field(default_factory=list)

    def check(
        self,
        selected_skus: dict[str, SKUItem],
        params: dict[str, float],
        allowed_origins: set[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Evaluate all declarative compatibility predicates and blocked SKU pairs."""
        errors: list[str] = []

        selected_ids = {item.sku_id for item in selected_skus.values()}
        for a, b in self.blocked_pairs:
            if a in selected_ids and b in selected_ids:
                errors.append(f"Blocked SKU pair selected: ({a}, {b})")

        if allowed_origins is not None:
            lift_motor = selected_skus.get("lift_motor")
            skip_esc_origin = bool(lift_motor and lift_motor.has_integrated_esc)
            for cat, item in selected_skus.items():
                if cat == "esc" and skip_esc_origin:
                    continue
                if item.origin_tier not in allowed_origins:
                    errors.append(
                        f"Origin policy violation for {cat} '{item.sku_id}': "
                        f"origin_tier='{item.origin_tier}' not in {sorted(allowed_origins)}"
                    )

        for rule in self.rules:
            p = rule.params
            if rule.rule_type == "voltage_window":
                src_cat = p["source_category"]
                load_cat = p["load_category"]
                if src_cat not in selected_skus or load_cat not in selected_skus:
                    continue
                if p.get("skip_if_integrated_esc", False):
                    lm = selected_skus.get("lift_motor")
                    if lm is not None and lm.has_integrated_esc:
                        continue

                src_sku = selected_skus[src_cat]
                load_sku = selected_skus[load_cat]
                n_series = params[p["series_var"]]

                v_nom_pack = n_series * src_sku.specs[p["cell_v_nom_field"]]
                v_max_pack = n_series * src_sku.specs[p["cell_v_max_field"]]
                v_min_load = load_sku.specs[p["load_v_min_field"]]
                v_max_load = load_sku.specs[p["load_v_max_field"]]

                if v_nom_pack < v_min_load - 1e-6:
                    errors.append(
                        f"[{rule.rule_id}] Nominal pack voltage {v_nom_pack:.2f} V < "
                        f"{load_sku.sku_id} min {v_min_load:.2f} V"
                    )
                if v_max_pack > v_max_load + 1e-6:
                    errors.append(
                        f"[{rule.rule_id}] Max charge pack voltage {v_max_pack:.2f} V > "
                        f"{load_sku.sku_id} max {v_max_load:.2f} V"
                    )

            elif rule.rule_type == "integrated_esc_cap":
                motor_cat = p.get("motor_category", "lift_motor")
                if motor_cat not in selected_skus:
                    continue
                motor_sku = selected_skus[motor_cat]
                n_series = params[p["series_var"]]
                if motor_sku.max_series_s is not None and n_series > motor_sku.max_series_s + 1e-6:
                    errors.append(
                        f"[{rule.rule_id}] {motor_sku.sku_id} caps pack topology at "
                        f"{motor_sku.max_series_s}S (got {int(round(n_series))}S)"
                    )

            elif rule.rule_type == "motor_kv_voltage":
                motor_cat = p["motor_category"]
                cell_cat = p.get("cell_category", "cell")
                if motor_cat not in selected_skus or cell_cat not in selected_skus:
                    continue
                motor_sku = selected_skus[motor_cat]
                cell_sku = selected_skus[cell_cat]
                n_series = params[p["series_var"]]
                v_max_pack = n_series * cell_sku.specs[p["cell_v_max_field"]]
                if "v_max_v" in motor_sku.specs and v_max_pack > motor_sku.specs["v_max_v"] + 1e-6:
                    errors.append(
                        f"[{rule.rule_id}] Pack {v_max_pack:.2f} V exceeds {motor_sku.sku_id} "
                        f"max voltage {motor_sku.specs['v_max_v']:.2f} V"
                    )

            elif rule.rule_type == "origin_policy":
                rule_origins = set(p.get("allowed_origins", []))
                if rule_origins:
                    for cat, item in selected_skus.items():
                        if item.origin_tier not in rule_origins:
                            errors.append(
                                f"[{rule.rule_id}] {cat} '{item.sku_id}' origin '{item.origin_tier}' disallowed"
                            )

            elif rule.rule_type == "blocked_pair":
                a, b = p["sku_a"], p["sku_b"]
                if a in selected_ids and b in selected_ids:
                    errors.append(f"[{rule.rule_id}] Blocked pair ({a}, {b})")

        return (len(errors) == 0), errors


class DesignIR(BaseModel):
    """Canonical domain-agnostic intermediate representation of a snapped candidate design."""

    design_id: str
    schema_id: str
    schema_version: str = "3.2.0"
    x_norm: list[float] = Field(default_factory=list)
    raw_vector: list[float]
    snapped_vector: list[float]
    params: dict[str, float]
    selected_skus: dict[str, str]
    topology: dict[str, int] = Field(default_factory=dict)
    airfoil_cst: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def export(self, fmt: str, out_path: str | Path | None = None) -> str:
        """Export DesignIR to '.json' or '.dat' (airfoil coords); post-Dec formats stubbed."""
        norm_fmt = fmt.lower().lstrip(".")
        if norm_fmt == "json":
            payload = self.model_dump_json(indent=2)
        elif norm_fmt == "dat":
            from veriaero.physics.airfoil_cst import kulfan_dict_to_coordinates

            if not self.airfoil_cst:
                raise ValueError("DesignIR has no airfoil_cst payload to export as .dat")
            coords = kulfan_dict_to_coordinates(self.airfoil_cst)
            lines = [f"{self.design_id}"] + [f" {x:10.6f} {y:10.6f}" for x, y in coords]
            payload = "\n".join(lines) + "\n"
        elif norm_fmt in ("vsp3", "inp", "step"):
            raise NotImplementedError(
                f"Post-December Tier-H3 export format '.{norm_fmt}' is intentionally deferred past Dec 23."
            )
        else:
            raise ValueError(f"Unsupported DesignIR export format: '{fmt}'")

        if out_path is not None:
            Path(out_path).write_text(payload, encoding="utf-8")
        return payload


class ProblemSchema(BaseModel):
    """Declarative specification of a design space, catalog, and constraints."""

    schema_id: str
    version: str
    domain: str
    description: str = ""
    catalog_categories_order: list[str] = Field(default_factory=list)
    variables: list[VariableSpec]
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec]
    catalog: SubsystemCatalog
    compatibility: CompatibilityGraph = Field(default_factory=CompatibilityGraph)
    scenario_defaults: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_variables(self) -> ProblemSchema:
        indices = [v.index for v in self.variables]
        names = [v.name for v in self.variables]
        if indices != list(range(len(self.variables))):
            raise ValueError(
                f"Variable indices must be contiguous 0..{len(self.variables) - 1}, got {indices}"
            )
        if len(set(names)) != len(names):
            raise ValueError("Duplicate variable names detected in ProblemSchema")
        return self

    @property
    def dim(self) -> int:
        return len(self.variables)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lowers = np.array([v.lower for v in self.variables], dtype=float)
        uppers = np.array([v.upper for v in self.variables], dtype=float)
        return lowers, uppers

    def baseline_vector(self) -> np.ndarray:
        return np.array([v.baseline for v in self.variables], dtype=float)

    def normalize(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        lowers, uppers = self.bounds()
        span = np.maximum(uppers - lowers, 1e-12)
        return np.clip((np.asarray(x, dtype=float) - lowers) / span, 0.0, 1.0)

    def denormalize(self, x_norm: Sequence[float] | np.ndarray) -> np.ndarray:
        lowers, uppers = self.bounds()
        return lowers + np.clip(np.asarray(x_norm, dtype=float), 0.0, 1.0) * (uppers - lowers)

    def decode_vector(self, x: Sequence[float] | np.ndarray) -> dict[str, float]:
        arr = np.asarray(x, dtype=float)
        if arr.shape != (self.dim,):
            raise ValueError(f"Expected vector of length {self.dim}, got shape {arr.shape}")
        return {v.name: float(arr[v.index]) for v in self.variables}

    def encode_dict(self, values: dict[str, float]) -> np.ndarray:
        vec = self.baseline_vector()
        for v in self.variables:
            if v.name in values:
                vec[v.index] = float(values[v.name])
        return vec

    def catalog_categories(self) -> list[str]:
        if self.catalog_categories_order:
            return list(self.catalog_categories_order)
        cats: list[str] = []
        for v in self.variables:
            if v.kind == VariableKind.CATALOG_FEATURE and v.catalog_category:
                if v.catalog_category not in cats:
                    cats.append(v.catalog_category)
        return cats

    def compute_m_safe(self, constraints: dict[str, float]) -> float:
        """Compute continuous signed safety margin m_safe = min_m (-g_m / scale_m)."""
        if not constraints:
            return 0.0
        scale_map = {c.name: max(c.m_safe_scale, 1e-9) for c in self.constraints}
        margins = [
            -val / scale_map.get(name, 1.0)
            for name, val in constraints.items()
            if not (name == "compat_violation_count" and val <= 0.0)
        ]
        if not margins:
            return 0.0
        return float(min(margins))

    def snap_design(
        self,
        x: Sequence[float] | np.ndarray,
        design_id: str = "design_0",
        allowed_origins: set[str] | None = None,
        excluded_skus: set[str] | None = None,
        override_skus: dict[str, str] | None = None,
    ) -> DesignIR:
        """Project raw continuous vector to bounds, integer topology, and nearest COTS SKUs."""
        raw = np.asarray(x, dtype=float)
        if raw.shape != (self.dim,):
            raise ValueError(f"Expected vector of length {self.dim}, got shape {raw.shape}")

        lowers, uppers = self.bounds()
        snapped = np.clip(raw.copy(), lowers, uppers)

        topology: dict[str, int] = {}
        for v in self.variables:
            if v.kind == VariableKind.INTEGER:
                ival = int(round(snapped[v.index]))
                snapped[v.index] = float(ival)
                topology[v.name] = ival

        selected_skus: dict[str, SKUItem] = {}
        snap_distances: dict[str, float] = {}
        for cat in self.catalog_categories():
            cat_vars = [
                v
                for v in self.variables
                if v.kind == VariableKind.CATALOG_FEATURE and v.catalog_category == cat
            ]
            if override_skus and cat in override_skus:
                sku = self.catalog.get(override_skus[cat])
                if allowed_origins is not None and sku.origin_tier not in allowed_origins:
                    raise ValueError(
                        f"Override SKU '{sku.sku_id}' violates allowed_origins={allowed_origins}"
                    )
                if excluded_skus is not None and sku.sku_id in excluded_skus:
                    raise ValueError(f"Override SKU '{sku.sku_id}' is excluded by hole mask")
                dist = 0.0
            else:
                raw_feats: dict[str, float] = {}
                scales: dict[str, float] = {}
                for v in cat_vars:
                    field = v.catalog_field or v.name
                    raw_feats[field] = float(snapped[v.index])
                    scales[field] = max(v.upper - v.lower, 1e-9)
                sku, dist = self.catalog.snap_feature_vector(
                    category=cat,
                    raw_features=raw_feats,
                    allowed_origins=allowed_origins,
                    excluded_skus=excluded_skus,
                    feature_scales=scales,
                )

            selected_skus[cat] = sku
            snap_distances[cat] = dist
            for v in cat_vars:
                field = v.catalog_field or v.name
                snapped[v.index] = float(sku.specs[field])

        var_idx = {v.name: v.index for v in self.variables}
        if "normalized_bom_cost" in var_idx:
            bom_usd = self._estimate_bom_cost_usd(selected_skus, topology)
            snapped[var_idx["normalized_bom_cost"]] = float(
                np.clip((bom_usd - 500.0) / 6000.0, 0.0, 1.0)
            )

        params = self.decode_vector(snapped)
        compat_ok, compat_errors = self.compatibility.check(
            selected_skus=selected_skus,
            params=params,
            allowed_origins=allowed_origins,
        )

        airfoil_cst: dict[str, Any] = {}
        if all(f"airfoil_pca_{k}" in params for k in range(4)):
            from veriaero.physics.airfoil_cst import reconstruct_kulfan_from_pca

            u_pca = np.array([params[f"airfoil_pca_{k}"] for k in range(4)], dtype=float)
            airfoil_cst = reconstruct_kulfan_from_pca(u_pca)

        return DesignIR(
            design_id=design_id,
            schema_id=self.schema_id,
            schema_version=self.version,
            x_norm=[float(val) for val in self.normalize(snapped)],
            raw_vector=[float(val) for val in raw],
            snapped_vector=[float(val) for val in snapped],
            params=params,
            selected_skus={cat: sku.sku_id for cat, sku in selected_skus.items()},
            topology=topology,
            airfoil_cst=airfoil_cst,
            metadata={
                "snap_distances": snap_distances,
                "compat_ok": compat_ok,
                "compat_errors": compat_errors,
            },
        )

    @staticmethod
    def _estimate_bom_cost_usd(
        selected_skus: dict[str, SKUItem],
        topology: dict[str, int],
    ) -> float:
        n_cells = topology.get("batt_n_series", 12) * topology.get("batt_n_parallel", 7)
        cost = 0.0
        if "cell" in selected_skus:
            cost += n_cells * selected_skus["cell"].unit_cost_usd
        lift_motor = selected_skus.get("lift_motor")
        if lift_motor is not None:
            cost += 6.0 * lift_motor.unit_cost_usd
        esc = selected_skus.get("esc")
        if esc is not None:
            n_escs = 1.0 if (lift_motor and lift_motor.has_integrated_esc) else 7.0
            cost += n_escs * esc.unit_cost_usd
        if "cruise_motor" in selected_skus:
            cost += selected_skus["cruise_motor"].unit_cost_usd
        return cost

    @classmethod
    def from_json_file(cls, path: str | Path) -> ProblemSchema:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


@runtime_checkable
class DisciplineOracle(Protocol):
    """Plugin interface contract implemented by Tier-L, Tier-H1, Tier-H2, and sub-oracles."""

    @property
    def name(self) -> str: ...

    @property
    def tier(self) -> OracleTier: ...

    @property
    def schema_id(self) -> str: ...

    @property
    def approx_cost_s(self) -> float: ...

    @property
    def provides_outputs(self) -> tuple[str, ...]: ...

    @property
    def sub_oracles(self) -> tuple[DisciplineOracle, ...]: ...

    def evaluate(
        self,
        design: DesignIR,
        scenario: dict[str, float] | None = None,
        ledger: BudgetLedger | None = None,
    ) -> EvalResult: ...
