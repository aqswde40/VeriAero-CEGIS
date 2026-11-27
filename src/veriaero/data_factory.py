"""Domain-agnostic Stage 0 Data Factory for VeriAero-CEGIS v3.2.

Synthesizes a deterministic, SHA-256-locked multi-fidelity corpus (`stage0_corpus.parquet`
and optional coupled Tier-H1 anchors `stage0_tier_h1_anchors.parquet`) strictly through
`ProblemSchema`, `DesignIR`, `DisciplineOracle`, `BudgetLedger`, and `HostProvenance`.

NEVER imports `veriaero.physics` or `veriaero.solvers`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable
import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import qmc

from veriaero.contracts.schema import (
    DesignIR,
    DisciplineOracle,
    ProblemSchema,
    VariableKind,
)
from veriaero.contracts.types import (
    BudgetLedger,
    EvalResult,
    HostProvenance,
    OracleTier,
    SolverStatus,
)

OracleFactory = Callable[[ProblemSchema], DisciplineOracle]

DEFAULT_MISSION_BOUNDS: dict[str, tuple[float, float]] = {
    "payload_kg": (2.0, 4.5),
    "mtow_limit_kg": (23.0, 27.0),
    "t_amb_c": (-15.0, 40.0),
    "rho_air": (0.95, 1.32),
    "gust_u_de_mps": (6.0, 14.0),
    "cruise_speed_mps": (21.0, 27.0),
    "headwind_mps": (0.0, 8.0),
}

_WORKER_SCHEMA: ProblemSchema | None = None
_WORKER_ORACLE: Any = None


def _init_worker(schema_json: str, factory: OracleFactory) -> None:
    """Initialize worker-local schema and oracle instance once per process."""
    global _WORKER_SCHEMA, _WORKER_ORACLE
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    _WORKER_SCHEMA = ProblemSchema.model_validate_json(schema_json)
    _WORKER_ORACLE = factory(_WORKER_SCHEMA)


def _eval_worker_task(
    task: tuple[str, np.ndarray, dict[str, str] | None, dict[str, float], str],
) -> dict[str, Any]:
    """Snap DesignIR and evaluate the worker's DisciplineOracle on one candidate."""
    assert _WORKER_SCHEMA is not None and _WORKER_ORACLE is not None
    design_id, raw_x, override_skus, scenario, phase = task
    t0 = time.perf_counter()
    ir = _WORKER_SCHEMA.snap_design(
        raw_x,
        design_id=design_id,
        override_skus=override_skus or None,
    )
    res = _WORKER_ORACLE.evaluate(ir, scenario=scenario, ledger=None)
    elapsed_s = time.perf_counter() - t0
    return {
        "design_id": design_id,
        "oracle_name": _WORKER_ORACLE.name,
        "sample_phase": phase,
        "ir": ir,
        "scenario": scenario,
        "status": res.status.value,
        "y_feasible": int(res.status == SolverStatus.RESOLVED_PASS),
        "m_safe": float(res.m_safe) if res.m_safe is not None else float("nan"),
        "objectives": dict(res.objectives),
        "constraints": dict(res.constraints),
        "elapsed_s": float(elapsed_s),
        "failure_reason": res.failure_reason,
    }


def _eval_h1_worker_task(
    task: tuple[DesignIR, dict[str, float], str],
) -> dict[str, Any]:
    """Evaluate coupled Tier-H1 oracle on an already-snapped DesignIR."""
    assert _WORKER_ORACLE is not None
    ir, scenario, anchor_bucket = task
    t0 = time.perf_counter()
    res = _WORKER_ORACLE.evaluate(ir, scenario=scenario, ledger=None)
    elapsed_s = time.perf_counter() - t0
    return {
        "design_id": ir.design_id,
        "oracle_name": _WORKER_ORACLE.name,
        "anchor_bucket": anchor_bucket,
        "status": res.status.value,
        "y_feasible": int(res.status == SolverStatus.RESOLVED_PASS),
        "m_safe": float(res.m_safe) if res.m_safe is not None else float("nan"),
        "objectives": dict(res.objectives),
        "constraints": dict(res.constraints),
        "elapsed_s": float(elapsed_s),
        "failure_reason": res.failure_reason,
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Stage0Config(BaseModel):
    """Configuration for deterministic Stage 0 corpus synthesis."""

    n_total: int = Field(default=50_000, ge=10)
    n_global_frac: float = Field(default=0.60, gt=0.0, le=1.0)
    n_h1_coupled: int = Field(default=600, ge=0)
    nominal_mission_frac: float = Field(default=0.25, ge=0.0, le=1.0)
    seed: int = 42
    workers: int = Field(default=8, ge=1)
    mission_bounds: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: dict(DEFAULT_MISSION_BOUNDS)
    )


class Stage0DataFactory:
    """Domain-agnostic generator of deterministic, SHA-256-locked multi-fidelity corpora."""

    def __init__(
        self,
        schema: ProblemSchema,
        tier_l_factory: OracleFactory,
        tier_h1_factory: OracleFactory | None = None,
        config: Stage0Config | None = None,
    ) -> None:
        self.schema = schema
        self.tier_l_factory = tier_l_factory
        self.tier_h1_factory = tier_h1_factory
        self.config = config or Stage0Config()
        self._mission_keys = sorted(self.config.mission_bounds.keys())
        self._cats = self.schema.catalog_categories()
        self._sku_pools = {
            cat: [item.sku_id for item in self.schema.catalog.by_category(cat)]
            for cat in self._cats
        }
        self._geom_cont_indices = [
            v.index
            for v in self.schema.variables
            if v.kind == VariableKind.CONTINUOUS
            and v.catalog_category is None
            and not v.is_derived
        ]
        self._hw_topo_indices = [
            v.index
            for v in self.schema.variables
            if (v.kind == VariableKind.INTEGER or v.catalog_category is not None)
            and not v.is_derived
        ]

    def _build_mission(self, u_mission: np.ndarray, use_nominal: bool) -> dict[str, float]:
        scen = dict(self.schema.scenario_defaults)
        if use_nominal:
            return scen
        for k, key in enumerate(self._mission_keys):
            lo, hi = self.config.mission_bounds[key]
            scen[key] = float(lo + np.clip(u_mission[k], 0.0, 1.0) * (hi - lo))
        return scen

    def _run_tasks(
        self,
        tasks: list[Any],
        factory: OracleFactory,
        worker_fn: Callable[[Any], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.config.workers <= 1 or len(tasks) < 8:
            _init_worker(self.schema.model_dump_json(), factory)
            return [worker_fn(t) for t in tasks]

        schema_json = self.schema.model_dump_json()
        chunksize = max(1, min(64, len(tasks) // (self.config.workers * 4) or 1))
        with ProcessPoolExecutor(
            max_workers=self.config.workers,
            initializer=_init_worker,
            initargs=(schema_json, factory),
        ) as pool:
            return list(pool.map(worker_fn, tasks, chunksize=chunksize))

    def _record_ledger(self, ledger: BudgetLedger, tier: OracleTier, row: dict[str, Any]) -> None:
        ledger.debit(
            EvalResult(
                design_id=row["design_id"],
                oracle_name=row["oracle_name"],
                tier=tier,
                status=SolverStatus(row["status"]),
                wall_time_s=float(row["elapsed_s"]),
                failure_reason=row.get("failure_reason"),
            )
        )

    def _rows_to_corpus_df(
        self,
        records: list[dict[str, Any]],
        splits: np.ndarray,
    ) -> pd.DataFrame:
        var_names = [v.name for v in self.schema.variables]
        obj_names = [o.name for o in self.schema.objectives]
        con_names = [c.name for c in self.schema.constraints]

        data: dict[str, Any] = {
            "design_id": [r["design_id"] for r in records],
            "split": [str(s) for s in splits],
            "sample_phase": [r["sample_phase"] for r in records],
            "tier_l_status": [r["status"] for r in records],
            "y_feasible": np.array([r["y_feasible"] for r in records], dtype=np.int8),
            "m_safe": np.array([r["m_safe"] for r in records], dtype=np.float32),
            "compat_ok": np.array(
                [1 if r["ir"].metadata.get("compat_ok", True) else 0 for r in records],
                dtype=np.int8,
            ),
        }

        x_norm_mat = np.array([r["ir"].x_norm for r in records], dtype=np.float32)
        s_snap_mat = np.array([r["ir"].snapped_vector for r in records], dtype=np.float32)
        for j, name in enumerate(var_names):
            data[f"x_norm_{name}"] = x_norm_mat[:, j]
            data[f"s_snap_{name}"] = s_snap_mat[:, j]

        for cat in self._cats:
            data[f"sku_{cat}"] = [r["ir"].selected_skus.get(cat, "") for r in records]

        for key in self._mission_keys:
            data[f"c_mission_{key}"] = np.array(
                [float(r["scenario"].get(key, 0.0)) for r in records], dtype=np.float32
            )

        for name in obj_names:
            data[f"obj_{name}"] = np.array(
                [float(r["objectives"].get(name, float("nan"))) for r in records],
                dtype=np.float32,
            )

        for name in con_names:
            data[f"con_{name}"] = np.array(
                [float(r["constraints"].get(name, float("nan"))) for r in records],
                dtype=np.float32,
            )

        return pd.DataFrame(data)

    def _select_h1_indices(self, records: list[dict[str, Any]], n_select: int) -> list[tuple[int, str]]:
        if n_select <= 0:
            return []
        rng = np.random.default_rng(self.config.seed + 777)

        feas = [
            (i, float(r["m_safe"]))
            for i, r in enumerate(records)
            if r["status"] == SolverStatus.RESOLVED_PASS.value and math.isfinite(r["m_safe"])
        ]
        near_viol = [
            (i, float(r["m_safe"]))
            for i, r in enumerate(records)
            if r["status"] == SolverStatus.RESOLVED_VIOLATION.value
            and math.isfinite(r["m_safe"])
            and -0.40 <= float(r["m_safe"]) <= 0.0
        ]
        other_viol = [
            (i, float(r["m_safe"]))
            for i, r in enumerate(records)
            if r["status"] == SolverStatus.RESOLVED_VIOLATION.value
            and math.isfinite(r["m_safe"])
            and float(r["m_safe"]) < -0.40
        ]

        chosen: list[tuple[int, str]] = []
        used: set[int] = set()

        # 60% feasible across 4 m_safe quartiles
        n_feas_target = min(len(feas), int(round(0.60 * n_select)))
        if feas and n_feas_target > 0:
            feas_sorted = sorted(feas, key=lambda x: x[1])
            n_bins = 4
            per_bin = max(1, n_feas_target // n_bins)
            bins = np.array_split(np.array([idx for idx, _ in feas_sorted]), n_bins)
            for b_idx, arr in enumerate(bins):
                if arr.size == 0:
                    continue
                k = min(int(arr.size), per_bin + (1 if len(chosen) + per_bin < n_feas_target else 0))
                picks = rng.choice(arr, size=min(k, int(arr.size)), replace=False)
                for idx in picks.tolist():
                    if len(chosen) < n_feas_target and idx not in used:
                        chosen.append((int(idx), f"feasible_q{b_idx + 1}"))
                        used.add(int(idx))

        # Fill remaining from near-boundary violations first, then remaining feasible/violated
        n_viol_target = n_select - len(chosen)
        if near_viol and n_viol_target > 0:
            pool_idx = np.array([idx for idx, _ in near_viol if idx not in used])
            if pool_idx.size > 0:
                k = min(int(pool_idx.size), n_viol_target)
                picks = rng.choice(pool_idx, size=k, replace=False)
                for idx in picks.tolist():
                    chosen.append((int(idx), "near_boundary_violation"))
                    used.add(int(idx))

        remaining_pool = [
            (i, "feasible_fill") for i, _ in feas if i not in used
        ] + [
            (i, "violation_fill") for i, _ in other_viol if i not in used
        ]
        if len(chosen) < n_select and remaining_pool:
            rem_arr = np.arange(len(remaining_pool))
            need = min(len(remaining_pool), n_select - len(chosen))
            for k in rng.choice(rem_arr, size=need, replace=False).tolist():
                idx, tag = remaining_pool[int(k)]
                chosen.append((idx, tag))
                used.add(idx)

        return sorted(chosen, key=lambda pair: pair[0])

    def run(
        self,
        out_dir: str | Path,
        manifest_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Generate Stage 0 corpus + coupled Tier-H1 anchors and return locked manifest."""
        t_wall_start = time.perf_counter()
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        man_path = Path(manifest_path) if manifest_path else (out_path / "stage0_manifest.json")
        man_path.parent.mkdir(parents=True, exist_ok=True)

        n_total = self.config.n_total
        n_global = max(1, min(n_total, int(round(n_total * self.config.n_global_frac))))
        n_enrich = n_total - n_global
        n_m = len(self._mission_keys)
        rng = np.random.default_rng(self.config.seed)
        ledger = BudgetLedger()

        # Phase 1: Global Sobol over [0, 1]^d x c_mission
        sobol = qmc.Sobol(d=self.schema.dim + n_m, scramble=True, seed=self.config.seed)
        m_pow = math.ceil(math.log2(max(n_global, 2)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u_all = sobol.random_base2(m=m_pow)[:n_global]

        phase1_tasks: list[tuple[str, np.ndarray, dict[str, str] | None, dict[str, float], str]] = []
        for i in range(n_global):
            x_raw = self.schema.denormalize(u_all[i, : self.schema.dim])
            use_nom = rng.random() < self.config.nominal_mission_frac
            scen = self._build_mission(u_all[i, self.schema.dim :], use_nominal=use_nom)
            phase1_tasks.append((f"s0_g_{i:06d}", x_raw, None, scen, "global_sobol"))

        records = self._run_tasks(phase1_tasks, self.tier_l_factory, _eval_worker_task)
        for rec in records:
            self._record_ledger(ledger, OracleTier.L, rec)

        # Phase 2: Multi-scale feasible-interior + 100%-SKU catalog coverage enrichment
        if n_enrich > 0:
            feas_records = [r for r in records if r["y_feasible"] == 1]
            if feas_records:
                seed_x_norms = [np.asarray(r["ir"].x_norm, dtype=float) for r in feas_records]
                seed_skus = [dict(r["ir"].selected_skus) for r in feas_records]
            else:
                base_x = self.schema.baseline_vector()
                base_ir = self.schema.snap_design(base_x, design_id="seed_baseline")
                seed_x_norms = [np.asarray(base_ir.x_norm, dtype=float)]
                seed_skus = [dict(base_ir.selected_skus)]

            sigmas = (0.03, 0.07, 0.14)
            phase2_tasks: list[
                tuple[str, np.ndarray, dict[str, str] | None, dict[str, float], str]
            ] = []
            # Build balanced SKU schedule ensuring every SKU in every category is covered
            all_sku_pairs = [
                (cat, sku_id) for cat in self._cats for sku_id in self._sku_pools[cat]
            ]
            n_sku_force = max(len(all_sku_pairs), int(round(0.35 * n_enrich)))

            for i in range(n_enrich):
                s_idx = int(rng.integers(0, len(seed_x_norms)))
                x_u = seed_x_norms[s_idx].copy()
                sig = sigmas[i % len(sigmas)]
                if self._geom_cont_indices:
                    x_u[self._geom_cont_indices] += rng.normal(
                        0.0, sig, size=len(self._geom_cont_indices)
                    )

                override: dict[str, str] | None = None
                if i < n_sku_force and all_sku_pairs:
                    cat_force, sku_force = all_sku_pairs[i % len(all_sku_pairs)]
                    override = {cat: seed_skus[s_idx][cat] for cat in self._cats}
                    override[cat_force] = sku_force
                    # Perturb discrete topology lightly to discover compatible & boundary wiring
                    if rng.random() < 0.50:
                        for h_idx in self._hw_topo_indices:
                            if self.schema.variables[h_idx].kind == VariableKind.INTEGER:
                                x_u[h_idx] = float(rng.random())
                elif rng.random() < 0.35 and self._hw_topo_indices:
                    # Mix hardware/topology draws around feasible geometry
                    x_u[self._hw_topo_indices] = rng.random(len(self._hw_topo_indices))
                else:
                    # Keep compatible feasible hardware/topology and vary geometry + mission
                    override = dict(seed_skus[s_idx])

                x_raw = self.schema.denormalize(np.clip(x_u, 0.0, 1.0))
                use_nom = rng.random() < self.config.nominal_mission_frac
                scen = self._build_mission(rng.random(n_m), use_nominal=use_nom)
                phase2_tasks.append((f"s0_e_{i:06d}", x_raw, override, scen, "feasible_enrich"))

            enrich_records = self._run_tasks(phase2_tasks, self.tier_l_factory, _eval_worker_task)
            for rec in enrich_records:
                self._record_ledger(ledger, OracleTier.L, rec)
            records.extend(enrich_records)

        # Deterministic 80 / 10 / 10 split
        perm = np.random.default_rng(self.config.seed + 999).permutation(len(records))
        splits = np.empty(len(records), dtype=object)
        n_tr = int(round(0.80 * len(records)))
        n_va = int(round(0.10 * len(records)))
        splits[perm[:n_tr]] = "train"
        splits[perm[n_tr : n_tr + n_va]] = "val"
        splits[perm[n_tr + n_va :]] = "test"

        df_corpus = self._rows_to_corpus_df(records, splits)
        corpus_path = out_path / "stage0_corpus.parquet"
        df_corpus.to_parquet(corpus_path, index=False, compression="snappy")

        files_sha256 = {str(corpus_path): _sha256_file(corpus_path)}
        split_counts = {
            sp: int(np.sum(splits == sp)) for sp in ("train", "val", "test")
        }

        # Phase 3: Coupled Tier-H1 anchors (OpenAeroStruct + 8-GL XFoil)
        h1_summary: dict[str, Any] = {"enabled": False, "n_requested": self.config.n_h1_coupled}
        if self.config.n_h1_coupled > 0 and self.tier_h1_factory is not None:
            anchor_sel = self._select_h1_indices(records, self.config.n_h1_coupled)
            h1_tasks = [
                (records[idx]["ir"], records[idx]["scenario"], bucket)
                for idx, bucket in anchor_sel
            ]
            h1_records = self._run_tasks(h1_tasks, self.tier_h1_factory, _eval_h1_worker_task)
            for h1_rec in h1_records:
                self._record_ledger(ledger, OracleTier.H1, h1_rec)

            obj_names = [o.name for o in self.schema.objectives]
            h1_rows: list[dict[str, Any]] = []
            for (idx, _), h1_rec in zip(anchor_sel, h1_records):
                l_rec = records[idx]
                row_dict: dict[str, Any] = {
                    "design_id": l_rec["design_id"],
                    "split": str(splits[idx]),
                    "anchor_bucket": h1_rec["anchor_bucket"],
                    "tier_l_status": l_rec["status"],
                    "tier_h1_status": h1_rec["status"],
                    "tier_l_y_feasible": l_rec["y_feasible"],
                    "tier_h1_y_feasible": h1_rec["y_feasible"],
                    "tier_l_m_safe": float(l_rec["m_safe"]),
                    "tier_h1_m_safe": float(h1_rec["m_safe"]),
                    "r_m_safe": float(h1_rec["m_safe"] - l_rec["m_safe"]),
                    "h1_eval_time_s": float(h1_rec["elapsed_s"]),
                }
                for oname in obj_names:
                    l_v = float(l_rec["objectives"].get(oname, float("nan")))
                    h_v = float(h1_rec["objectives"].get(oname, float("nan")))
                    row_dict[f"l_obj_{oname}"] = l_v
                    row_dict[f"h1_obj_{oname}"] = h_v
                    row_dict[f"r_obj_{oname}"] = float(h_v - l_v)
                h1_rows.append(row_dict)

            df_h1 = pd.DataFrame(h1_rows)
            h1_path = out_path / "stage0_tier_h1_anchors.parquet"
            df_h1.to_parquet(h1_path, index=False, compression="snappy")
            files_sha256[str(h1_path)] = _sha256_file(h1_path)

            resolved_h1 = df_h1[df_h1["tier_h1_status"] != SolverStatus.UNRESOLVED_NUMERICAL.value]
            res_stats: dict[str, dict[str, float]] = {}
            for col in ["r_m_safe", *[f"r_obj_{o}" for o in obj_names]]:
                vals = resolved_h1[col].to_numpy(dtype=float)
                finite = vals[np.isfinite(vals)]
                if finite.size > 0:
                    res_stats[col] = {
                        "mean": float(np.mean(finite)),
                        "std": float(np.std(finite)),
                        "p05": float(np.percentile(finite, 5)),
                        "p95": float(np.percentile(finite, 95)),
                    }
            h1_summary = {
                "enabled": True,
                "n_anchors": int(len(df_h1)),
                "status_counts": {
                    k: int(v) for k, v in df_h1["tier_h1_status"].value_counts().items()
                },
                "y_feasible_count": int(df_h1["tier_h1_y_feasible"].sum()),
                "label_flip_count": int(
                    (df_h1["tier_l_y_feasible"] != df_h1["tier_h1_y_feasible"]).sum()
                ),
                "residual_stats": res_stats,
                "anchors_parquet": str(h1_path),
            }

        sku_coverage: dict[str, dict[str, int]] = {}
        for cat in self._cats:
            counts = df_corpus[f"sku_{cat}"].value_counts().to_dict()
            sku_coverage[cat] = {
                sku_id: int(counts.get(sku_id, 0)) for sku_id in self._sku_pools[cat]
            }

        n_feas = int(df_corpus["y_feasible"].sum())
        manifest = {
            "schema_id": self.schema.schema_id,
            "schema_version": self.schema.version,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host_provenance": HostProvenance.capture().model_dump(),
            "config": self.config.model_dump(),
            "wall_time_s": float(time.perf_counter() - t_wall_start),
            "corpus_stats": {
                "n_total": int(len(df_corpus)),
                "split_counts": split_counts,
                "status_counts": {
                    k: int(v) for k, v in df_corpus["tier_l_status"].value_counts().items()
                },
                "y_feasible_count": n_feas,
                "y_feasible_frac": float(n_feas / max(len(df_corpus), 1)),
                "compat_ok_count": int(df_corpus["compat_ok"].sum()),
                "sku_coverage_counts": sku_coverage,
                "all_skus_covered": all(
                    all(cnt > 0 for cnt in cat_map.values())
                    for cat_map in sku_coverage.values()
                ),
            },
            "tier_h1_anchors": h1_summary,
            "budget_ledger": ledger.summary(),
            "files_sha256": files_sha256,
        }
        man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
