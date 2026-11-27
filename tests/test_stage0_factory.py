"""Tests for domain-agnostic Stage 0 Data Factory (`veriaero.data_factory`)."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from veriaero.contracts.schema import ProblemSchema
from veriaero.data_factory import Stage0Config, Stage0DataFactory
from veriaero.physics.tier_l_uav import TierLUAVOracle

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "configs" / "uav25kg_schema.json"


def _make_l(schema: ProblemSchema) -> TierLUAVOracle:
    return TierLUAVOracle(schema=schema)


def test_stage0_data_factory_domain_agnostic_firewall() -> None:
    """data_factory.py must NEVER import UAV physics, external solvers, or eval_quarantine."""
    target = REPO_ROOT / "src" / "veriaero" / "data_factory.py"
    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    forbidden = ("veriaero.physics", "veriaero.solvers", "eval_quarantine")
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == p or alias.name.startswith(p + ".") for p in forbidden):
                    violations.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod == p or mod.startswith(p + ".") for p in forbidden):
                violations.append(f"line {node.lineno}: from {mod}")
    assert not violations, "Domain-agnostic boundary breached:\n" + "\n".join(violations)


def test_stage0_factory_smoke_determinism_and_sha256_manifest(tmp_path: Path) -> None:
    """Stage 0 produces deterministic 80/10/10 Parquet corpus + SHA-256 locked manifest."""
    schema = ProblemSchema.from_json_file(SCHEMA_PATH)
    cfg = Stage0Config(n_total=40, n_h1_coupled=0, workers=1, seed=123)

    out_a = tmp_path / "run_a"
    man_a = Stage0DataFactory(schema=schema, tier_l_factory=_make_l, config=cfg).run(
        out_dir=out_a, manifest_path=out_a / "manifest.json"
    )
    out_b = tmp_path / "run_b"
    man_b = Stage0DataFactory(schema=schema, tier_l_factory=_make_l, config=cfg).run(
        out_dir=out_b, manifest_path=out_b / "manifest.json"
    )

    assert man_a["corpus_stats"]["n_total"] == 40
    assert man_a["corpus_stats"]["split_counts"] == {"train": 32, "val": 4, "test": 4}
    assert man_a["corpus_stats"]["y_feasible_count"] >= 1
    assert man_a["budget_ledger"]["total_calls"] == 40

    parquet_a = out_a / "stage0_corpus.parquet"
    parquet_b = out_b / "stage0_corpus.parquet"
    assert parquet_a.is_file()

    # SHA-256 lock verification
    sha = hashlib.sha256(parquet_a.read_bytes()).hexdigest()
    assert man_a["files_sha256"][str(parquet_a)] == sha

    # Deterministic feature & split reproducibility for identical seed
    df_a = pd.read_parquet(parquet_a)
    df_b = pd.read_parquet(parquet_b)
    assert list(df_a["split"]) == list(df_b["split"])
    assert list(df_a["y_feasible"]) == list(df_b["y_feasible"])
    assert np.allclose(
        df_a["x_norm_wing_area_m2"].to_numpy(),
        df_b["x_norm_wing_area_m2"].to_numpy(),
        atol=1e-7,
    )
    assert man_a["corpus_stats"]["y_feasible_count"] == man_b["corpus_stats"]["y_feasible_count"]
