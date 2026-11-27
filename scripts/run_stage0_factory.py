"""CLI entrypoint for VeriAero-CEGIS Stage 0 Data Factory on C2-WSL2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from veriaero.contracts.schema import ProblemSchema
from veriaero.data_factory import Stage0Config, Stage0DataFactory
from veriaero.physics.tier_h1_uav import TierH1UAVOracle
from veriaero.physics.tier_l_uav import TierLUAVOracle


def make_tier_l_oracle(schema: ProblemSchema) -> TierLUAVOracle:
    return TierLUAVOracle(schema=schema)


def make_tier_h1_oracle(schema: ProblemSchema) -> TierH1UAVOracle:
    # Each worker process evaluates one design at a time; use 1 inner XFoil thread
    # when multi-processing across workers to prevent CPU oversubscription.
    return TierH1UAVOracle(schema=schema, xfoil_workers=1)


def main() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    parser = argparse.ArgumentParser(description="Run VeriAero-CEGIS v3.2 Stage 0 Data Factory")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/uav25kg_schema.json"),
    )
    parser.add_argument("--n-total", type=int, default=50_000)
    parser.add_argument("--n-h1", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/stage0"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/stage0/stage0_manifest.json"),
    )
    args = parser.parse_args()

    schema = ProblemSchema.from_json_file(args.schema)
    cfg = Stage0Config(
        n_total=args.n_total,
        n_h1_coupled=args.n_h1,
        seed=args.seed,
        workers=args.workers,
    )
    factory = Stage0DataFactory(
        schema=schema,
        tier_l_factory=make_tier_l_oracle,
        tier_h1_factory=make_tier_h1_oracle if args.n_h1 > 0 else None,
        config=cfg,
    )
    manifest = factory.run(out_dir=args.out_dir, manifest_path=args.manifest)
    cs = manifest["corpus_stats"]
    print(
        f"Stage 0 complete in {manifest['wall_time_s']:.1f}s on host={manifest['host_provenance']['host_id']}:\n"
        f"  Total={cs['n_total']} | Splits={cs['split_counts']} | "
        f"Feasible={cs['y_feasible_count']} ({100.0 * cs['y_feasible_frac']:.1f}%) | "
        f"All 44 SKUs covered={cs['all_skus_covered']}\n"
        f"  Tier-H1 anchors={manifest['tier_h1_anchors']}\n"
        f"  Manifest saved: {args.manifest}"
    )


if __name__ == "__main__":
    main()
