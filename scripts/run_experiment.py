"""Entry point for the main experiment and the exploratory adaptive set.

    python scripts/run_experiment.py [--max-cases N] [--dry-run] [--adaptive]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import run_experiment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="run only the first N cases (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="check config and data without calling any model")
    ap.add_argument("--adaptive", action="store_true",
                    help="run the small defense-aware set instead of the main set")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.adaptive:
        cfg["data"]["cases_file"] = cfg["data"]["adaptive_cases_file"]
        cfg["paths"]["results_dir"] = str(Path(cfg["paths"]["results_dir"]) / "adaptive")

    stats = run_experiment(cfg, max_cases=args.max_cases, dry_run=args.dry_run)
    if args.dry_run:
        print("dry-run complete — no models were called.")
        return
    print(f"\n{stats['records']} responses written.")
    print(stats["report"])


if __name__ == "__main__":
    main()
