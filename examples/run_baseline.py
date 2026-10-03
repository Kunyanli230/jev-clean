"""Compatibility entry point for the packaged, explicitly rule-only baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from idac.baseline import BaselineRunner, run_baseline  # noqa: F401
from idac.config import ConfigError
from idac.storage import StorageError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed rule baseline (no API calls).")
    parser.add_argument("--input", default="examples/data/dirty.csv")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--output", default="runs/baseline")
    args = parser.parse_args()
    try:
        summary = run_baseline(args.input, args.config, args.output)
    except (ConfigError, StorageError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"wrote {Path(args.output) / 'cleaned.csv'} "
        f"({summary['rows']} rows, {summary['versions']} commits)"
    )
    print("model calls: 0; probability distributions: N/A")
    if summary["hard_failures"]:
        raise SystemExit(f"hard validation failures: {summary['hard_failures']}")


if __name__ == "__main__":
    main()
