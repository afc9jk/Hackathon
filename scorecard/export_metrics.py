"""Export scorecard metric definitions for review and presentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scorecard.metrics import metric_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export scorecard metric definitions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metric_definitions.csv"),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows()).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
