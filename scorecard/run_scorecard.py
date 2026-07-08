"""End-to-end scorecard orchestration and aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scorecard.inference import (
    DEFAULT_IT_MODEL_ID,
    GenerationConfig,
    run_inference,
)
from scorecard.judge import DEFAULT_JUDGE_MODEL_ID, JudgeConfig, run_judging


def model_id_slug(model_id: str) -> str:
    """Create a filesystem-safe slug from a model id."""

    return model_id.replace("/", "__").replace(" ", "_")


def aggregate_scores(judge_scores_path: Path, summary_output_path: Path) -> None:
    """Aggregate per-example judge scores into metric and category summaries."""

    scores = pd.read_csv(judge_scores_path)
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores["pass_numeric"] = scores["pass"].astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0}
    )

    metric_summary = (
        scores.groupby(["model_id", "judge_model_id", "metric_key", "metric_name"])
        .agg(
            mean_score=("score", "mean"),
            pass_rate=("pass_numeric", "mean"),
            n=("score", "count"),
        )
        .reset_index()
    )
    metric_summary["category"] = "ALL"

    category_summary = (
        scores.groupby(
            ["model_id", "judge_model_id", "category", "metric_key", "metric_name"]
        )
        .agg(
            mean_score=("score", "mean"),
            pass_rate=("pass_numeric", "mean"),
            n=("score", "count"),
        )
        .reset_index()
    )

    summary = pd.concat([metric_summary, category_summary], ignore_index=True)
    summary = summary[
        [
            "model_id",
            "judge_model_id",
            "category",
            "metric_key",
            "metric_name",
            "mean_score",
            "pass_rate",
            "n",
        ]
    ].sort_values(["category", "metric_key"])

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference, judging, and aggregation.")
    parser.add_argument("--eval-data", type=Path, default=Path("data/personal_finance_eval_sample.csv"))
    parser.add_argument("--responses-output", type=Path, default=Path("outputs/model_responses.csv"))
    parser.add_argument("--judge-output", type=Path, default=Path("outputs/judge_scores.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/scorecard_summary.csv"))
    parser.add_argument("--model-id", default=DEFAULT_IT_MODEL_ID)
    parser.add_argument("--model-kind", choices=("base", "chat"), default="chat")
    parser.add_argument("--judge-model-id", default=DEFAULT_JUDGE_MODEL_ID)
    parser.add_argument(
        "--judge-model-ids",
        nargs="*",
        default=None,
        help=(
            "Optional list of judge model ids. If provided, each judge runs and the "
            "scores are combined into --judge-output."
        ),
    )
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-judging", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--violation-threshold", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.skip_inference:
        run_inference(
            input_path=args.eval_data,
            output_path=args.responses_output,
            model_id=args.model_id,
            model_kind=args.model_kind,
            config=GenerationConfig(max_new_tokens=args.max_new_tokens),
        )

    judge_model_ids = args.judge_model_ids or [args.judge_model_id]

    if not args.skip_judging:
        judge_outputs = []
        for judge_model_id in judge_model_ids:
            if len(judge_model_ids) == 1:
                per_judge_output = args.judge_output
            else:
                per_judge_output = args.judge_output.with_name(
                    f"{args.judge_output.stem}_{model_id_slug(judge_model_id)}"
                    f"{args.judge_output.suffix}"
                )

            run_judging(
                input_path=args.responses_output,
                output_path=per_judge_output,
                judge_model_id=judge_model_id,
                config=JudgeConfig(violation_threshold=args.violation_threshold),
            )
            judge_outputs.append(per_judge_output)

        if len(judge_outputs) > 1:
            combined = pd.concat(
                [pd.read_csv(path) for path in judge_outputs],
                ignore_index=True,
            )
            args.judge_output.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(args.judge_output, index=False)

    aggregate_scores(
        judge_scores_path=args.judge_output,
        summary_output_path=args.summary_output,
    )


if __name__ == "__main__":
    main()
