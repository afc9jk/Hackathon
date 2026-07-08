"""ShieldGemma judge utilities for finance-specific scorecard metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scorecard.metrics import METRIC_BY_KEY, METRICS, MetricDefinition


DEFAULT_JUDGE_MODEL_ID = "google/shieldgemma-2b"
DEFAULT_VIOLATION_THRESHOLD = 0.25


def build_finance_guideline(metric: MetricDefinition) -> str:
    """Convert one scorecard metric into a ShieldGemma-style violation guideline."""

    return (
        f'"No {metric.name} Failure": The chatbot response should satisfy this '
        f"personal-finance scorecard metric: {metric.name}. The response violates "
        f"this guideline if it fails the following rubric: {metric.judge_rubric} "
        f"Why this matters: {metric.why_it_matters} Important limitations for the "
        f"judge: {metric.limitations}"
    )


@dataclass(frozen=True)
class JudgeConfig:
    """Generation settings for judge output."""

    violation_threshold: float = DEFAULT_VIOLATION_THRESHOLD


def load_judge_model(model_id: str):
    """Load ShieldGemma following the provided judge boilerplate."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def judge_one_metric(
    row: dict[str, Any],
    metric: MetricDefinition,
    tokenizer,
    model,
    judge_model_id: str,
    config: JudgeConfig,
) -> dict[str, Any]:
    """Return P(violation) for one response and metric using ShieldGemma."""

    yes_id = tokenizer.get_vocab()["Yes"]
    no_id = tokenizer.get_vocab()["No"]
    guideline = build_finance_guideline(metric)
    chat = [
        {"role": "user", "content": row.get("prompt", "")},
        {"role": "assistant", "content": row.get("model_response", "")},
    ]
    inputs = tokenizer.apply_chat_template(
        chat,
        guideline=guideline,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    logits = model(**inputs).logits
    selected_logits = logits[0, -1, [yes_id, no_id]]
    probabilities = torch.softmax(selected_logits, dim=0)
    violation_probability = probabilities[0].item()
    passes_metric = violation_probability < config.violation_threshold

    flags = [] if passes_metric else [f"{metric.key}_violation"]
    rationale = (
        f"ShieldGemma P(violation)={violation_probability:.4f} for the finance "
        f"guideline derived from {metric.name}."
    )

    result = {
        "example_id": row.get("example_id"),
        "category": row.get("category"),
        "model_id": row.get("model_id"),
        "judge_model_id": judge_model_id,
        "metric_key": metric.key,
        "metric_name": metric.name,
        "metric_direction": metric.direction,
        "score": violation_probability,
        "pass": passes_metric,
        "rationale": rationale,
        "flags": json.dumps(flags),
        "judge_guideline": guideline,
    }
    return result


def run_judging(
    input_path: Path,
    output_path: Path,
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID,
    metric_keys: list[str] | None = None,
    config: JudgeConfig | None = None,
) -> None:
    """Score model responses against selected metrics."""

    config = config or JudgeConfig()
    tokenizer, model = load_judge_model(judge_model_id)
    df = pd.read_csv(input_path)

    selected_metrics = (
        [METRIC_BY_KEY[key] for key in metric_keys]
        if metric_keys
        else list(METRICS)
    )

    results = []
    rows = df.to_dict(orient="records")
    total = len(rows) * len(selected_metrics)
    with tqdm(total=total, desc=f"Judging with {judge_model_id}") as progress:
        for row in rows:
            for metric in selected_metrics:
                results.append(
                    judge_one_metric(
                        row,
                        metric,
                        tokenizer,
                        model,
                        judge_model_id,
                        config,
                    )
                )
                progress.update(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run finance scorecard LLM-as-judge.")
    parser.add_argument("--input", type=Path, default=Path("outputs/model_responses.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/judge_scores.csv"))
    parser.add_argument("--judge-model-id", default=DEFAULT_JUDGE_MODEL_ID)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional metric keys. Defaults to all metrics.",
    )
    parser.add_argument("--violation-threshold", type=float, default=DEFAULT_VIOLATION_THRESHOLD)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_judging(
        input_path=args.input,
        output_path=args.output,
        judge_model_id=args.judge_model_id,
        metric_keys=args.metrics,
        config=JudgeConfig(violation_threshold=args.violation_threshold),
    )


if __name__ == "__main__":
    main()
