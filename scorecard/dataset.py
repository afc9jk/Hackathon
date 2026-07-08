"""Dataset loading and sampling utilities for the finance scorecard."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset


DATASET_ID = "Akhil-Theerthala/PersonalFinance_v2"
DEFAULT_SPLIT = "train"


@dataclass(frozen=True)
class EvalExample:
    """Normalized evaluation example used by the scorecard pipeline."""

    example_id: str
    category: str
    prompt: str
    reference_response: str
    reference_chain_of_thought: str
    source_dataset: str = DATASET_ID


def _first_present(row: dict, field_names: Iterable[str], default: str = "") -> str:
    """Return the first non-empty value from a row using candidate field names."""

    for field_name in field_names:
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def normalize_row(row: dict, index: int) -> EvalExample:
    """Convert a raw Hugging Face dataset row into the scorecard schema."""

    category = _first_present(row, ("category", "Category"), default="unknown")
    prompt = _first_present(row, ("query", "prompt", "question", "input"))
    reference_response = _first_present(
        row,
        ("response", "answer", "reference_response", "output"),
    )
    reference_chain_of_thought = _first_present(
        row,
        ("chain_of_thought", "cot", "reasoning", "rationale"),
    )

    safe_category = category.lower().replace(" ", "_").replace("/", "_")
    example_id = f"{safe_category}-{index:05d}"

    return EvalExample(
        example_id=example_id,
        category=category,
        prompt=prompt,
        reference_response=reference_response,
        reference_chain_of_thought=reference_chain_of_thought,
    )


def load_finance_dataset(split: str = DEFAULT_SPLIT) -> list[EvalExample]:
    """Load and normalize the PersonalFinance_v2 dataset."""

    dataset = load_dataset(DATASET_ID, split=split)
    return [normalize_row(row, index) for index, row in enumerate(dataset)]


def sample_by_category(
    examples: list[EvalExample],
    per_category: int,
    random_seed: int = 6051,
) -> list[EvalExample]:
    """Return a balanced random sample across categories."""

    if per_category < 1:
        raise ValueError("per_category must be at least 1")

    df = pd.DataFrame(asdict(example) for example in examples)
    sampled_groups = []
    for _, group in df.groupby("category", sort=True):
        sampled_groups.append(
            group.sample(
                n=min(per_category, len(group)),
                random_state=random_seed,
            )
        )

    sampled = (
        pd.concat(sampled_groups, ignore_index=True)
        .sort_values(["category", "example_id"])
        .reset_index(drop=True)
    )
    return [EvalExample(**row) for row in sampled.to_dict(orient="records")]


def write_examples(examples: list[EvalExample], output_path: Path) -> None:
    """Write examples as CSV or JSONL based on the output file extension."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(example) for example in examples]

    if output_path.suffix == ".csv":
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return

    if output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    raise ValueError("output_path must end in .csv or .jsonl")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for exporting a balanced evaluation sample."""

    parser = argparse.ArgumentParser(
        description="Create a balanced PersonalFinance_v2 evaluation sample."
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6051)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/personal_finance_eval_sample.csv"),
    )
    return parser


def main() -> None:
    """CLI entrypoint for creating a local evaluation sample."""

    args = build_arg_parser().parse_args()
    examples = load_finance_dataset(split=args.split)
    sample = sample_by_category(
        examples,
        per_category=args.per_category,
        random_seed=args.seed,
    )
    write_examples(sample, args.output)
    print(f"Wrote {len(sample)} examples to {args.output}")


if __name__ == "__main__":
    main()
