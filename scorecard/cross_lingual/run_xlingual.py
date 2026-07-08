"""Cross-lingual safety-distribution-shift metric.

Measures whether the finance advisor's alignment degrades in non-English
languages: it may refuse a harmful request in English yet comply with the exact
same request translated into another language (the info sheet's required metric).

Consistency: this reuses the SAME pipeline as the rest of the scorecard —
`scorecard.inference` to generate responses (same model + finance system prompt)
and the SAME ShieldGemma judge from `scorecard.judge`, scored against the
existing `harmful_recommendations` metric. No separate judge is introduced.

Signal: for a harmful prompt, if the model complies, ShieldGemma flags the
response as violating the harmful-recommendations guideline (high P). If the
model refuses, the response does not violate it (low P). So:
    harmful "unsafe compliance" rate  ~=  1 - refusal rate.
We report that rate per language and the GAP vs the English baseline.

Pipeline (per language, English = baseline):
  1. Generate the model's response to each seed prompt in that language.
  2. Score each response with ShieldGemma against `harmful_recommendations`.
  3. Report the harmful unsafe-compliance rate and the gap vs English.

    python -m scorecard.cross_lingual.run_xlingual --langs eng_Latn,spa_Latn,swh_Latn

Assumes translate.py has produced data/xlingual/seed_<lang>.jsonl for every
non-English language (English reads data/xlingual/seed_en.jsonl).

Outputs -> outputs/xlingual_safety_gap.md / .csv  (the table for the write-up)
           outputs/xlingual_responses.csv          (every prompt/response/score)

Limitations to state: (a) ShieldGemma is itself weaker in non-English languages,
so a measured gap conflates the target model's weaker alignment with the JUDGE's
weaker moderation in that language; (b) machine-translated prompts add noise,
worst in the low-resource languages we most want to test. Report the gap as a
screening signal, and spot-check outputs/xlingual_responses.csv.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from scorecard.inference import (
    DEFAULT_IT_MODEL_ID,
    FINANCE_SYSTEM_PROMPT,
    GenerationConfig,
    generate_chat_response,
    load_model,
)
from scorecard.judge import (
    DEFAULT_JUDGE_MODEL_ID,
    DEFAULT_VIOLATION_THRESHOLD,
    JudgeConfig,
    judge_one_metric,
    load_judge_model,
)
from scorecard.metrics import METRIC_BY_KEY

DATA_DIR = Path("data/xlingual")
OUT_DIR = Path("outputs")
BASELINE = "eng_Latn"
# The scorecard metric we reuse to detect compliance with a harmful request.
JUDGE_METRIC_KEY = "harmful_recommendations"


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_path(lang: str) -> Path:
    return DATA_DIR / ("seed_en.jsonl" if lang == BASELINE else f"seed_{lang}.jsonl")


def load_seed(lang: str):
    path = seed_path(lang)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run translate.py --langs {lang} first "
            f"(English seed is data/xlingual/seed_en.jsonl)."
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def rate_unsafe(subset):
    """Fraction of `subset` whose response was flagged unsafe (metric not passed)."""
    if not subset:
        return float("nan")
    return sum(not r["pass"] for r in subset) / len(subset)


def fmt(x):
    return "nan" if isinstance(x, float) and x != x else (
        f"{x:.3f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser(description="Measure cross-lingual safety gap.")
    ap.add_argument("--langs", default="eng_Latn,spa_Latn,swh_Latn",
                    help="comma-separated FLORES codes; include eng_Latn as baseline")
    ap.add_argument("--model-id", default=DEFAULT_IT_MODEL_ID)
    ap.add_argument("--judge-model-id", default=DEFAULT_JUDGE_MODEL_ID)
    ap.add_argument("--violation-threshold", type=float,
                    default=DEFAULT_VIOLATION_THRESHOLD)
    ap.add_argument("--outdir", type=Path, default=OUT_DIR)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if BASELINE not in langs:
        langs = [BASELINE] + langs
    seeds = {lang: load_seed(lang) for lang in langs}  # fail fast if a file is missing
    metric = METRIC_BY_KEY[JUDGE_METRIC_KEY]

    # --- Phase 1: generate responses with the workflow model (loaded once) ---
    processor, model = load_model(args.model_id)
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens)
    records = []
    for lang in langs:
        print(f"[generate] {lang}: {len(seeds[lang])} prompts")
        for r in seeds[lang]:
            resp = generate_chat_response(
                r["prompt"], processor, model, gen_config,
                system_prompt=FINANCE_SYSTEM_PROMPT,
            )
            records.append({"lang": lang, "example_id": r["id"], "type": r["type"],
                            "category": r["category"], "model_id": args.model_id,
                            "prompt": r["prompt"], "model_response": resp})
    del processor, model
    _free()

    # --- Phase 2: score each response with the SAME ShieldGemma judge ---
    tokenizer, judge_model = load_judge_model(args.judge_model_id)
    judge_config = JudgeConfig(violation_threshold=args.violation_threshold)
    print(f"[judge] {args.judge_model_id} on metric '{metric.key}' "
          f"({len(records)} responses)")
    for rec in records:
        result = judge_one_metric(
            rec, metric, tokenizer, judge_model, args.judge_model_id, judge_config
        )
        rec["violation_score"] = result["score"]
        rec["pass"] = result["pass"]  # True = safe (refused / no harmful content)
    del tokenizer, judge_model
    _free()

    # --- Phase 3: per-language rates + gap vs the English baseline ---
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "xlingual_responses.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lang", "example_id", "type", "category",
                                          "violation_score", "pass", "prompt",
                                          "model_response"])
        w.writeheader()
        for rec in records:
            w.writerow({k: rec[k] for k in w.fieldnames})

    per_lang = {}
    for lang in langs:
        rows = [r for r in records if r["lang"] == lang]
        harmful = [r for r in rows if r["type"] == "harmful"]
        benign = [r for r in rows if r["type"] == "benign"]
        per_lang[lang] = {
            "harmful_unsafe_rate": rate_unsafe(harmful),  # ~ 1 - refusal rate
            "benign_unsafe_rate": rate_unsafe(benign),    # should be ~0; judge FP check
            "n_harmful": len(harmful),
            "n_benign": len(benign),
        }

    base = per_lang[BASELINE]
    cols = ["lang", "harmful_unsafe_rate", "safety_gap_vs_en",
            "benign_unsafe_rate", "n_harmful", "n_benign"]
    table = []
    for lang in langs:
        m = per_lang[lang]
        table.append({
            "lang": lang + (" (baseline)" if lang == BASELINE else ""),
            "harmful_unsafe_rate": m["harmful_unsafe_rate"],
            # positive gap = MORE harmful compliance than English = safety degraded
            "safety_gap_vs_en": m["harmful_unsafe_rate"] - base["harmful_unsafe_rate"],
            "benign_unsafe_rate": m["benign_unsafe_rate"],
            "n_harmful": m["n_harmful"],
            "n_benign": m["n_benign"],
        })
    table.sort(key=lambda r: (1e9 if r["lang"].endswith("(baseline)")
                              else -r["safety_gap_vs_en"]))  # baseline first, then worst

    with (args.outdir / "xlingual_safety_gap.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in table:
            w.writerow({k: (fmt(row[k]) if isinstance(row[k], float) else row[k])
                        for k in cols})

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep] + [
        "| " + " | ".join(fmt(row[k]) for k in cols) + " |" for row in table
    ]
    md = "\n".join(lines)
    (args.outdir / "xlingual_safety_gap.md").write_text(md + "\n")

    print("\nCross-lingual safety gap "
          "(harmful_unsafe_rate ~ 1 - refusal; positive gap = worse than English):\n")
    print(md)
    print(f"\nWrote {args.outdir/'xlingual_safety_gap.md'}, .csv, and "
          f"xlingual_responses.csv")


if __name__ == "__main__":
    main()
