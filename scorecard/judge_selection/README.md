# Judge Selection — which guard model do we trust?

This submodule picks the content-safety guard model for the finance-advisor
scorecard **empirically**, instead of assuming ShieldGemma is best.

## Why it exists (how it fits the rest of the package)

`scorecard/metrics.py` defines the metrics. Two of them —
`harmful_recommendations` and `unsupported_claims` — are scored by an
"LLM-as-judge policy check." But a judge is only as trustworthy as the model
behind it, and the off-the-shelf guard models (ShieldGemma, Llama Guard,
Qwen3Guard, Granite Guardian) are all trained on generic taxonomies —
**dangerous content, hate, harassment, sexual** — that say nothing about
financial crime. So we can't just grab one; we measure them.

Two distinct judge roles, don't conflate:
- **Role A — content-safety guard (binary safe/unsafe).** Powers the
  harmful-request metric. This submodule ranks these. We inject a custom
  `FINANCE_POLICY` (tax evasion, structuring, securities fraud, loan fraud,
  identity fraud, elder abuse, hiding assets…) so we measure finance harm,
  not generic harm.
- **Role B — open-ended rubric judge.** Scores the 1–5 quality metrics in
  `metrics.py` (accuracy, reasoning, completeness…). That's a general instruct
  LLM + rubric, and the info sheet asks for 3 of them with inter-judge
  agreement — handled elsewhere, not here.

## Run it (from repo root)

```bash
python -m scorecard.judge_selection.run_bakeoff                      # T4/Rivanna-friendly guards
python -m scorecard.judge_selection.run_bakeoff --only granite-guardian-3.2-5b   # heavy one, alone
```

Ranks by: (1) harm recall, (2) false-positive rate on correct refusals +
legitimate advice, (3) AUC. Outputs land in `outputs/`:
`judge_bakeoff.md` (ranked table for the write-up), `judge_bakeoff.csv`, and
`judge_per_example.csv` (per-row scores for eyeballing disagreements).

## Data

`data/judge_calibration_set.jsonl` — 30 labeled `(user_prompt,
assistant_response)` pairs: 14 unsafe (tax evasion, structuring,
securities/loan/identity fraud, elder abuse, one self-harm sanity check) and 16
safe (correct refusals of those same prompts + legitimate finance advice + two
informational edge cases). The refusals-labeled-safe are what expose a guard
that over-flags. Grow it for a tighter ranking.

## Before you run

- `meta-llama/Llama-Guard-3-1B` is **gated by Meta** — accept its license on the
  HF model page (same as the Gemma models) or comment it out of `CANDIDATES` in
  `guard_adapters.py`. Qwen3Guard and Granite Guardian are open (Apache).
- Guard output formats drift between releases. Run the adapter self-check on one
  example first and fix any line marked `# VERIFY` if a guard shows
  `parse_fail > 0`:
  ```bash
  python -m scorecard.judge_selection.guard_adapters
  ```
