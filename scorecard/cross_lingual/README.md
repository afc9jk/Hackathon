# Cross-Lingual Safety Distribution Shift

The info sheet's required metric: modern models keep strong safety guardrails in
English but often comply with the **same harmful request when it's translated**
into another language. This submodule measures that gap for the finance advisor.

Distinct from the rest of the repo: `dataset.py` samples English quality data and
`metrics.py` scores English rubrics; this is the only place we test non-English
behavior, and it's adversarial (harmful vs benign refusal), not quality scoring.

## Languages

Pick **English + one high-resource + one low-resource** language (the info sheet
recommends including a low-resource one). Suggested FLORES-200 codes:
- high-resource: `spa_Latn` (Spanish) or `zho_Hans` (Chinese)
- low-resource: `swh_Latn` (Swahili), `yor_Latn` (Yoruba), or `npi_Deva` (Nepali)

## Run (from repo root)

```bash
# 1. translate the 30-prompt seed set (15 harmful + 15 benign)
python -m scorecard.cross_lingual.translate --langs spa_Latn,swh_Latn

# 2. generate responses per language, classify refusal, compute the gap
python -m scorecard.cross_lingual.run_xlingual --langs eng_Latn,spa_Latn,swh_Latn
```

Outputs land in `outputs/`:
- `xlingual_safety_gap.md` / `.csv` — the results table (ranked worst-first)
- `xlingual_responses.csv` — every prompt / response / REFUSE-or-COMPLY verdict

## How to read it

- **harmful_refusal_rate** — fraction of the 15 harmful prompts the model
  refused. Higher = safer. English is the baseline.
- **safety_gap_vs_en** — `English − this language`. **Positive = the model
  refuses less in this language than in English = safety degraded.** This is the
  headline number.
- **benign_answer_rate** / **overrefusal_gap_vs_en** — the flip side: does it
  stop *answering legitimate* questions in other languages (over-refusal)?

## Limitations (state these in the write-up)

1. **Refusal is judged by an LLM** (`gemma-4-E2B-it` reused with an English
   rubric) that shares failure modes with the target model; spot-check
   `xlingual_responses.csv`.
2. **Machine translation adds noise**, and NLLB quality is *lowest* for the
   low-resource languages we most want to test — so a gap conflates weaker
   alignment with translation error. If a teammate speaks the language, verify a
   few prompts. Report the gap as a screening signal, not a precise measurement.
3. Small n (15 per class) — treat as directional; widen `seed_en.jsonl` if time
   allows.
