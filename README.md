# Personal Finance LLM Safety Scorecard

DS6051 hackathon — a safety-evaluation scorecard for **`google/gemma-4-E2B-it`
deployed as a personal-finance advisor**. We treat this general-purpose model as
if it were built for that use case and measure whether it is safe to deploy:
*what* to measure, *why*, and *how* — plus each measurement's limits.

This is an **evaluation** project, not training. Nothing here fine-tunes the model.

## Repository structure

```
data/
  personal_finance_eval_sample.csv   # balanced sample of PersonalFinance_v2 (benign, quality eval)
  xlingual/seed_en.jsonl             # 15 harmful + 15 benign prompts for the cross-lingual test
scorecard/
  dataset.py                         # load + balance-sample the HF dataset
  metrics.py                         # the 8 scorecard metric definitions + rubrics
  export_metrics.py                  # metric defs -> outputs/metric_definitions.csv
  inference.py                       # run gemma-4-E2B-it over the eval prompts
  judge.py                           # ShieldGemma LLM-as-judge, one guideline per metric
  run_scorecard.py                   # inference -> judge -> aggregate (end-to-end)
  cross_lingual/                     # required: does safety degrade in other languages?
outputs/                             # all generated results land here (committed for the write-up)
requirements.txt                     # pinned environment
bash_command_basics.txt              # Rivanna / git / HF login cheatsheet
```

## The scorecard at a glance

- **Quality & advice metrics (8)** — `scorecard/metrics.py`: advice accuracy,
  hallucination, harmful recommendations, completeness, reasoning, robustness,
  referral, personalization. Scored by the ShieldGemma LLM-as-judge in `judge.py`.
- **Cross-lingual safety shift** — `scorecard/cross_lingual/`: does the model
  refuse a harmful finance request in English but comply when it's translated?
  Reuses the same inference pipeline and the same ShieldGemma judge.

## Setup

Accept the gated model licenses on Hugging Face (while logged in), then log in:
`google/gemma-4-E2B`, `google/gemma-4-E2B-it`, and `google/shieldgemma-2b`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login          # paste your HF token
```

On Rivanna, grab a GPU first (see `bash_command_basics.txt`):
```bash
srun -A ds6051-summer -p interactive --time=6:00:00 --mem=24G \
  --reservation=ds6051-summer-hackathon --gres=gpu:1 --pty bash
```

## Run order (all commands from the repo root)

```bash
# 0. (once) export the eval sample and metric definitions
python -m scorecard.dataset --per-category 15 --output data/personal_finance_eval_sample.csv
python -m scorecard.export_metrics

# 1. core scorecard: generate responses, judge them with ShieldGemma, aggregate.
#    Pass extra ShieldGemma sizes to --judge-model-ids to compare judges (info-sheet requirement).
python -m scorecard.run_scorecard --judge-model-ids google/shieldgemma-2b

# 2. cross-lingual safety gap (English + 1 high-resource + 1 low-resource).
#    Reuses the same model + ShieldGemma judge as step 1.
python -m scorecard.cross_lingual.translate   --langs spa_Latn,swh_Latn
python -m scorecard.cross_lingual.run_xlingual --langs eng_Latn,spa_Latn,swh_Latn
```

## Outputs (results table for the write-up)

| File | Produced by | Contents |
| --- | --- | --- |
| `outputs/metric_definitions.csv` | `export_metrics` | the 8 metrics + rubrics |
| `outputs/model_responses.csv` | `run_scorecard` | model answer per eval prompt |
| `outputs/judge_scores.csv` | `run_scorecard` | per-example judge score per metric |
| `outputs/scorecard_summary.csv` | `run_scorecard` | mean score / pass rate per metric & category |
| `outputs/xlingual_safety_gap.md` / `.csv` | `cross_lingual` | harmful-compliance gap per language |

After a run, commit the results so the graded table is in the repo:
```bash
git add outputs && git commit -m "Add scorecard run results" && git push origin main
```

## Limitations we track

The benchmark reference answers are not perfect ground truth; the ShieldGemma
judge shares failure modes with the target model and is itself weaker in
non-English languages; and machine translation adds noise to the cross-lingual
test, worst in low-resource languages. Each submodule's README states its
specific caveats.
