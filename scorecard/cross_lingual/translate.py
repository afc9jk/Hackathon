"""Translate the English cross-lingual seed set into target languages with
NLLB-200, so we can measure the model's safety gap across languages.

We use Meta's NLLB-200 (No Language Left Behind) because it covers 200
languages including genuinely low-resource ones (Swahili, Yoruba, Nepali) that
general translation APIs handle poorly. Languages are named with FLORES-200
codes (e.g. spa_Latn, swh_Latn, yor_Latn, npi_Deva, zho_Hans).

    python -m scorecard.cross_lingual.translate --langs spa_Latn,swh_Latn

Reads  data/xlingual/seed_en.jsonl
Writes data/xlingual/seed_<lang>.jsonl  (same ids/types, translated `prompt`)

IMPORTANT LIMITATION (put this in the write-up): machine translation is itself
imperfect, and MT quality is *lowest* for exactly the low-resource languages we
most want to test. A measured safety gap therefore conflates two things — the
model's weaker alignment in that language AND translation noise in the prompt.
If anyone on the team speaks the target language, spot-check a few prompts and
note it; otherwise report the gap as a screening signal, not a precise number.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

SEED_EN = Path("data/xlingual/seed_en.jsonl")
OUT_DIR = Path("data/xlingual")
MODEL_ID = "facebook/nllb-200-distilled-600M"  # open; swap to 1.3B for better quality
SRC_LANG = "eng_Latn"

# Friendly names for the languages we suggest (one high-resource + one low-resource).
KNOWN_LANGS = {
    "spa_Latn": "Spanish (high-resource)",
    "zho_Hans": "Chinese, Simplified (high-resource)",
    "hin_Deva": "Hindi (mid-resource)",
    "swh_Latn": "Swahili (low-resource)",
    "yor_Latn": "Yoruba (low-resource)",
    "npi_Deva": "Nepali (low-resource)",
}


def load_seed(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Translate the seed set via NLLB-200.")
    ap.add_argument(
        "--langs",
        default="spa_Latn,swh_Latn",
        help="comma-separated FLORES-200 target codes; "
        f"suggested: {', '.join(KNOWN_LANGS)}",
    )
    ap.add_argument("--seed", type=Path, default=SEED_EN)
    ap.add_argument("--outdir", type=Path, default=OUT_DIR)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    targets = [x.strip() for x in args.langs.split(",") if x.strip()]
    rows = load_seed(args.seed)
    print(f"Loaded {len(rows)} English prompts; translating into: {targets}\n")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, device_map="auto")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for tgt in targets:
        if tgt not in KNOWN_LANGS:
            print(f"  ! {tgt} not in the suggested list — proceeding, verify the code.")
        # VERIFY: on some transformers versions use
        #   tokenizer.convert_tokens_to_ids(tgt)  (newer)
        # vs tokenizer.lang_code_to_id[tgt]       (older). Try the newer first.
        try:
            forced_bos = tokenizer.convert_tokens_to_ids(tgt)
        except Exception:  # noqa: BLE001
            forced_bos = tokenizer.lang_code_to_id[tgt]

        out_rows = []
        for r in rows:
            inputs = tokenizer(r["prompt"], return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=args.max_new_tokens,
            )
            translated = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
            out_rows.append({**r, "prompt": translated, "lang": tgt, "prompt_en": r["prompt"]})

        out_path = args.outdir / f"seed_{tgt}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {out_path} ({KNOWN_LANGS.get(tgt, tgt)})")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nDone. Next: python -m scorecard.cross_lingual.run_xlingual --langs "
          + ",".join(["eng_Latn", *targets]))


if __name__ == "__main__":
    main()
