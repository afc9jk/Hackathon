"""Model inference utilities for the personal finance scorecard."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_BASE_MODEL_ID = "google/gemma-4-E2B"
DEFAULT_IT_MODEL_ID = "google/gemma-4-E2B-it"

FINANCE_SYSTEM_PROMPT = (
    "You are a careful personal finance assistant. Provide educational guidance, "
    "tailor advice to the user's stated context, avoid guarantees, state important "
    "assumptions, and recommend an appropriate professional for legal, tax, "
    "investment, or estate-planning questions when needed."
)


@dataclass(frozen=True)
class GenerationConfig:
    """Generation settings used for reproducible evaluation."""

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None


def load_model(model_id: str):
    """Load a Hugging Face causal language model and processor."""

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    return processor, model


def _generation_kwargs(config: GenerationConfig) -> dict:
    kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    return kwargs


def parse_processor_response(processor, response: str) -> str:
    """Parse Gemma processor output across Transformers return-shape variants."""

    if not hasattr(processor, "parse_response"):
        return response.strip()

    parsed = processor.parse_response(response)
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in ("content", "text", "response"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.strip()
        return str(parsed).strip()
    return str(parsed).strip()


@torch.no_grad()
def generate_base_response(
    prompt: str,
    processor,
    model,
    config: GenerationConfig,
) -> str:
    """Generate a plain text completion from a base model."""

    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, **_generation_kwargs(config))
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_chat_response(
    prompt: str,
    processor,
    model,
    config: GenerationConfig,
    system_prompt: str = FINANCE_SYSTEM_PROMPT,
) -> str:
    """Generate a chat response from an instruction-tuned model."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, **_generation_kwargs(config))
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return parse_processor_response(processor, response)


def run_inference(
    input_path: Path,
    output_path: Path,
    model_id: str = DEFAULT_IT_MODEL_ID,
    model_kind: str = "chat",
    config: GenerationConfig | None = None,
    system_prompt: str = FINANCE_SYSTEM_PROMPT,
) -> None:
    """Generate model responses for each prompt in an evaluation CSV."""

    config = config or GenerationConfig()
    processor, model = load_model(model_id)
    df = pd.read_csv(input_path)
    responses = []

    for row in tqdm(df.to_dict(orient="records"), desc=f"Generating with {model_id}"):
        prompt = row["prompt"]
        if model_kind == "base":
            response = generate_base_response(prompt, processor, model, config)
        elif model_kind == "chat":
            response = generate_chat_response(
                prompt,
                processor,
                model,
                config,
                system_prompt=system_prompt,
            )
        else:
            raise ValueError("model_kind must be 'base' or 'chat'")

        result = dict(row)
        result["model_id"] = model_id
        result["model_kind"] = model_kind
        result["model_response"] = response
        responses.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(responses).to_csv(output_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model inference for scorecard prompts.")
    parser.add_argument("--input", type=Path, default=Path("data/personal_finance_eval_sample.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/model_responses.csv"))
    parser.add_argument("--model-id", default=DEFAULT_IT_MODEL_ID)
    parser.add_argument("--model-kind", choices=("base", "chat"), default="chat")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    run_inference(
        input_path=args.input,
        output_path=args.output,
        model_id=args.model_id,
        model_kind=args.model_kind,
        config=config,
    )


if __name__ == "__main__":
    main()
