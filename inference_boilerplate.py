# Model Inference Boilerplate — Gemma 4 (E2B / E2B-it)
#
# Loads Google's `gemma-4-E2B` (base) and `gemma-4-E2B-it` (instruction-tuned)
# models and runs text generation on both, so you can compare their behavior.

# 1. Import Libraries
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# 2. Load Models and Processors
# Loads both the base (pre-trained) and instruction-tuned variants.
BASE_MODEL_ID = "google/gemma-4-E2B"
IT_MODEL_ID = "google/gemma-4-E2B-it"

base_processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype="auto",
    device_map="auto",
)

it_processor = AutoProcessor.from_pretrained(IT_MODEL_ID)
it_model = AutoModelForCausalLM.from_pretrained(
    IT_MODEL_ID,
    dtype="auto",
    device_map="auto",
)

# 3. Define Input
prompt = "Write a short joke about saving RAM."

# 4. Base Model (Non-Instruction-Tuned) Inference
# The base model does plain text completion — no chat template.
inputs = base_processor(text=prompt, return_tensors="pt").to(base_model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = base_model.generate(**inputs, max_new_tokens=256)
base_response = base_processor.decode(outputs[0][input_len:], skip_special_tokens=True)
print(base_response)

# 5. Instruction-Tuned Model Inference
# The `-it` model expects chat-formatted messages via `apply_chat_template`.
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

text = it_processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = it_processor(text=text, return_tensors="pt").to(it_model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = it_model.generate(**inputs, max_new_tokens=256)
response = it_processor.decode(outputs[0][input_len:], skip_special_tokens=False)
it_response = it_processor.parse_response(response)
print(it_response)

# 6. Compare Outputs
print("Base model:\n", base_response)
print("\nInstruction-tuned model:\n", it_response)