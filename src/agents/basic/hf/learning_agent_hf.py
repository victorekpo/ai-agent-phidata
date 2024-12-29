import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

# Initialize the model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token explicitly
tokenizer.pad_token = tokenizer.eos_token

# Corpora management
corpora_file = "autocompletion_corpora.json"

def load_corpora():
    if not os.path.exists(corpora_file):
        return []
    with open(corpora_file, "r") as file:
        return json.load(file)

def save_corpora(corpora):
    with open(corpora_file, "w") as file:
        json.dump(corpora, file, indent=2)

def add_to_corpora(prefix, expected_output):
    corpora = load_corpora()
    if {"prefix": prefix, "expected_output": expected_output} not in corpora:
        corpora.append({"prefix": prefix, "expected_output": expected_output})
        save_corpora(corpora)
        print(f"Added: '{prefix}' -> '{expected_output}'")
    else:
        print(f"Duplicate entry skipped: '{prefix}' -> '{expected_output}'")

def autocomplete(prefix, max_length=20):
    inputs = tokenizer(prefix, return_tensors="pt", padding=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=len(inputs.input_ids[0]) + max_length,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_autocomplete():
    corpora = load_corpora()
    for item in corpora:
        prefix = item["prefix"]
        expected_output = item["expected_output"]
        prediction = autocomplete(prefix)
        print(f"Prefix: '{prefix}' | Prediction: '{prediction}' | Expected: '{expected_output}'")

def fine_tune_on_corpora():
    corpora = load_corpora()
    print("Simulating fine-tuning on corpora...")
    for item in corpora:
        print(f"Prefix: {item['prefix']}, Expected: {item['expected_output']}")

def main():
    add_to_corpora("What is the capital of Tex", "Texas is Austin")
    add_to_corpora("The quick brown fox", "jumps over the lazy dog")

    print("\nAutocompletion Tests:")
    test_autocomplete()

    print("\nSimulated Fine-Tuning:")
    fine_tune_on_corpora()

if __name__ == "__main__":
    main()
