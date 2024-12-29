import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_name = "./fine_tuned_model"  # Path to the fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token explicitly
tokenizer.pad_token = tokenizer.eos_token

# Test data for autocompletion (using examples from corpora)
test_data = [
    {"prefix": "What is the capital of Tex", "expected_output": "Texas is Austin"},
    {"prefix": "The quick brown fox", "expected_output": "jumps over the lazy dog"}
]

# Autocompletion function using the fine-tuned model
def autocomplete(prefix, max_length=20):
    inputs = tokenizer(prefix, return_tensors="pt", padding=True)
    outputs = model.generate(
        inputs.input_ids,
        max_length=len(inputs.input_ids[0]) + max_length,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the fine-tuned model
def test_fine_tuned_model():
    for item in test_data:
        prefix = item["prefix"]
        expected_output = item["expected_output"]
        prediction = autocomplete(prefix)
        print(f"Prefix: '{prefix}'")
        print(f"Prediction: '{prediction}'")
        print(f"Expected: '{expected_output}'")
        print("Test passed!" if expected_output in prediction else "Test failed!")
        print("=" * 50)

def main():
    print("\nTesting Fine-Tuned Model:")
    test_fine_tuned_model()

if __name__ == "__main__":
    main()
