from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch
import json
import os

# Initialize the model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token explicitly
tokenizer.pad_token = tokenizer.eos_token

# Ensure model is on the correct device (MPS, GPU, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

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


def prepare_training_data(corpora):
    # Combine all the prefix and expected_output into a single text corpus
    training_data = []
    for item in corpora:
        # Ensure the format clarifies the expected output with special tokens
        training_data.append({"text": item["prefix"] + " [SEP] " + item["expected_output"]})

    # Save the data into a Hugging Face dataset
    dataset = Dataset.from_list(training_data)
    dataset.save_to_disk("training_data")

    print("Training data prepared.")


def tokenize_data(example):
    # Tokenize the 'text' column using the GPT2 tokenizer
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=128, return_tensors="pt")


def fine_tune_on_corpora():
    corpora = load_corpora()
    prepare_training_data(corpora)

    # Load the dataset from the saved directory
    train_dataset = Dataset.load_from_disk("training_data")

    print(f"Training dataset length: {len(train_dataset)}")  # Debug: check dataset length

    # Tokenize the dataset
    train_dataset = train_dataset.map(tokenize_data, batched=True)
    print(f"Tokenized dataset length: {len(train_dataset)}")  # Debug: check tokenized dataset length

    # Use DataCollator for Language Modeling for padding and formatting
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=10,  # Increased epochs
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Fine-tune the model
    print("Starting fine-tuning...")
    trainer.train()

    # Save the model after fine-tuning
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuning completed and model saved.")


def autocomplete(prefix, max_length=20):
    # Move model to the device (MPS, GPU, or CPU)
    device = model.device  # Ensure we're using the same device as the model

    # Tokenize input and move to the same device as the model
    inputs = tokenizer(prefix, return_tensors="pt", padding=True).to(device)

    # Generate output
    outputs = model.generate(
        inputs.input_ids,
        max_length=len(inputs.input_ids[0]) + max_length,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_fine_tuned_model():
    test_data = [
        {"prefix": "What is the capital of Tex", "expected_output": "Texas is Austin"},
        {"prefix": "The quick brown fox", "expected_output": "jumps over the lazy dog"}
    ]
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
    add_to_corpora("What is the capital of Tex", "Texas is Austin")
    add_to_corpora("The quick brown fox", "jumps over the lazy dog")

    # Fine-tune the model on the new corpora
    fine_tune_on_corpora()

    # Test the autocompletion
    print("\nTesting Fine-Tuned Model:")
    test_fine_tuned_model()


if __name__ == "__main__":
    main()
