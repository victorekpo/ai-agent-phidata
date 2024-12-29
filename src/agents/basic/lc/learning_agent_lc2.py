import spacy
import json
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import numpy as np

# Load spaCy model for NLP processing (no LLM involved)
nlp = spacy.load("en_core_web_sm")

# Example initial corpora (could be loaded from a file)
corpora = [
    "The capital of Texas is Austin.",
    "The quick brown fox jumps over the lazy dog."
]

# Initialize the SentenceTransformer model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")


# Process text using spaCy (tokenization, NER, etc.)
def process_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]  # Tokenized text
    return tokens


# Create FAISS index from the processed corpus
def create_faiss_index(corpora):
    # Generate embeddings for the corpus texts using SentenceTransformer directly
    corpus_embeddings = sentence_model.encode(corpora)

    # Print the embeddings to debug
    print("Generated embeddings:")
    print(corpus_embeddings)

    # Ensure that embeddings are generated correctly and not empty
    if not corpus_embeddings or len(corpus_embeddings[0]) == 0:
        raise ValueError("Embeddings generation failed or returned empty embeddings.")

    # Create FAISS index using the embeddings
    faiss_index = FAISS.from_embeddings(corpus_embeddings, sentence_model)
    return faiss_index


# Function to add feedback to corpora and update the FAISS index
def add_feedback_to_corpora(corpora, faiss_index, new_feedback):
    corpora.append(new_feedback)
    new_embedding = sentence_model.encode([new_feedback])  # Generate embedding for new feedback

    # Print the new embedding to debug
    print("Generated embedding for feedback:")
    print(new_embedding)

    faiss_index.add_texts([new_feedback])  # Add new text feedback to FAISS index
    print(f"Added new feedback: {new_feedback}")
    return faiss_index


# Function for continuous learning: Retrieval + Feedback Loop
def continuous_learning(corpora, faiss_index, prompt, feedback=None):
    # Retrieve relevant information from the FAISS index based on the prompt
    result = faiss_index.similarity_search(prompt, k=1)
    print(f"Retrieved from corpora: {result}")

    # If feedback is provided, update the corpora and FAISS index
    if feedback:
        faiss_index = add_feedback_to_corpora(corpora, faiss_index, feedback)

    return result, faiss_index


# Function to save and load corpus from file
def save_corpora(corpora, filename="corpora.json"):
    with open(filename, "w") as f:
        json.dump(corpora, f)


def load_corpora(filename="corpora.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def main():
    # Load corpora from file if exists
    corpora = load_corpora()

    # Create FAISS index from initial corpora
    faiss_index = create_faiss_index(corpora)

    # Example: Adding new feedback and performing a query
    prompt = "Where is the capital of Texas?"
    feedback = "Texas is Austin."  # New data to add to the corpora

    # Call the continuous learning function
    result, faiss_index = continuous_learning(corpora, faiss_index, prompt, feedback)

    # Print the result
    print(f"Final result: {result}")

    # Optionally, save the updated corpora to a file
    save_corpora(corpora)


if __name__ == "__main__":
    main()
