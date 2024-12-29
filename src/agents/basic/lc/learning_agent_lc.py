from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Initialize SentenceTransformerEmbeddings (No OpenAI API key required)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Using a free transformer model

# Load corpora from a file
def load_corpora_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

# Save corpora to a file
def save_corpora_to_file(corpora, file_path):
    with open(file_path, 'w') as file:
        for line in corpora:
            file.write(line + "\n")

# Add new feedback to corpora and update FAISS index
def add_feedback_to_corpora(corpora, faiss_index, new_feedback):
    """Add new feedback to corpora and update FAISS index"""
    # Add the new feedback to the corpora
    corpora.append(new_feedback)
    # Update the FAISS index with the new feedback
    faiss_index.add_texts([new_feedback])
    print(f"Added new feedback: {new_feedback}")
    return faiss_index

# Continuous learning function (retrieval and feedback addition)
def continuous_learning(corpora, faiss_index, prompt, feedback=None):
    """Retrieve relevant information from the corpora using FAISS and add feedback if provided"""
    # Retrieve relevant information from the corpora using FAISS
    result = faiss_index.similarity_search(prompt, k=1)
    print(f"Retrieved from corpora: {result}")

    # If feedback is provided, update the corpus and FAISS index
    if feedback:
        faiss_index = add_feedback_to_corpora(corpora, faiss_index, feedback)

    return result, faiss_index

def main():
    # Load initial corpora from file or define it directly for the first run
    corpora_file = 'corpora.txt'
    try:
        corpora = load_corpora_from_file(corpora_file)
        print(f"Loaded corpora: {corpora}")
    except FileNotFoundError:
        # Initialize with default values if the file doesn't exist
        corpora = [
            "What is the capital of Texas?",
            "The capital of Texas is Austin",
            "What does the quick brown fox do?",
            "The quick brown fox jumps over the lazy dog",
        ]
        save_corpora_to_file(corpora, corpora_file)  # Save the initial corpus to file
        print("Initialized corpora and saved to file.")

    # Create FAISS index from initial corpora
    faiss_index = FAISS.from_texts(corpora, embeddings)

    # Test prompt and feedback
    prompt = "Where is the capital of Texas?"
    feedback = "Texas is Austin."  # This is the new feedback you want to add

    # Call continuous learning function
    result, faiss_index = continuous_learning(corpora, faiss_index, prompt, feedback)

    # Print the result
    print(f"Final result: {result}")

    # Save updated corpora to file
    save_corpora_to_file(corpora, corpora_file)

    # Additional test to check if learning was successful
    test_prompt = "What is the capital of Texas?"
    result, _ = continuous_learning(corpora, faiss_index, test_prompt)
    assert "Austin" in result[0][0], "Test failed: Expected 'Austin' to be retrieved from the corpus."

    # Another test case to verify learning from new feedback
    test_prompt = "What does the quick brown fox do?"
    result, _ = continuous_learning(corpora, faiss_index, test_prompt)
    assert "jumps over the lazy dog" in result[0][0], "Test failed: Expected the correct phrase from corpus."

    print("All tests passed!")

if __name__ == "__main__":
    main()
