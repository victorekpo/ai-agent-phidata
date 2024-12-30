import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import json

class ConversationalAgent:
    def __init__(self, model_name="EleutherAI/gpt-neo-2.7B"):
        print("Initializing ConversationalAgent...")
        self.knowledge_base = {}  # Local knowledge base
        print("Loading model...")
        self.model = pipeline("text-generation", model=model_name)
        print("Model loaded.")
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("ConversationalAgent initialized.")

    def similar_question(self, question):
        """Find the most similar question in the knowledge base."""
        question_embedding = self.sentence_model.encode(question, convert_to_tensor=True)
        best_match = None
        highest_similarity = 0
        for saved_question in self.knowledge_base:
            saved_question_embedding = self.sentence_model.encode(saved_question, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(question_embedding, saved_question_embedding).item()
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = saved_question
        return best_match if highest_similarity > 0.7 else None

    def ask(self, question):
        """Answer a question and ask for feedback."""
        print(f"Asking question: {question}")
        # First, try to find a similar question in the knowledge base
        matched_question = self.similar_question(question)
        if matched_question:
            answer = self.knowledge_base[matched_question]
            source = "local knowledge base"
        else:
            # Use the text-generation model to generate an answer
            prompt = f"Q: {question}\nA:"
            answer = self.model(prompt, max_length=50, num_return_sequences=1)[0]['generated_text'].split("A:")[1].strip()
            source = "pretrained model"

        print(f"Answer ({source}): {answer}")

        # Ask for feedback
        feedback = int(input("Rate the answer from 1-10: "))
        if feedback >= 7:
            # Save the correct answer in the local knowledge base
            self.knowledge_base[question] = answer
            print("Answer saved to knowledge base!")
        else:
            # Ask for the correct answer and update the knowledge base
            correct_answer = input("What is the correct answer? ")
            self.knowledge_base[question] = correct_answer
            print("Correct answer saved to knowledge base!")

    def save_knowledge_base(self, filename="knowledge_base.json"):
        """Save the knowledge base to a file."""
        with open(filename, "w") as f:
            json.dump(self.knowledge_base, f)
        print("Knowledge base saved!")

    def load_knowledge_base(self, filename="knowledge_base.json"):
        """Load the knowledge base from a file."""
        try:
            with open(filename, "r") as f:
                self.knowledge_base = json.load(f)
            print("Knowledge base loaded!")
        except FileNotFoundError:
            print("No saved knowledge base found. Starting fresh.")

# Initialize the agent
print("Initializing agent...")
agent = ConversationalAgent()

# Load any saved knowledge base
print("Loading knowledge base...")
agent.load_knowledge_base()

# Main loop
try:
    while True:
        user_question = input("\nAsk a question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break
        agent.ask(user_question)
except KeyboardInterrupt:
    print("\nExiting...")

# Save the knowledge base before exiting
print("Saving knowledge base...")
agent.save_knowledge_base()
print("Knowledge base saved. Exiting...")