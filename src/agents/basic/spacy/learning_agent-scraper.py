import spacy
from sentence_transformers import SentenceTransformer, util
import json
import requests
from bs4 import BeautifulSoup
import time

class ConversationalAgent:
    def __init__(self):
        self.knowledge_base = {}  # Local knowledge base
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
        """Answer a question."""
        matched_question = self.similar_question(question)
        if matched_question:
            answer = self.knowledge_base[matched_question]
            source = "local knowledge base"
        else:
            answer = "I don't know the answer to that question yet."
            source = "unknown"

        print(f"Answer ({source}): {answer}")

    def add_to_knowledge_base(self, text):
        """Add text to the knowledge base."""
        doc = self.nlp(text)
        for sent in doc.sents:
            self.knowledge_base[sent.text] = sent.text
        print("Information added to knowledge base.")

    def scrape_and_add(self, url):
        """Scrape a webpage and add its content to the knowledge base."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        self.add_to_knowledge_base(text)
        print("Webpage content added to knowledge base.")

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

def main():
    agent = ConversationalAgent()
    agent.load_knowledge_base()

    while True:
        print("\nMenu:")
        print("1. Train")
        print("2. Chat")
        print("3. Sleep")
        print("4. Exit")
        choice = input("Select an option: ")

        if choice == "1":
            while True:
                training_input = input("Enter URL or text (or type 'done' to finish): ")
                if training_input.lower() == "done":
                    break
                elif training_input.startswith("http"):
                    agent.scrape_and_add(training_input)
                else:
                    agent.add_to_knowledge_base(training_input)
        elif choice == "2":
            while True:
                user_question = input("\nAsk a question (or type 'exit' to quit): ")
                if user_question.lower() == "exit":
                    break
                agent.ask(user_question)
        elif choice == "3":
            input("Sleeping... Press any key to wake up.")
        elif choice == "4":
            agent.save_knowledge_base()
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()