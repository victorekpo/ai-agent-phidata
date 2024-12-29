import spacy
from sentence_transformers import SentenceTransformer, util
import json
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import os

class ConversationalAgent:
    def __init__(self, name):
        self.name = name
        self.dob = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.brain = {}
        self.residence = None
        self.bank_balance = {category: 0 for category in ["income", "expenses", "savings", "misc"]}
        self.bank_budget = {category: 0 for category in ["income", "expenses", "savings", "misc"]}
        self.knowledge_base = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"ConversationalAgent {self.name} initialized.")

    def similar_question(self, question):
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
        matched_question = self.similar_question(question)
        if (matched_question):
            answer = self.knowledge_base[matched_question]
            source = "local knowledge base"
        else:
            answer = "I don't know the answer to that question yet."
            source = "unknown"
        print(f"Answer ({source}): {answer}")

    def add_to_knowledge_base(self, text):
        doc = self.nlp(text)
        for sent in doc.sents:
            self.knowledge_base[sent.text] = sent.text
        print("Information added to knowledge base.")

    def scrape_and_add(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        self.add_to_knowledge_base(text)
        self.process_information("scraped", url, text)
        print("Webpage content added to knowledge base and processed to brain.")

    def save_knowledge_base(self, filename="knowledge_base.json"):
        with open(filename, "w") as f:
            json.dump(self.knowledge_base, f)
        print("Knowledge base saved!")

    def load_knowledge_base(self, filename="knowledge_base.json"):
        try:
            with open(filename, "r") as f:
                self.knowledge_base = json.load(f)
            print("Knowledge base loaded!")
        except FileNotFoundError:
            print("No saved knowledge base found. Starting fresh.")

    def _try_to_parse(self, obj):
        """Attempt to parse JSON in multiple ways."""
        try:
            return json.loads(obj)
        except Exception:
            pass
        try:
            return json.loads(f"[{obj}]")
        except Exception:
            pass
        return None

    def process_information(self, _type, key, raw_value, limit=-1, cache=True):
        value = "\r\n".join(raw_value) if isinstance(raw_value, list) else raw_value
        formatted_value = value  # Optional formatting logic can be added here.

        if value and limit != 0:
            if _type not in self.brain or self.brain.get(_type, {}).get(key) in ["{}", "[]"]:
                self.brain[_type] = {}

            if key not in self.brain[_type] or self.brain[_type][key] in ["{}", "[]"]:
                self.brain[_type][key] = []

            item_exists = any(
                isinstance(entry, dict) and (
                            entry["value"] == formatted_value or formatted_value in entry["value"].split("\r\n"))
                for entry in self.brain[_type][key]
            )

            if not item_exists and len(formatted_value) == 1 and len(self.brain[_type][key]) < 1 and cache:
                self.brain[_type][key] = [{"time": datetime.now().isoformat(), "value": value}]

            if (not item_exists or not cache) and (
                    len(formatted_value) > 1 or isinstance(self._try_to_parse(self.brain[_type][key]), dict)
            ):
                to_add = (
                        "\r\n".join(
                            {
                                line
                                for line in formatted_value.split("\r\n")
                                if "@file:" in line or not any(
                                line in entry["value"] for entry in self.brain[_type][key] if isinstance(entry, dict)
                            )
                            }
                        )
                        .replace("\r\n\r\n", "\r\n")
                        or None
                )
                if to_add:
                    self.brain[_type][key] = list(
                        {
                            tuple(entry.items()) for entry in self.brain[_type][key] if isinstance(entry, dict)
                        } | {("time", datetime.now().isoformat()), ("value", to_add)}
                    )

            if limit > 0 and isinstance(self._try_to_parse(self.brain[_type][key]), list):
                while len(self.brain[_type][key]) > limit:
                    self.brain[_type][key].pop(0)

        self.save_brain()

    def save_brain(self, filename="brain.json"):
        with open(filename, "w") as f:
            json.dump(self.brain, f)
        print("Brain saved!")

    def load_brain(self, filename="brain.json"):
        try:
            with open(filename, "r") as f:
                self.brain = json.load(f)
            print("Brain loaded!")
        except FileNotFoundError:
            print("No saved brain found. Starting fresh.")

    def get_age(self):
        dob = datetime.strptime(self.dob, "%Y-%m-%d %H:%M:%S")
        age = datetime.now() - dob
        return f"My name is {self.name}, my age is {age}. I was born {self.dob}"

    def live(self, neighborhood):
        self.residence = neighborhood
        print(f"{self.name} just moved into {neighborhood} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return str(neighborhood)

    def learn(self, _type, item, value, limit=-1):
        self.process_information(_type, item, value, limit)
        return "Item processed."

    def set_monthly_budget(self, budget):
        self.bank_budget.update(budget)
        self.bank_budget["savings"] = self.bank_budget["income"] - self.bank_budget["expenses"]
        print("Updated Monthly Budget:", self.bank_budget)
        return self.bank_budget

    def get_account_balance(self):
        print("Current Account Balance:", self.bank_balance)
        return self.bank_balance

    def get_monthly_budget(self):
        print("Monthly Budget:", self.bank_budget)
        return self.bank_budget

# Example usage
agent = ConversationalAgent("Victor")
agent.load_knowledge_base()
agent.load_brain()
agent.scrape_and_add("https://www.telerik.com/blogs/mastering-typescript-benefits-best-practices")
agent.ask("What is typescript?")
agent.set_monthly_budget({"income": 5000, "expenses": 3000})
agent.get_account_balance()
agent.get_monthly_budget()
agent.learn("type", "key", "value")
agent.save_knowledge_base()
agent.save_brain()