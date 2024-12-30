import json
import time
from datetime import datetime, timedelta

import requests
import spacy
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util


class ConversationalAgent:
    def __init__(self, name):
        self.name = name
        self.dob = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.brain = []
        self.brain_extended = []
        self.residence = None
        self.bank_balance = {category: 0 for category in ["income", "expenses", "savings", "misc"]}
        self.bank_budget = {category: 0 for category in ["income", "expenses", "savings", "misc"]}
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"ConversationalAgent {self.name} initialized.")

    def similar_question(self, question):
        question_embedding = self.sentence_model.encode(question, convert_to_tensor=True)
        best_match = None
        highest_similarity = 0

        # Check brain_extended
        for saved_info in self.brain_extended:
            print("Saved information in brain extended", saved_info["url"])
            for detail in saved_info["details"]:
                #  print("Detailed information", detail)
                saved_question_embedding = self.sentence_model.encode(detail, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(question_embedding, saved_question_embedding).item()
                #   print("Similarity", similarity, "Highest Similarity", highest_similarity)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = detail

        # Check brain
        # for key in self.brain:
        #     print("Key", key)
        #     brain_doc_embedding = self.sentence_model.encode(key, convert_to_tensor=True)
        #     similarity = util.pytorch_cos_sim(question_embedding, brain_doc_embedding).item()
        #     if similarity > highest_similarity:
        #         highest_similarity = similarity
        #         best_match = key

        return best_match if highest_similarity > 0.7 else None

    def ask(self, question):
        matched_entry = self.similar_question(question)
        if matched_entry:
            answer = matched_entry
            source = "local knowledge base"
        else:
            answer = "I don't know the answer to that question yet."
            source = "unknown"
        print(f"Answer ({source}): {answer}")

    def add_to_brain_extended(self, url, text):
        doc = self.nlp(text)
        details = [sent.text for sent in doc.sents]
        entry = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.brain_extended.append(entry)
        print("Information added to knowledge base.")

    def scrape_and_add(self, url):
        # Check if the URL was scraped within the last 3 days
        for entry in self.brain_extended:
            if entry["url"] == url:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if datetime.now() - timestamp < timedelta(days=3):
                    print("Data is less than 3 days old. Skipping scrape.")
                    return

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        self.add_to_brain_extended(url, text)
        # self.process_information("scraped", url, text)
        print("Webpage content added to knowledge base and processed to brain.")

    def save_brain_extended(self, filename="brain_extended.json"):
        with open(filename, "w") as f:
            json.dump(self.brain_extended, f)
        print("Knowledge base saved!")

    def load_brain_extended(self, filename="brain_extended.json"):
        try:
            with open(filename, "r") as f:
                self.brain_extended = json.load(f)
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
            item_exists = any(
                isinstance(entry, dict) and (
                        entry["type"] == _type and entry["key"] == key and entry["value"] == formatted_value
                )
                for entry in self.brain
            )

            if item_exists:
                print("Item already exists in brain.", formatted_value)
                return

            if not item_exists:
                self.brain.append({
                    "type": _type,
                    "key": key,
                    "value": formatted_value,
                    "timestamp": datetime.now().isoformat()
                })

            if limit > 0:
                self.brain = self.brain[-limit:]

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


def main():
    start_time = time.time()

    load_start = time.time()
    agent = ConversationalAgent("vicBot")
    agent.load_brain_extended()
    agent.load_brain()
    agent.live("Milwaukee, Wisconsin")
    load_end = time.time()
    print(f"Loading agent took {load_end - load_start:.2f} seconds")

    scrape_start = time.time()
    agent.scrape_and_add("https://www.telerik.com/blogs/mastering-typescript-benefits-best-practices")
    scrape_end = time.time()
    print(f"Scraping and adding took {scrape_end - scrape_start:.2f} seconds")

    ask_start = time.time()
    agent.ask("What is typescript?")
    ask_end = time.time()
    print(f"Asking question took {ask_end - ask_start:.2f} seconds")

    budget_start = time.time()
    agent.set_monthly_budget({"income": 5000, "expenses": 3000})
    budget_end = time.time()
    print(f"Setting monthly budget took {budget_end - budget_start:.2f} seconds")

    balance_start = time.time()
    agent.get_account_balance()
    balance_end = time.time()
    print(f"Getting account balance took {balance_end - balance_start:.2f} seconds")

    monthly_budget_start = time.time()
    agent.get_monthly_budget()
    monthly_budget_end = time.time()
    print(f"Getting monthly budget took {monthly_budget_end - monthly_budget_start:.2f} seconds")

    learn_start = time.time()
    agent.learn("coding", "Java-Completeable-Futures",
                "Completeable-Futures are awesome, they allow you to run asynchronous code in java")
    agent.learn("coding", "Java-Completeable-Futures",
                "Completeable-Futures are awesome, they allow you to run asynchronous code in java")
    agent.learn("coding", "Java-Completeable-Futures",
                "You can create Completeable Futures by...CompletableFuture.supplyAsync(() -> {return 42;})")
    learn_end = time.time()
    print(f"Learning took {learn_end - learn_start:.2f} seconds")

    save_start = time.time()
    agent.save_brain_extended()
    agent.save_brain()
    save_end = time.time()
    print(f"Saving brain took {save_end - save_start:.2f} seconds")

    end_time = time.time()
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
