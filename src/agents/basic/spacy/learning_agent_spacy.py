import spacy
from spacy.training import Example

# Initialize spaCy NLP pipeline
nlp = spacy.blank("en")

# Add the NER component to the pipeline if it's not already present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Example training data
train_data = [
    ("The capital of Texas is Austin.", {"entities": [(17, 23, "LOCATION")]}),
    ("The president of the United States is Joe Biden.", {"entities": [(31, 40, "PERSON")]})
]

# Add labels to the NER component
for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipelines during training to only train NER
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for epoch in range(20):
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, sgd=optimizer)

# Save the trained model to disk
nlp.to_disk("./model")

# Load the trained model from disk
nlp_loaded = spacy.load("./model")

# Function to ask a question and get the answer
def ask_question(question):
    doc = nlp_loaded(question)
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")

# Example usage
ask_question("Where is the capital of Texas?")
ask_question("Who is the president of the United States?")