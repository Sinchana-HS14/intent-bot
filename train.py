import json
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

patterns = []
tags = []

# Extract patterns and tags
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Convert text into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, tags)

# Save model and vectorizer
pickle.dump(model, open("chatbot_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")