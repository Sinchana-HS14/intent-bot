import json
import pickle
import nltk
import re #added re for cleaning
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4') #Recommended for lemmatization

#Intialize lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    tokens=nltk.word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

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

#Apply the cleaning function to patern Before vectorization
cleaned_patterns = [clean_text(pattern) for pattern in patterns]

# Used cleaned pattern instead of raw patterns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_patterns)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, tags)

# Save model and vectorizer
pickle.dump(model, open("chatbot_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully with lemmatiztion!")