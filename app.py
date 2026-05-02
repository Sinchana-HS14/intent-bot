import streamlit as st
import pickle
import json
import random
import nltk
import re
from nltk.stem import WordNetLemmatizer

# Load trained model and vectorizer
try:
    model = pickle.load(open("chatbot_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found. Please run train.py first!")

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    tokens = nltk.word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in tokens])

# --- Streamlit UI ---
st.set_page_config(page_title="ML Chatbot", page_icon="🤖")
st.title("🤖 Intelligent ML Chatbot")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me about CSE or movies..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and Predict
    cleaned_input = clean_text(prompt)
    X = vectorizer.transform([cleaned_input])
    
    # Get probability to check confidence
    probs = model.predict_proba(X)
    max_prob = max(probs[0])

    if max_prob < 0.3:
        response = "I'm not sure I understand. Could you rephrase that?"
    else:
        prediction = model.predict(X)[0]
        for intent in data["intents"]:
            if intent["tag"] == prediction:
                response = random.choice(intent["responses"])

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})