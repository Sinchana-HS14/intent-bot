import streamlit as st
import pickle
import json
import random

# Load trained model
model = pickle.load(open("chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

st.title("🤖 ML Chatbot")
st.write("Type something and I will respond!")

user_input = st.text_input("You:")

if user_input:
    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]

    for intent in data["intents"]:
        if intent["tag"] == prediction:
            response = random.choice(intent["responses"])
            st.success(response)