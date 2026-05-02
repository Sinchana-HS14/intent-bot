# ML Chatbot 

A robust, intent-based chatbot built using Python, Scikit-learn, and Streamlit. This project uses Natural Language Processing (NLP) techniques like Lemmatization and TF-IDF vectorization to understand and respond to user queries effectively.

## Features
*   **Natural Language Processing:** Utilizes `WordNetLemmatizer` to understand word roots.
*   **Machine Learning:** Powered by a Logistic Regression model for intent classification.
*   **Modern UI:** A clean, interactive chat interface built with Streamlit.
*   **Session Management:** Maintains conversation history during the session.
*   **Confidence Threshold:** Filters out low-confidence inputs to ensure response accuracy.

## Tech Stack
*   **Language:** Python 3.13
*   **Frontend:** Streamlit
*   **ML Libraries:** Scikit-learn, NLTK
*   **Data Format:** JSON (for intents and patterns)

## Project Structure
*   `app.py`: The main Streamlit application script.
*   `train.py`: Script to preprocess data and train the ML model.
*   `intents.json`: The knowledge base containing patterns and responses.
*   `chatbot_model.pkl`: The trained Logistic Regression model.
*   `vectorizer.pkl`: The saved TF-IDF vectorizer.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Sinchana-HS14/intent-bot.git](https://github.com/Sinchana-HS14/intent-bot.git)
   cd intent-bot
