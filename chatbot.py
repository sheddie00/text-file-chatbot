# chatbot.py
import streamlit as st
import nltk
import string
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources (only first run)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

# -------------------- 1. Load Text --------------------
def load_text():
    if os.path.exists("sample.txt"):
        with open("sample.txt", "r", encoding="utf-8") as f:
            return f.read()
    else:
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file is not None:
            return uploaded_file.read().decode("utf-8")
    return None

# -------------------- 2. Preprocess --------------------
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -------------------- 3. Chatbot Functions --------------------
def build_bot(raw_text):
    sentences = nltk.sent_tokenize(raw_text)
    corpus = [preprocess(s) for s in sentences]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    def get_most_relevant_sentence(query):
        query_processed = preprocess(query)
        query_vec = vectorizer.transform([query_processed])
        similarity_scores = cosine_similarity(query_vec, X).flatten()
        idx = np.argmax(similarity_scores)
        return sentences[idx]

    def chatbot(user_input):
        return get_most_relevant_sentence(user_input)

    return chatbot

# -------------------- 4. Streamlit App --------------------
def main():
    st.title("📚 Chatbot from Your Text File")

    raw_text = load_text()

    if raw_text:
        chatbot = build_bot(raw_text)

        user_input = st.text_input("You:", "")

        if st.button("Ask"):
            if user_input.strip():
                st.write(f" Bot: {chatbot(user_input)}")
            else:
                st.write("Please enter a question.")
    else:
        st.info("Please upload a text file to start chatting.")

if __name__ == "__main__":
    main()
