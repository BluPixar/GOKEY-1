import streamlit as st
import fitz  # PyMuPDF
import os
import nltk
import json
from datetime import datetime
from groq import Groq
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Ensure NLTK data is downloaded
nltk.data.path.append("nltk_data")
nltk.download("punkt", download_dir="nltk_data")
nltk.download("stopwords", download_dir="nltk_data")
nltk.download("wordnet", download_dir="nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="nltk_data")
nltk.download("omw-1.4", download_dir="nltk_data")

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# History log for current session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return preprocess_text(text)

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    return " ".join(lemmatized)

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": "a", "N": "n", "V": "v", "R": "r"}
    return tag_dict.get(tag, "n")

def generate_response(document_text, user_question):
    prompt = f"Document: {document_text}\n\nQuestion: {user_question}\n\nAnswer:"
    chat_completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return chat_completion.choices[0].message.content

st.set_page_config(page_title="AI PDF Q&A", layout="centered")
st.title("📄 AI-Powered Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    document_text = extract_text_from_pdf(uploaded_file)
    user_question = st.text_input("Ask a question about the document")

    if user_question:
        ai_response = generate_response(document_text, user_question)
        st.markdown(f"**Answer:** {ai_response}")

        # Save to session history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": ai_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Display History
    if st.session_state.chat_history:
        st.subheader("📝 Chat History")
        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Q{i+1}:** {entry['question']}")
            st.markdown(f"**A{i+1}:** {entry['answer']}")
            st.markdown(f"⏱ {entry['timestamp']}")
            st.markdown("---")
