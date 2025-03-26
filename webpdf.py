import os
os.environ["STREAMLIT_RUN_CONTEXT"] = "1"

import streamlit as st
import faiss
import numpy as np
import fitz  # PyMuPDF for PDF parsing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from groq import Groq
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text):
    """Tokenization, Lemmatization, and Cleaning"""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return preprocess_text(text)

def create_faiss_index(text_chunks):
    """Convert text chunks to vectorized form and store in FAISS"""
    vectorizer = CountVectorizer(binary=True)
    vectors = vectorizer.fit_transform(text_chunks).toarray()
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors, dtype=np.float32))
    return index, vectorizer

def search_faiss(query, index, vectorizer, text_chunks):
    """Find the closest matching text in FAISS index"""
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    _, I = index.search(query_vector, 1)  # Get closest match
    return text_chunks[I[0][0]]

def query_groq(question, context):
    """Send query to Groq API with extracted context"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.generate(f"Context: {context}\nQuestion: {question}")
    return response

# Streamlit UI
st.title("AI-Powered Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF Uploaded Successfully!")
    document_text = extract_text_from_pdf(uploaded_file)
    text_chunks = document_text.split(". ")
    faiss_index, vectorizer = create_faiss_index(text_chunks)
    
    query = st.text_input("Ask a question from the document:")
    if query:
        context = search_faiss(query, faiss_index, vectorizer, text_chunks)
        answer = query_groq(query, context)
        st.write("### Answer:")
        st.write(answer)
import sys

# Check if running as a script directly
if __name__ == "__main__":
    if "streamlit" not in sys.argv[0]:  # Ensure it’s not already in Streamlit mode
        os.system(f"streamlit run {sys.argv[0]}")
        sys.exit()  # Exit the normal Python execution after launching Streamlit
