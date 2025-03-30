import os
import sys
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
from dotenv import load_dotenv  # Load environment variables

# Load .env variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key not found! Make sure your .env file is correctly set.")

# Download NLTK resources

# Set NLTK data path
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

# Download required NLTK resources
nltk.download('punkt', download_dir="nltk_data")
nltk.download('wordnet', download_dir="nltk_data")
nltk.download('omw-1.4', download_dir="nltk_data")
nltk.download('averaged_perceptron_tagger', download_dir="nltk_data")

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

def truncate_text(text, max_tokens=1024):
    """Truncate text to fit within the model's token limit."""
    words = text.split()[:max_tokens]  # Take only the first max_tokens words
    return " ".join(words)

def search_faiss(query, index, vectorizer, text_chunks):
    """Find the closest matching text in FAISS index"""
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    D, I = index.search(query_vector, 1)  # Get closest match
    if I[0][0] == -1:
        return "No relevant context found."
    return text_chunks[I[0][0]]

def query_groq(question, context):
    """Send query to Groq API with truncated context"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    truncated_context = truncate_text(context, max_tokens=1024)  # Limit context size
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Context: {truncated_context}\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content


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

# Run Streamlit automatically if executed as a script
if __name__ == "__main__":
    if not any(arg.endswith("streamlit") for arg in sys.argv):  # Ensure not already in Streamlit mode
        os.system(f"streamlit run {sys.argv[0]}")
        sys.exit()
