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

# Set NLTK data path
nltk_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_path, exist_ok=True)
nltk.data.path.append(nltk_path)

# Download required NLTK resources safely
for resource in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_path)

# Force-load Punkt tokenizer if needed (fix for punkt_tab issue)
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk import data as nltk_data
import pickle

try:
    _ = PunktSentenceTokenizer()
except LookupError:
    print("Punkt tokenizer not found. Training a fallback tokenizer.")
    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train("This is a sentence. Here's another one. And a third!")
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    
    tokenizer_path = os.path.join(nltk_path, "tokenizers", "punkt", "english.pickle")
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    nltk_data.load("tokenizers/punkt/english.pickle")


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
    words = text.split()[:max_tokens]
    return " ".join(words)

def search_faiss(query, index, vectorizer, text_chunks):
    """Find the closest matching text in FAISS index"""
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    D, I = index.search(query_vector, 1)
    if I[0][0] == -1:
        return "No relevant context found."
    return text_chunks[I[0][0]]

def query_groq(question, context):
    """Send query to Groq API with truncated context"""
    client = Groq(api_key=groq_api_key)

    truncated_context = truncate_text(context, max_tokens=1024)
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": f"Context: {truncated_context}\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content


# ========== Streamlit App ==========

st.title("AI-Powered Document Q&A")

# Use session state to keep history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# To create conversation context
def build_conversation_context():
    """Builds conversation context from the chat history."""
    context = ""
    for q, a in st.session_state.chat_history:
        context += f"Q: {q}\nA: {a}\n\n"
    return context


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("✅ PDF Uploaded Successfully!")
    document_text = extract_text_from_pdf(uploaded_file)
    text_chunks = document_text.split(". ")
    faiss_index, vectorizer = create_faiss_index(text_chunks)

    query = st.text_input("Ask a question from the document:")
    if query:
        context = build_conversation_context()  # Get ongoing conversation context
        if context:
            context += f"Q: {query}\n"  # Add the current query to the context
        
        # Search for relevant context from the document
        document_context = search_faiss(query, faiss_index, vectorizer, text_chunks)
        
        # If the document context is relevant, add it to the conversation context
        if document_context:
            context += f"A (from document): {document_context}\n"

        # Send to Groq
        answer = query_groq(query, context)

        # Save to history
        st.session_state.chat_history.append((query, answer))

        st.markdown("Answer:")
        st.write(answer)

    # Display History Log
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")
            st.markdown("---")

# Auto run
if __name__ == "__main__":
    if not any(arg.endswith("streamlit") for arg in sys.argv):
        os.system(f"streamlit run {sys.argv[0]}")
        sys.exit()
