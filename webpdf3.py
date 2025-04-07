import os
import streamlit as st
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# 💡 Safely download NLTK data
def safe_nltk_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

# ✅ Ensure all required NLTK resources are available
safe_nltk_download('tokenizers/punkt')
safe_nltk_download('corpora/wordnet')
safe_nltk_download('corpora/omw-1.4')
safe_nltk_download('taggers/averaged_perceptron_tagger')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# 📚 Text preprocessing functions
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(text):
    from nltk import word_tokenize, pos_tag
    tokens = word_tokenize(text.lower())
    tagged_tokens = pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]
    return " ".join(lemmatized_tokens)

# 🧠 Embedder using Bag-of-Words
class SimpleBoWEmbedder:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def fit_transform(self, documents):
        return self.vectorizer.fit_transform(documents).toarray()

    def transform(self, documents):
        return self.vectorizer.transform(documents).toarray()

# 📖 PDF Loader
def load_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# 📌 Main app
st.title("📄 AI PDF Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = load_pdf(uploaded_file)
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
        processed_chunks = [preprocess(chunk) for chunk in chunks]

        embedder = SimpleBoWEmbedder()
        doc_vectors = embedder.fit_transform(processed_chunks)

        # 🎯 Create FAISS index
        dimension = doc_vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(doc_vectors).astype(np.float32))

    st.success("PDF processed successfully!")

    query = st.text_input("Ask a question:")

    if query:
        processed_query = preprocess(query)
        query_vector = embedder.transform([processed_query])
        D, I = index.search(np.array(query_vector).astype(np.float32), k=3)

        answers = [chunks[i] for i in I[0]]
        st.subheader("Top Answers:")
        for i, ans in enumerate(answers, 1):
            st.markdown(f"**{i}.** {ans}")
