import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts, _model):
    return _model.encode(texts)

@st.cache_data
def load_documents_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]
    documents = [f"{lines[i]} {lines[i+1]}" for i in range(0, len(lines), 2)]
    return documents

@st.cache_resource
def create_faiss_index(_documents, _model):
    document_embeddings = generate_embeddings(_documents, _model)
    d = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(document_embeddings))
    return index, document_embeddings

def retrieve(query, _model, index, documents, top_k=2):
    query_embedding = generate_embeddings([query], _model)
    D, I = index.search(np.array(query_embedding), top_k)
    return [(documents[i], D[0][idx]) for idx, i in enumerate(I[0])]

def main():
    st.title("🔍 Semantic Search with Sentence Transformers + FAISS")
    st.markdown("Upload a `.txt` file with Q&A pairs (2 lines per pair).")

    uploaded_file = st.file_uploader("Upload your text file", type="txt")

    if uploaded_file:
        filepath = "uploaded_documents.txt"
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        _model = load_model()
        documents = load_documents_from_file(filepath)
        index, _ = create_faiss_index(documents, _model)

        query = st.text_input("Enter a query to search documents:")

        if query:
            results = retrieve(query, _model, index, documents)
            st.subheader("🔎 Search Results:")
            for doc, score in results:
                st.write(f"**Score**: {score:.4f}")
                st.write(f"{doc}")
                st.markdown("---")

if __name__ == "__main__":
    main()