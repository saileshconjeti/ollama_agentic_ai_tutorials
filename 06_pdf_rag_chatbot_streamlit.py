import os
import tempfile
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Local PDF RAG Chatbot", layout="wide")

st.title("Local PDF RAG Chatbot")
st.write("Upload PDFs, build an index locally, and ask questions.")

st.sidebar.header("Settings")
chat_model_name = st.sidebar.selectbox(
    "Chat model",
    ["qwen3:4b", "gemma3:4b", "llama3.1:8b"],
    index=0,
)
embedding_model_name = st.sidebar.selectbox(
    "Embedding model",
    ["nomic-embed-text", "qwen3-embedding:0.6b"],
    index=0,
)

chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 50, 500, 200, 50)
top_k = st.sidebar.slider("Top-k", 2, 8, 4, 1)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

def load_uploaded_pdfs(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = Path(tmp.name)

        try:
            loader = PyPDFLoader(str(tmp_path))
            docs.extend(loader.load())
        finally:
            # Keep temp file lifecycle explicit for tutorial clarity.
            if tmp_path.exists():
                os.unlink(tmp_path)
    return docs

def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=embedding_model_name)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        # Unique name avoids mixing chunks across repeated "Build index" clicks.
        collection_name=f"pdf_rag_demo_{uuid4().hex[:8]}",
    )
    return vectorstore, chunks

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Build index"):
        with st.spinner("Building index..."):
            try:
                docs = load_uploaded_pdfs(uploaded_files)
                if not docs:
                    st.error("No text could be extracted from the uploaded PDFs.")
                else:
                    vectorstore, chunks = build_vector_store(docs)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chunk_count = len(chunks)
                    st.success(f"Built index with {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Failed to build index: {e}")

question = st.text_input("Ask a question about your PDFs")

if question and st.session_state.vectorstore is not None:
    try:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join(
            f"[Page {doc.metadata.get('page', 'unknown')}]\n{doc.page_content}"
            for doc in retrieved_docs
        )

        prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say you do not know.

Question:
{question}

Context:
{context}
"""

        llm = ChatOllama(model=chat_model_name, temperature=0)
        answer = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(answer.content)

        st.subheader("Retrieved context")
        for i, doc in enumerate(retrieved_docs, start=1):
            st.markdown(f"**Chunk {i} — page {doc.metadata.get('page', 'unknown')}**")
            st.code(doc.page_content[:1200])
    except Exception as e:
        st.error(f"Failed to answer question: {e}")
elif question and st.session_state.vectorstore is None:
    st.warning("Please upload PDFs and click Build index first.")

if st.session_state.vectorstore is None:
    st.info("Upload PDFs and click Build index.")
else:
    st.caption(f"Index ready: {st.session_state.chunk_count} chunks")

if st.button("Clear index"):
    st.session_state.vectorstore = None
    st.session_state.chunk_count = 0
    st.success("Index cleared.")
