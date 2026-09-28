"""
Embeddings & Vector Store Module for LearnLens AI.
Creates FAISS vector stores from document chunks using Google Generative AI embeddings.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def _get_api_key() -> str:
    """Retrieve the Gemini API key from session state or environment."""
    try:
        import streamlit as st
        key = st.session_state.get("gemini_api_key", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Return a GoogleGenerativeAIEmbeddings instance.

    Returns:
        An embeddings object configured with the Gemini API key.
    """
    api_key = _get_api_key()
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )


def create_vector_store(chunks: List[Document]) -> FAISS:
    """
    Create a FAISS vector store from a list of document chunks.

    Args:
        chunks: List of LangChain Document objects (already split).

    Returns:
        A FAISS vector store populated with the chunk embeddings.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_retriever(vector_store: FAISS, k: int = 5):
    """
    Create a retriever from a FAISS vector store.

    Args:
        vector_store: A FAISS vector store instance.
        k: Number of documents to retrieve per query.

    Returns:
        A LangChain retriever.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
