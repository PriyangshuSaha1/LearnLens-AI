"""
Embeddings & Vector Store Module for LearnLens AI.
Creates FAISS vector stores from document chunks using HuggingFace local embeddings.
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance.
    Uses the lightweight and fast all-MiniLM-L6-v2 model.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


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
