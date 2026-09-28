"""
RAG Chain Module for LearnLens AI.
Builds retrieval-augmented generation chains using Google Gemini.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


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


# ---------------------------------------------------------------------------
# Prompt templates for different modes
# ---------------------------------------------------------------------------

_PROMPTS = {
    "chat": PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are LearnLens AI, an intelligent study assistant. "
            "Use the following context extracted from the student's study materials "
            "to answer the question. Always cite which source document and page/slide "
            "the information came from. If the context does not contain enough "
            "information, say 'I don't have enough information in your uploaded "
            "documents to answer this question.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer (with source references):"
        ),
    ),
    "explain": PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are LearnLens AI, a friendly study tutor. "
            "Using the following context from the student's study materials, "
            "explain the requested concept in simple, easy-to-understand terms. "
            "Use analogies, real-world examples, and step-by-step breakdowns. "
            "Make it accessible to a beginner.\n\n"
            "Context:\n{context}\n\n"
            "Topic to explain: {question}\n\n"
            "Simplified Explanation:"
        ),
    ),
    "exam_notes": PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are LearnLens AI, an exam preparation specialist. "
            "Using the following context from the student's study materials, "
            "generate concise, exam-oriented notes. Include:\n"
            "• Key definitions and terminology\n"
            "• Important formulas or equations\n"
            "• Critical concepts and their relationships\n"
            "• Potential exam questions and brief answers\n"
            "• Mnemonics or memory aids where helpful\n\n"
            "Format the notes in a clear, scannable structure with headers and bullet points.\n\n"
            "Context:\n{context}\n\n"
            "Topic: {question}\n\n"
            "Exam-Oriented Notes:"
        ),
    ),
    "summary": PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are LearnLens AI, a study summarization expert. "
            "Using the following context from the student's study materials, "
            "provide a comprehensive and well-structured summary. Cover all major "
            "topics and sub-topics. Organize the summary with clear headings, "
            "sub-headings, and bullet points. Highlight the most important takeaways.\n\n"
            "Context:\n{context}\n\n"
            "Request: {question}\n\n"
            "Comprehensive Summary:"
        ),
    ),
}


def get_qa_chain(retriever, mode: str = "chat") -> RetrievalQA:
    """
    Build and return a RetrievalQA chain for the given mode.

    Args:
        retriever: A LangChain retriever (e.g. from FAISS).
        mode: One of 'chat', 'explain', 'exam_notes', 'summary'.

    Returns:
        A RetrievalQA chain that returns the result and source documents.
    """
    api_key = _get_api_key()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True,
    )

    prompt = _PROMPTS.get(mode, _PROMPTS["chat"])

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


def run_chain(chain: RetrievalQA, query: str) -> dict:
    """
    Run a RetrievalQA chain and return the result.

    Args:
        chain: A RetrievalQA chain instance.
        query: The user's question or request.

    Returns:
        A dict with 'result' (str) and 'source_documents' (list of Documents).
    """
    try:
        response = chain.invoke({"query": query})
        return {
            "result": response.get("result", ""),
            "source_documents": response.get("source_documents", []),
        }
    except Exception as e:
        return {
            "result": f"An error occurred: {str(e)}",
            "source_documents": [],
        }
