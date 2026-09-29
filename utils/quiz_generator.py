"""
Quiz Generator Module for LearnLens AI.
Generates MCQ quizzes from uploaded study materials using Gemini.
"""

import json
import os
import re
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI


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


def generate_quiz(
    retriever,
    topic: str,
    num_questions: int = 5,
    difficulty: str = "medium",
) -> List[Dict]:
    """
    Generate a multiple-choice quiz based on the uploaded study materials.

    Args:
        retriever: A LangChain retriever to fetch relevant context.
        topic: The topic for the quiz.
        num_questions: Number of questions to generate (3-10).
        difficulty: Difficulty level - 'easy', 'medium', or 'hard'.

    Returns:
        A list of dicts, each with keys:
            - question (str)
            - options (list of 4 str)
            - correct_answer (str)
            - explanation (str)
    """
    # Retrieve relevant context
    relevant_docs = retriever.invoke(topic)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    if not context.strip():
        return []

    prompt = f"""You are a quiz generator for students. Based on the following study material context, 
generate exactly {num_questions} multiple-choice questions at {difficulty} difficulty level about "{topic}".

Context from study materials:
{context}

Generate the quiz as a JSON array. Each element must have exactly these keys:
- "question": the question text
- "options": an array of exactly 4 answer options (strings)
- "correct_answer": the correct option text (must match one of the options exactly)
- "explanation": a brief explanation of why this is the correct answer

Return ONLY the JSON array, no other text. Example format:
[
  {{
    "question": "What is ...?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Because ..."
  }}
]"""

    api_key = _get_api_key()
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.7,
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Try to extract JSON from the response
        # Remove markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        quiz_data = json.loads(content)

        # Validate structure
        validated = []
        for item in quiz_data:
            if (
                isinstance(item, dict)
                and "question" in item
                and "options" in item
                and "correct_answer" in item
                and isinstance(item["options"], list)
                and len(item["options"]) == 4
            ):
                validated.append(
                    {
                        "question": str(item["question"]),
                        "options": [str(o) for o in item["options"]],
                        "correct_answer": str(item["correct_answer"]),
                        "explanation": str(item.get("explanation", "No explanation provided.")),
                    }
                )
        return validated

    except json.JSONDecodeError:
        # Fallback: try to parse partial JSON
        try:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                quiz_data = json.loads(match.group())
                validated = []
                for item in quiz_data:
                    if isinstance(item, dict) and "question" in item and "options" in item:
                        validated.append(
                            {
                                "question": str(item.get("question", "")),
                                "options": [str(o) for o in item.get("options", [])],
                                "correct_answer": str(item.get("correct_answer", "")),
                                "explanation": str(item.get("explanation", "")),
                            }
                        )
                return validated
        except Exception:
            pass
        return []
    except Exception:
        return []
