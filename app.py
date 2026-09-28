"""
🎓 LearnLens AI — A RAG-Powered Personalized Study Assistant
Main Streamlit application.
"""

import os
import streamlit as st

from utils.document_loader import load_multiple_documents
from utils.text_splitter import split_documents
from utils.embeddings import create_vector_store, get_retriever
from utils.rag_chain import get_qa_chain, run_chain
from utils.quiz_generator import generate_quiz

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LearnLens AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #6C63FF 0%, #48C6EF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 25px;
    }
    /* Source reference cards */
    .source-card {
        background: #1E1E2E;
        border-left: 4px solid #6C63FF;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    /* Quiz correct / wrong */
    .quiz-correct { color: #4CAF50; font-weight: 600; }
    .quiz-wrong   { color: #FF5252; font-weight: 600; }
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px 0 10px;
        color: #666;
        font-size: 0.85rem;
    }
    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    .stat-label {
        color: #aaa;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "gemini_api_key": "",
    "vector_store": None,
    "chat_history": [],
    "processed_files": [],
    "num_chunks": 0,
    "quiz_data": [],
    "quiz_submitted": False,
    "quiz_answers": {},
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 LearnLens AI")
    st.caption("RAG-Powered Study Assistant")
    st.divider()

    # API key
    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        value=st.session_state["gemini_api_key"],
        help="Get your free API key from https://aistudio.google.com/apikey",
    )
    if api_key:
        st.session_state["gemini_api_key"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    # File uploader
    uploaded_files = st.file_uploader(
        "📄 Upload Study Materials",
        type=["pdf", "pptx", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload PDFs, PPTs, Word docs, or text files",
    )

    # Process button
    if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key first.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner("Loading documents…"):
                docs = load_multiple_documents(uploaded_files)
            if not docs:
                st.error("Could not extract any text from the uploaded files.")
            else:
                with st.spinner("Splitting into chunks…"):
                    chunks = split_documents(docs)
                progress = st.progress(0, text="Creating embeddings…")
                try:
                    vector_store = create_vector_store(chunks)
                    progress.progress(100, text="Done!")
                    st.session_state["vector_store"] = vector_store
                    st.session_state["num_chunks"] = len(chunks)
                    st.session_state["processed_files"] = [f.name for f in uploaded_files]
                    st.success(
                        f"✅ Processed {len(uploaded_files)} file(s) → {len(chunks)} chunks"
                    )
                except Exception as e:
                    progress.empty()
                    st.error(f"Error creating embeddings: {e}")

    # Show processed files
    if st.session_state["processed_files"]:
        st.divider()
        st.markdown("**📚 Processed Files**")
        for fname in st.session_state["processed_files"]:
            ext = fname.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📕", "PPTX": "📊", "DOCX": "📘", "TXT": "📝"}.get(ext, "📄")
            st.markdown(f"{icon} `{fname}`")

    # Clear button
    if st.session_state["vector_store"] is not None:
        st.divider()
        if st.button("🗑️ Clear All", use_container_width=True):
            for key, val in _DEFAULTS.items():
                st.session_state[key] = val
            st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🎓 LearnLens AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload your study materials and let AI help you learn smarter</p>',
    unsafe_allow_html=True,
)

# Gate: need API key + processed docs for most features
_ready = st.session_state["vector_store"] is not None and st.session_state["gemini_api_key"]

if not st.session_state["gemini_api_key"]:
    st.info("👈 Start by entering your **Google Gemini API key** in the sidebar.")
elif st.session_state["vector_store"] is None:
    st.info("👈 Upload study materials and click **Process Documents** to get started.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_notes, tab_quiz, tab_docs = st.tabs(
    ["💬 Chat", "📝 Summary & Notes", "🧠 Quiz", "📚 Documents"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — CHAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_chat:
    if not _ready:
        st.warning("Please upload & process documents first.")
    else:
        # Display chat history
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📖 Source References"):
                        for src in msg["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>{src["source"]}</b>'
                                f'{" — Page " + str(src["page"]) if src.get("page") else ""}'
                                f'{" — Slide " + str(src["slide"]) if src.get("slide") else ""}'
                                f'<br/><small>{src["snippet"]}</small></div>',
                                unsafe_allow_html=True,
                            )

        # Chat input
        if user_query := st.chat_input("Ask anything about your study materials…"):
            # Show user message
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    chain = get_qa_chain(retriever, mode="chat")
                    response = run_chain(chain, user_query)

                answer = response["result"]
                st.markdown(answer)

                # Build source list
                sources = []
                for doc in response["source_documents"]:
                    sources.append(
                        {
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page"),
                            "slide": doc.metadata.get("slide"),
                            "snippet": doc.page_content[:200] + "…",
                        }
                    )
                if sources:
                    with st.expander("📖 Source References"):
                        for src in sources:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>{src["source"]}</b>'
                                f'{" — Page " + str(src["page"]) if src.get("page") else ""}'
                                f'{" — Slide " + str(src["slide"]) if src.get("slide") else ""}'
                                f'<br/><small>{src["snippet"]}</small></div>',
                                unsafe_allow_html=True,
                            )

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — SUMMARY & NOTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_notes:
    if not _ready:
        st.warning("Please upload & process documents first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Comprehensive Summary")
            if st.button("Generate Summary", key="btn_summary", use_container_width=True):
                with st.spinner("Generating summary…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    chain = get_qa_chain(retriever, mode="summary")
                    response = run_chain(
                        chain,
                        "Provide a comprehensive summary of all the uploaded study material.",
                    )
                st.markdown(response["result"])

        with col2:
            st.subheader("🎯 Exam-Oriented Notes")
            if st.button("Generate Exam Notes", key="btn_exam", use_container_width=True):
                with st.spinner("Generating exam notes…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    chain = get_qa_chain(retriever, mode="exam_notes")
                    response = run_chain(
                        chain,
                        "Generate exam-oriented notes covering all key topics from the study material.",
                    )
                st.markdown(response["result"])

        st.divider()
        st.subheader("💡 Simplified Explanation")
        explain_topic = st.text_input(
            "Enter a topic or concept to simplify",
            placeholder="e.g. Binary Search Trees, Newton's Laws, etc.",
        )
        if st.button("Simplify", key="btn_explain", use_container_width=True):
            if not explain_topic:
                st.warning("Please enter a topic first.")
            else:
                with st.spinner("Simplifying…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    chain = get_qa_chain(retriever, mode="explain")
                    response = run_chain(chain, explain_topic)
                st.markdown(response["result"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — QUIZ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_quiz:
    if not _ready:
        st.warning("Please upload & process documents first.")
    else:
        st.subheader("🧠 Auto-Generated Quiz")

        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            quiz_topic = st.text_input(
                "Quiz Topic",
                placeholder="e.g. Data Structures",
                key="quiz_topic_input",
            )
        with qcol2:
            num_q = st.slider("Number of Questions", 3, 10, 5, key="quiz_num")
        with qcol3:
            difficulty = st.selectbox(
                "Difficulty", ["Easy", "Medium", "Hard"], index=1, key="quiz_diff"
            )

        if st.button("🎲 Generate Quiz", use_container_width=True, type="primary"):
            if not quiz_topic:
                st.warning("Please enter a quiz topic.")
            else:
                with st.spinner("Generating quiz questions…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    quiz = generate_quiz(
                        retriever,
                        quiz_topic,
                        num_questions=num_q,
                        difficulty=difficulty.lower(),
                    )
                if quiz:
                    st.session_state["quiz_data"] = quiz
                    st.session_state["quiz_submitted"] = False
                    st.session_state["quiz_answers"] = {}
                    st.success(f"✅ Generated {len(quiz)} questions!")
                else:
                    st.error("Could not generate quiz. Try a different topic.")

        # Display quiz
        if st.session_state["quiz_data"]:
            st.divider()
            for i, q in enumerate(st.session_state["quiz_data"]):
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                answer = st.radio(
                    f"Select your answer for Q{i + 1}",
                    q["options"],
                    key=f"quiz_q_{i}",
                    label_visibility="collapsed",
                )
                st.session_state["quiz_answers"][i] = answer
                st.markdown("---")

            if st.button("📊 Submit Quiz", use_container_width=True, type="primary"):
                st.session_state["quiz_submitted"] = True

            if st.session_state["quiz_submitted"]:
                score = 0
                total = len(st.session_state["quiz_data"])
                st.divider()
                st.subheader("📊 Results")
                for i, q in enumerate(st.session_state["quiz_data"]):
                    user_ans = st.session_state["quiz_answers"].get(i, "")
                    is_correct = user_ans == q["correct_answer"]
                    if is_correct:
                        score += 1
                        st.success(f"✅ Q{i + 1}: Correct!")
                    else:
                        st.error(
                            f"❌ Q{i + 1}: Wrong — Your answer: {user_ans} | "
                            f"Correct: {q['correct_answer']}"
                        )
                    with st.expander(f"Explanation for Q{i + 1}"):
                        st.write(q["explanation"])

                # Score display
                pct = int(score / total * 100) if total else 0
                if pct >= 80:
                    st.balloons()
                st.markdown(
                    f"### 🏆 Your Score: **{score}/{total}** ({pct}%)"
                )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — DOCUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_docs:
    if not _ready:
        st.warning("Please upload & process documents first.")
    else:
        st.subheader("📚 Document Overview")

        # Stats
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.markdown(
                f'<div class="stat-card"><div class="stat-number">'
                f'{len(st.session_state["processed_files"])}</div>'
                f'<div class="stat-label">Documents</div></div>',
                unsafe_allow_html=True,
            )
        with dcol2:
            st.markdown(
                f'<div class="stat-card"><div class="stat-number">'
                f'{st.session_state["num_chunks"]}</div>'
                f'<div class="stat-label">Text Chunks</div></div>',
                unsafe_allow_html=True,
            )
        with dcol3:
            types = set(
                f.rsplit(".", 1)[-1].upper()
                for f in st.session_state["processed_files"]
            )
            st.markdown(
                f'<div class="stat-card"><div class="stat-number">'
                f'{", ".join(types) if types else "—"}</div>'
                f'<div class="stat-label">File Types</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # File list
        st.markdown("**Uploaded Files:**")
        for fname in st.session_state["processed_files"]:
            ext = fname.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📕", "PPTX": "📊", "DOCX": "📘", "TXT": "📝"}.get(ext, "📄")
            st.markdown(f"- {icon} **{fname}** (`.{ext.lower()}`)")

        st.divider()

        # Semantic search
        st.subheader("🔍 Search Within Documents")
        search_query = st.text_input(
            "Search your documents",
            placeholder="Type a keyword or phrase to search…",
        )
        if st.button("Search", key="btn_search", use_container_width=True):
            if not search_query:
                st.warning("Please enter a search query.")
            else:
                with st.spinner("Searching…"):
                    retriever = get_retriever(st.session_state["vector_store"], k=5)
                    results = retriever.invoke(search_query)
                if results:
                    for i, doc in enumerate(results):
                        with st.expander(
                            f"Result {i + 1} — {doc.metadata.get('source', 'Unknown')}"
                        ):
                            st.write(doc.page_content)
                            st.caption(
                                f"Source: {doc.metadata.get('source', '?')} | "
                                f"Type: {doc.metadata.get('file_type', '?')}"
                            )
                else:
                    st.info("No relevant results found.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit, LangChain &amp; Google Gemini</div>',
    unsafe_allow_html=True,
)
