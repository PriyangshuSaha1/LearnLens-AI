"""
🎓 LearnLens AI — A RAG-Powered Personalized Study Assistant
Main Streamlit application with premium UI.
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

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ─── Import Google Font ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Root variables ─── */
:root {
    --primary: #7C3AED;
    --primary-light: #A78BFA;
    --accent: #06B6D4;
    --accent2: #F59E0B;
    --bg-dark: #0B0F19;
    --bg-card: #111827;
    --bg-card-hover: #1F2937;
    --border: #1F2937;
    --text: #F9FAFB;
    --text-muted: #9CA3AF;
    --success: #10B981;
    --error: #EF4444;
    --warning: #F59E0B;
}

/* ─── Global overrides ─── */
.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ─── Hide default header & footer ─── */
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ─── Sidebar styling ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%) !important;
    border-right: 1px solid rgba(124, 58, 237, 0.2) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    background: linear-gradient(135deg, #A78BFA, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

/* ─── Animated gradient hero ─── */
.hero-container {
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 20px;
    padding: 40px 50px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(124,58,237,0.1) 0%, transparent 50%);
    animation: pulse-bg 4s ease-in-out infinite;
}
@keyframes pulse-bg {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #A78BFA 0%, #06B6D4 50%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    position: relative;
    z-index: 1;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #9CA3AF;
    margin: 0;
    position: relative;
    z-index: 1;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(6,182,212,0.3));
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #A78BFA;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
    position: relative;
    z-index: 1;
}

/* ─── Glass-morphism cards ─── */
.glass-card {
    background: rgba(17, 24, 39, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(124, 58, 237, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(124, 58, 237, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(124, 58, 237, 0.15);
}

/* ─── Stat cards with glow ─── */
.stat-card {
    background: linear-gradient(145deg, #111827, #1E1B4B);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.stat-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7C3AED, #06B6D4);
    border-radius: 16px 16px 0 0;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(124, 58, 237, 0.2);
}
.stat-icon {
    font-size: 2rem;
    margin-bottom: 8px;
}
.stat-number {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #A78BFA, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.stat-label {
    color: #9CA3AF;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ─── Tab styling ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(17, 24, 39, 0.6);
    border-radius: 14px;
    padding: 6px;
    border: 1px solid rgba(124, 58, 237, 0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #9CA3AF;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(6,182,212,0.2)) !important;
    color: #F9FAFB !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ─── Chat messages ─── */
.stChatMessage {
    background: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid rgba(124, 58, 237, 0.1) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}

/* ─── Source reference cards ─── */
.source-ref {
    background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(6,182,212,0.05));
    border-left: 3px solid #7C3AED;
    padding: 12px 16px;
    border-radius: 0 12px 12px 0;
    margin-bottom: 10px;
    font-size: 0.85rem;
    color: #D1D5DB;
    transition: all 0.2s ease;
}
.source-ref:hover {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(6,182,212,0.1));
    border-left-color: #06B6D4;
}
.source-ref b {
    color: #A78BFA;
}

/* ─── Feature cards ─── */
.feature-card {
    background: linear-gradient(145deg, rgba(17,24,39,0.9), rgba(30,27,75,0.5));
    border: 1px solid rgba(124,58,237,0.15);
    border-radius: 16px;
    padding: 30px 24px;
    text-align: center;
    transition: all 0.3s ease;
    min-height: 160px;
}
.feature-card:hover {
    border-color: rgba(124,58,237,0.5);
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(124,58,237,0.15);
}
.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 12px;
    display: block;
}
.feature-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #F9FAFB;
    margin-bottom: 6px;
}
.feature-desc {
    font-size: 0.85rem;
    color: #9CA3AF;
    line-height: 1.5;
}

/* ─── Quiz styling ─── */
.quiz-question {
    background: linear-gradient(145deg, #111827, #1E1B4B);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}
.quiz-number {
    display: inline-block;
    background: linear-gradient(135deg, #7C3AED, #06B6D4);
    color: white;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    line-height: 32px;
    text-align: center;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 10px;
}

/* ─── Buttons ─── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C3AED, #06B6D4) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5) !important;
}

/* ─── Section headers ─── */
.section-header {
    font-size: 1.5rem;
    font-weight: 800;
    color: #F9FAFB;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.section-desc {
    color: #9CA3AF;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* ─── Processed file pills ─── */
.file-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(6,182,212,0.1));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 8px;
    padding: 6px 12px;
    margin: 4px 2px;
    font-size: 0.8rem;
    color: #D1D5DB;
    font-weight: 500;
}

/* ─── Score display ─── */
.score-display {
    background: linear-gradient(145deg, #111827, #1E1B4B);
    border: 2px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}
.score-number {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #A78BFA, #06B6D4, #F59E0B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.score-label {
    color: #9CA3AF;
    font-size: 1rem;
    font-weight: 500;
}

/* ─── Footer ─── */
.app-footer {
    text-align: center;
    padding: 30px 0 15px;
    color: #6B7280;
    font-size: 0.85rem;
    border-top: 1px solid rgba(124,58,237,0.1);
    margin-top: 40px;
}
.app-footer a {
    color: #A78BFA;
    text-decoration: none;
}

/* ─── Divider ─── */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.4), rgba(6,182,212,0.4), transparent);
    border: none;
    margin: 25px 0;
    border-radius: 2px;
}

/* ─── Welcome animation ─── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.6s ease-out;
}

/* ─── Expander styling ─── */
.streamlit-expanderHeader {
    background: rgba(17, 24, 39, 0.6) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* ─── Inputs ─── */
.stTextInput > div > div {
    border-radius: 12px !important;
    border-color: rgba(124,58,237,0.2) !important;
}
.stTextInput > div > div:focus-within {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
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
    st.markdown(
        '<p style="color:#9CA3AF; font-size:0.85rem; margin-top:-10px;">'
        "Your AI-Powered Study Companion</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # API key
    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=st.session_state["gemini_api_key"],
        placeholder="Paste your API key here…",
        help="Get your free key → https://aistudio.google.com/apikey",
    )
    if api_key:
        st.session_state["gemini_api_key"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # File uploader
    uploaded_files = st.file_uploader(
        "📂 Upload Study Materials",
        type=["pdf", "pptx", "docx", "txt"],
        accept_multiple_files=True,
        help="Drag & drop PDFs, PPTs, Word docs, or text files",
    )

    # Process button
    if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
        if not api_key:
            st.error("⚠️ Please enter your Gemini API key first.")
        elif not uploaded_files:
            st.error("⚠️ Please upload at least one document.")
        else:
            with st.status("🔄 Processing your documents…", expanded=True) as status:
                st.write("📄 Loading documents…")
                docs = load_multiple_documents(uploaded_files)
                if not docs:
                    st.error("Could not extract text from the uploaded files.")
                else:
                    st.write(f"✅ Loaded {len(docs)} document sections")
                    st.write("✂️ Splitting into chunks…")
                    chunks = split_documents(docs)
                    st.write(f"✅ Created {len(chunks)} chunks")
                    st.write("🧬 Generating embeddings & building index…")
                    try:
                        vector_store = create_vector_store(chunks)
                        st.session_state["vector_store"] = vector_store
                        st.session_state["num_chunks"] = len(chunks)
                        st.session_state["processed_files"] = [
                            f.name for f in uploaded_files
                        ]
                        status.update(
                            label="✅ Processing complete!", state="complete"
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")
                        status.update(label="❌ Error", state="error")

    # Show processed files
    if st.session_state["processed_files"]:
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("**📚 Knowledge Base**")
        for fname in st.session_state["processed_files"]:
            ext = fname.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📕", "PPTX": "📊", "DOCX": "📘", "TXT": "📝"}.get(
                ext, "📄"
            )
            st.markdown(
                f'<span class="file-pill">{icon} {fname}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<p style="color:#6B7280; font-size:0.8rem; margin-top:8px;">'
            f'{st.session_state["num_chunks"]} chunks indexed</p>',
            unsafe_allow_html=True,
        )

    # Clear button
    if st.session_state["vector_store"] is not None:
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            for key, val in _DEFAULTS.items():
                st.session_state[key] = val
            st.rerun()

    # Sidebar footer
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#4B5563; font-size:0.75rem; text-align:center;">'
        "LearnLens AI v1.0 · Built with Streamlit</p>",
        unsafe_allow_html=True,
    )

# ── Main content ──────────────────────────────────────────────────────────────

# Gate: need API key + processed docs for most features
_ready = (
    st.session_state["vector_store"] is not None
    and st.session_state["gemini_api_key"]
)

# ── HERO SECTION ──────────────────────────────────────────────────────────────
if not _ready:
    # Show landing / welcome page
    st.markdown(
        """
        <div class="hero-container animate-in">
            <div class="hero-badge">✨ AI-Powered Learning</div>
            <h1 class="hero-title">LearnLens AI</h1>
            <p class="hero-subtitle">
                Upload your study materials and unlock AI-powered learning.
                Get instant answers, smart summaries, exam notes, and auto-generated quizzes
                — all grounded in your own documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    cols = st.columns(4)
    features = [
        ("💬", "Smart Chat", "Ask questions and get precise answers with source citations from your materials"),
        ("📋", "Auto Summary", "Generate comprehensive summaries and concise exam-oriented revision notes"),
        ("💡", "ELI5 Mode", "Complex topics broken down into simple explanations with analogies"),
        ("🧠", "Quiz Generator", "Auto-generated MCQ quizzes with difficulty levels and instant grading"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card animate-in">
                    <span class="feature-icon">{icon}</span>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # Getting started steps
    if not st.session_state["gemini_api_key"]:
        st.info("**Step 1:** Enter your Google Gemini API key in the sidebar → Get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)")
    elif st.session_state["vector_store"] is None:
        st.info("**Step 2:** Upload your study materials (PDF, PPTX, DOCX, TXT) and click **⚡ Process Documents**")

else:
    # Compact header when ready
    st.markdown(
        """
        <div style="margin-bottom: 20px;" class="animate-in">
            <div class="hero-badge">✨ Knowledge Base Active</div>
            <h1 class="hero-title" style="font-size: 2rem;">LearnLens AI</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_notes, tab_quiz, tab_docs = st.tabs(
    ["💬 Chat", "📝 Summary & Notes", "🧠 Quiz", "📚 Documents"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — CHAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_chat:
    if not _ready:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:50px 30px;">
                <span style="font-size:3rem;">💬</span>
                <h3 style="color:#F9FAFB; margin-top:10px;">Chat with Your Documents</h3>
                <p style="color:#9CA3AF;">Upload and process your study materials to start chatting.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="section-header">💬 Ask Your Documents</p>'
            '<p class="section-desc">Get precise, cited answers from your study materials</p>',
            unsafe_allow_html=True,
        )

        # Display chat history
        for msg in st.session_state["chat_history"]:
            with st.chat_message(
                msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"
            ):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📖 View Source References"):
                        for src in msg["sources"]:
                            loc = ""
                            if src.get("page"):
                                loc = f" · Page {src['page']}"
                            if src.get("slide"):
                                loc = f" · Slide {src['slide']}"
                            st.markdown(
                                f'<div class="source-ref">'
                                f'<b>📄 {src["source"]}</b>{loc}<br/>'
                                f'<span style="color:#6B7280;">{src["snippet"]}</span></div>',
                                unsafe_allow_html=True,
                            )

        # Chat input
        if user_query := st.chat_input("Ask anything about your study materials…"):
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_query}
            )
            with st.chat_message("user", avatar="🧑‍🎓"):
                st.markdown(user_query)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("🔍 Searching & generating answer…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    chain = get_qa_chain(retriever, mode="chat")
                    response = run_chain(chain, user_query)

                answer = response["result"]
                st.markdown(answer)

                sources = []
                for doc in response["source_documents"]:
                    sources.append(
                        {
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page"),
                            "slide": doc.metadata.get("slide"),
                            "snippet": doc.page_content[:180] + "…",
                        }
                    )
                if sources:
                    with st.expander("📖 View Source References"):
                        for src in sources:
                            loc = ""
                            if src.get("page"):
                                loc = f" · Page {src['page']}"
                            if src.get("slide"):
                                loc = f" · Slide {src['slide']}"
                            st.markdown(
                                f'<div class="source-ref">'
                                f'<b>📄 {src["source"]}</b>{loc}<br/>'
                                f'<span style="color:#6B7280;">{src["snippet"]}</span></div>',
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
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:50px 30px;">
                <span style="font-size:3rem;">📝</span>
                <h3 style="color:#F9FAFB; margin-top:10px;">Smart Summaries & Notes</h3>
                <p style="color:#9CA3AF;">Upload and process your study materials first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="section-header">📝 Summary & Notes Generator</p>'
            '<p class="section-desc">Transform your documents into study-ready content</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(
                """
                <div class="glass-card">
                    <span style="font-size:1.5rem;">📋</span>
                    <span style="font-size:1.1rem; font-weight:700; color:#F9FAFB;"> Comprehensive Summary</span>
                    <p style="color:#9CA3AF; font-size:0.85rem; margin-top:4px;">
                    Get a detailed overview of all your uploaded materials</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "✨ Generate Summary", key="btn_summary", use_container_width=True, type="primary"
            ):
                with st.spinner("📋 Generating comprehensive summary…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    chain = get_qa_chain(retriever, mode="summary")
                    response = run_chain(
                        chain,
                        "Provide a comprehensive summary of all the uploaded study material.",
                    )
                st.markdown(response["result"])

        with col2:
            st.markdown(
                """
                <div class="glass-card">
                    <span style="font-size:1.5rem;">🎯</span>
                    <span style="font-size:1.1rem; font-weight:700; color:#F9FAFB;"> Exam-Oriented Notes</span>
                    <p style="color:#9CA3AF; font-size:0.85rem; margin-top:4px;">
                    Key points, definitions, and formulas for quick revision</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "🎯 Generate Exam Notes",
                key="btn_exam",
                use_container_width=True,
                type="primary",
            ):
                with st.spinner("🎯 Generating exam notes…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    chain = get_qa_chain(retriever, mode="exam_notes")
                    response = run_chain(
                        chain,
                        "Generate exam-oriented notes covering all key topics from the study material.",
                    )
                st.markdown(response["result"])

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="glass-card">
                <span style="font-size:1.5rem;">💡</span>
                <span style="font-size:1.1rem; font-weight:700; color:#F9FAFB;"> Simplified Explanation</span>
                <p style="color:#9CA3AF; font-size:0.85rem; margin-top:4px;">
                Break down complex topics into simple, easy-to-understand language</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ecol1, ecol2 = st.columns([3, 1])
        with ecol1:
            explain_topic = st.text_input(
                "Topic to simplify",
                placeholder="e.g. Binary Search Trees, Newton's Laws, Photosynthesis…",
                label_visibility="collapsed",
            )
        with ecol2:
            explain_btn = st.button(
                "💡 Simplify", key="btn_explain", use_container_width=True, type="primary"
            )
        if explain_btn:
            if not explain_topic:
                st.warning("Please enter a topic first.")
            else:
                with st.spinner("💡 Simplifying…"):
                    retriever = get_retriever(st.session_state["vector_store"])
                    chain = get_qa_chain(retriever, mode="explain")
                    response = run_chain(chain, explain_topic)
                st.markdown(response["result"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — QUIZ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_quiz:
    if not _ready:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:50px 30px;">
                <span style="font-size:3rem;">🧠</span>
                <h3 style="color:#F9FAFB; margin-top:10px;">AI Quiz Generator</h3>
                <p style="color:#9CA3AF;">Upload and process your study materials first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="section-header">🧠 Auto-Generated Quiz</p>'
            '<p class="section-desc">Test your knowledge with AI-generated questions from your materials</p>',
            unsafe_allow_html=True,
        )

        # Quiz settings in a glass card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        qcol1, qcol2, qcol3 = st.columns([2, 1, 1])
        with qcol1:
            quiz_topic = st.text_input(
                "📌 Quiz Topic",
                placeholder="e.g. Data Structures, Machine Learning…",
                key="quiz_topic_input",
            )
        with qcol2:
            num_q = st.slider("📊 Questions", 3, 10, 5, key="quiz_num")
        with qcol3:
            difficulty = st.selectbox(
                "⚡ Difficulty",
                ["Easy", "Medium", "Hard"],
                index=1,
                key="quiz_diff",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button(
            "🎲 Generate Quiz",
            use_container_width=True,
            type="primary",
        ):
            if not quiz_topic:
                st.warning("Please enter a quiz topic.")
            else:
                with st.spinner("🧠 Creating quiz questions from your materials…"):
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
                    st.success(f"✅ Generated {len(quiz)} questions! Scroll down to start.")
                else:
                    st.error(
                        "Could not generate quiz. Try a different or more specific topic."
                    )

        # Display quiz
        if st.session_state["quiz_data"]:
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

            for i, q in enumerate(st.session_state["quiz_data"]):
                st.markdown(
                    f'<div class="quiz-question">'
                    f'<span class="quiz-number">{i + 1}</span>'
                    f'<span style="font-weight:600; color:#F9FAFB; font-size:1rem;">'
                    f"{q['question']}</span></div>",
                    unsafe_allow_html=True,
                )
                answer = st.radio(
                    f"Select answer for Q{i + 1}",
                    q["options"],
                    key=f"quiz_q_{i}",
                    label_visibility="collapsed",
                )
                st.session_state["quiz_answers"][i] = answer

            st.markdown("")
            if st.button(
                "📊 Submit & Check Answers",
                use_container_width=True,
                type="primary",
            ):
                st.session_state["quiz_submitted"] = True

            if st.session_state["quiz_submitted"]:
                st.markdown(
                    '<div class="gradient-divider"></div>', unsafe_allow_html=True
                )
                score = 0
                total = len(st.session_state["quiz_data"])

                for i, q in enumerate(st.session_state["quiz_data"]):
                    user_ans = st.session_state["quiz_answers"].get(i, "")
                    is_correct = user_ans == q["correct_answer"]
                    if is_correct:
                        score += 1
                        st.success(f"✅ **Q{i + 1}:** Correct!")
                    else:
                        st.error(
                            f"❌ **Q{i + 1}:** Your answer: *{user_ans}* → "
                            f"Correct: **{q['correct_answer']}**"
                        )
                    with st.expander(f"💡 Explanation for Q{i + 1}"):
                        st.write(q["explanation"])

                # Score display
                pct = int(score / total * 100) if total else 0
                emoji = "🏆" if pct >= 80 else "👏" if pct >= 50 else "📚"
                msg = (
                    "Excellent! You've mastered this topic!"
                    if pct >= 80
                    else "Good job! Keep practicing!"
                    if pct >= 50
                    else "Keep studying, you'll get there!"
                )
                if pct >= 80:
                    st.balloons()

                st.markdown(
                    f"""
                    <div class="score-display">
                        <div style="font-size:2rem;">{emoji}</div>
                        <div class="score-number">{score}/{total}</div>
                        <div class="score-label">{pct}% · {msg}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — DOCUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_docs:
    if not _ready:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:50px 30px;">
                <span style="font-size:3rem;">📚</span>
                <h3 style="color:#F9FAFB; margin-top:10px;">Document Overview</h3>
                <p style="color:#9CA3AF;">Upload and process your study materials first.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="section-header">📚 Knowledge Base Overview</p>'
            '<p class="section-desc">Your indexed study materials at a glance</p>',
            unsafe_allow_html=True,
        )

        # Stats
        dcol1, dcol2, dcol3 = st.columns(3, gap="large")
        with dcol1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-icon">📄</div>
                    <div class="stat-number">{len(st.session_state["processed_files"])}</div>
                    <div class="stat-label">Documents</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with dcol2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-icon">🧩</div>
                    <div class="stat-number">{st.session_state["num_chunks"]}</div>
                    <div class="stat-label">Text Chunks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with dcol3:
            types = set(
                f.rsplit(".", 1)[-1].upper()
                for f in st.session_state["processed_files"]
            )
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-icon">📁</div>
                    <div class="stat-number" style="font-size:1.5rem;">{", ".join(sorted(types)) if types else "—"}</div>
                    <div class="stat-label">File Types</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # File list
        st.markdown(
            '<p style="font-weight:700; color:#F9FAFB; font-size:1.1rem;">📑 Indexed Files</p>',
            unsafe_allow_html=True,
        )
        for fname in st.session_state["processed_files"]:
            ext = fname.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📕", "PPTX": "📊", "DOCX": "📘", "TXT": "📝"}.get(
                ext, "📄"
            )
            st.markdown(
                f'<div class="source-ref"><b>{icon} {fname}</b> · '
                f'<span style="color:#6B7280;">{ext} file</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # Semantic search
        st.markdown(
            '<p style="font-weight:700; color:#F9FAFB; font-size:1.1rem;">🔍 Semantic Search</p>'
            '<p style="color:#9CA3AF; font-size:0.85rem; margin-top:-5px;">Find specific information across all your documents</p>',
            unsafe_allow_html=True,
        )
        scol1, scol2 = st.columns([4, 1])
        with scol1:
            search_query = st.text_input(
                "Search",
                placeholder="Type a keyword or phrase to search…",
                label_visibility="collapsed",
            )
        with scol2:
            search_btn = st.button(
                "🔍 Search", key="btn_search", use_container_width=True, type="primary"
            )
        if search_btn:
            if not search_query:
                st.warning("Please enter a search query.")
            else:
                with st.spinner("🔍 Searching…"):
                    retriever = get_retriever(st.session_state["vector_store"], k=5)
                    results = retriever.invoke(search_query)
                if results:
                    for i, doc in enumerate(results):
                        src = doc.metadata.get("source", "Unknown")
                        with st.expander(f"Result {i + 1} — {src}"):
                            st.write(doc.page_content)
                            st.caption(
                                f"Source: {doc.metadata.get('source', '?')} · "
                                f"Type: {doc.metadata.get('file_type', '?')}"
                            )
                else:
                    st.info("No relevant results found. Try a different query.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-footer">
        Built with ❤️ using
        <a href="https://streamlit.io" target="_blank">Streamlit</a>,
        <a href="https://www.langchain.com" target="_blank">LangChain</a> &
        <a href="https://ai.google.dev" target="_blank">Google Gemini</a>
    </div>
    """,
    unsafe_allow_html=True,
)
