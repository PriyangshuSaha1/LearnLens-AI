# 🎓 LearnLens AI — A RAG-Powered Personalized Study Assistant

> **LearnLens AI** is a RAG-based personalized study assistant that allows students to upload study materials such as PDFs, textbooks, lecture notes, and PPTs. The system retrieves relevant information from the uploaded documents and uses an AI language model to provide context-based answers, simplified explanations, summaries, exam-oriented notes, and automatically generated quizzes. The system also provides source references to improve the reliability and transparency of the generated responses.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-Format Upload** | Upload PDFs, PPTX, DOCX, and TXT files |
| 💬 **AI Chat** | Ask questions and get context-based answers with source references |
| 📋 **Smart Summaries** | Generate comprehensive summaries of your study materials |
| 🎯 **Exam Notes** | Auto-generate concise, exam-oriented revision notes |
| 💡 **Simplified Explanations** | Get complex topics explained in simple terms with analogies |
| 🧠 **Auto Quizzes** | Generate MCQ quizzes with configurable difficulty and instant grading |
| 🔍 **Semantic Search** | Search through your documents using natural language |
| 📖 **Source References** | Every answer includes citations back to the original documents |

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) — Interactive Python web UI
- **LLM**: [Google Gemini 2.0 Flash](https://ai.google.dev/) — Fast, high-quality language model
- **RAG Framework**: [LangChain](https://www.langchain.com/) — Orchestration of retrieval & generation
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- **Embeddings**: Google Generative AI Embeddings (`embedding-001`)
- **Document Parsing**: PyPDF2, python-pptx, python-docx

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/apikey))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/LearnLens-AI.git
cd LearnLens-AI

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### Configuration

You can provide your Gemini API key in two ways:
1. **In the app**: Enter it in the sidebar text input (recommended for quick use)
2. **Environment variable**: Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

---

## 📁 Project Structure

```
LearnLens-AI/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Procfile                    # Render deployment config
├── render.yaml                 # Render blueprint
├── .env.example                # Environment variables template
├── .gitignore
├── .streamlit/
│   └── config.toml             # Streamlit theme & server config
├── utils/
│   ├── __init__.py
│   ├── document_loader.py      # PDF, PPTX, DOCX, TXT extraction
│   ├── text_splitter.py        # Document chunking
│   ├── embeddings.py           # Google AI embeddings & FAISS
│   ├── rag_chain.py            # RAG chains (chat, explain, summary, exam)
│   └── quiz_generator.py       # MCQ quiz generation
└── README.md
```

---

## 📖 Usage Guide

1. **Enter API Key** — Paste your Google Gemini API key in the sidebar
2. **Upload Documents** — Upload one or more PDF, PPTX, DOCX, or TXT files
3. **Process** — Click "Process Documents" to create the vector index
4. **Chat** — Ask questions in the Chat tab and get answers with citations
5. **Summary & Notes** — Generate summaries, exam notes, or simplified explanations
6. **Quiz** — Pick a topic, set difficulty, and take an auto-generated quiz
7. **Search** — Use semantic search in the Documents tab

---

## 🌐 Deployment

### Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy
4. Add `GOOGLE_API_KEY` in Streamlit secrets

### Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Render will auto-detect the `render.yaml` or use the Procfile
5. Add `GOOGLE_API_KEY` as an environment variable

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Built with ❤️ using Streamlit, LangChain & Google Gemini</p>
