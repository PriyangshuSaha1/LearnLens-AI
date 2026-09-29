<div align="center">
  
  # 🎓 LearnLens AI
  ### A RAG-Powered Personalized Study Assistant
  
  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://learnlens-ai-cshwmbrfareb3oajdzstan.streamlit.app/)
  
</div>

---

## 📖 Overview

**LearnLens AI** is an advanced Retrieval-Augmented Generation (RAG) study assistant designed to help students interact with their study materials intelligently. By uploading textbooks, lecture notes, PDFs, or presentations, students can instantly generate accurate summaries, exam-oriented notes, and automated quizzes—all grounded in their specific course material. 

The application utilizes **Google's Gemini 3.8 Flash** model for high-speed, accurate generation, and local **HuggingFace** models for secure, fast vector embeddings.

---

## ✨ Key Features

- **📄 Multi-Format Document Support**: Upload multiple `.pdf`, `.pptx`, `.docx`, and `.txt` files simultaneously.
- **💬 Conversational Chat (RAG)**: Ask complex questions about your documents and receive answers with precise source citations (page/slide numbers).
- **📝 Automated Summarization**: Generate comprehensive, well-structured summaries of uploaded materials.
- **🎯 Exam Notes Generator**: Extract key definitions, formulas, and concepts into rapid-revision study notes.
- **💡 "ELI5" Mode**: Break down complex academic concepts into simple, easy-to-understand explanations using analogies.
- **🧠 Intelligent Quiz Generator**: Automatically generate interactive Multiple Choice Question (MCQ) quizzes based on any topic found in your documents, complete with difficulty settings and automated grading.
- **🎨 Premium UI/UX**: Built with a sleek, glassmorphism-inspired dark theme, interactive animations, and responsive design.

---

## 🛠️ Architecture & Tech Stack

LearnLens AI is built on a modern, robust AI stack:

* **Frontend**: [Streamlit](https://streamlit.io/) (with custom CSS/HTML injection for premium styling)
* **LLM Engine**: [Google Gemini 3.8 Flash](https://ai.google.dev/) (via `langchain-google-genai`)
* **Embeddings**: Local HuggingFace `all-MiniLM-L6-v2` (via `sentence-transformers`) - *chosen for zero API latency and rate-limit immunity.*
* **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
* **Orchestration**: [LangChain](https://www.langchain.com/)

---

## 🚀 Live Demo

You can try the live application here:  
**👉 [LearnLens AI - Streamlit Cloud](https://learnlens-ai-cshwmbrfareb3oajdzstan.streamlit.app/)**

*(Note: You will need a free Google Gemini API key to use the application).*

---

## 💻 Local Installation

To run this project locally on your machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/PriyangshuSaha1/LearnLens-AI.git
cd LearnLens-AI
```

### 2. Set up a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 🔑 Configuration

To utilize the generative features, you must provide a Google Gemini API Key. 
1. Get a free API key from **[Google AI Studio](https://aistudio.google.com/apikey)**.
2. You can input the key directly into the secure sidebar of the web app.
3. Alternatively, create a `.env` file in the root directory for local development:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
   ```

---

## 📁 Project Structure

```text
LearnLens-AI/
├── app.py                      # Main Streamlit UI & Application Logic
├── requirements.txt            # Project dependencies
├── utils/
│   ├── document_loader.py      # Parses PDF, PPTX, DOCX, and TXT files
│   ├── text_splitter.py        # LangChain RecursiveCharacterTextSplitter
│   ├── embeddings.py           # FAISS vector store & HuggingFace embeddings
│   ├── rag_chain.py            # Custom prompt templates & Gemini 3.8 logic
│   └── quiz_generator.py       # JSON-structured Gemini quiz generation
└── README.md                   # Project documentation
```

---
<div align="center">
  <p><i>Built for educational purposes.</i></p>
</div>
