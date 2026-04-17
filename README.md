# 🧠 AI Medical RAG Assistant

A simple AI-powered medical assistant that answers questions based on custom documents using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- Ask medical questions via API or web interface
- Uses OpenAI LLM for responses
- Retrieves relevant information from custom data
- Prevents hallucinations by answering only from context
- Simple chat UI (HTML + JS)

## 🛠 Tech Stack

- Python
- FastAPI
- OpenAI API
- FAISS (vector search)
- HTML / JavaScript

## 🧩 How it works

1. Text data is loaded from `data.txt`
2. Text is converted into embeddings
3. FAISS is used to search relevant chunks
4. Context is passed to the LLM
5. AI generates answer based only on context

## ▶️ Run locally

```bash
git clone <your-repo>
cd ai-medical-rag-assistant

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

uvicorn main:app --reload

Open in browser:

http://127.0.0.1:8000/

📌 Example questions
What is gingivitis?
What are symptoms of flu?

⚠️ Note
This is a demo project and not intended for real medical use.