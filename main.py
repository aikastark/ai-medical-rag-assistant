from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import faiss
import numpy as np

from fastapi.responses import FileResponse

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Question(BaseModel):
    question: str

# --- Loading data ---
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [c.strip() for c in text.split("\n") if c.strip()]

# --- embeddings ---
embeddings = []
for chunk in chunks:
    if chunk.strip():
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(emb.data[0].embedding)

# --- FAISS ---
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# --- API ---
@app.post("/ask")
def ask_ai(q: Question):
    # embedding question
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=q.question
    ).data[0].embedding

    # search
    D, I = index.search(np.array([q_emb]), k=2)

    context = "\n".join([chunks[i] for i in I[0]])

    # answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
            """
            You are a medical assistant.

            Rules:
            - Answer ONLY using the provided context
            - If answer is not in context, say: "I don't know based on the provided information"
            - Be concise and clear
            """
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q.question}"}
        ]
    )

    return {"answer": response.choices[0].message.content}

@app.get("/")
def read_root():
    return FileResponse("index.html")