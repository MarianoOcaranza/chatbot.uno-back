from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from utils import normalize_text
import json

with open("./faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

faq_questions, faq_answers, faq_links = [], [], []

for f in faqs:
    variants = [f["question"]] + f.get("alternatives", [])
    for q in variants:
        faq_questions.append(normalize_text(q))
        faq_answers.append(f["answer"])
        faq_links.append(f.get("link"))

model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

@app.post("/query")
def ask_question(query: Query):
    user_question = normalize_text(query.question)
    query_embedding = model.encode(user_question, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    threshold = 0.55 
    if best_score < threshold:
        top_k = min(5, len(scores))
        best_indices = scores.topk(top_k).indices.tolist()
        suggestions = [faq_questions[i] for i in best_indices]

        return {
            "answer": "No encontré una respuesta exacta, pero quizás estas preguntas sean útiles:",
            "suggestions": suggestions,
            "link": "https://www.uno.edu.ar/"
        }

    return {
        "question": faq_questions[best_idx],
        "answer": faq_answers[best_idx],
        "link": faq_links[best_idx],
        "score": best_score
    }
