from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import json
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn') 
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text.strip()

with open("./faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

faq_questions = [normalize_text(f["question"]) for f in faqs]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def get_answer(query: Query):
    user_question = normalize_text(query.question)
    query_embedding = model.encode(user_question, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    if best_score < 0.5:
        return {"answer": "No encontré una respuesta adecuada.", "score": best_score}

    return {
        "question": faqs[best_idx]["question"],
        "answer": faqs[best_idx]["answer"],
        "score": best_score
    }
