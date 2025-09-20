from fastapi import APIRouter
from models.query import Query
from services import faqs, embeddings, llm
from config.config import THRESHOLD, MIN_THRESHOLD
from utils.utils import normalize_text

faq_questions, faq_answers, faq_links = faqs.load_faqs()
faq_embeddings = embeddings.encode_questions(faq_questions)


router = APIRouter()

@router.post("/query")
def ask_question(query: Query):

    user_question = normalize_text(query.question)

    scores, best_idx, best_score = embeddings.get_best_match(
        user_question, faq_embeddings, faq_questions, THRESHOLD
    )

    #Caso 1: no hay respuesta coincidente
    if best_score < MIN_THRESHOLD:
        top_k = min(5, len(scores))
        best_indices = scores.topk(top_k).indices.tolist()
        suggestions = [faq_questions[i] for i in best_indices]
        return {
            "answer": "Che... no encontré una respuesta exacta a tu pregunta :(, pero capaz que quisiste preguntarme:",
            "suggestions": suggestions,
            "link": "https://www.uno.edu.ar/"
        }
    
    #Caso 2: respuesta altamente coincidente
    if best_score >= THRESHOLD:
        top_indices = [best_idx]
    #Caso 3: respuestas debilmente coincidentes
    else:
        k = min(query.top_k or 2, len(scores))  # hasta 2 respuestas si hay duda
        top_indices = scores.topk(k).indices.tolist()

    retrieved_contexts = [faq_answers[i] for i in top_indices]
    context_text = "\n\n---\n\n".join(retrieved_contexts)

    prompt = f"""
        Sos un agente que se dedica a responder dudas para la carrera de Licenciatura en Informática de la Universidad Nacional del Oeste.
        Instrucciones:
        - Responde en español argentino, tono informal y cordial.
        - No incluyas saludos ni links en la respuesta.
        - Máximo 200 palabras.
        - Si no hay info suficiente: "No tengo suficiente información para responder".

        Pregunta del usuario:
        {query.question}

        Contexto relevante:
        {context_text}
    """

    response_text = llm.generate_response(prompt)

    #Se devuelve el JSON para ser procesado en el frontend
    return {
        "answer": response_text,
        "link": faq_links[best_idx],
    }
