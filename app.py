from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from utils import normalize_text
from google import genai
import json

#---PRECARGA---
#Cliente de Google AI Studio
client = genai.Client()

#cargo las FAQS
with open("./faqs.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

#Aca voy a guardar las preguntas, las respuestas y los links
faq_questions, faq_answers, faq_links = [], [], []

#en questions guardo las preguntas y sus variantes o alternativas
#en answers guardo las respuestas
#en links guardo los links si es que los hay
for f in faqs:
    variants = [f["question"]] + f.get("alternatives", [])
    for q in variants:
        faq_questions.append(normalize_text(q))
        faq_answers.append(f["answer"])
        faq_links.append(f.get("link"))

#Una vez que tengo mis preguntas y mis respuestas guardadas, cargo el modelo de embeddings
embedding_model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
#Aca guardo las FAQS ya transformadas en tensor con el embedder
faq_embeddings = embedding_model.encode(faq_questions, convert_to_tensor=True)


#--INICIO LA APP DE FAST API--
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"]
)

#Defino la interfaz de mi consulta, que tendra una pregunta y un top_k modificable para establecer cuantos contextos quiero recuperar
class Query(BaseModel):
    question: str
    top_k: Optional[int] = 1

#Este va a ser mi unico endpoint, que recibira y procesara la consulta del usuario
@app.post("/query")
def ask_question(query: Query):

    #Normalizo el texto y lo paso a tensor con el modelo de embeddings
    user_question = normalize_text(query.question)
    query_embedding = embedding_model.encode(user_question, convert_to_tensor=True)

    #Obtengo los scores usando similitud coseno entre la query y todas las FAQS, asimismo guardo el mejor de todos los scores
    scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    #Defino un umbral minimo, si es menor, entonces mi top_k van a ser 5 como máximo (pueden ser menos si el array de scores es mas pequeño)
    threshold = 0.55 
    if best_score < threshold:
        top_k = min(5, len(scores))
        #Obtengo las preguntas mas cercanas a la del usuario que puedan usarse como sugerencia
        best_indices = scores.topk(top_k).indices.tolist()
        suggestions = [faq_questions[i] for i in best_indices]
        #Devuelvo una respuesta estandar de que no se encontro una respuesta exacta, con algunas sugerencias segun el score de la similitud coseno
        return {
            "answer": "Che... no encontré una respuesta exacta a tu pregunta :(, pero capaz que quisiste preguntarme:",
            "suggestions": suggestions,
            "link": "https://www.uno.edu.ar/"
        }
    
    #Si efectivamente se encuentra una respuesta (porque el score mas alto dio mayor a 0.55), paso a este bloque
    #Me traigo la mejor respuesta si el score es mayor a 0.7, sino, me traigo hasta dos respuestas como contexto
    if best_score >= 0.7:
        top_indices = [best_idx]  # solo la mejor
    else:
        k = min(query.top_k or 2, len(scores))  # hasta 2 respuestas si hay duda
        top_indices = scores.topk(k).indices.tolist()


    retrieved_contexts = [faq_answers[i] for i in top_indices]
    #El contexto obtenido es guardado en un texto plano
    context_text = "\n\n---\n\n".join(retrieved_contexts)

    #Prompt para generar una respuesta
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

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    #Se devuelve el JSON para ser procesado en el frontend
    return {
        "answer": response.text,
        "link": faq_links[best_idx],
    }
