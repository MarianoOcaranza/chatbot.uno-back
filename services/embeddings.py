from sentence_transformers import SentenceTransformer, util


embedding_model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")

def encode_questions(questions):
    return embedding_model.encode(questions, convert_to_tensor=True)


def get_best_match(query, faq_embeddings, faq_questions, threshold):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    return scores, best_idx, best_score


