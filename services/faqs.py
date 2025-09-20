import json
from utils.utils import normalize_text


def load_faqs(path='data/faqs.json'):
    with open(path, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    faq_questions, faq_answers, faq_links = [], [], []

    for f in faqs:
        variants = [f["question"]] + f.get("alternatives", [])
        for q in variants:
            faq_questions.append(normalize_text(q))
            faq_answers.append(f["answer"])
            faq_links.append(f.get("link"))

    return faq_questions, faq_answers, faq_links
