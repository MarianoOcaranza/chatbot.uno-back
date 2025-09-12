import unicodedata

def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn') 
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text.strip()
