from pydantic import BaseModel
from typing import Optional

class Query(BaseModel):
    question: str
    top_k: Optional[int] = 1
