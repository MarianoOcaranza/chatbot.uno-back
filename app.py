from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(query.router)