import os
import unkey

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import documents

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(documents.router, prefix="/documents")


@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}
