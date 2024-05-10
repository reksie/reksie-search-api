import os
import unkey

from dotenv import load_dotenv
from typing import Annotated, Any, Dict, Optional
from fastapi import Body, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pinecone_text.sparse import BM25Encoder 

load_dotenv()

app = FastAPI()
bm25 = BM25Encoder().default()

origins = [
  '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def key_extractor(*args: Any, **kwargs: Any) -> Optional[str]:
    if isinstance(auth := kwargs.get("authorization"), str):
        return auth.split(" ")[-1]

    return None

class Document(BaseModel):
    content: str | None = Field(
        default=None, title="Content to embed"
    )

@app.get("/")
async def read_root():
    return {"status": "Okay!"}

@app.get("/healthcheck")
def read_root():
    return {"status": "ok"}

@app.post("/embeddings")
@unkey.protected(os.environ["UNKEY_API_ID"], key_extractor)
async def protected_route(
    document: Annotated[Document, Body(embed=True)],
    authorization: str = Header(None),
    unkey_verification: Any = None,
):
    assert isinstance(unkey_verification, unkey.ApiKeyVerification)
    assert unkey_verification.valid

    doc_sparse_vector = bm25.encode_documents(document.content)

    return doc_sparse_vector