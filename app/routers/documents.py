import os
from openai.types.chat import ChatCompletionChunk
import unkey
import json
import numpy as np

from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, root_validator
from typing import  Any, List, Dict, Optional
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pinecone import Pinecone
from openai import OpenAI
from ..dependencies import create_embeddings, split_text

load_dotenv()

router = APIRouter()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
openai = OpenAI()

class Match(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]]

class DocumentType(str, Enum):
    RAG = 'rag'
    HYBRID = 'hybrid'

class UpsertInput(BaseModel):
    type: List[DocumentType]
    content: str
    metadata: Optional[Dict[str, Any]]


class DeleteInput(BaseModel):
    ids: Optional[list[str]] = None
    all: Optional[bool] = None

    @root_validator(pre=True)
    def check_ids_or_all(cls, values):
        ids, all_flag = values.get("ids"), values.get("all")
        if ids is not None and all_flag is not None:
            raise ValueError("ids and all cannot be provided together.")
        if ids is None and all_flag is None:
            raise ValueError("ids or all must be provided.")
        return values


class QueryInput(BaseModel):
    type: DocumentType
    query: str
    top_k: Optional[int] = None
    alpha: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class PromptInput(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    stream: Optional[bool] = False
    model: Optional[str] = "gpt-3.5-turbo"


def key_extractor(*args: Any, **kwargs: Any) -> Optional[str]:
    if isinstance(auth := kwargs.get("authorization"), str):
        return auth.split(" ")[-1]

    return None

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ChatCompletionChunk):
            # Assuming 'ChatCompletionChunk' has a method to_dict() for serialization.
            # If not, you'll have to manually construct a dictionary that represents the object.
            return o.to_dict()
        # Fall back to the super class's default handler for other types
        return super().default(o)

def generate_stream(response):
    for event in response:
        event_json = json.dumps(event, cls=EnhancedJSONEncoder)
        yield f"data: {event_json}\n\n"

@router.post("/upsert", tags=["documents"])
# @unkey.protected(os.environ["UNKEY_API_ID"], key_extractor)
async def upsert_documents(
    input: UpsertInput,
    authorization: str = Header(None),
    unkey_verification: Any = None,
):
    # assert isinstance(unkey_verification, unkey.ApiKeyVerification)
    # assert unkey_verification.valid
    # owner_id = unkey_verification.owner_id
    owner_id = "user_2f4XwosSCKSJ7sS99MOXEayr47q"
    index = pc.Index("reksie-search-dev")
    hybrid_index = pc.Index("reksie-search-dev-hybrid")

    chunks = await split_text(input.content)
    metadata_dict = input.metadata if input.metadata else {}

    vectors = []
    hybrid_vectors = []

    # We will always chunk up the content and store it in the RAG index
    for chunk_index, chunk in enumerate(chunks):
        dense_vector, sparse_vector = await create_embeddings(chunk.page_content)
        vectors.append(
            {
                "id": f"{owner_id}#{metadata_dict.get('id')}#chunk_{chunk_index}",
                "values": dense_vector,
                "sparse_values": sparse_vector,
                "metadata": {
                    **metadata_dict,
                    "content": chunk.page_content,
                },
            }
        )

    # If the document type is hybrid, we will also store the full content in the hybrid index
    # This is useful for when we want to retrieve the full content of the document
    if DocumentType.HYBRID in input.type:
        response = openai.embeddings.create(input=input.content, model="text-embedding-3-small")
        dense_vector = response.data[0].embedding
        _, sparse_vector = await create_embeddings(input.content)
        hybrid_vectors.append(
            {
                "id": f"{owner_id}#{metadata_dict.get('id')}",
                "values": dense_vector,
                "sparse_values": sparse_vector,
                "metadata": {
                    **metadata_dict,
                    "content": input.content,
                },
            }
        )

    # We need to look into calling these upserts in parallel
    index.upsert(vectors, namespace=owner_id)
    hybrid_index.upsert(hybrid_vectors, namespace=owner_id)

    return {"status": "success"}


@router.post("/delete", tags=["documents"])
# @unkey.protected(os.environ["UNKEY_API_ID"], key_extractor)
async def delete_documents(
    input: DeleteInput,
    authorization: str = Header(None),
    unkey_verification: Any = None,
):
    # assert isinstance(unkey_verification, unkey.ApiKeyVerification)
    # assert unkey_verification.valid
    # owner_id = unkey_verification.owner_id
    owner_id = "user_2f4XwosSCKSJ7sS99MOXEayr47q"
    index = pc.Index("reksie-search-dev")

    if input.all:
        index.delete(delete_all=True, namespace=owner_id)

    if input.ids:
        index.delete(ids=input.ids, namespace=owner_id)

    return {"status": "success"}


@router.post("/query", tags=["documents"])
# @unkey.protected(os.environ["UNKEY_API_ID"], key_extractor)
async def query_documents(
    input: QueryInput, authorization: str = Header(None), unkey_verification: Any = None
):
    # assert isinstance(unkey_verification, unkey.ApiKeyVerification)
    # assert unkey_verification.valid
    # owner_id = unkey_verification.owner_id
    owner_id = "user_2f4XwosSCKSJ7sS99MOXEayr47q"
    index = pc.Index("reksie-search-dev")

    if input.type == DocumentType.HYBRID:
        index = pc.Index("reksie-search-dev-hybrid")

    top_k = input.top_k if input.top_k else 3
    alpha = input.alpha if input.alpha else 0.8

    dense_vector, sparse_vector = await create_embeddings(
        query=input.query, alpha=alpha
    )

    if input.type == DocumentType.HYBRID:
        response = openai.embeddings.create(input=input.query, model="text-embedding-3-small")
        dense_vector = response.data[0].embedding

    results = index.query(
        namespace=owner_id,
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True,
    )

    matches = [
        Match(id=match["id"], score=match["score"], metadata=match.get("metadata"))
        for match in results.get("matches", [])
    ]

    return {"matches": matches}

@router.post("/completion", tags=["documents"])
# @unkey.protected(os.environ["UNKEY_API_ID"], key_extractor)
async def prompt_completion(
    input: PromptInput, authorization: str = Header(None), unkey_verification: Any = None
):
    # assert isinstance(unkey_verification, unkey.ApiKeyVerification)
    # assert unkey_verification.valid
    # owner_id = unkey_verification.owner_id
    owner_id = "user_2f4XwosSCKSJ7sS99MOXEayr47q"
    index = pc.Index("reksie-search-dev")

    dense_vector, sparse_vector = await create_embeddings(
        query=input.prompt, alpha=0.8
    )

    results = index.query(
        namespace=owner_id,
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=3,
        include_metadata=True,
    )

    matches = [
        Match(id=match["id"], score=match["score"], metadata=match.get("metadata"))
        for match in results.get("matches", [])
    ]

    system_prompt = input.system_prompt if input.system_prompt else "Do your best to give an explanation or answer the following question based on the user prompt given and the information provided in the system prompt."

    completion = openai.chat.completions.create(model=input.model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": " ".join([match.metadata["content"] for match in matches if match.metadata and "content" in match.metadata])},
        {"role": "user", "content": input.prompt}
    ], stream=input.stream)

    if input.stream:
        return StreamingResponse(generate_stream(completion), media_type="text/event-stream")


    return {"completion": completion}
