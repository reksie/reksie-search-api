import os
import unkey

from dotenv import load_dotenv
from pydantic import BaseModel, root_validator
from typing import  Any, Dict, Optional
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


class UpsertInput(BaseModel):
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
    query: str
    top_k: Optional[int] = None
    alpha: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class PromptInput(BaseModel):
    query: str
    stream: Optional[bool] = False


def key_extractor(*args: Any, **kwargs: Any) -> Optional[str]:
    if isinstance(auth := kwargs.get("authorization"), str):
        return auth.split(" ")[-1]

    return None

def generate_stream(response):
    for event in response:
        if hasattr(event.choices[0].delta, 'content'):
            current_response = event.choices[0].delta.content
            yield f"data: {current_response}\n\n"

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

    chunks = await split_text(input.content)
    metadata_dict = input.metadata if input.metadata else {}

    vectors = []

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

    index.upsert(vectors, namespace=owner_id)

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
    top_k = input.top_k if input.top_k else 3
    alpha = input.alpha if input.alpha else 0.8

    dense_vector, sparse_vector = await create_embeddings(
        query=input.query, alpha=alpha
    )

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
        query=input.query, alpha=0.8
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

    completion = openai.chat.completions.create(model="gpt-4o", messages=[
        {"role": "system", "content": "Use the following content to answer the prompt"},
        {"role": "system", "content": " ".join([match.metadata["content"] for match in matches if match.metadata and "content" in match.metadata])},
        {"role": "user", "content": input.query}
    ], stream=input.stream)

    if input.stream:
        return StreamingResponse(generate_stream(completion), media_type="text/event-stream")

    return {"completion": completion.choices[0].message.content}
