import os
import unkey

from pydantic import BaseModel, Extra, root_validator, ValidationError
from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, Body, Header
from pinecone import Pinecone, ServerlessSpec, QueryResponse
from ..dependencies import create_embeddings, split_text

router = APIRouter()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])


class Match(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]]


class Metadata(BaseModel):
    id: str

    class Config:
        extra = Extra.allow


class UpsertInput(BaseModel):
    content: str
    metadata: Optional[Metadata]


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


def key_extractor(*args: Any, **kwargs: Any) -> Optional[str]:
    if isinstance(auth := kwargs.get("authorization"), str):
        return auth.split(" ")[-1]

    return None


@router.post("/upsert", tags=["documents"])
# @unkey.protected(os.environ["UNKEY_API_ID"], key_extractor)
async def read_documents(
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
    metadata_dict = input.metadata.dict() if input.metadata else {}

    vectors = []

    for chunk_index, chunk in enumerate(chunks):
        dense_vector, sparse_vector = await create_embeddings(chunk.page_content)
        vectors.append(
            {
                "id": f"{owner_id}#{metadata_dict.get('id')}#chunk_{chunk_index}",
                "values": dense_vector,
                "sparse_values": sparse_vector,
                "metadata": metadata_dict,
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
