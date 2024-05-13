from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import SpladeEncoder
from pinecone_text.dense import SentenceTransformerEncoder
from langchain_text_splitters import CharacterTextSplitter

splade = SpladeEncoder()
sentence_transformer = SentenceTransformerEncoder(
    "sentence-transformers/all-MiniLM-L6-v2"
)
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


async def create_embeddings(query: str, alpha: float = 0.8):
    sparse_vector = splade.encode_queries(query)
    dense_vector = sentence_transformer.encode_queries(query)

    return hybrid_convex_scale(dense_vector, sparse_vector, alpha)


async def split_text(query: str):
    return text_splitter.create_documents([query])
