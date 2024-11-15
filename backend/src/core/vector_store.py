import pinecone
from .config import config
from .embeddings import get_embeddings

pinecone.init(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENVIRONMENT)
index = pinecone.Index(config.PINECONE_INDEX)

def store_embeddings(text, metadata):
    embeddings = get_embeddings(text)
    index.upsert([(metadata['id'], embeddings, metadata)])

def search_embeddings(query):
    query_embeddings = get_embeddings(query)
    results = index.query(query_embeddings, top_k=5, include_metadata=True)
    return results['matches']
