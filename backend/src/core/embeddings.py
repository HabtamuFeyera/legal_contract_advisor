from langchain.embeddings import OpenAIEmbeddings
from .config import config

def get_embeddings(text):
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=config.OPENAI_API_KEY)
    return embedding_model.embed(text)
